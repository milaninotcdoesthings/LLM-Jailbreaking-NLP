"""
13_attack_extract_llama8b.py
=============================
Attacco a Llama 3.1 8B Instruct con estrazione contestuale degli hidden states.
Versione ottimizzata per throughput: batching, ordinamento per lunghezza,
sharding opzionale su più pod.

TRE OTTIMIZZAZIONI RISPETTO ALLA VERSIONE INGENUA (~60h stimate):

1. BATCHING. Processare un prompt alla volta lascia la GPU in attesa fra un
   token e l'altro. Con BATCH_SIZE sequenze in parallelo la scheda lavora
   davvero. Guadagno ~4-6x.

2. ORDINAMENTO PER LUNGHEZZA. In un batch tutte le sequenze vengono riempite
   fino alla più lunga; mescolando prompt da 20 e da 300 token si calcola
   soprattutto padding. Ordinando prima, ogni batch è omogeneo. Guadagno
   ~1.5-2x, gratis.

3. SHARDING. Se un solo pod non basta, si avviano N pod identici cambiando
   solo SHARD_INDEX: ogni pod prende 1/N del pool. Speedup lineare, nessuna
   configurazione distribuita. I file di output si concatenano alla fine.

COSA NON È STATO TOCCATO. MAX_NEW_TOKENS resta 512, il budget HarmBench
standard. È il parametro che il Capitolo 1 identifica come il singolo fattore
che più altera la misura: sotto quella soglia le risposte vengono troncate
prima che rifiuto o compliance emergano, e Mazeika et al. mostrano che l'ASR
misurata può spostarsi di 30 punti. Ridurlo per guadagnare tempo
reintrodurrebbe proprio il confound che la campagna corretta ha eliminato.

Gli hidden states restano estratti dal forward pass sul PROMPT, prima della
generazione — è lo stato del modello nel momento della decisione, ed è anche
la parte computazionalmente economica (nessuna generazione autoregressiva).

Output: llama8b_responses.parquet + llama8b_hidden_states.h5
"""

import os
import json
import time

import numpy as np
import pandas as pd
import h5py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Configurazione ────────────────────────────────────────────────────────────

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

DATA_DIR   = "data"
INPUT_FILE = os.path.join(DATA_DIR, "prompt_pool_sampled.parquet")

# Sharding: per usare più pod, avviarli tutti con lo stesso SHARD_COUNT e
# SHARD_INDEX diverso (0, 1, 2...). Con un solo pod lasciare i default.
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))

suffix = f"_shard{SHARD_INDEX}" if SHARD_COUNT > 1 else ""
RESP_FILE   = os.path.join(DATA_DIR, f"llama8b_responses{suffix}.parquet")
HIDDEN_FILE = os.path.join(DATA_DIR, f"llama8b_hidden_states{suffix}.h5")

MAX_NEW_TOKENS = 512     # budget HarmBench standard — non ridurre
BATCH_SIZE     = 48      # A6000 48GB: il modello ne usa ~16, resta margine
LAST_N_TOKENS  = 32      # token del prompt su cui fare mean-pooling
MAX_PROMPT_TOKENS = 1024 # tronca prompt patologicamente lunghi
CHECKPOINT_EVERY_BATCHES = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise SystemExit("Nessuna GPU rilevata. Controlla il pod prima di procedere.")


# ── Caricamento modello ───────────────────────────────────────────────────────

def load_model():
    print(f"Caricamento {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # I modelli decoder-only richiedono padding a sinistra per la generazione
    # batched: il token successivo deve trovarsi in fondo alla sequenza. Ha
    # anche un effetto collaterale utile qui — gli ultimi LAST_N_TOKENS della
    # sequenza sono sempre token reali, mai padding, quindi il mean-pooling
    # non ha bisogno di mascheramento aggiuntivo.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",     # più veloce dell'attenzione eager
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    n_layers = model.config.num_hidden_layers
    layer_indices = sorted(set([
        n_layers // 4,
        n_layers // 2,
        (3 * n_layers) // 4,
        n_layers - 1,
    ]))
    print(f"  {n_layers} layer, hidden dim {model.config.hidden_size}")
    print(f"  layer estratti: {layer_indices} "
          f"({[f'{i/(n_layers-1)*100:.0f}%' for i in layer_indices]})")

    return tokenizer, model, layer_indices, n_layers


# ── Elaborazione di un batch ──────────────────────────────────────────────────

def process_batch(prompts: list[str], tokenizer, model,
                  layer_indices: list[int]) -> list[dict]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": str(p)}],
            tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    encoded = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_PROMPT_TOKENS,
    ).to(device)

    # ── Passo 1: forward pass sul prompt ─────────────────────────────────────
    # Restituisce sia gli hidden states sia i logit dell'ultima posizione, che
    # predicono il primo token generato. num_logits_to_keep=1 evita di
    # materializzare i logit per tutta la sequenza: con vocabolario da 128k
    # sarebbero gigabyte di memoria sprecata.
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            num_logits_to_keep=1,
        )

    batch_hidden = []
    for b in range(len(prompts)):
        vectors = {}
        for layer_idx in layer_indices:
            # hidden_states[0] è l'embedding layer, da qui il +1
            layer_hidden = outputs.hidden_states[layer_idx + 1][b]
            pooled = layer_hidden[-LAST_N_TOKENS:].mean(dim=0)
            vectors[layer_idx] = pooled.float().cpu().numpy()
        batch_hidden.append(vectors)

    # Feature comportamentali sul primo token generato, ricavate dai logit già
    # calcolati qui. La versione precedente le estraeva da generate() con
    # output_scores=True, che conserva i logit di TUTTI i 512 passi: diversi GB
    # di allocazione per batch, per usarne venti. Il primo token è comunque il
    # più informativo per rifiuto/compliance, dato che è lì che la decisione si
    # manifesta ("I" di "I can't" contro qualunque altro incipit).
    first_logits = outputs.logits[:, -1, :].float()
    probs = torch.softmax(first_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).cpu().numpy()
    top2 = torch.topk(probs, k=2, dim=-1).values.cpu().numpy()
    top1_prob = top2[:, 0]
    prob_gap = top2[:, 0] - top2[:, 1]

    del outputs, first_logits, probs
    torch.cuda.empty_cache()

    # ── Passo 2: generazione batched ─────────────────────────────────────────
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    results = []

    for b in range(len(prompts)):
        generated_ids = gen_output[b][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        real_len = (generated_ids != tokenizer.pad_token_id).sum().item()

        results.append({
            "response": response,
            "finish_reason": "length" if real_len >= MAX_NEW_TOKENS else "stop",
            "n_generated_tokens": int(real_len),
            "first_token_entropy": float(entropy[b]),
            "first_token_top1_prob": float(top1_prob[b]),
            "first_token_prob_gap": float(prob_gap[b]),
            "hidden_vectors": batch_hidden[b],
        })

    del gen_output
    torch.cuda.empty_cache()

    return results


# ── HDF5 ──────────────────────────────────────────────────────────────────────

def get_processed_ids(h5_path: str) -> set:
    if not os.path.exists(h5_path):
        return set()
    with h5py.File(h5_path, "r") as f:
        return set(f.keys())


def save_hidden_batch(h5_path: str, prompt_ids: list[str], vectors_list: list[dict]):
    """Un'apertura per batch invece che per riga — l'I/O su HDF5 ha overhead."""
    with h5py.File(h5_path, "a") as f:
        for pid, vectors in zip(prompt_ids, vectors_list):
            if pid in f:
                continue
            grp = f.create_group(pid)
            for layer_idx, vec in vectors.items():
                grp.create_dataset(f"layer_{layer_idx}", data=vec,
                                   compression="gzip", compression_opts=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)
    print(f"Pool completo: {len(df)} prompt")

    if SHARD_COUNT > 1:
        df = df.iloc[SHARD_INDEX::SHARD_COUNT].reset_index(drop=True)
        print(f"Shard {SHARD_INDEX}/{SHARD_COUNT}: {len(df)} prompt assegnati")

    tokenizer, model, layer_indices, n_layers = load_model()

    # Ripresa
    processed = get_processed_ids(HIDDEN_FILE)
    if os.path.exists(RESP_FILE):
        existing = pd.read_parquet(RESP_FILE)
        processed &= set(existing["prompt_id"])
    else:
        existing = pd.DataFrame()

    remaining = df[~df["prompt_id"].isin(processed)].copy()
    print(f"Da processare: {len(remaining)} (già fatti: {len(processed)})")

    if len(remaining) == 0:
        print("Nulla da fare.")
        return

    # Ordinamento per lunghezza: batch omogenei, padding minimo. È la
    # differenza fra calcolare token reali e calcolare riempimento.
    if "token_length_native" in remaining.columns:
        remaining = remaining.sort_values("token_length_native")
    else:
        remaining["_len"] = remaining["text_native"].str.len()
        remaining = remaining.sort_values("_len")
    remaining = remaining.reset_index(drop=True)

    new_rows = []
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()

    pbar = tqdm(range(n_batches), desc="Attacco", unit="batch")
    for i in pbar:
        chunk = remaining.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        prompt_ids = chunk["prompt_id"].tolist()

        try:
            results = process_batch(
                chunk["text_native"].tolist(), tokenizer, model, layer_indices)

            save_hidden_batch(HIDDEN_FILE, prompt_ids,
                              [r["hidden_vectors"] for r in results])

            for pid, r in zip(prompt_ids, results):
                new_rows.append({
                    "prompt_id": pid,
                    "response": r["response"],
                    "finish_reason": r["finish_reason"],
                    "n_generated_tokens": r["n_generated_tokens"],
                    "first_token_entropy": r["first_token_entropy"],
                    "first_token_top1_prob": r["first_token_top1_prob"],
                    "first_token_prob_gap": r["first_token_prob_gap"],
                })

        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"  OOM sul batch {i} — riduci BATCH_SIZE e riavvia "
                       f"(la ripresa è automatica)")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            tqdm.write(f"  errore sul batch {i}: {e}")
            for pid in prompt_ids:
                new_rows.append({
                    "prompt_id": pid, "response": None,
                    "finish_reason": "error", "n_generated_tokens": None,
                    "first_token_entropy": None, "first_token_top1_prob": None,
                    "first_token_prob_gap": None,
                })

        done = len(new_rows)
        if done:
            rate = done / (time.time() - t_start)
            eta_h = (len(remaining) - done) / rate / 3600 if rate > 0 else 0
            pbar.set_postfix({"prompt/s": f"{rate:.2f}", "ETA": f"{eta_h:.1f}h"})

        if (i + 1) % CHECKPOINT_EVERY_BATCHES == 0:
            pd.concat([existing, pd.DataFrame(new_rows)],
                      ignore_index=True).to_parquet(RESP_FILE, engine="pyarrow")

    pbar.close()

    final = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    final.to_parquet(RESP_FILE, engine="pyarrow")

    with open(os.path.join(DATA_DIR, f"llama8b_geometry{suffix}.json"), "w") as f:
        json.dump({"model": MODEL_ID, "n_layers": n_layers,
                   "layer_indices": layer_indices,
                   "hidden_dim": model.config.hidden_size,
                   "max_new_tokens": MAX_NEW_TOKENS,
                   "last_n_tokens_pooled": LAST_N_TOKENS}, f, indent=2)

    elapsed = (time.time() - t_start) / 3600
    print(f"\nCompletato in {elapsed:.1f}h — {len(final)} risposte")
    print(f"  {RESP_FILE}")
    print(f"  {HIDDEN_FILE}")


if __name__ == "__main__":
    main()