"""
14_attack_extract_qwen3b.py
=============================
Attacco a Qwen2.5-3B-Instruct con estrazione contestuale degli hidden states.

Stesso schema esatto del file 13_attack_extract_llama8b.py — stessa logica di
pooling, stesse feature comportamentali, stesso formato di output. Solo il
modello e la geometria dei layer cambiano.

PERCHÉ LO STESSO SCHEMA CONTA. Llama 3.1 8B ha 32 layer da 4096 dimensioni,
Qwen2.5 3B ne ha 36 da 2048. I vettori non sono confrontabili elemento per
elemento — non esiste corrispondenza tra "layer 16 di Llama" e "layer 16 di
Qwen", sono due spazi organizzati indipendentemente. Per questo si addestrano
DUE classificatori separati, uno per modello, mai uno unico sui vettori misti.

Il confronto tra i due avviene non sui vettori ma sui RISULTATI dei due probe,
allineati per PROFONDITÀ RELATIVA (25%/50%/75%/100% del percorso) invece che
per indice di layer assoluto — è l'unico modo per mettere le due curve
sull'asse stesso nonostante l'architettura diversa. Vedi 15_train_probes.py.

Eseguibile sia su RunPod (stesso pod dell'8B, in sequenza) sia in locale su
Mac con MPS — Qwen 3B è abbastanza leggero da girare comodamente anche lì.

Output identico per struttura a quello di Llama:
  - qwen3b_responses.parquet
  - qwen3b_hidden_states.h5
"""

import os
import json

import numpy as np
import pandas as pd
import h5py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Configurazione ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR    = "data"
INPUT_FILE  = os.path.join(DATA_DIR, "prompt_pool_with_metrics.parquet")
RESP_FILE   = os.path.join(DATA_DIR, "qwen3b_responses.parquet")
HIDDEN_FILE = os.path.join(DATA_DIR, "qwen3b_hidden_states.h5")

MAX_NEW_TOKENS   = 512
LAST_N_TOKENS    = 32
ENTROPY_WINDOW   = 20
CHECKPOINT_EVERY = 50

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


# ── Caricamento modello ───────────────────────────────────────────────────────

def load_model():
    print(f"Caricamento {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer_indices = sorted(set([
        n_layers // 4,
        n_layers // 2,
        (3 * n_layers) // 4,
        n_layers - 1,
    ]))
    print(f"Modello caricato: {n_layers} layer totali "
          f"(Llama 8B ne ha 32 — numeri diversi per costruzione, normale)")
    print(f"Layer estratti (profondità relativa): {layer_indices} "
          f"({[f'{i/(n_layers-1)*100:.0f}%' for i in layer_indices]})")

    return tokenizer, model, layer_indices, n_layers


# ── Estrazione per singolo prompt ─────────────────────────────────────────────

def process_one(prompt_text: str, tokenizer, model, layer_indices: list[int]) -> dict:
    messages = [{"role": "user", "content": str(prompt_text)}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_vectors = {}
    for layer_idx in layer_indices:
        layer_hidden = outputs.hidden_states[layer_idx + 1][0]
        pooled = layer_hidden[-LAST_N_TOKENS:].mean(dim=0)
        hidden_vectors[layer_idx] = pooled.float().cpu().numpy()

    del outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    with torch.no_grad():
        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = gen_output.sequences[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    entropies, top1_probs, gaps = [], [], []
    for step_scores in gen_output.scores[:ENTROPY_WINDOW]:
        probs = torch.softmax(step_scores[0].float(), dim=-1)
        top_probs, _ = torch.topk(probs, k=5)
        top_probs = top_probs.cpu().numpy()

        entropy = -np.sum(probs.cpu().numpy() * np.log(probs.cpu().numpy() + 1e-12))
        entropies.append(entropy)
        top1_probs.append(top_probs[0])
        if len(top_probs) > 1:
            gaps.append(top_probs[0] - top_probs[1])

    return {
        "response": response_text,
        "finish_reason": "stop" if len(generated_ids) < MAX_NEW_TOKENS else "length",
        "entropy_first_tokens": float(np.mean(entropies)) if entropies else None,
        "mean_top1_prob": float(np.mean(top1_probs)) if top1_probs else None,
        "mean_prob_gap": float(np.mean(gaps)) if gaps else None,
        "hidden_vectors": hidden_vectors,
    }


# ── Gestione HDF5 (identico a Llama) ──────────────────────────────────────────

def get_processed_ids(h5_path: str) -> set:
    if not os.path.exists(h5_path):
        return set()
    with h5py.File(h5_path, "r") as f:
        return set(f.keys())


def save_hidden_vectors(h5_path: str, prompt_id: str, vectors: dict):
    with h5py.File(h5_path, "a") as f:
        if prompt_id in f:
            return
        grp = f.create_group(prompt_id)
        for layer_idx, vec in vectors.items():
            grp.create_dataset(f"layer_{layer_idx}", data=vec, compression="gzip")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)
    print(f"Pool caricato: {len(df)} prompt")

    tokenizer, model, layer_indices, n_layers = load_model()

    processed_hidden = get_processed_ids(HIDDEN_FILE)
    if os.path.exists(RESP_FILE):
        existing_resp = pd.read_parquet(RESP_FILE)
        processed_resp = set(existing_resp["prompt_id"])
    else:
        existing_resp = pd.DataFrame()
        processed_resp = set()

    already_done = processed_hidden & processed_resp
    remaining = df[~df["prompt_id"].isin(already_done)]
    print(f"Da processare: {len(remaining)} / {len(df)} "
          f"(già fatti: {len(already_done)})")

    new_responses = []

    for i, row in tqdm(remaining.iterrows(), total=len(remaining), desc="Attacco Qwen"):
        try:
            result = process_one(row["text_native"], tokenizer, model, layer_indices)
            save_hidden_vectors(HIDDEN_FILE, row["prompt_id"], result["hidden_vectors"])

            new_responses.append({
                "prompt_id": row["prompt_id"],
                "response": result["response"],
                "finish_reason": result["finish_reason"],
                "entropy_first_tokens": result["entropy_first_tokens"],
                "mean_top1_prob": result["mean_top1_prob"],
                "mean_prob_gap": result["mean_prob_gap"],
            })
        except Exception as e:
            tqdm.write(f"Errore su {row['prompt_id']}: {e}")
            new_responses.append({
                "prompt_id": row["prompt_id"], "response": None,
                "finish_reason": "error", "entropy_first_tokens": None,
                "mean_top1_prob": None, "mean_prob_gap": None,
            })

        if len(new_responses) % CHECKPOINT_EVERY == 0:
            combined = pd.concat(
                [existing_resp, pd.DataFrame(new_responses)], ignore_index=True)
            combined.to_parquet(RESP_FILE, engine="pyarrow")
            tqdm.write(f"  checkpoint: {len(combined)} risposte salvate")

    combined = pd.concat(
        [existing_resp, pd.DataFrame(new_responses)], ignore_index=True)
    combined.to_parquet(RESP_FILE, engine="pyarrow")

    # Metadati di geometria — servono al passo di confronto per allineare le
    # profondità relative tra i due modelli.
    with open(os.path.join(DATA_DIR, "qwen3b_geometry.json"), "w") as f:
        json.dump({"n_layers": n_layers, "layer_indices": layer_indices,
                   "hidden_dim": model.config.hidden_size}, f, indent=2)

    print(f"\nCompletato: {len(combined)} risposte salvate in {RESP_FILE}")
    print(f"Hidden states salvati in {HIDDEN_FILE}")


if __name__ == "__main__":
    main()