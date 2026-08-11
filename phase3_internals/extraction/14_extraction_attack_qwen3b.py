"""
14_attack_extract_qwen3b.py
===========================
Attacks Qwen2.5-3B-Instruct with contextual hidden-state extraction.

Same structure as the Llama script: batched, length-sorted, resumable, hidden
states pooled over the last 32 prompt tokens before generation.

Why a second model. On its own the Llama result is a fact about one model. If
the depth curve has the same shape on a different architecture, trained by a
different lab on different data, it becomes a regularity. The layer counts
differ (Qwen 36, Llama 32), so the curves are compared on relative depth, not
layer index — which is what layer_pct in the results is for.

SUBSAMPLE. The learning curve showed both targets saturate by ~30% of the data,
so the full 19,539 rows are not needed here. N_SAMPLE keeps the run to about an
hour of GPU while staying well above the saturation point. Sampling is by
source_prompt_id so language variants stay together, matching the split logic
used downstream.

Output: qwen3b_responses.parquet + qwen3b_hidden_states.h5
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

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR   = "data"
INPUT_FILE = os.path.join(DATA_DIR, "prompt_pool_sampled.parquet")

RESP_FILE     = os.path.join(DATA_DIR, "qwen3b_responses.parquet")
HIDDEN_FILE   = os.path.join(DATA_DIR, "qwen3b_hidden_states.h5")
GEOMETRY_FILE = os.path.join(DATA_DIR, "qwen3b_geometry.json")
SAMPLE_FILE   = os.path.join(DATA_DIR, "qwen3b_sample_ids.csv")

N_SAMPLE = 9000          # set to None to run the full pool
RANDOM_SEED = 42

MAX_NEW_TOKENS = 512
BATCH_SIZE     = 48      # 3B model, plenty of headroom on a 48GB card
LAST_N_TOKENS  = 32
MAX_PROMPT_TOKENS = 1024
CHECKPOINT_EVERY_BATCHES = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise SystemExit("No GPU detected. Check the pod before continuing.")


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    n_layers = model.config.num_hidden_layers
    layer_indices = sorted(set([
        n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1,
    ]))
    print(f"  {n_layers} layers, hidden dim {model.config.hidden_size}")
    print(f"  extracted layers: {layer_indices} "
          f"({[f'{i/(n_layers-1)*100:.0f}%' for i in layer_indices]})")
    print(f"  Llama has 32 layers / 4096 dims — different geometry by design, "
          f"compared on relative depth")

    return tokenizer, model, layer_indices, n_layers


# ── Batch processing ──────────────────────────────────────────────────────────

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
            layer_hidden = outputs.hidden_states[layer_idx + 1][b]
            pooled = layer_hidden[-LAST_N_TOKENS:].mean(dim=0)
            vectors[layer_idx] = pooled.float().cpu().numpy()
        batch_hidden.append(vectors)

    first_logits = outputs.logits[:, -1, :].float()
    probs = torch.softmax(first_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).cpu().numpy()
    top2 = torch.topk(probs, k=2, dim=-1).values.cpu().numpy()
    top1_prob, prob_gap = top2[:, 0], top2[:, 0] - top2[:, 1]

    del outputs, first_logits, probs
    torch.cuda.empty_cache()

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

def get_processed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def save_hidden_batch(path: str, prompt_ids: list[str], vectors_list: list[dict]):
    with h5py.File(path, "a") as f:
        for pid, vectors in zip(prompt_ids, vectors_list):
            if pid in f:
                continue
            grp = f.create_group(pid)
            for layer_idx, vec in vectors.items():
                grp.create_dataset(f"layer_{layer_idx}", data=vec,
                                   compression="gzip", compression_opts=4)


# ── Sampling ──────────────────────────────────────────────────────────────────

def subsample(df: pd.DataFrame) -> pd.DataFrame:
    """Sampled by group so language variants of one prompt stay together."""
    if N_SAMPLE is None or len(df) <= N_SAMPLE:
        return df

    if os.path.exists(SAMPLE_FILE):
        keep = pd.read_csv(SAMPLE_FILE)["prompt_id"]
        out = df[df["prompt_id"].isin(keep)].reset_index(drop=True)
        print(f"Reusing saved sample: {len(out)} rows")
        return out

    rng = np.random.default_rng(RANDOM_SEED)
    groups = df["source_prompt_id"].unique()
    rng.shuffle(groups)

    sizes = df.groupby("source_prompt_id").size()
    chosen, total = [], 0
    for g in groups:
        if total >= N_SAMPLE:
            break
        chosen.append(g)
        total += sizes[g]

    out = df[df["source_prompt_id"].isin(chosen)].reset_index(drop=True)
    out[["prompt_id"]].to_csv(SAMPLE_FILE, index=False)

    print(f"Subsampled: {len(out)} of {len(df)} rows "
          f"({len(chosen)} groups)")
    print(out["language"].value_counts().to_string())
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)
    print(f"Pool: {len(df)} rows")
    df = subsample(df)

    tokenizer, model, layer_indices, n_layers = load_model()

    processed = get_processed_ids(HIDDEN_FILE)
    if os.path.exists(RESP_FILE):
        existing = pd.read_parquet(RESP_FILE)
        processed &= set(existing["prompt_id"])
    else:
        existing = pd.DataFrame()

    remaining = df[~df["prompt_id"].isin(processed)].copy()
    print(f"\nTo process: {len(remaining)} (done: {len(processed)})")

    if len(remaining) == 0:
        print("Nothing to do.")
        return

    if "token_length_native" in remaining.columns:
        remaining = remaining.sort_values("token_length_native")
    else:
        remaining["_len"] = remaining["text_native"].str.len()
        remaining = remaining.sort_values("_len")
    remaining = remaining.reset_index(drop=True)

    new_rows = []
    n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()

    pbar = tqdm(range(n_batches), desc="Qwen attack", unit="batch")
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
            tqdm.write(f"  OOM at batch {i} — lower BATCH_SIZE and rerun "
                       f"(resume is automatic)")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            tqdm.write(f"  error at batch {i}: {e}")
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
            eta_h = (len(remaining) - done) / rate / 3600 if rate else 0
            pbar.set_postfix({"prompt/s": f"{rate:.2f}", "ETA": f"{eta_h:.1f}h"})

        if (i + 1) % CHECKPOINT_EVERY_BATCHES == 0:
            pd.concat([existing, pd.DataFrame(new_rows)],
                      ignore_index=True).to_parquet(RESP_FILE, engine="pyarrow")

    pbar.close()

    final = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    final.to_parquet(RESP_FILE, engine="pyarrow")

    with open(GEOMETRY_FILE, "w") as f:
        json.dump({"model": MODEL_ID, "n_layers": n_layers,
                   "layer_indices": layer_indices,
                   "hidden_dim": model.config.hidden_size,
                   "max_new_tokens": MAX_NEW_TOKENS,
                   "last_n_tokens_pooled": LAST_N_TOKENS,
                   "n_sampled": len(df)}, f, indent=2)

    print(f"\nDone in {(time.time() - t_start)/3600:.1f}h — {len(final)} responses")
    print(f"  {RESP_FILE}")
    print(f"  {HIDDEN_FILE}")
    print(f"  {GEOMETRY_FILE}")


if __name__ == "__main__":
    main()