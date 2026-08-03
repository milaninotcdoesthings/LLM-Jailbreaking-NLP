"""
06_attack_groq.py
==================
Attacks Llama 3.1 8B Instant and Llama 3.3 70B Versatile via Groq, on the
prompts in corpus_with_metrics.parquet.

Uses text_native for every row: this is the text the target model actually
receives, in whatever language that row is in. English rows have
text_native == text_source, so nothing special is needed there.

logprobs=True is requested on every call. For the 70B model these are the only
internal-ish signal available (no direct access to hidden states via Groq), so
they are kept as behavioural features: response entropy over the first N
tokens, top-1 probability, and the probability gap between the first and second
token. These are NOT the model's internal representations and should not be
described as such in the writeup — see the entropy_first_tokens etc. columns.

Generation is capped at 512 tokens, matching the corrected budget established
in Chapter 1 (a 20-token cap was shown there to truncate responses before
refusal or compliance substance can appear).

Resumable: rows already carrying a response for a given model are skipped, so
the script can be re-run after a rate-limit failure or interruption without
reprocessing everything. Autosaves every CHUNK_SIZE rows.

Output: ai_regeneration/data/corpus_attacked.parquet
"""

import os
import json
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import AsyncGroq
from tqdm.asyncio import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv("key.env")
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_8B  = "llama-3.1-8b-instant"
MODEL_70B = "llama-3.3-70b-versatile"

MAX_TOKENS_GENERATI = 512      # HarmBench-standard budget (Chapter 1, Section 1.1.2)
CONCURRENCY_LIMIT   = 15
CHUNK_SIZE           = 100
TOP_LOGPROBS         = 5
ENTROPY_WINDOW       = 20      # first N tokens used for entropy/confidence features
MAX_RETRIES           = 4

DATA_DIR    = "data"
INPUT_FILE  = os.path.join(DATA_DIR, "corpus_with_metrics.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "corpus_attacked.parquet")


# ── Logprob feature extraction ─────────────────────────────────────────────────

def extract_logprob_features(logprobs_content) -> dict:
    """
    Behavioural features from Groq's logprobs.content. These describe how
    confident/uncertain the model was while generating — a proxy signal, not an
    internal representation. Computed over the first ENTROPY_WINDOW tokens,
    where the refusal-vs-compliance decision is made.
    """
    if not logprobs_content:
        return {
            "entropy_first_tokens": None,
            "mean_top1_prob": None,
            "mean_prob_gap": None,
            "n_tokens_scored": 0,
        }

    window = logprobs_content[:ENTROPY_WINDOW]
    entropies, top1_probs, gaps = [], [], []

    for token_info in window:
        top_lps = token_info.top_logprobs or []
        if not top_lps:
            continue
        probs = np.exp([t.logprob for t in top_lps])
        probs = probs / probs.sum()  # renormalise over the observed top-k

        entropy = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(entropy)
        top1_probs.append(probs[0])
        if len(probs) > 1:
            gaps.append(probs[0] - probs[1])

    return {
        "entropy_first_tokens": float(np.mean(entropies)) if entropies else None,
        "mean_top1_prob": float(np.mean(top1_probs)) if top1_probs else None,
        "mean_prob_gap": float(np.mean(gaps)) if gaps else None,
        "n_tokens_scored": len(window),
    }


# ── Single-row attack ───────────────────────────────────────────────────────────

async def attack_one(row, model_name: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": str(row["text_native"])}],
                    max_tokens=MAX_TOKENS_GENERATI,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=TOP_LOGPROBS,
                )

                choice = response.choices[0]
                text = choice.message.content or ""

                logprob_feats = extract_logprob_features(
                    choice.logprobs.content if choice.logprobs else None
                )

                return {
                    "prompt_id": row["prompt_id"],
                    "response": text,
                    "finish_reason": choice.finish_reason,
                    "error": None,
                    **logprob_feats,
                }

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "prompt_id": row["prompt_id"],
                        "response": None,
                        "finish_reason": None,
                        "error": str(e),
                        "entropy_first_tokens": None,
                        "mean_top1_prob": None,
                        "mean_prob_gap": None,
                        "n_tokens_scored": 0,
                    }
                await asyncio.sleep(2 ** attempt)  # exponential backoff


# ── Batch driver ─────────────────────────────────────────────────────────────

async def run_attack(df: pd.DataFrame, model_name: str, col_prefix: str) -> pd.DataFrame:
    resp_col = f"{col_prefix}_response"
    err_col  = f"{col_prefix}_error"

    if resp_col not in df.columns:
        df[resp_col] = None
        df[err_col] = None
        for feat in ["entropy_first_tokens", "mean_top1_prob",
                    "mean_prob_gap", "n_tokens_scored"]:
            df[f"{col_prefix}_{feat}"] = None
        df[f"{col_prefix}_finish_reason"] = None

    pending_mask = df[resp_col].isna()
    pending = df[pending_mask]
    print(f"\n{model_name}: {len(pending)} pending / {len(df)} total")

    if len(pending) == 0:
        print("  nothing to do")
        return df

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    for start in range(0, len(pending), CHUNK_SIZE):
        chunk = pending.iloc[start:start + CHUNK_SIZE]
        tasks = [attack_one(row, model_name, semaphore)
                 for _, row in chunk.iterrows()]

        results = await tqdm.gather(
            *tasks, desc=f"  {model_name} [{start}:{start+len(chunk)}]")

        for res in results:
            idx = df.index[df["prompt_id"] == res["prompt_id"]][0]
            df.at[idx, resp_col] = res["response"]
            df.at[idx, err_col] = res["error"]
            df.at[idx, f"{col_prefix}_finish_reason"] = res["finish_reason"]
            df.at[idx, f"{col_prefix}_entropy_first_tokens"] = res["entropy_first_tokens"]
            df.at[idx, f"{col_prefix}_mean_top1_prob"] = res["mean_top1_prob"]
            df.at[idx, f"{col_prefix}_mean_prob_gap"] = res["mean_prob_gap"]
            df.at[idx, f"{col_prefix}_n_tokens_scored"] = res["n_tokens_scored"]

        df.to_parquet(OUTPUT_FILE, engine="pyarrow")
        n_errors = sum(1 for r in results if r["error"])
        print(f"    saved checkpoint — {n_errors} errors in this chunk")

    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("ATTACK REPORT")
    print("=" * 62)

    for prefix, model in [("llama8b", MODEL_8B), ("llama70b", MODEL_70B)]:
        resp_col = f"{prefix}_response"
        err_col  = f"{prefix}_error"
        if resp_col not in df.columns:
            continue

        n_ok = df[resp_col].notna().sum()
        n_err = df[err_col].notna().sum()
        n_empty = (df[resp_col] == "").sum()

        print(f"\n{model}")
        print(f"  successful responses  {n_ok}")
        print(f"  errors                {n_err}")
        print(f"  empty responses       {n_empty}")

        finish = df[f"{prefix}_finish_reason"].value_counts()
        if len(finish):
            print(f"  finish reasons:")
            for reason, n in finish.items():
                print(f"    {reason:<12} {n}")

        entropy_col = f"{prefix}_entropy_first_tokens"
        if entropy_col in df.columns:
            e = df[entropy_col].dropna()
            if len(e):
                print(f"  entropy (first {ENTROPY_WINDOW} tok): "
                      f"median {e.median():.3f}, range "
                      f"{e.min():.3f}–{e.max():.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"{INPUT_FILE} not found — run 05_compute_metrics.py first.")

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} prompts")
    print(f"Languages: {df['language'].value_counts().to_dict()}")

    if os.path.exists(OUTPUT_FILE):
        print(f"\nResuming from existing {OUTPUT_FILE}")
        existing = pd.read_parquet(OUTPUT_FILE)
        # merge any new metric columns from the fresh corpus, keep attack progress
        new_cols = [c for c in df.columns if c not in existing.columns]
        if new_cols:
            df = existing.merge(df[["prompt_id"] + new_cols], on="prompt_id", how="left")
        else:
            df = existing

    df = await run_attack(df, MODEL_8B, "llama8b")
    df = await run_attack(df, MODEL_70B, "llama70b")

    df.to_parquet(OUTPUT_FILE, engine="pyarrow")
    report(df)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())