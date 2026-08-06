"""
03_deduplicate_merge.py
=======================
Merges the two generator arms into a single English corpus, applying
cross-generator deduplication.

The individual generation scripts deduplicated only against their own output,
since neither had visibility of the other. Qwen and Mistral can independently
converge on similar framings, so a corpus-level pass is required before the
corpus is used for anything.

Optionally folds in the 1,395 unique prompts retained from the Chapter 2 corpus.

Output: ai_regeneration/data/corpus_english.csv
"""

import os
import re
import json
import hashlib
import unicodedata

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = "data"

QWEN_FILE    = os.path.join(DATA_DIR, "qwen_generated.csv")
MISTRAL_FILE = os.path.join(DATA_DIR, "mistral_generated.csv")

# Chapter 2 survivors. Set to None to skip.
LEGACY_FILE   = os.path.join(DATA_DIR, "chapter2_unique.csv")
LEGACY_COLUMN = "raw_prompt"     # column holding the prompt text in that file

OUTPUT_FILE = os.path.join(DATA_DIR, "corpus_english.csv")
STATS_FILE  = os.path.join(DATA_DIR, "merge_stats.json")

SIM_THRESHOLD = 0.92
BATCH_SIZE    = 128

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def prompt_id(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def load_arm(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  {label}: file not found at {path} — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  {label}: {len(df)} prompts")
    return df


def load_legacy() -> pd.DataFrame:
    """Chapter 2 prompts, reshaped into the current schema."""
    if LEGACY_FILE is None or not os.path.exists(LEGACY_FILE):
        print(f"  legacy: not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(LEGACY_FILE)
    if LEGACY_COLUMN not in df.columns:
        print(f"  legacy: column '{LEGACY_COLUMN}' absent "
              f"(found: {list(df.columns)}) — skipping")
        return pd.DataFrame()

    out = pd.DataFrame({
        "prompt_id":    df[LEGACY_COLUMN].apply(prompt_id),
        "prompt_text":  df[LEGACY_COLUMN],
        "source":       "ai_generated_chapter2",
        "language":     "english",
        "category":     df["category"] if "category" in df.columns else "unknown",
        "persona_hint": None,
        "framing_hint": None,
        "generator":    "qwen2.5:latest",
        "timestamp":    df["timestamp"] if "timestamp" in df.columns else None,
    })
    print(f"  legacy (Chapter 2): {len(out)} prompts")
    return out


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Two passes. Exact match is vectorised and cheap, so it runs first and
    shrinks the input to the expensive semantic pass.
    """
    stats = {"input": len(df)}

    # Pass 1 — exact
    df = df.copy()
    df["_norm"] = df["prompt_text"].apply(normalise)
    before = len(df)
    df = df.drop_duplicates(subset="_norm", keep="first").reset_index(drop=True)
    stats["removed_exact"] = before - len(df)
    print(f"\nExact duplicates removed: {stats['removed_exact']}")

    # Pass 2 — semantic
    print(f"Embedding {len(df)} prompts for semantic deduplication...")
    embedder = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device=device)
    embeddings = embedder.encode(
        df["prompt_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,      # lets cosine reduce to a dot product
    )

    keep_idx = []
    kept_embeddings = []

    for i in tqdm(range(len(df)), desc="Semantic pass", unit="prompt"):
        emb = embeddings[i]
        if kept_embeddings:
            sims = np.vstack(kept_embeddings) @ emb
            if sims.max() > SIM_THRESHOLD:
                continue
        keep_idx.append(i)
        kept_embeddings.append(emb)

    stats["removed_semantic"] = len(df) - len(keep_idx)
    print(f"Near-duplicates removed: {stats['removed_semantic']}")

    df = df.iloc[keep_idx].drop(columns="_norm").reset_index(drop=True)
    stats["output"] = len(df)
    return df, stats


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, stats: dict) -> None:
    print("\n" + "=" * 62)
    print("MERGE REPORT")
    print("=" * 62)
    print(f"Input prompts              {stats['input']}")
    print(f"Removed — exact duplicate  {stats['removed_exact']}")
    print(f"Removed — near-duplicate   {stats['removed_semantic']}")
    print(f"Final corpus               {stats['output']}")

    total_removed = stats["removed_exact"] + stats["removed_semantic"]
    if stats["input"]:
        print(f"Cross-corpus overlap       "
              f"{total_removed / stats['input'] * 100:.1f}%")

    print(f"\nBy source:")
    for src, n in df["source"].value_counts().items():
        print(f"  {n:>5}  {src}")

    print(f"\nBy category:")
    for cat, n in df["category"].value_counts().sort_index().items():
        print(f"  {n:>5}  {cat}")

    lengths = df["prompt_text"].str.split().str.len()
    print(f"\nPrompt length (words): median {lengths.median():.0f}, "
          f"range {lengths.min()}–{lengths.max()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading generator arms...")
    frames = [
        load_arm(QWEN_FILE, "qwen"),
        load_arm(MISTRAL_FILE, "mistral"),
        load_legacy(),
    ]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise SystemExit("No input files found — nothing to merge.")

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(merged)} prompts before deduplication")

    final, stats = deduplicate(merged)

    # Regenerate ids so they are consistent across the whole corpus
    final["prompt_id"] = final["prompt_text"].apply(prompt_id)

    final.to_csv(OUTPUT_FILE, index=False)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    report(final, stats)
    print(f"\nSaved to {OUTPUT_FILE}")