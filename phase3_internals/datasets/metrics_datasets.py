"""
step2_metrics.py - metrics on AdvBench / XSTest / Do-Not-Answer.

Same models as metrics_computation.py (phase 1-2), so values are comparable.
Adds Flesch readability via textstat.

Input:  benchmarks_raw.csv   (from step0_download.py)
Output: benchmarks_metrics.parquet

pip install textstat sentence-transformers
"""

import gc
import pandas as pd
import torch
import textstat
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

BASE = Path("/Users/tommasomilanino/Developer/THESIS")
IN = BASE / "benchmarks_raw.csv"
OUT = BASE / "benchmarks_metrics.parquet"
device = "mps" if torch.backends.mps.is_available() else "cpu"


def free_memory():
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def save_checkpoint(df, phase):
    p = BASE / f"ckpt_bm_{phase}.parquet"
    df.to_parquet(p, engine="pyarrow")
    print(f"saved: {p.name}")


df = pd.read_csv(IN)
df["raw_prompt"] = df["text"].astype(str)
df["language"] = "english"          # all three benchmarks are English
print(f"loaded {len(df)} prompts | device: {device}")

# ── PHASE 1: Token Length ───────────────────────────────────────────────
try:
    print("\n=== PHASE 1: Token Length ===")
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    df["prompt_token_length"] = df["raw_prompt"].progress_apply(
        lambda x: len(tk.encode(str(x)))
    )
    del tk
    free_memory()
    save_checkpoint(df, 1)
except Exception as e:
    print(f"Phase 1 failed: {e}")

# ── PHASE 2: Semantic Embeddings ────────────────────────────────────────
try:
    print("\n=== PHASE 2: Embeddings ===")
    emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    df["semantic_embedding"] = list(
        emb.encode(df["raw_prompt"].tolist(), show_progress_bar=True, batch_size=32)
    )
    del emb
    free_memory()
    save_checkpoint(df, 2)
except Exception as e:
    print(f"Phase 2 failed: {e}")

# ── PHASE 3: Zero-Shot Intent ───────────────────────────────────────────
try:
    print("\n=== PHASE 3: Intent ===")
    clf = pipeline("zero-shot-classification",
                   model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=device)
    labels = ["factual", "procedural", "opinion", "causal"]

    def get_intent(t):
        try:
            return clf(str(t), labels, multi_label=False)["labels"][0]
        except Exception:
            return None

    df["intent_category"] = df["raw_prompt"].progress_apply(get_intent)
    del clf
    free_memory()
    save_checkpoint(df, 3)
except Exception as e:
    print(f"Phase 3 failed: {e}")

# ── PHASE 4: Toxicity ───────────────────────────────────────────────────
try:
    print("\n=== PHASE 4: Toxicity ===")
    tox = pipeline("text-classification",
                   model="martin-ha/toxic-comment-model", device=device)

    def get_toxicity(t):
        try:
            for d in tox(str(t)[:512], top_k=None):
                if d["label"].lower() == "toxic":
                    return d["score"]
            return 0.0
        except Exception:
            return None

    df["toxicity_score"] = df["raw_prompt"].progress_apply(get_toxicity)
    del tox
    free_memory()
    save_checkpoint(df, 4)
except Exception as e:
    print(f"Phase 4 failed: {e}")

# ── PHASE 5: Sentiment (English model, as in phase 1-2) ─────────────────
try:
    print("\n=== PHASE 5: Sentiment ===")
    sent = pipeline("text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device)

    def get_sentiment(t):
        try:
            r = sent(str(t)[:512])
            return pd.Series([r[0]["score"], r[0]["label"].lower()])
        except Exception:
            return pd.Series([None, None])

    res = df["raw_prompt"].progress_apply(get_sentiment)
    df["sentiment_score"] = res.iloc[:, 0].values
    df["sentiment_label"] = res.iloc[:, 1].values
    del sent
    free_memory()
    save_checkpoint(df, 5)
except Exception as e:
    print(f"Phase 5 failed: {e}")

# ── PHASE 6: Perplexity ─────────────────────────────────────────────────
try:
    print("\n=== PHASE 6: Perplexity ===")
    pid = "Qwen/Qwen2.5-0.5B"
    ptk = AutoTokenizer.from_pretrained(pid)
    pm = AutoModelForCausalLM.from_pretrained(pid, torch_dtype=torch.float16).to(device)

    def perplexity(t):
        try:
            inp = ptk(str(t), return_tensors="pt", truncation=True,
                      max_length=512).to(device)
            with torch.no_grad():
                o = pm(**inp, labels=inp["input_ids"])
            return torch.exp(o.loss).item()
        except Exception:
            return None

    df["perplexity_score"] = df["raw_prompt"].progress_apply(perplexity)
    del pm, ptk
    free_memory()
    save_checkpoint(df, 6)
except Exception as e:
    print(f"Phase 6 failed: {e}")

# ── PHASE 7: Readability (new) ──────────────────────────────────────────
try:
    print("\n=== PHASE 7: Readability ===")
    df["flesch_reading_ease"] = df["raw_prompt"].progress_apply(
        lambda x: textstat.flesch_reading_ease(str(x))
    )
    df["flesch_kincaid_grade"] = df["raw_prompt"].progress_apply(
        lambda x: textstat.flesch_kincaid_grade(str(x))
    )
    save_checkpoint(df, 7)
except Exception as e:
    print(f"Phase 7 failed: {e}")

df.to_parquet(OUT, engine="pyarrow")
print(f"\ndone -> {OUT}  ({len(df)} rows)")

cols = ["prompt_token_length", "toxicity_score", "sentiment_score",
        "perplexity_score", "flesch_reading_ease"]
print(df[[c for c in cols if c in df]].describe().round(2).to_string())
print("\nmissing values:")
print(df[[c for c in cols if c in df]].isna().sum().to_string())
