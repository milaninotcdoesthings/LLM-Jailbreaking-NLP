"""
compute_rerun_metrics.py - NLP metrics for the multilingual_rerun corpus.

Reuses the exact same functions/models as metrics_computation.py (phase 1-2),
applied to the 22,257-prompt corpus behind multilingual_rerun_results.csv.
This is a distinct corpus from native_multi_metrics.parquet (verified: zero
exact prompt overlap), so metrics must be computed fresh rather than joined.

Prompts are deduplicated before scoring (same prompt appears twice, once per
target model) — metrics depend only on the prompt text, not on the target.

Output: multilingual_rerun/data/rerun_metrics.parquet
        (join back to multilingual_rerun_results.csv on `prompt`)
"""

import gc
import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

ROOT = Path("/Users/tommasomilanino/Developer/THESIS")
HERE = ROOT / "multilingual_rerun"
IN = HERE / "results" / "multilingual_rerun_results.csv"
OUT = HERE / "data" / "rerun_metrics.parquet"
device = "mps" if torch.backends.mps.is_available() else "cpu"


def free_memory():
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def save_checkpoint(df, phase):
    p = HERE / "data" / f"ckpt_metrics_{phase}.parquet"
    df.to_parquet(p, engine="pyarrow")
    print(f"  saved: {p.name}")


# ── load + deduplicate ───────────────────────────────────────────────
rerun = pd.read_csv(IN)
print(f"rerun rows (both targets): {len(rerun)}")

df = rerun[["prompt", "language"]].drop_duplicates(subset="prompt").reset_index(drop=True)
df["raw_prompt"] = df["prompt"]
print(f"unique prompts to score: {len(df)}")
print(df.language.value_counts().to_string())

# ── PHASE 1: Token Length ───────────────────────────────────────────
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

# ── PHASE 2: Semantic Embeddings ────────────────────────────────────
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

# ── PHASE 3: Zero-Shot Intent ───────────────────────────────────────
try:
    print("\n=== PHASE 3: Intent ===")
    clf = pipeline("zero-shot-classification",
                   model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=device)
    labels = ["factual", "procedural", "opinion", "causal"]

    def get_intent(t):
        try:
            return clf(str(t)[:512], labels, multi_label=False)["labels"][0]
        except Exception:
            return None

    df["intent_category"] = df["raw_prompt"].progress_apply(get_intent)
    del clf
    free_memory()
    save_checkpoint(df, 3)
except Exception as e:
    print(f"Phase 3 failed: {e}")

# ── PHASE 4: Toxicity ───────────────────────────────────────────────
try:
    print("\n=== PHASE 4: Toxicity ===")
    tox = pipeline("text-classification",
                   model="martin-ha/toxic-comment-model", device=device)

    def get_toxicity(t):
        try:
            for dd in tox(str(t)[:512], top_k=None):
                if dd["label"].lower() == "toxic":
                    return dd["score"]
            return 0.0
        except Exception:
            return None

    df["toxicity_score"] = df["raw_prompt"].progress_apply(get_toxicity)
    del tox
    free_memory()
    save_checkpoint(df, 4)
except Exception as e:
    print(f"Phase 4 failed: {e}")

# ── PHASE 5: Language-Specific Sentiment ────────────────────────────
try:
    print("\n=== PHASE 5: Sentiment ===")
    sentiment_models = {
        "arabic": "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
        "german": "oliverguhr/german-sentiment-bert",
        "russian": "BlancheFort/rubert-base-cased-sentiment",
    }
    df["sentiment_score"] = None
    df["sentiment_label"] = None

    for lang, model_name in sentiment_models.items():
        mask = df["language"] == lang
        if mask.sum() == 0:
            continue
        print(f"  -> {lang}: {model_name}")
        sp = pipeline("text-classification", model=model_name, device=device)

        def apply_sent(t):
            try:
                r = sp(str(t)[:512])
                return pd.Series([r[0]["score"], r[0]["label"].lower()])
            except Exception:
                return pd.Series([None, None])

        res = df.loc[mask, "raw_prompt"].progress_apply(apply_sent)
        df.loc[mask, "sentiment_score"] = res.iloc[:, 0].values
        df.loc[mask, "sentiment_label"] = res.iloc[:, 1].values
        del sp
        free_memory()

    # spanish: pysentimiento (same as Chapter 1), not the multilingual pipeline
    mask = df["language"] == "spanish"
    if mask.sum():
        print("  -> spanish: pysentimiento")
        from pysentimiento import create_analyzer
        an = create_analyzer(task="sentiment", lang="es")

        def apply_sent_es(t):
            try:
                r = an.predict(str(t)[:512])
                # pysentimiento returns .output in {POS, NEG, NEU} and .probas dict
                label = r.output.lower()
                score = r.probas[r.output]
                return pd.Series([score, label])
            except Exception:
                return pd.Series([None, None])

        res = df.loc[mask, "raw_prompt"].progress_apply(apply_sent_es)
        df.loc[mask, "sentiment_score"] = res.iloc[:, 0].values
        df.loc[mask, "sentiment_label"] = res.iloc[:, 1].values
        del an
        free_memory()

    save_checkpoint(df, 5)
except Exception as e:
    print(f"Phase 5 failed: {e}")

# ── PHASE 6: Perplexity ─────────────────────────────────────────────
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

# ── save + report ───────────────────────────────────────────────────
df.to_parquet(OUT, engine="pyarrow")
print(f"\ndone -> {OUT}  ({len(df)} unique prompts)")

cols = ["prompt_token_length", "toxicity_score", "perplexity_score"]
print(df[[c for c in cols if c in df]].describe().round(2).to_string())
print("\nmissing values:")
print(df[[c for c in cols if c in df]].isna().sum().to_string())