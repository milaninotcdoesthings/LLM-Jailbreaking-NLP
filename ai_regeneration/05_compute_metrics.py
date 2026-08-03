"""
05_compute_metrics.py
=====================
Computes NLP metrics over the English and multilingual corpora.

Two design decisions carry over from the limitations documented in Chapters 1–2.

DUAL MEASUREMENT. Every metric is computed twice: once on the prompt in its own
language (suffix _native) and once on its English source (suffix _source). The
corpus is parallel, so every translated prompt has an English original. Native
values describe the text the target model actually receives; source values are
measured with a single instrument across every row and are therefore comparable
across languages without per-language correction. For English rows the two
coincide. Chapter 1 could not do this because its prompts arrived pre-translated.

NORMALISATION. Chapter 1 established that raw cross-language comparison of these
metrics is unsound: the toxicity classifier's output is governed largely by
language rather than content, and perplexity from a small scorer principally
ranks how well that scorer models each language. Perplexity is therefore also
reported normalised against a per-language reference median.

Two further corrections relative to the Chapter 1 pipeline:
  - toxicity extracts p(toxic) rather than the winning label's score, which is
    what produced the spurious 0.95–1.00 concentration in Figure 9
  - sentiment is reported both from language-specific models (more accurate
    within a language) and from one multilingual model (comparable across them)

Output: ai_regeneration/data/corpus_with_metrics.parquet
"""

import os
import gc
import json

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR      = "data"
ENGLISH_FILE  = os.path.join(DATA_DIR, "corpus_english.csv")
MULTILING_FILE = os.path.join(DATA_DIR, "corpus_multilingual.csv")
OUTPUT_FILE   = os.path.join(DATA_DIR, "corpus_with_metrics.parquet")

# Optional. A parallel corpus of benign prompts, same schema, used as the
# perplexity reference. If absent, the corpus's own per-language median is used
# instead — see normalise_perplexity() for the trade-off.
BASELINE_FILE = os.path.join(DATA_DIR, "corpus_benign_parallel.csv")

CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# Multilingual, WordPiece-based — avoids the sentencepiece/protobuf failure that
# XLM-RoBERTa-based models trigger in this environment.
TOXICITY_MODEL  = "citizenlab/distilbert-base-multilingual-cased-toxicity"
SENTIMENT_MULTI = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
INTENT_MODEL    = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
PERPLEXITY_MODEL = "Qwen/Qwen2.5-0.5B"
TOKENIZER_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

SENTIMENT_MONO = {
    "english": "distilbert-base-uncased-finetuned-sst-2-english",
    "german":  "oliverguhr/german-sentiment-bert",
    "russian": "seara/rubert-tiny2-russian-sentiment",
    "arabic":  "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
}

INTENT_LABELS = ["factual", "procedural", "opinion", "causal"]

MAX_CHARS = 512

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


# ── Infrastructure ────────────────────────────────────────────────────────────

def free_memory():
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def checkpoint(df: pd.DataFrame, phase: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"metrics_{phase}.parquet")
    df.to_parquet(path, engine="pyarrow")
    print(f"  checkpoint saved: {path}")


def load_corpus() -> pd.DataFrame:
    """
    Unifies both corpora into one frame with a text_native / text_source pair.
    """
    frames = []

    if os.path.exists(ENGLISH_FILE):
        en = pd.read_csv(ENGLISH_FILE)
        en["text_native"] = en["prompt_text"]
        en["text_source"] = en["prompt_text"]        # identical for English
        en["source_prompt_id"] = en["prompt_id"]     # its own source
        frames.append(en)
        print(f"English corpus: {len(en)} prompts")

    if os.path.exists(MULTILING_FILE):
        ml = pd.read_csv(MULTILING_FILE)
        ml["text_native"] = ml["prompt_text"]
        ml["text_source"] = ml["prompt_text_en"]
        frames.append(ml)
        print(f"Multilingual corpus: {len(ml)} prompts")

    if not frames:
        raise SystemExit("No corpus files found.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(df)} rows\n")
    return df


# ── Phase 1: token length ─────────────────────────────────────────────────────

def phase_token_length(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 1: Token length ===")
    print("Note: Qwen2.5-7B tokenizer, not either target model's. This is a "
          "consistent relative measure, not the count Llama actually processes.")
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        df["token_length_native"] = df["text_native"].progress_apply(
            lambda x: len(tok.encode(str(x))))
        df["token_length_source"] = df["text_source"].progress_apply(
            lambda x: len(tok.encode(str(x))))
        del tok
        free_memory()
        checkpoint(df, "01_tokens")
    except Exception as e:
        print(f"  FAILED: {e}")
    return df


# ── Phase 2: perplexity ───────────────────────────────────────────────────────

def phase_perplexity(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 2: Perplexity ===")
    try:
        tok = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            PERPLEXITY_MODEL, torch_dtype=torch.float16).to(device)
        model.eval()

        def perplexity(text):
            try:
                inputs = tok(str(text), return_tensors="pt",
                             truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    out = model(**inputs, labels=inputs["input_ids"])
                    return torch.exp(out.loss).item()
            except Exception:
                return None

        df["perplexity_native"] = df["text_native"].progress_apply(perplexity)
        df["perplexity_source"] = df["text_source"].progress_apply(perplexity)

        del model, tok
        free_memory()
        checkpoint(df, "02_perplexity")
    except Exception as e:
        print(f"  FAILED: {e}")
    return df


def normalise_perplexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divides raw perplexity by a per-language reference median, so that 1.0 means
    'typical for this language' in every language.

    The reference is drawn from a benign parallel corpus if one is supplied,
    which is the stronger option because the reference is independent of attack
    content. Failing that, the corpus's own per-language median is used; this
    still removes the scorer's per-language competence offset, but cannot
    separate 'this language is genuinely more perplexing here' from 'the scorer
    models this language poorly', so it should be described as within-corpus
    normalisation rather than as a language baseline.
    """
    print("\n--- Perplexity normalisation ---")

    if os.path.exists(BASELINE_FILE):
        baseline_df = pd.read_csv(BASELINE_FILE)
        if "perplexity_native" in baseline_df.columns:
            refs = baseline_df.groupby("language")["perplexity_native"].median()
            mode = "benign reference corpus"
        else:
            print(f"  {BASELINE_FILE} has no perplexity column — "
                  f"falling back to corpus median")
            refs = df.groupby("language")["perplexity_native"].median()
            mode = "within-corpus median (fallback)"
    else:
        refs = df.groupby("language")["perplexity_native"].median()
        mode = "within-corpus median"

    print(f"  reference: {mode}")
    for lang, val in refs.items():
        print(f"    {lang:<10} {val:.1f}")

    df["perplexity_norm"] = df.apply(
        lambda r: r["perplexity_native"] / refs.get(r["language"], np.nan)
        if pd.notna(r["perplexity_native"]) else None,
        axis=1,
    )
    df.attrs["perplexity_reference_mode"] = mode
    return df


# ── Phase 3: toxicity ─────────────────────────────────────────────────────────

def phase_toxicity(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 3: Toxicity ===")
    print("Extracting p(toxic), not the winning label's score. The latter is "
          "what produced the spurious 0.95–1.00 concentration in Chapter 1.")
    try:
        pipe = pipeline("text-classification", model=TOXICITY_MODEL,
                        device=device, top_k=None)

        def toxicity(text):
            try:
                res = pipe(str(text)[:MAX_CHARS], top_k=None)
                scores = res[0] if isinstance(res[0], list) else res
                for d in scores:
                    if "toxic" in d["label"].lower() and "non" not in d["label"].lower():
                        return d["score"]
                return 0.0
            except Exception:
                return None

        df["toxicity_native"] = df["text_native"].progress_apply(toxicity)
        df["toxicity_source"] = df["text_source"].progress_apply(toxicity)

        del pipe
        free_memory()
        checkpoint(df, "03_toxicity")
    except Exception as e:
        print(f"  FAILED: {e}")
    return df


# ── Phase 4: sentiment ────────────────────────────────────────────────────────

def phase_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 4: Sentiment ===")

    # 4a — one multilingual model, comparable across languages
    print("  4a: multilingual model (cross-language comparable)")
    try:
        pipe = pipeline("text-classification", model=SENTIMENT_MULTI,
                        device=device)

        def sentiment(text):
            try:
                res = pipe(str(text)[:MAX_CHARS])
                return pd.Series([res[0]["score"], res[0]["label"].lower()])
            except Exception:
                return pd.Series([None, None])

        res_native = df["text_native"].progress_apply(sentiment)
        df["sentiment_multi_score"] = res_native.iloc[:, 0].values
        df["sentiment_multi_label"] = res_native.iloc[:, 1].values

        res_source = df["text_source"].progress_apply(sentiment)
        df["sentiment_multi_score_source"] = res_source.iloc[:, 0].values

        del pipe
        free_memory()
    except Exception as e:
        print(f"  4a FAILED: {e}")

    # 4b — language-specific models, more accurate within a language
    print("  4b: language-specific models (within-language accuracy)")
    df["sentiment_mono_score"] = None
    df["sentiment_mono_label"] = None

    for lang, model_id in SENTIMENT_MONO.items():
        mask = df["language"].str.lower() == lang
        if mask.sum() == 0:
            continue
        print(f"    {lang} ({mask.sum()} rows): {model_id}")
        try:
            pipe = pipeline("text-classification", model=model_id, device=device)

            def sentiment(text):
                try:
                    res = pipe(str(text)[:MAX_CHARS])
                    return pd.Series([res[0]["score"], res[0]["label"].lower()])
                except Exception:
                    return pd.Series([None, None])

            res = df.loc[mask, "text_native"].progress_apply(sentiment)
            df.loc[mask, "sentiment_mono_score"] = res.iloc[:, 0].values
            df.loc[mask, "sentiment_mono_label"] = res.iloc[:, 1].values

            del pipe
            free_memory()
        except Exception as e:
            print(f"    {lang} FAILED: {e}")

    checkpoint(df, "04_sentiment")
    return df


# ── Phase 5: readability and complexity ───────────────────────────────────────

def phase_readability(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 5: Readability and complexity ===")

    # Flesch is defined for English only, so it is computed on the source text.
    # This is precisely why it exists for every row: the corpus is parallel.
    try:
        import textstat
        df["flesch_source"] = df["text_source"].progress_apply(
            lambda x: textstat.flesch_reading_ease(str(x)))
        print("  Flesch computed on English source (English-only metric)")
    except ImportError:
        print("  textstat not installed — skipping Flesch. "
              "Install with: pip install textstat")
        df["flesch_source"] = None

    # Language-agnostic complexity proxies, valid in any script.
    def avg_word_length(text):
        words = str(text).split()
        return np.mean([len(w) for w in words]) if words else None

    def type_token_ratio(text):
        words = str(text).split()
        return len(set(words)) / len(words) if words else None

    df["avg_word_length_native"] = df["text_native"].apply(avg_word_length)
    df["ttr_native"] = df["text_native"].apply(type_token_ratio)
    print("  Language-agnostic proxies computed (avg word length, TTR)")

    checkpoint(df, "05_readability")
    return df


# ── Phase 6: intent ───────────────────────────────────────────────────────────

def phase_intent(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 6: Zero-shot intent ===")
    try:
        clf = pipeline("zero-shot-classification", model=INTENT_MODEL,
                       device=device)

        def intent(text):
            try:
                res = clf(str(text)[:MAX_CHARS], INTENT_LABELS, multi_label=False)
                return pd.Series([res["labels"][0], res["scores"][0]])
            except Exception:
                return pd.Series([None, None])

        res_native = df["text_native"].progress_apply(intent)
        df["intent_native"] = res_native.iloc[:, 0].values
        df["intent_confidence"] = res_native.iloc[:, 1].values

        res_source = df["text_source"].progress_apply(intent)
        df["intent_source"] = res_source.iloc[:, 0].values

        del clf
        free_memory()
        checkpoint(df, "06_intent")
    except Exception as e:
        print(f"  FAILED: {e}")
    return df


# ── Phase 7: embeddings ───────────────────────────────────────────────────────

def phase_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PHASE 7: Semantic embeddings ===")
    try:
        embedder = SentenceTransformer(EMBED_MODEL, device=device)
        emb = embedder.encode(
            df["text_native"].astype(str).tolist(),
            batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        df["embedding_native"] = [e.tolist() for e in emb]
        del embedder
        free_memory()
        checkpoint(df, "07_embeddings")
    except Exception as e:
        print(f"  FAILED: {e}")
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("METRICS REPORT")
    print("=" * 62)
    print(f"Rows: {len(df)}")

    print(f"\nCoverage:")
    for col in sorted(c for c in df.columns
                      if any(k in c for k in
                             ["toxicity", "perplexity", "sentiment",
                              "flesch", "intent", "token_length", "ttr"])):
        n = df[col].notna().sum()
        print(f"  {col:<32} {n:>6} / {len(df)}")

    print(f"\nPer-language medians (native):")
    cols = [c for c in ["toxicity_native", "perplexity_native",
                        "perplexity_norm", "token_length_native"]
            if c in df.columns]
    if cols:
        print(df.groupby("language")[cols].median().round(3).to_string())

    if "toxicity_native" in df.columns:
        tox = df["toxicity_native"].dropna()
        if len(tox):
            print(f"\nToxicity sanity check:")
            print(f"  min {tox.min():.4f}  median {tox.median():.4f}  "
                  f"max {tox.max():.4f}")
            print(f"  A minimum near 0 confirms p(toxic) is being read, not the "
                  f"winning label's confidence (Chapter 1's minimum was 0.538).")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {device}\n")

    df = load_corpus()

    df = phase_token_length(df)
    df = phase_perplexity(df)
    df = normalise_perplexity(df)
    df = phase_toxicity(df)
    df = phase_sentiment(df)
    df = phase_readability(df)
    df = phase_intent(df)
    df = phase_embeddings(df)

    df.to_parquet(OUTPUT_FILE, engine="pyarrow")

    meta = {
        "rows": len(df),
        "perplexity_reference_mode": df.attrs.get(
            "perplexity_reference_mode", "unknown"),
        "models": {
            "toxicity": TOXICITY_MODEL,
            "sentiment_multilingual": SENTIMENT_MULTI,
            "sentiment_monolingual": SENTIMENT_MONO,
            "intent": INTENT_MODEL,
            "perplexity": PERPLEXITY_MODEL,
            "tokenizer": TOKENIZER_MODEL,
            "embeddings": EMBED_MODEL,
        },
    }
    with open(os.path.join(DATA_DIR, "metrics_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    report(df)
    print(f"\nSaved to {OUTPUT_FILE}")