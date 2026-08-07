"""
11_compute_metrics.py
======================
Calcola le metriche NLP sul prompt_pool unificato, con checkpoint per-chunk
invece che solo per-fase. Riscritto dopo uno spegnimento imprevisto del Mac a
metà esecuzione: con la versione precedente, perdere il pc a metà di una fase
lunga (perplexity, intent) significava perdere ore di lavoro anche se il
checkpoint di fase esisteva. Qui ogni CHUNK_SIZE righe vengono salvate,
indipendentemente da dove ci si trova dentro la fase.

RIPRESA AUTOMATICA. Ogni fase controlla prima se esiste già un file parziale
per quella fase; se sì, salta le righe già processate e continua da lì. Basta
rilanciare lo script dopo un crash — non serve intervento manuale.

Le logiche di calcolo (doppia misurazione native/source, correzioni toxicity e
sentiment, normalizzazione perplexity) sono identiche alla versione precedente.

Output: prompt_pool_with_metrics.parquet
"""

import os
import gc
import json
import time
import time as time_module

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "dataset_building")

POOL_FILE     = os.path.join(DATA_DIR, "prompt_pool.parquet")
BASELINE_FILE = os.path.join(DATA_DIR, "benign_baseline.parquet")  # opzionale
OUTPUT_FILE   = os.path.join(DATA_DIR, "prompt_pool_with_metrics.parquet")

CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Modelli ───────────────────────────────────────────────────────────────────

TOXICITY_MODEL   = "citizenlab/distilbert-base-multilingual-cased-toxicity"
SENTIMENT_MULTI  = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
INTENT_MODEL     = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
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

# Righe per checkpoint. Più basso = più resiliente ma più I/O overhead.
# 1500 è un compromesso ragionevole: nel caso peggiore si perdono 1-2 minuti
# di lavoro, non ore.
CHUNK_SIZE = 1000

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


# ── Infrastruttura ────────────────────────────────────────────────────────────

def free_memory():
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def log_system_state(label: str):
    mem = psutil.virtual_memory()
    try:
        temps = psutil.sensors_temperatures()
        temp_str = ""
        for name, entries in (temps or {}).items():
            for e in entries:
                if e.current:
                    temp_str = f", {e.current:.0f}°C ({name})"
                    break
            if temp_str:
                break
    except (AttributeError, Exception):
        temp_str = ""  # sensors_temperatures non disponibile su macOS

    print(f"  [{label}] RAM: {mem.percent:.0f}% usata, "
          f"{mem.available / 1e9:.1f}GB liberi{temp_str}")


def chunk_path(phase: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{phase}_partial.parquet")


def load_partial(phase: str, id_col: str = "prompt_id") -> pd.DataFrame | None:
    path = chunk_path(phase)
    if os.path.exists(path):
        partial = pd.read_parquet(path)
        print(f"  ripresa: {len(partial)} righe già processate per '{phase}'")
        return partial
    return None


def save_partial(df_chunk: pd.DataFrame, phase: str, append: bool):
    path = chunk_path(phase)
    if append and os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df_chunk], ignore_index=True)
        combined.to_parquet(path, engine="pyarrow")
    else:
        df_chunk.to_parquet(path, engine="pyarrow")


def finalise_phase(df: pd.DataFrame, phase: str, new_cols: list[str],
                   id_col: str = "prompt_id") -> pd.DataFrame:
    """Unisce il parziale calcolato nel dataframe principale e libera il chunk."""
    partial = pd.read_parquet(chunk_path(phase))
    df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
    df = df.merge(partial[[id_col] + new_cols], on=id_col, how="left")
    os.remove(chunk_path(phase))
    return df


# ── Runner generico a chunk con ripresa ───────────────────────────────────────

def run_chunked(df: pd.DataFrame, phase: str, compute_fn, new_cols: list[str],
                text_cols: list[str], id_col: str = "prompt_id") -> pd.DataFrame:
    """
    Applica compute_fn a blocchi di CHUNK_SIZE righe, salvando dopo ogni
    blocco. compute_fn riceve un dataframe (le colonne di testo) e restituisce
    un dataframe con le stesse righe e le colonne in new_cols.

    Se un file parziale esiste già per questa fase, le righe già processate
    vengono saltate — è così che la ripresa funziona senza intervento manuale.
    """
    print(f"\n=== {phase} ===")
    log_system_state(f"{phase} — inizio")
    time_module.sleep(3)

    partial = load_partial(phase, id_col)
    done_ids = set(partial[id_col]) if partial is not None else set()

    remaining = df[~df[id_col].isin(done_ids)]
    print(f"  da processare: {len(remaining)} / {len(df)} righe")

    if len(remaining) == 0:
        print(f"  già completo, salto il calcolo")
    else:
        n_chunks = (len(remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in tqdm(range(n_chunks), desc=f"  {phase}", unit="chunk"):
            start = i * CHUNK_SIZE
            chunk = remaining.iloc[start:start + CHUNK_SIZE]

            t0 = time.time()
            result_cols = compute_fn(chunk)
            elapsed = time.time() - t0

            out_chunk = pd.DataFrame({id_col: chunk[id_col].values})
            for col in new_cols:
                out_chunk[col] = result_cols[col].values

            save_partial(out_chunk, phase, append=True)

            if i % 3 == 0:
                log_system_state(f"{phase} — chunk {i+1}/{n_chunks} "
                                 f"({elapsed:.0f}s)")
                free_memory()

    df = finalise_phase(df, phase, new_cols, id_col)
    print(f"  {phase} completata: {df[new_cols[0]].notna().sum()} / {len(df)} "
          f"non-null su '{new_cols[0]}'")
    return df


# ── Fase 1: token length ──────────────────────────────────────────────────────

def phase_token_length(df: pd.DataFrame) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    def compute(chunk):
        return pd.DataFrame({
            "token_length_native": chunk["text_native"].apply(
                lambda x: len(tok.encode(str(x)))),
            "token_length_source": chunk["text_source"].apply(
                lambda x: len(tok.encode(str(x)))),
        })

    df = run_chunked(df, "01_token_length", compute,
                     ["token_length_native", "token_length_source"],
                     ["text_native", "text_source"])
    del tok
    free_memory()
    return df


# ── Fase 2: perplexity ────────────────────────────────────────────────────────

def phase_perplexity(df: pd.DataFrame) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PERPLEXITY_MODEL, torch_dtype=torch.float16).to(device)
    model.eval()

    def perplexity_one(text):
        try:
            inputs = tok(str(text), return_tensors="pt",
                         truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
                return torch.exp(out.loss).item()
        except Exception:
            return None

    def compute(chunk):
        return pd.DataFrame({
            "perplexity_native": chunk["text_native"].apply(perplexity_one),
            "perplexity_source": chunk["text_source"].apply(perplexity_one),
        })

    df = run_chunked(df, "02_perplexity", compute,
                     ["perplexity_native", "perplexity_source"],
                     ["text_native", "text_source"])
    del model, tok
    free_memory()
    return df


def normalise_perplexity(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Normalizzazione perplexity ---")

    if os.path.exists(BASELINE_FILE):
        baseline_df = pd.read_parquet(BASELINE_FILE)
        if "perplexity_native" in baseline_df.columns:
            refs = baseline_df.groupby("language")["perplexity_native"].median()
            mode = "corpus di riferimento benigno indipendente"
        else:
            refs = df.groupby("language")["perplexity_native"].median()
            mode = "mediana within-corpus (fallback)"
    else:
        refs = df.groupby("language")["perplexity_native"].median()
        mode = "mediana within-corpus (fallback — nessun baseline fornito)"

    print(f"  riferimento: {mode}")
    for lang, val in refs.items():
        print(f"    {lang:<10} {val:.1f}")

    df["perplexity_norm"] = df.apply(
        lambda r: r["perplexity_native"] / refs.get(r["language"], np.nan)
        if pd.notna(r["perplexity_native"]) else None,
        axis=1,
    )
    df.attrs["perplexity_reference_mode"] = mode
    return df


# ── Fase 3: toxicity ──────────────────────────────────────────────────────────

def phase_toxicity(df: pd.DataFrame) -> pd.DataFrame:
    pipe = pipeline("text-classification", model=TOXICITY_MODEL,
                    device=device, top_k=None)

    def toxicity_one(text):
        try:
            res = pipe(str(text)[:MAX_CHARS], top_k=None)
            scores = res[0] if isinstance(res[0], list) else res
            for d in scores:
                if "toxic" in d["label"].lower() and "non" not in d["label"].lower():
                    return d["score"]
            return 0.0
        except Exception:
            return None

    def compute(chunk):
        return pd.DataFrame({
            "toxicity_native": chunk["text_native"].apply(toxicity_one),
            "toxicity_source": chunk["text_source"].apply(toxicity_one),
        })

    df = run_chunked(df, "03_toxicity", compute,
                     ["toxicity_native", "toxicity_source"],
                     ["text_native", "text_source"])
    del pipe
    free_memory()
    return df


# ── Fase 4: sentiment multilingue ─────────────────────────────────────────────

def phase_sentiment_multi(df: pd.DataFrame) -> pd.DataFrame:
    pipe = pipeline("text-classification", model=SENTIMENT_MULTI, device=device)

    def sentiment_one(text):
        try:
            res = pipe(str(text)[:MAX_CHARS])
            return res[0]["score"], res[0]["label"].lower()
        except Exception:
            return None, None

    def compute(chunk):
        native = chunk["text_native"].apply(sentiment_one)
        source = chunk["text_source"].apply(sentiment_one)
        return pd.DataFrame({
            "sentiment_multi_score": [s for s, _ in native],
            "sentiment_multi_label": [l for _, l in native],
            "sentiment_multi_score_source": [s for s, _ in source],
        })

    df = run_chunked(df, "04a_sentiment_multi", compute,
                     ["sentiment_multi_score", "sentiment_multi_label",
                      "sentiment_multi_score_source"],
                     ["text_native", "text_source"])
    del pipe
    free_memory()
    return df


# ── Fase 4b: sentiment monolingue (per lingua) ────────────────────────────────

def phase_sentiment_mono(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== 04b_sentiment_mono ===")
    print(f"  copertura: {list(SENTIMENT_MONO.keys())}")
    print(f"  francese/spagnolo restano solo su sentiment_multi")

    if "sentiment_mono_score" not in df.columns:
        df["sentiment_mono_score"] = None
        df["sentiment_mono_label"] = None

    for lang, model_id in SENTIMENT_MONO.items():
        mask = df["language"].str.lower() == lang
        already_done = df.loc[mask, "sentiment_mono_score"].notna().sum()
        remaining_mask = mask & df["sentiment_mono_score"].isna()

        if remaining_mask.sum() == 0:
            print(f"    {lang}: già completo ({already_done} righe)")
            continue

        print(f"    {lang}: {remaining_mask.sum()} righe da processare")
        try:
            pipe = pipeline("text-classification", model=model_id, device=device)

            def sentiment_one(text):
                try:
                    res = pipe(str(text)[:MAX_CHARS])
                    return res[0]["score"], res[0]["label"].lower()
                except Exception:
                    return None, None

            lang_df = df.loc[remaining_mask]
            n_chunks = (len(lang_df) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for i in tqdm(range(n_chunks), desc=f"      {lang}", unit="chunk"):
                idx = lang_df.index[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
                results = df.loc[idx, "text_native"].apply(sentiment_one)
                df.loc[idx, "sentiment_mono_score"] = [s for s, _ in results]
                df.loc[idx, "sentiment_mono_label"] = [l for _, l in results]

                # checkpoint dell'intero df ogni 3 chunk per questa lingua —
                # più pesante del chunk_path dedicato ma semplice e sicuro
                if i % 3 == 0:
                    df.to_parquet(os.path.join(CHECKPOINT_DIR,
                                               "04b_sentiment_mono_wip.parquet"),
                                 engine="pyarrow")
                    free_memory()

            del pipe
            free_memory()
        except Exception as e:
            print(f"    {lang} FALLITA: {e}")

    wip_path = os.path.join(CHECKPOINT_DIR, "04b_sentiment_mono_wip.parquet")
    if os.path.exists(wip_path):
        os.remove(wip_path)

    return df


# ── Fase 5: readability e complessità ─────────────────────────────────────────

def phase_readability(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 05_readability ===")
    try:
        import textstat
        if "flesch_source" not in df.columns or df["flesch_source"].isna().all():
            df["flesch_source"] = df["text_source"].apply(
                lambda x: textstat.flesch_reading_ease(str(x)))
        print("  Flesch calcolato sul sorgente inglese")
    except ImportError:
        print("  textstat non installato — skip")
        df["flesch_source"] = None

    def avg_word_length(text):
        words = str(text).split()
        return np.mean([len(w) for w in words]) if words else None

    def type_token_ratio(text):
        words = str(text).split()
        return len(set(words)) / len(words) if words else None

    df["avg_word_length_native"] = df["text_native"].apply(avg_word_length)
    df["ttr_native"] = df["text_native"].apply(type_token_ratio)
    print("  proxy language-agnostic calcolati")

    df.to_parquet(os.path.join(CHECKPOINT_DIR, "05_readability.parquet"),
                 engine="pyarrow")
    return df


# ── Fase 6: intent ────────────────────────────────────────────────────────────

def phase_intent(df: pd.DataFrame) -> pd.DataFrame:
    clf = pipeline("zero-shot-classification", model=INTENT_MODEL, device=device)

    def intent_one(text):
        try:
            res = clf(str(text)[:MAX_CHARS], INTENT_LABELS, multi_label=False)
            return res["labels"][0], res["scores"][0]
        except Exception:
            return None, None

    def compute(chunk):
        native = chunk["text_native"].apply(intent_one)
        source = chunk["text_source"].apply(intent_one)
        return pd.DataFrame({
            "intent_native": [l for l, _ in native],
            "intent_confidence": [s for _, s in native],
            "intent_source": [l for l, _ in source],
        })

    df = run_chunked(df, "06_intent", compute,
                     ["intent_native", "intent_confidence", "intent_source"],
                     ["text_native", "text_source"])
    del clf
    free_memory()
    return df


# ── Fase 7: embeddings ────────────────────────────────────────────────────────

def phase_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    def compute(chunk):
        emb = embedder.encode(
            chunk["text_native"].astype(str).tolist(),
            batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return pd.DataFrame({"embedding_native": [e.tolist() for e in emb]})

    df = run_chunked(df, "07_embeddings", compute,
                     ["embedding_native"], ["text_native"])
    del embedder
    free_memory()
    return df


# ── Report ────────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("REPORT METRICHE")
    print("=" * 66)
    print(f"Righe: {len(df)}")

    print(f"\nCopertura:")
    for col in sorted(c for c in df.columns
                      if any(k in c for k in
                             ["toxicity", "perplexity", "sentiment",
                              "flesch", "intent", "token_length", "ttr"])):
        n = df[col].notna().sum()
        print(f"  {col:<32} {n:>6} / {len(df)}")

    if "toxicity_native" in df.columns:
        tox = df["toxicity_native"].dropna()
        if len(tox):
            print(f"\nSanity check toxicity:")
            print(f"  min {tox.min():.4f}  mediana {tox.median():.4f}  "
                  f"max {tox.max():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(POOL_FILE):
        raise SystemExit(f"{POOL_FILE} non trovato.")

    print(f"Device: {device}")
    log_system_state("avvio")

    df = pd.read_parquet(POOL_FILE)
    print(f"Pool caricato: {len(df)} prompt")
    print(df["language"].value_counts().to_string())

    df = phase_token_length(df)
    df = phase_perplexity(df)
    df = normalise_perplexity(df)
    df = phase_toxicity(df)
    df = phase_sentiment_multi(df)
    df = phase_sentiment_mono(df)
    df = phase_readability(df)
    df = phase_intent(df)
    df = phase_embeddings(df)

    df.to_parquet(OUTPUT_FILE, engine="pyarrow")

    meta = {
        "rows": len(df),
        "perplexity_reference_mode": df.attrs.get(
            "perplexity_reference_mode", "sconosciuta"),
        "sentiment_mono_coverage": list(SENTIMENT_MONO.keys()),
        "chunk_size": CHUNK_SIZE,
    }
    with open(os.path.join(DATA_DIR, "metrics_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    report(df)
    log_system_state("completato")
    print(f"\nSalvato in {OUTPUT_FILE}")