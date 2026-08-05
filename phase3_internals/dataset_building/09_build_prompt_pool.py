"""
09_build_prompt_pool.py
=======================
Merges every prompt corpus from Chapters 1 and 2 into one deduplicated pool.

Prompts only. No responses, no labels, no target models. Those are regenerated
downstream by a single attack campaign under one protocol, which removes the
comparability problem documented in Section 1.4.1 — the legacy responses were
produced under three different generation budgets and two different judging
procedures, so nothing derived from them is comparable across sources anyway.
The prompts themselves are the reusable asset.

Output schema:
    prompt_id          sha1 of the text
    source_prompt_id   the English original this row derives from
    text_native        the prompt in its own language
    text_source        the same prompt in English
    language
    category           unified taxonomy
    source             which corpus it came from
    is_translated      True if machine-translated, False if natively written

CONFIGURATION. Fill in SOURCES below with the real column names from your
files — run 08_inventory.py first if you don't have them to hand. Two patterns
are supported:

  Pattern A — one file, one prompt column, optionally with the English
              original in a second column of the same file (Aya, and the
              ai_regeneration corpus, both look like this).

  Pattern B — a set of files sharing row order, one per language, with one
              designated English (the merged_*.csv multilingual benchmark).

Output: data/prompt_pool.parquet
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

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT  = "/Users/tommasomilanino/Developer/THESIS"
OUTPUT_DIR = OUTPUT_DIR = "/Users/tommasomilanino/Developer/THESIS/phase3_internals/dataset_building/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prompt_pool.parquet")
STATS_FILE  = os.path.join(OUTPUT_DIR, "prompt_pool_stats.json")

SIM_THRESHOLD = 0.94      # higher than in generation: only near-identical text
EMBED_BATCH   = 128

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ── Source configuration ──────────────────────────────────────────────────────
# Edit paths and column names to match your files. Set "enabled": False to skip
# a source without deleting its entry.

SOURCES = [

    # ── AI-generated corpus (this thesis, regeneration phase) ─────────────────

    {
        "enabled": True,
        "pattern": "A",
        "name": "ai_generated",
        "path": "ai_regeneration/data/corpus_english.csv",
        "prompt_col": "prompt_text",
        "source_text_col": None,          # already English
        "category_col": "category",
        "language": "english",
        "is_translated": False,
    },
    {
        "enabled": True,
        "pattern": "A",
        "name": "ai_generated",
        "path": "ai_regeneration/data/corpus_multilingual.csv",
        "prompt_col": "prompt_text",
        "source_text_col": "prompt_text_en",
        "category_col": "category",
        "language_col": "language",
        "is_translated": True,
    },

    # ── Aya red-teaming — natively authored, human-translated pairs ───────────
    # Only source with genuine native prompts paired to real translations,
    # not machine translations of an English original. English rows carry no
    # literal_translation; the adapter's empty-value fallback uses the native
    # text as its own source for those rows, which is correct since it is
    # already English.

    {
        "enabled": True,
        "pattern": "A",
        "name": "aya_redteaming",
        "path": "native_vs_translated/data/aya_raw.parquet",
        "prompt_col": "prompt",
        "source_text_col": "literal_translation",
        "category_col": "harm_category",
        "language_col": "language",
        "is_translated": False,
    },

    # ── AdvBench + DoNotAnswer + XSTest, already unified in one file ──────────
    # benchmarks_raw.csv already combines all three (938 + 450 + 388 = 1776
    # rows checks out against the individual sources). source_from_col keeps
    # the three distinguishable in the pool instead of collapsing them.

    {
        "enabled": True,
        "pattern": "A",
        "name": None,                      # overridden by source_from_col
        "path": "benchmarks_raw.csv",
        "prompt_col": "text",
        "source_text_col": None,
        "category_col": "category",
        "source_from_col": "source",
        "language": "english",
        "is_translated": False,
    },

    # ── HarmBench ──────────────────────────────────────────────────────────────

    {
        "enabled": True,
        "pattern": "A",
        "name": "harmbench",
        "path": "harmbench_dataset.csv",
        "prompt_col": "prompt",
        "source_text_col": None,
        "category_col": "category",
        "language": "english",
        "is_translated": False,
    },

    # ── WildJailbreak — keep the harmful arm only, drop the benign controls ───

    {
        "enabled": True,
        "pattern": "A",
        "name": "wildjailbreak",
        "path": "wildjailbreak_full.csv",
        "prompt_col": "adversarial",
        "source_text_col": None,
        "category_col": None,
        "filter_col": "data_type",
        "filter_value": "adversarial_harmful",
        "language": "english",
        "is_translated": False,
    },

    # ── Multilingual Safety benchmark, English arm (Chapter 1, Section 1.1) ───

    {
        "enabled": True,
        "pattern": "A",
        "name": "multilingual_safety_en",
        "path": "multilingual_benchmark_en_full.csv",
        "prompt_col": "prompt",
        "source_text_col": None,
        "category_col": None,
        "language": "english",
        "is_translated": False,
    },

    # ── Chapter 2 AI-generated corpus, original run ───────────────────────────

    {
        "enabled": True,
        "pattern": "A",
        "name": "chapter2_ai_generated",
        "path": "redteaming_results_2000.parquet",
        "prompt_col": "raw_prompt",
        "source_text_col": None,
        "category_col": "category",
        "language": "english",
        "is_translated": False,
    },
    {
        # No real English original exists for these — they were generated
        # directly in the target language (Chapter 2, "Approach B"). The
        # fallback sets text_source = text_native, which is honest about what
        # this row is: there is no translation to compare against.
        "enabled": True,
        "pattern": "A",
        "name": "chapter2_native_multilingual",
        "path": "redteaming_native_multilingual.parquet",
        "prompt_col": "raw_prompt",
        "source_text_col": "original_prompt",   # empty for every row → fallback fires
        "category_col": "category",
        "language_col": "language",
        "is_translated": False,
    },
    {
        "enabled": True,
        "pattern": "A",
        "name": "chapter2_translated",
        "path": "redteaming_translated.parquet",
        "prompt_col": "raw_prompt",
        "source_text_col": "original_prompt",
        "category_col": "category",
        "language_col": "language",
        "is_translated": True,
    },

    # merged_en/de/ru/ar.csv and rerun_metrics.parquet are deliberately
    # excluded. Row inspection shows column position does not correspond to
    # the same content across language files (the same English sentence
    # appears at a different column index in each language's file), so pairing
    # by row/column index — the assumption Pattern B depends on — produces
    # incorrect native/source matches. rerun_metrics.parquet inherits this
    # corruption: its raw_prompt field contains a whole mis-parsed CSV line
    # rather than a single prompt. Salvaging these would require embedding-
    # based cross-lingual matching, which is out of scope here given the pool
    # is already large without them.
]


# ── Category normalisation ────────────────────────────────────────────────────
# Maps each source's own taxonomy onto a shared one. Anything unmatched becomes
# "other" rather than being forced into a nearby bucket — a wrong category is
# worse than an honest absence.

CATEGORY_MAP = {
    # HarmBench semantic categories
    "chemical_biological": "physical_harm",
    "cybercrime_intrusion": "cybersecurity",
    "harassment_bullying": "harassment",
    "misinformation_disinformation": "misinformation",
    "copyright": "copyright",
    "illegal": "illegal_activity",
    "harmful": "other",

    # AI-generated taxonomy (Chapter 2)
    "financial fraud and tax evasion": "financial_fraud",
    "cybersecurity and malware generation": "cybersecurity",
    "hate speech and harassment": "harassment",
    "misinformation and fake news generation": "misinformation",
    "physical harm and weapons manufacturing": "physical_harm",
    "social engineering and phishing": "social_engineering",
    "bypassing copyright and piracy": "copyright",
    "illegal drug synthesis": "drug_synthesis",
    "privacy violation and doxxing": "privacy",
    "unsafe roleplay and identity theft": "roleplay_identity",

    # Multilingual Safety benchmark
    "crimes_and_illegal_activities": "illegal_activity",
    "ethics_and_morality": "other",
    "inquiry_with_unsafe_opinion": "other",
    "insult": "harassment",
    "mental_health": "mental_health",
    "physical_harm": "physical_harm",
    "privacy_and_property": "privacy",
    "unfairness_and_discrimination": "harassment",
    "role_play_instruction": "roleplay_identity",
}


def normalise_category(raw) -> str:
    if pd.isna(raw) or not str(raw).strip():
        return "unspecified"
    key = str(raw).strip().lower()

    # Aya stores categories as a stringified list, e.g. '["violence, ..."]'
    if key.startswith("["):
        key = re.sub(r'^\["|"\]$', "", key).split(",")[0].strip().lower()

    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    key_us = key.replace(" ", "_").replace("-", "_")
    return CATEGORY_MAP.get(key_us, "other")


# ── Text helpers ──────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def make_id(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def read_any(path: str) -> pd.DataFrame:
    """Handles the ';'-separated CSVs that appear in this project."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, on_bad_lines="skip", engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"could not parse {path}")


# ── Adapters ──────────────────────────────────────────────────────────────────

def adapt_pattern_a(cfg: dict) -> pd.DataFrame:
    path = os.path.join(REPO_ROOT, cfg["path"])
    if not os.path.exists(path):
        print(f"    not found: {cfg['path']}")
        return pd.DataFrame()

    df = read_any(path)

    # Optional row filter, e.g. wildjailbreak's benign/harmful split
    filter_col = cfg.get("filter_col")
    if filter_col and filter_col in df.columns:
        before = len(df)
        df = df[df[filter_col] == cfg["filter_value"]].reset_index(drop=True)
        print(f"    filtered {filter_col}=={cfg['filter_value']}: "
              f"{before} → {len(df)} rows")

    if cfg["prompt_col"] not in df.columns:
        print(f"    column '{cfg['prompt_col']}' absent in {cfg['path']}")
        print(f"    available: {list(df.columns)[:10]}")
        return pd.DataFrame()

    native = df[cfg["prompt_col"]].astype(str)

    # Source text, with fallback to native when the translation is missing or
    # empty (Aya's English rows carry no literal_translation; the natively
    # generated Chapter 2 corpus carries no back-translation at all).
    src_col = cfg.get("source_text_col")
    if src_col and src_col in df.columns:
        source_text = df[src_col].astype(str)
        empty = source_text.isna() | (source_text.str.strip() == "") | \
                (source_text.str.lower() == "nan")
        source_text = source_text.where(~empty, native)
    else:
        source_text = native

    if "language_col" in cfg and cfg["language_col"] in df.columns:
        language = df[cfg["language_col"]].astype(str).str.lower()
    else:
        language = cfg.get("language", "english")

    cat_col = cfg.get("category_col")
    category = (df[cat_col] if cat_col and cat_col in df.columns
                else "unspecified")

    # Provenance can come from a fixed name or from a column already present
    # in the file (benchmarks_raw.csv already distinguishes advbench /
    # do_not_answer / xstest internally).
    source_col = cfg.get("source_from_col")
    if source_col and source_col in df.columns:
        source_name = df[source_col].astype(str)
    else:
        source_name = cfg.get("name") or os.path.splitext(
            os.path.basename(cfg["path"]))[0]

    out = pd.DataFrame({
        "text_native": native,
        "text_source": source_text,
        "language": language,
        "category": category,
        "source": source_name,
        "is_translated": cfg.get("is_translated", False),
    })

    print(f"    {len(out)} rows")
    return out


def adapt_pattern_b(cfg: dict) -> pd.DataFrame:
    """
    Parallel files aligned by row index. The English file supplies text_source
    for every language, which is what makes cross-language metrics comparable
    downstream without per-language correction.
    """
    english_key = cfg["english_key"]
    files = cfg["files"]

    en_path = os.path.join(REPO_ROOT, files[english_key])
    if not os.path.exists(en_path):
        print(f"    English anchor not found: {files[english_key]}")
        return pd.DataFrame()

    en_df = read_any(en_path)
    if cfg["prompt_col"] not in en_df.columns:
        print(f"    column '{cfg['prompt_col']}' absent in the English file")
        print(f"    available: {list(en_df.columns)[:10]}")
        return pd.DataFrame()

    en_text = en_df[cfg["prompt_col"]].astype(str)
    cat_col = cfg.get("category_col")
    en_cat = (en_df[cat_col] if cat_col and cat_col in en_df.columns
              else "unspecified")

    frames = []

    for language, filename in files.items():
        path = os.path.join(REPO_ROOT, filename)
        if not os.path.exists(path):
            print(f"    {language}: not found ({filename})")
            continue

        df = read_any(path)
        if cfg["prompt_col"] not in df.columns:
            print(f"    {language}: prompt column absent")
            continue

        native = df[cfg["prompt_col"]].astype(str)

        # Row alignment is an assumption of this pattern; if the lengths differ
        # the files are not parallel and pairing them would be wrong.
        n = min(len(native), len(en_text))
        if len(native) != len(en_text):
            print(f"    {language}: {len(native)} rows vs {len(en_text)} English "
                  f"— truncating to {n}, verify alignment")

        frames.append(pd.DataFrame({
            "text_native": native.iloc[:n].values,
            "text_source": en_text.iloc[:n].values,
            "language": language,
            "category": (en_cat.iloc[:n].values
                         if hasattr(en_cat, "iloc") else en_cat),
            "source": cfg["name"],
            "is_translated": (language != english_key
                              and cfg.get("is_translated", True)),
        }))
        print(f"    {language}: {n} rows")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Exact match first, then semantic — but semantic runs WITHIN each language.
    The embedding model is cross-lingual by design, so a German prompt and its
    English source sit close together in the space; deduplicating across
    languages would delete the parallel corpus.
    """
    stats = {"input": len(df)}

    df = df.copy()
    df["_norm"] = df["text_native"].apply(normalise_text)
    before = len(df)
    df = df.drop_duplicates(subset=["_norm", "language"], keep="first")
    df = df.reset_index(drop=True)
    stats["removed_exact"] = before - len(df)
    print(f"\nExact duplicates removed: {stats['removed_exact']}")

    print("Semantic deduplication, within language:")
    embedder = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device=device)

    keep_global = []

    for language, group in df.groupby("language"):
        idx = group.index.tolist()
        embeddings = embedder.encode(
            group["text_native"].astype(str).tolist(),
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        kept_idx, kept_emb = [], []
        for i, row_idx in enumerate(tqdm(idx, desc=f"  {language}",
                                         unit="prompt", leave=False)):
            emb = embeddings[i]
            if kept_emb and (np.vstack(kept_emb) @ emb).max() > SIM_THRESHOLD:
                continue
            kept_idx.append(row_idx)
            kept_emb.append(emb)

        removed = len(idx) - len(kept_idx)
        print(f"  {language:<10} {len(kept_idx):>6} kept, {removed} near-duplicates")
        keep_global.extend(kept_idx)

    stats["removed_semantic"] = len(df) - len(keep_global)
    df = df.loc[sorted(keep_global)].drop(columns="_norm").reset_index(drop=True)
    stats["output"] = len(df)
    return df, stats


# ── Identifiers ───────────────────────────────────────────────────────────────

def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    source_prompt_id groups every language version of one request. It is what
    the train/test split must be stratified on: four translations of the same
    prompt are near-identical, and splitting by row would put some in train and
    some in test, inflating measured accuracy.
    """
    df["prompt_id"] = df["text_native"].apply(make_id)
    df["source_prompt_id"] = df["text_source"].apply(make_id)
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, stats: dict) -> None:
    print("\n" + "=" * 66)
    print("PROMPT POOL")
    print("=" * 66)
    print(f"Input rows                 {stats['input']}")
    print(f"Removed — exact duplicate  {stats['removed_exact']}")
    print(f"Removed — near-duplicate   {stats['removed_semantic']}")
    print(f"Final pool                 {stats['output']}")

    print(f"\nBy source:")
    for src, n in df["source"].value_counts().items():
        print(f"  {n:>6}  {src}")

    print(f"\nBy language:")
    for lang, n in df["language"].value_counts().items():
        print(f"  {n:>6}  {lang}")

    print(f"\nBy category:")
    for cat, n in df["category"].value_counts().items():
        print(f"  {n:>6}  {cat}")

    print(f"\nNatively written vs machine-translated:")
    for flag, n in df["is_translated"].value_counts().items():
        label = "machine-translated" if flag else "natively written"
        print(f"  {n:>6}  {label}")

    n_groups = df["source_prompt_id"].nunique()
    print(f"\nUnique source prompts: {n_groups}")
    print(f"Mean language versions per prompt: {len(df) / n_groups:.2f}")

    lengths = df["text_native"].str.split().str.len()
    print(f"\nPrompt length (words): median {lengths.median():.0f}, "
          f"range {lengths.min()}–{lengths.max()}")

    print(f"\nAttacking {len(df)} prompts against 2 models = "
          f"{len(df) * 2} generations.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for cfg in SOURCES:
        if not cfg.get("enabled", True):
            continue
        print(f"\n{cfg['name']} ({cfg.get('path', 'parallel set')})")
        adapt = adapt_pattern_a if cfg["pattern"] == "A" else adapt_pattern_b
        frame = adapt(cfg)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit("\nNo sources loaded. Check the paths in SOURCES.")

    pool = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(pool)} rows before deduplication")

    pool["category"] = pool["category"].apply(normalise_category)
    pool = pool[pool["text_native"].notna()]
    pool = pool[pool["text_native"].str.split().str.len() >= 3]

    pool, stats = deduplicate(pool)
    pool = assign_ids(pool)

    pool = pool[["prompt_id", "source_prompt_id", "text_native", "text_source",
                 "language", "category", "source", "is_translated"]]

    pool.to_parquet(OUTPUT_FILE, engine="pyarrow")
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    report(pool, stats)
    print(f"\nSaved to {OUTPUT_FILE}")