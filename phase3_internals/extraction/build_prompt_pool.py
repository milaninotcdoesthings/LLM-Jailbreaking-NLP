"""
build_prompt_pool.py - extract unique prompts from every data file under THESIS.

Keeps only (prompt, lang, label, source). Discards old metrics (they'll be
recomputed uniformly). Handles wrong separators, different column names,
and deduplicates across files.

Output: prompt_pool.csv
"""

import re, unicodedata
import pandas as pd
from pathlib import Path

BASE = Path("/Users/tommasomilanino/Developer/THESIS")
OUT = BASE / "prompt_pool.csv"

SKIP_DIRS = {"advbench", "donotanswer", "Multilingual_safety_benchmark",
             ".git", "qwen_3b_local", "__pycache__", ".venv"}

# column-name candidates
PROMPT_NAMES = ["raw_prompt", "prompt", "text", "question", "goal",
                "instruction", "behavior", "adversarial", "query", "attack"]
LABEL_NAMES = ["label", "y", "harmful", "unsafe", "success", "jailbroken", "compliance"]
LANG_NAMES = ["lang", "language"]

# infer language from filename when the file has no lang column
FILE_LANG = {
    "arabic": "ar", "_ar": "ar", "arab": "ar",
    "german": "de", "_de": "de", "germa": "de",
    "russian": "ru", "_ru": "ru", "russ": "ru",
    "french": "fr", "_fr": "fr", "franc": "fr",
    "spanish": "es", "_sp": "es", "_es": "es", "span": "es",
    "english": "en", "_en": "en", "bengali": "bn", "_bn": "bn",
    "hindi": "hi", "_hi": "hi", "japanese": "ja", "_ja": "ja",
    "chinese": "zh", "_zh": "zh",
}

def guess_lang_from_name(name):
    low = name.lower()
    for key, code in FILE_LANG.items():
        if key in low:
            return code
    return None

def robust_read(p):
    """Try comma, then semicolon, then python engine. Return df or None."""
    for kw in ({}, {"sep": ";"}, {"sep": None, "engine": "python"}):
        try:
            df = pd.read_parquet(p) if p.suffix == ".parquet" \
                 else pd.read_csv(p, on_bad_lines="skip", **kw)
            if df.shape[1] > 1 or p.suffix == ".parquet":
                return df
        except Exception:
            continue
    # last resort: single-column text file (like AdvBench)
    try:
        lines = [l.strip() for l in p.open(encoding="utf-8", errors="ignore") if l.strip()]
        return pd.DataFrame({"text": lines})
    except Exception:
        return None

def pick(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None

def norm_label(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true", "yes", "harmful", "unsafe", "success", "jailbroken"}:
        return 1
    if s in {"0", "0.0", "false", "no", "safe", "benign", "refused", "fail"}:
        return 0
    return None

def normkey(text):
    t = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t))

# --- scan all files ----------------------------------------------------
files = [p for p in BASE.rglob("*")
         if p.suffix.lower() in {".csv", ".parquet"}
         and not any(part in SKIP_DIRS for part in p.parts)
         and p.name not in {OUT.name, "file_inventory.csv"}]

records = []
per_file = []
for p in sorted(files):
    df = robust_read(p)
    if df is None or len(df) == 0:
        per_file.append((p.name, 0, "unreadable")); continue

    pcol = pick(df.columns, PROMPT_NAMES)
    if pcol is None:
        per_file.append((p.name, 0, "no prompt col")); continue

    lcol = pick(df.columns, LABEL_NAMES)
    lang_col = pick(df.columns, LANG_NAMES)
    file_lang = guess_lang_from_name(p.name)

    n = 0
    for _, row in df.iterrows():
        text = str(row[pcol]).strip()
        if not (5 <= len(text) <= 2000):
            continue
        lang = None
        if lang_col and pd.notna(row.get(lang_col)):
            lang = str(row[lang_col]).lower()[:2]
        lang = lang or file_lang
        label = norm_label(row[lcol]) if lcol else None
        records.append({"raw_prompt": text, "lang": lang, "label": label,
                        "source_file": p.name})
        n += 1
    per_file.append((p.name, n, "ok"))

pool = pd.DataFrame(records)
print(f"raw records extracted: {len(pool)}")

# --- deduplicate: same normalized text -> keep the row that HAS a label -
pool["_key"] = pool["raw_prompt"].map(normkey)
pool["_has_label"] = pool["label"].notna().astype(int)
pool = (pool.sort_values("_has_label", ascending=False)
            .drop_duplicates("_key")
            .drop(columns=["_key", "_has_label"])
            .reset_index(drop=True))
pool["id"] = ["pool-" + str(i).zfill(6) for i in range(len(pool))]

pool.to_csv(OUT, index=False)

# --- report ------------------------------------------------------------
print(f"\nunique prompts after dedup: {len(pool)}")
print("\nby language:")
print(pool.lang.fillna("unknown").value_counts().to_string())
print("\nlabel coverage:")
print(f"  with label: {pool.label.notna().sum()}  |  without: {pool.label.isna().sum()}")
print("\ntop source files by contribution:")
pf = pd.DataFrame(per_file, columns=["file", "n_prompts", "status"])
print(pf[pf.n_prompts > 0].sort_values("n_prompts", ascending=False).head(15).to_string(index=False))
print(f"\nunreadable / skipped: {(pf.status != 'ok').sum()} files")
print(f"\n[+] saved -> {OUT}")