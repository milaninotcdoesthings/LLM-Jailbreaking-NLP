"""
08_inventory.py
===============
Step 0 of building the training dataset.

Scans every CSV and Parquet file under the given roots and reports what each
one contains: row count, columns, and a guess at which column holds the prompt,
the response, and the label. Writes the result to inventory.json.

This exists because the adapters in 09_build_dataset.py have to map each
dataset's own column names onto the common schema, and those names cannot be
recalled reliably from memory — the repository has dozens of files with names
like german_processed_new_risultati_groq_llm.csv whose contents are not
deducible from the filename.

Run this, read the output, then write one adapter per dataset you want to keep.

Usage:
    python 08_inventory.py /path/to/repo /path/to/other/dir
"""

import os
import re
import sys
import json

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_ROOTS = [
    "/Users/tommasomilanino/Developer/THESIS",
]

OUTPUT_FILE = "inventory.json"

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
             "checkpoints", ".venv", "venv"}

MAX_PREVIEW_CHARS = 90

# Heuristics for guessing a column's role. Order matters: first match wins.
PROMPT_HINTS   = ["raw_prompt", "prompt_text", "text_native", "prompt",
                  "question", "instruction", "behavior", "adversarial",
                  "goal", "query", "input"]
RESPONSE_HINTS = ["response", "risposta", "output", "completion",
                  "generation", "answer"]
LABEL_HINTS    = ["score", "status", "success", "jailbreak", "label",
                  "verdict", "esito", "giudizio"]
LANG_HINTS     = ["language", "lang", "lingua"]
CATEGORY_HINTS = ["category", "categoria", "semantic", "topic", "type", "harm"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def guess_column(columns, hints):
    """Return every column whose name contains one of the hint substrings."""
    lowered = {c: c.lower() for c in columns}
    matches = []
    for hint in hints:
        for col, low in lowered.items():
            if hint in low and col not in matches:
                matches.append(col)
    return matches


def read_head(path, n=200):
    """
    Read enough rows to characterise the file without loading a large CSV
    entirely. Tries several separators, since some files in this project were
    written with ';'.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(n), len(df), None

    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, nrows=n, on_bad_lines="skip",
                             engine="python")
            if df.shape[1] > 1:
                total = sum(1 for _ in open(path, encoding="utf-8",
                                            errors="ignore")) - 1
                return df, total, sep
        except Exception:
            continue

    raise ValueError("could not parse with , ; or tab")


def preview_value(series):
    """First non-null value, truncated, for eyeballing what a column holds."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    val = str(non_null.iloc[0]).replace("\n", " ").strip()
    return val[:MAX_PREVIEW_CHARS] + ("…" if len(val) > MAX_PREVIEW_CHARS else "")


def describe_label(df, col):
    """
    For a candidate label column, show the value distribution. A column with
    values {0,1,2} is a graded judge verdict; {0,1} is binary; anything else is
    probably not a label at all.
    """
    try:
        vals = df[col].dropna()
        if len(vals) == 0:
            return None
        uniq = vals.unique()
        if len(uniq) <= 8:
            return {str(k): int(v) for k, v in vals.value_counts().items()}
        return f"{len(uniq)} distinct values"
    except Exception:
        return None


# ── Scanning ──────────────────────────────────────────────────────────────────

def scan_file(path):
    entry = {"path": path, "filename": os.path.basename(path)}

    try:
        df, total_rows, sep = read_head(path)
    except Exception as e:
        entry["error"] = str(e)
        return entry

    entry["rows"] = total_rows
    entry["separator"] = sep
    entry["n_columns"] = df.shape[1]
    entry["columns"] = list(df.columns)

    entry["likely_prompt"] = guess_column(df.columns, PROMPT_HINTS)
    entry["likely_response"] = guess_column(df.columns, RESPONSE_HINTS)
    entry["likely_label"] = guess_column(df.columns, LABEL_HINTS)
    entry["likely_language"] = guess_column(df.columns, LANG_HINTS)
    entry["likely_category"] = guess_column(df.columns, CATEGORY_HINTS)

    entry["previews"] = {}
    for role in ["likely_prompt", "likely_response", "likely_category"]:
        for col in entry[role][:2]:
            entry["previews"][col] = preview_value(df[col])

    entry["label_distributions"] = {}
    for col in entry["likely_label"][:3]:
        dist = describe_label(df, col)
        if dist is not None:
            entry["label_distributions"][col] = dist

    return entry


def scan_roots(roots):
    entries = []
    for root in roots:
        if not os.path.exists(root):
            print(f"  skipping {root} — does not exist")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                if not fn.lower().endswith((".csv", ".parquet")):
                    continue
                entries.append(scan_file(os.path.join(dirpath, fn)))
    return entries


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(entries):
    ok = [e for e in entries if "error" not in e]
    bad = [e for e in entries if "error" in e]

    print("\n" + "=" * 72)
    print(f"INVENTORY — {len(ok)} readable files, {len(bad)} unreadable")
    print("=" * 72)

    # Files carrying a graded 0/1/2 verdict are the ones eligible for training
    graded = []
    for e in ok:
        for col, dist in e.get("label_distributions", {}).items():
            if isinstance(dist, dict) and set(dist.keys()) <= {"0", "1", "2",
                                                               "0.0", "1.0", "2.0"}:
                graded.append((e, col, dist))
                break

    print(f"\n--- Files with a graded 0/1/2 label ({len(graded)}) ---")
    print("These are the candidates for protocol_valid = True.\n")
    for e, col, dist in sorted(graded, key=lambda x: -x[0].get("rows", 0)):
        print(f"  {e['filename']}")
        print(f"    rows {e['rows']}   label column '{col}'   {dist}")
        if e["likely_prompt"]:
            print(f"    prompt: {e['likely_prompt'][0]}")
        if e["likely_response"]:
            print(f"    response: {e['likely_response'][:2]}")
        print()

    print(f"\n--- All readable files, by size ---\n")
    for e in sorted(ok, key=lambda x: -x.get("rows", 0)):
        cols = ", ".join(e["columns"][:6])
        more = f" (+{len(e['columns'])-6})" if len(e["columns"]) > 6 else ""
        print(f"  {e['rows']:>7}  {e['filename']}")
        print(f"           {cols}{more}")

    if bad:
        print(f"\n--- Unreadable ---")
        for e in bad:
            print(f"  {e['filename']}: {e['error']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    roots = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ROOTS
    print(f"Scanning: {', '.join(roots)}")

    entries = scan_roots(roots)
    print_report(entries)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)

    print(f"\nFull inventory written to {OUTPUT_FILE}")
    print("Read it, decide which files to keep, then write one adapter each.")