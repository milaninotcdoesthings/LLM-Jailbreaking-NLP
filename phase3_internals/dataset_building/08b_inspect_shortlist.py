"""
08b_inspect_shortlist.py
========================
Prints the actual contents of the files shortlisted for the prompt pool, so the
adapters in 09_build_prompt_pool.py can be written against what is there rather
than against what the column names suggest.

08_inventory.py counts lines, not dataframe rows — multi-line text fields
inflate the count badly — so this script reports true shapes. It also reads the
merged_*.csv family with header=None, since those files appear to carry data in
their first row rather than column names.

Usage:
    python 08b_inspect_shortlist.py
"""

import os
import pandas as pd

pd.set_option("display.max_colwidth", 200)

REPO = "/Users/tommasomilanino/Developer/THESIS"

MAX_CHARS = 220
N_SAMPLES = 2

# Files to inspect, in the order they matter for the pool.
SHORTLIST = [
    "ai_regeneration/data/corpus_english.csv",
    "ai_regeneration/data/corpus_multilingual.csv",
    "aya_raw.parquet",
    "benchmarks_raw.csv",
    "harmbench_dataset.csv",
    "wildjailbreak_full.csv",
    "multilingual_benchmark_en_full.csv",
    "rerun_metrics.parquet",
    "redteaming_results_2000.parquet",
    "redteaming_native_multilingual.parquet",
    "redteaming_translated.parquet",
    "harmful_behaviors.csv",
    "do_not_answer_en.csv",
    "xstest_prompts.csv",
]

# Read with header=None: the inventory shows their first row was consumed as
# column names, which means the real structure is still unknown.
HEADERLESS = [
    "merged_en.csv",
    "merged_de.csv",
    "merged_ru.csv",
    "merged_ar.csv",
]


def find(filename: str) -> str | None:
    """Locate a file anywhere under the repo — paths differ across branches."""
    target = os.path.basename(filename)
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in
                       {".git", "__pycache__", ".ipynb_checkpoints"}]
        if target in filenames:
            candidate = os.path.join(dirpath, target)
            # prefer an exact suffix match when the shortlist gave a subpath
            if filename.endswith(target) and candidate.endswith(filename):
                return candidate
            return candidate
    return None


def read_any(path: str, header="infer") -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, header=header,
                             on_bad_lines="skip", engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, on_bad_lines="skip", engine="python", header=header)


def truncate(val) -> str:
    if isinstance(val, (list, tuple)) or (hasattr(val, "__len__") and not isinstance(val, str)):
        return f"<array len={len(val)}>"
    if pd.isna(val):
        return "<NaN>"
    s = str(val).replace("\n", " ⏎ ").strip()
    return s[:MAX_CHARS] + ("…" if len(s) > MAX_CHARS else "")


def inspect(path: str, label: str, header="infer") -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    try:
        df = read_any(path, header=header)
    except Exception as e:
        print(f"  could not read: {e}")
        return

    print(f"shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"columns: {list(df.columns)}")

    # Value counts for anything that looks like a grouping variable
    for col in df.columns:
        if df[col].dtype != object:
            continue
        try:
            n_unique = df[col].nunique()
        except TypeError:
            continue  # colonna con valori non hashable (es. embedding), salta

        if n_unique <= 15:
            counts = df[col].value_counts(dropna=False).head(15)
            print(f"\n  '{col}' — {n_unique} distinct:")
            for k, v in counts.items():
                print(f"      {v:>6}  {truncate(k)[:60]}")

    print(f"\n  sample rows:")
    for i in range(min(N_SAMPLES, len(df))):
        print(f"\n  [row {i}]")
        for col in df.columns:
            print(f"    {col:<26} {truncate(df.iloc[i][col])}")


if __name__ == "__main__":
    print(f"Inspecting shortlisted files under {REPO}")

    for rel in SHORTLIST:
        path = find(rel)
        if path is None:
            print(f"\n{'=' * 78}\n{rel}\n{'=' * 78}\n  NOT FOUND")
            continue
        inspect(path, f"{rel}\n  → {path}")

    print("\n\n" + "#" * 78)
    print("# merged_*.csv read with header=None")
    print("#" * 78)

    for rel in HEADERLESS:
        path = find(rel)
        if path is None:
            print(f"\n{rel}: NOT FOUND")
            continue
        inspect(path, f"{rel} (header=None)\n  → {path}", header=None)