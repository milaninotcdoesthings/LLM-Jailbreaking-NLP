from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT = Path("native_vs_translated/data")
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["arabic", "russian", "english", "french", "spanish"]
frames = []
for lg in LANGS:
    df = load_dataset("CohereForAI/aya_redteaming", split=lg).to_pandas()
    df["language"] = lg
    frames.append(df)
    print(f"{lg:10s} {len(df):5d}")

aya = pd.concat(frames, ignore_index=True)
aya.to_parquet(OUT / "aya_raw.parquet")

print("\ncolonne:", aya.columns.tolist())
print(aya.head(2).to_string())

for c in aya.columns:
    if aya[c].dtype == object and aya[c].nunique() < 30:
        print(f"\n{c}:")
        print(aya[c].value_counts().head(12).to_string())