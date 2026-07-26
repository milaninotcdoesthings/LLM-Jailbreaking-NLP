# judge_validation/download_sorrybench.py
from datasets import load_dataset
from pathlib import Path

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

for split in ["train", "test"]:
    d = load_dataset("sorry-bench/sorry-bench-human-judgment-202503", split=split)
    df = d.to_pandas()
    df.to_parquet(OUT / f"human_judgment_{split}.parquet")
    print(f"{split}: {len(df)} rows, cols: {list(df.columns)}")