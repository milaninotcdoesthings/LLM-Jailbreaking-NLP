"""
20_learning_curve.py
====================
Tests whether the probe is data-limited or signal-limited.

The pool has 8,747 rows that were never attacked. Adding them costs ~2h of GPU
plus re-judging. Worth it only if performance is still climbing with sample
size; if the curve has flattened, the extra rows change nothing.

Subsamples by group, not by row, so the same leakage control as training holds.

Output: learning_curve.csv
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "extraction", "Data")
OUT_DIR   = os.path.join(REPO_ROOT, "phase3_internals", "ml_classifier")

JUDGED_FILE = os.path.join(DATA_DIR, "llama8b_judged.parquet")
CACHE_DIR   = os.path.join(OUT_DIR, "cache")
OUTPUT_FILE = os.path.join(OUT_DIR, "learning_curve.csv")

LAYER = 16
FRACTIONS = [0.15, 0.30, 0.50, 0.70, 0.85, 1.00]
N_REPEATS = 3
N_FOLDS = 5
RANDOM_SEED = 42


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                           class_weight="balanced", random_state=RANDOM_SEED),
    )


def evaluate_subset(X, y, groups, idx):
    """Fixed 5-fold CV on the subsample; test folds stay within it."""
    cv = GroupKFold(n_splits=N_FOLDS)
    aucs = []
    for tr, te in cv.split(X[idx], y[idx], groups[idx]):
        m = make_model().fit(X[idx][tr], y[idx][tr])
        p = m.predict_proba(X[idx][te])[:, 1]
        if len(np.unique(y[idx][te])) > 1:
            aucs.append(roc_auc_score(y[idx][te], p))
    return np.mean(aucs) if aucs else np.nan


def main():
    df = pd.read_parquet(JUDGED_FILE)
    df = df[~df["is_safe_control"]]
    df = df[df["score"].notna()].reset_index(drop=True)

    X = np.load(os.path.join(CACHE_DIR, f"layer_{LAYER}.npy"))
    groups = df["source_prompt_id"].values
    unique_groups = df["source_prompt_id"].unique()

    targets = {
        "compliance": (df["score"] == 2).astype(int).values,
        "refusal_english": None,
    }

    # Refusal only makes sense on English — the detector is English-only.
    eng_mask = (df["language"] == "english").values

    rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for target_name in ["compliance", "refusal_english"]:
        if target_name == "compliance":
            y = targets["compliance"]
            sub_mask = np.ones(len(df), dtype=bool)
        else:
            y = df["is_refusal"].astype(int).values
            sub_mask = eng_mask

        pool_groups = df.loc[sub_mask, "source_prompt_id"].unique()

        print(f"\n{'=' * 60}")
        print(f"{target_name}: {sub_mask.sum()} rows, "
              f"{len(pool_groups)} groups, {y[sub_mask].mean()*100:.1f}% positive")
        print(f"{'=' * 60}")

        for frac in FRACTIONS:
            scores = []
            for rep in range(N_REPEATS if frac < 1.0 else 1):
                n_groups = max(N_FOLDS, int(len(pool_groups) * frac))
                chosen = rng.choice(pool_groups, size=n_groups, replace=False)
                idx = np.where(sub_mask & df["source_prompt_id"].isin(chosen).values)[0]

                if len(np.unique(y[idx])) < 2:
                    continue
                scores.append(evaluate_subset(X, y, groups, idx))

            if not scores:
                continue

            n_rows = int(sub_mask.sum() * frac)
            print(f"  {frac*100:>5.0f}%  n≈{n_rows:>6}  "
                  f"AUC {np.mean(scores):.3f} ± {np.std(scores):.3f}")

            rows.append({"target": target_name, "fraction": frac,
                         "n_rows": n_rows, "auc_mean": np.mean(scores),
                         "auc_std": np.std(scores), "n_repeats": len(scores)})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")

    for target, g in results.groupby("target"):
        g = g.sort_values("fraction")
        last_two = g.tail(2)
        if len(last_two) < 2:
            continue
        delta = last_two["auc_mean"].iloc[1] - last_two["auc_mean"].iloc[0]
        frac_delta = last_two["fraction"].iloc[1] - last_two["fraction"].iloc[0]

        # Extrapolate the gain from the 8,747 unattacked rows (+45% of current)
        projected = delta / frac_delta * 0.45 if frac_delta else 0

        print(f"\n{target}:")
        print(f"  slope over the last segment: {delta:+.4f} AUC "
              f"per {frac_delta*100:.0f}% of data")
        print(f"  projected gain from +45% more rows: {projected:+.3f} AUC")

        if abs(projected) < 0.01:
            print(f"  → saturated. Attacking the remaining 8,747 rows is not "
                  f"worth the GPU time.")
        else:
            print(f"  → still climbing. The extra rows would help.")


if __name__ == "__main__":
    main()