"""
18_validate_probes.py
=====================
Validation suite for the linear probes.

Seven checks, from sanity to hardest generalisation:
  A  shuffled labels          — confirms no leakage in the pipeline
  B  per-source breakdown     — in-distribution, does it hold everywhere
  C  per-language breakdown   — in-distribution
  D  leave-one-source-out     — out-of-distribution, the real test
  E  cross-lingual transfer   — train English, test elsewhere
  F  direction consistency    — is the learned direction shared across languages
  G  calibration              — are the probabilities meaningful

Output: validation_results.csv
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "extraction", "Data")
OUT_DIR   = os.path.join(REPO_ROOT, "phase3_internals", "ml_classifier")

JUDGED_FILE   = os.path.join(DATA_DIR, "llama8b_judged.parquet")
GEOMETRY_FILE = os.path.join(DATA_DIR, "llama8b_geometry.json")
CACHE_DIR     = os.path.join(OUT_DIR, "cache")

RESULTS_FILE     = os.path.join(OUT_DIR, "validation_results.csv")
CALIBRATION_FILE = os.path.join(OUT_DIR, "calibration_data.csv")

# ── Configuration ─────────────────────────────────────────────────────────────

BEST_LAYER  = 16      # compliance peaks here; refusal is near-saturated
N_FOLDS     = 5
RANDOM_SEED = 42
MIN_GROUP_N = 100     # skip strata too small for a stable AUC

TARGETS = {
    "compliance": lambda df: (df["score"] == 2).astype(int),
    "refusal":    lambda df: df["is_refusal"].astype(int),
}


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                           class_weight="balanced", random_state=RANDOM_SEED),
    )


def safe_auc(y_true, proba):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, proba)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(JUDGED_FILE)
    df = df[~df["is_safe_control"]]
    df = df[df["score"].notna()].reset_index(drop=True)

    X = np.load(os.path.join(CACHE_DIR, f"layer_{BEST_LAYER}.npy"))
    assert len(X) == len(df), f"cache mismatch: {len(X)} vs {len(df)}"

    print(f"Rows: {len(df)}   features: {X.shape[1]}   layer: {BEST_LAYER}")
    return df, X


# ── A. Shuffled labels ────────────────────────────────────────────────────────

def check_shuffled(X, y, groups, target):
    rng = np.random.default_rng(RANDOM_SEED)
    y_shuf = rng.permutation(y)

    cv = GroupKFold(n_splits=N_FOLDS)
    aucs = []
    for tr, te in cv.split(X, y_shuf, groups):
        m = make_model().fit(X[tr], y_shuf[tr])
        aucs.append(safe_auc(y_shuf[te], m.predict_proba(X[te])[:, 1]))

    mean = np.mean(aucs)
    verdict = "OK" if abs(mean - 0.5) < 0.05 else "SOSPETTO"
    print(f"  shuffled labels: AUC {mean:.3f} — {verdict}")
    return [{"check": "shuffled_labels", "target": target,
             "stratum": "all", "auc": mean, "n": len(y)}]


# ── B/C. Per-stratum breakdown, in-distribution ───────────────────────────────

def check_per_stratum(X, y, groups, df, column, target):
    """Trained on everything, scored separately within each stratum."""
    cv = GroupKFold(n_splits=N_FOLDS)
    proba = np.zeros(len(y))

    for tr, te in cv.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]

    rows = []
    for stratum, idx in df.groupby(column).groups.items():
        idx = np.array(idx)
        if len(idx) < MIN_GROUP_N:
            continue
        auc = safe_auc(y[idx], proba[idx])
        rows.append({"check": f"per_{column}", "target": target,
                     "stratum": str(stratum), "auc": auc, "n": len(idx),
                     "positive_rate": float(y[idx].mean())})
        print(f"    {str(stratum):<30} n={len(idx):>5}  "
              f"pos={y[idx].mean()*100:>5.1f}%  AUC={auc:.3f}")
    return rows


# ── D. Leave-one-source-out ───────────────────────────────────────────────────

def check_leave_one_source_out(X, y, df, target):
    """
    Hardest test: the probe never sees the held-out source during training.
    A drop here means it learned source-specific quirks rather than the
    underlying phenomenon.
    """
    rows = []
    for source in sorted(df["source"].unique()):
        te = np.where(df["source"] == source)[0]
        tr = np.where(df["source"] != source)[0]

        if len(te) < MIN_GROUP_N or len(np.unique(y[te])) < 2:
            continue

        m = make_model().fit(X[tr], y[tr])
        auc = safe_auc(y[te], m.predict_proba(X[te])[:, 1])

        rows.append({"check": "leave_one_source_out", "target": target,
                     "stratum": source, "auc": auc, "n": len(te),
                     "positive_rate": float(y[te].mean())})
        print(f"    held out {source:<30} n={len(te):>5}  AUC={auc:.3f}")
    return rows


# ── E. Cross-lingual transfer ─────────────────────────────────────────────────

def check_cross_lingual(X, y, df, target):
    """Train on English only, test on each other language."""
    tr = np.where(df["language"] == "english")[0]
    if len(tr) == 0:
        return []

    m = make_model().fit(X[tr], y[tr])
    rows = []

    for lang in sorted(df["language"].unique()):
        if lang == "english":
            continue
        te = np.where(df["language"] == lang)[0]
        if len(te) < MIN_GROUP_N or len(np.unique(y[te])) < 2:
            continue
        auc = safe_auc(y[te], m.predict_proba(X[te])[:, 1])
        rows.append({"check": "cross_lingual_from_english", "target": target,
                     "stratum": lang, "auc": auc, "n": len(te),
                     "positive_rate": float(y[te].mean())})
        print(f"    english → {lang:<12} n={len(te):>5}  AUC={auc:.3f}")
    return rows


# ── F. Direction consistency ──────────────────────────────────────────────────

def check_direction_consistency(X, y, df, target):
    """
    A probe per language, then cosine similarity between weight vectors. High
    similarity means the languages share one direction in activation space;
    low means each has its own.
    """
    directions = {}
    for lang in sorted(df["language"].unique()):
        idx = np.where(df["language"] == lang)[0]
        if len(idx) < 500 or len(np.unique(y[idx])) < 2:
            continue
        m = make_model().fit(X[idx], y[idx])
        w = m.named_steps["logisticregression"].coef_[0]
        directions[lang] = w / np.linalg.norm(w)

    if len(directions) < 2:
        return []

    langs = list(directions)
    print(f"    cosine similarity between per-language directions:")
    header = " " * 12 + "".join(f"{l[:6]:>8}" for l in langs)
    print(header)

    rows = []
    for a in langs:
        line = f"    {a[:8]:<8}"
        for b in langs:
            cos = float(np.dot(directions[a], directions[b]))
            line += f"{cos:>8.2f}"
            if a < b:
                rows.append({"check": "direction_cosine", "target": target,
                             "stratum": f"{a}|{b}", "auc": cos, "n": 0})
        print(line)

    off = [r["auc"] for r in rows]
    if off:
        print(f"    mean off-diagonal: {np.mean(off):.3f}")
    return rows


# ── G. Calibration ────────────────────────────────────────────────────────────

def check_calibration(X, y, groups, target):
    cv = GroupKFold(n_splits=N_FOLDS)
    proba = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]

    brier = brier_score_loss(y, proba)
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10)

    print(f"    Brier score: {brier:.4f}  (più basso è meglio)")

    cal = pd.DataFrame({"target": target, "mean_predicted": mean_pred,
                        "fraction_positive": frac_pos})
    return [{"check": "calibration_brier", "target": target,
             "stratum": "all", "auc": brier, "n": len(y)}], cal


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df, X = load_data()
    groups = df["source_prompt_id"].values

    all_rows = []
    all_cal = []
    t0 = time.time()

    for target_name, target_fn in TARGETS.items():
        y = target_fn(df).values
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target_name}  ({y.sum()} positives, {y.mean()*100:.1f}%)")
        print(f"{'=' * 70}")

        print(f"\n[A] Sanity — shuffled labels")
        all_rows += check_shuffled(X, y, groups, target_name)

        print(f"\n[B] Per source (in-distribution)")
        all_rows += check_per_stratum(X, y, groups, df, "source", target_name)

        print(f"\n[C] Per language (in-distribution)")
        all_rows += check_per_stratum(X, y, groups, df, "language", target_name)

        print(f"\n[D] Leave-one-source-out (out-of-distribution)")
        all_rows += check_leave_one_source_out(X, y, df, target_name)

        print(f"\n[E] Cross-lingual transfer")
        all_rows += check_cross_lingual(X, y, df, target_name)

        print(f"\n[F] Direction consistency")
        all_rows += check_direction_consistency(X, y, df, target_name)

        print(f"\n[G] Calibration")
        rows, cal = check_calibration(X, y, groups, target_name)
        all_rows += rows
        all_cal.append(cal)

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_FILE, index=False)
    pd.concat(all_cal).to_csv(CALIBRATION_FILE, index=False)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for target in TARGETS:
        sub = results[(results["target"] == target) &
                      (results["check"].isin(["per_source", "per_language",
                                              "leave_one_source_out",
                                              "cross_lingual_from_english"]))]
        print(f"\n{target}:")
        for check, g in sub.groupby("check"):
            vals = g["auc"].dropna()
            print(f"  {check:<30} AUC {vals.mean():.3f}  "
                  f"[{vals.min():.3f} – {vals.max():.3f}]")

    print(f"\nDone in {(time.time() - t0)/60:.1f} minutes")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()