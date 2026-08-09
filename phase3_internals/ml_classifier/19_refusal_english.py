"""
19_refusal_english.py
=====================
Re-runs the refusal probe on English rows only.

The is_refusal detector matches English refusal markers, so it finds nothing in
Arabic, French, German or Russian — where the model refuses in those languages.
Positives were therefore ~100% English, and a probe trained on the full pool
could reach 0.98 AUC largely by detecting language. Restricting to English
removes that confound.

Also reports the same validation checks as 18, plus a language-detection
control that quantifies how much of the original number was language.

Output: refusal_english_results.csv
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
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "extraction", "Data")
OUT_DIR   = os.path.join(REPO_ROOT, "phase3_internals", "ml_classifier")

JUDGED_FILE   = os.path.join(DATA_DIR, "llama8b_judged.parquet")
GEOMETRY_FILE = os.path.join(DATA_DIR, "llama8b_geometry.json")
CACHE_DIR     = os.path.join(OUT_DIR, "cache")
RESULTS_FILE  = os.path.join(OUT_DIR, "refusal_english_results.csv")

# ── Configuration ─────────────────────────────────────────────────────────────

N_FOLDS = 5
RANDOM_SEED = 42
MIN_GROUP_N = 100

SURFACE_FEATURES = [
    "toxicity_native", "toxicity_source",
    "perplexity_native", "perplexity_norm",
    "token_length_native", "token_length_source",
    "flesch_source", "avg_word_length_native", "ttr_native",
    "sentiment_multi_score",
]
BEHAVIOURAL_FEATURES = [
    "first_token_entropy", "first_token_top1_prob", "first_token_prob_gap",
]


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                           class_weight="balanced", random_state=RANDOM_SEED),
    )


def safe_auc(y, p):
    return np.nan if len(np.unique(y)) < 2 else roc_auc_score(y, p)


def cv_eval(X, y, groups, label):
    cv = GroupKFold(n_splits=N_FOLDS)
    aucs, f1s = [], []
    for tr, te in cv.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(safe_auc(y[te], p))
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int), zero_division=0))
    print(f"    {label:<34} AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return np.mean(aucs), np.std(aucs), np.mean(f1s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(GEOMETRY_FILE) as f:
        geom = json.load(f)
    n_layers = geom["n_layers"]
    layer_indices = geom["layer_indices"]

    df_all = pd.read_parquet(JUDGED_FILE)
    df_all = df_all[~df_all["is_safe_control"]]
    df_all = df_all[df_all["score"].notna()].reset_index(drop=True)

    print("Refusal detection by language (full pool):")
    by_lang = df_all.groupby("language")["is_refusal"].agg(["sum", "mean", "size"])
    for lang, r in by_lang.iterrows():
        print(f"  {lang:<10} {int(r['sum']):>5} / {int(r['size']):<5} "
              f"({r['mean']*100:>5.1f}%)")
    print("  The near-zero non-English rates are a detector artefact, not model"
          " behaviour.\n")

    # ── Language-detection control ───────────────────────────────────────────
    # Quantifies how much of the original 0.98 was separable by language alone.
    print("Control: can the probe recover is_refusal from language alone?")
    lang_dummies = pd.get_dummies(df_all["language"]).values.astype(float)
    y_all = df_all["is_refusal"].astype(int).values
    groups_all = df_all["source_prompt_id"].values
    auc_lang, _, _ = cv_eval(lang_dummies, y_all, groups_all, "language one-hot")
    print(f"    An AUC near the reported 0.982 confirms the confound.\n")

    # ── English subset ───────────────────────────────────────────────────────
    mask = (df_all["language"] == "english").values
    df = df_all[mask].reset_index(drop=True)
    y = df["is_refusal"].astype(int).values
    groups = df["source_prompt_id"].values

    print(f"{'=' * 66}")
    print(f"ENGLISH ONLY — {len(df)} rows, {y.sum()} refusals "
          f"({y.mean()*100:.1f}%)")
    print(f"{'=' * 66}\n")

    rows = []

    print("Feature sets:")
    from sklearn.impute import SimpleImputer

    def surface_matrix(cols):
        avail = [c for c in cols if c in df.columns]
        X = df[avail].apply(pd.to_numeric, errors="coerce").values
        return SimpleImputer(strategy="median").fit_transform(X)

    sets = {
        "surface": surface_matrix(SURFACE_FEATURES),
        "behavioural": surface_matrix(BEHAVIOURAL_FEATURES),
    }
    if "embedding_native" in df.columns:
        sets["prompt_embedding"] = np.vstack(
            df["embedding_native"].values).astype(np.float32)

    for layer in layer_indices:
        full = np.load(os.path.join(CACHE_DIR, f"layer_{layer}.npy"))
        sets[f"hidden_layer_{layer}"] = full[mask]

    for name, X in sets.items():
        auc, std, f1 = cv_eval(X, y, groups, name)
        layer = (int(name.split("_")[-1])
                 if name.startswith("hidden_layer_") else None)
        rows.append({"analysis": "english_only", "feature_set": name,
                     "layer": layer,
                     "layer_pct": (layer / (n_layers - 1) * 100
                                   if layer is not None else None),
                     "auc": auc, "auc_std": std, "f1": f1, "n": len(y)})

    # ── Per-source within English ────────────────────────────────────────────
    best = max((k for k in sets if k.startswith("hidden_layer_")),
               key=lambda k: [r["auc"] for r in rows
                              if r["feature_set"] == k][0])
    X_best = sets[best]

    print(f"\nPer source, within English (layer from {best}):")
    cv = GroupKFold(n_splits=N_FOLDS)
    proba = np.zeros(len(y))
    for tr, te in cv.split(X_best, y, groups):
        m = make_model().fit(X_best[tr], y[tr])
        proba[te] = m.predict_proba(X_best[te])[:, 1]

    for src, idx in df.groupby("source").groups.items():
        idx = np.array(idx)
        if len(idx) < MIN_GROUP_N:
            continue
        auc = safe_auc(y[idx], proba[idx])
        print(f"    {src:<30} n={len(idx):>5}  "
              f"pos={y[idx].mean()*100:>5.1f}%  AUC={auc:.3f}")
        rows.append({"analysis": "per_source_english", "feature_set": best,
                     "layer": int(best.split("_")[-1]),
                     "layer_pct": None, "auc": auc, "auc_std": np.nan,
                     "f1": np.nan, "n": len(idx), "stratum": src})

    # ── Leave-one-source-out within English ──────────────────────────────────
    print(f"\nLeave-one-source-out, within English:")
    for src in sorted(df["source"].unique()):
        te = np.where(df["source"] == src)[0]
        tr = np.where(df["source"] != src)[0]
        if len(te) < MIN_GROUP_N or len(np.unique(y[te])) < 2:
            continue
        m = make_model().fit(X_best[tr], y[tr])
        auc = safe_auc(y[te], m.predict_proba(X_best[te])[:, 1])
        print(f"    held out {src:<30} n={len(te):>5}  AUC={auc:.3f}")
        rows.append({"analysis": "loso_english", "feature_set": best,
                     "layer": int(best.split("_")[-1]), "layer_pct": None,
                     "auc": auc, "auc_std": np.nan, "f1": np.nan,
                     "n": len(te), "stratum": src})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)

    print(f"\n{'=' * 66}")
    print("SUMMARY — English only")
    print(f"{'=' * 66}")
    main_rows = results[results["analysis"] == "english_only"]
    print(main_rows[["feature_set", "auc", "auc_std"]]
          .round(3).to_string(index=False))

    print(f"\nLanguage-only control (full pool): AUC {auc_lang:.3f}")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()