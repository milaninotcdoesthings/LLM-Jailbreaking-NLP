"""
21_probe_adversarial.py
=======================
Retrains the probes on the adversarial subset only.

The learning curve showed both targets saturate by ~30% of the data, so the
0.80 ceiling on compliance is not a sample-size limit. The likely cause is task
definition: ~12,000 of 19,289 rows carry score 0 without the model having
refused anything, because the prompt asked for nothing harmful. The probe is
then learning two questions at once — is this prompt dangerous, and will the
model yield — and the first dilutes the second.

Restricting to rows where the model took a safety-relevant position (refused,
or produced partial or full compliance) isolates the second question.

Reports both framings side by side: the full pool result is the conservative
one, this is the sharper one.

Output: probe_adversarial_results.csv
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
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "extraction", "Data")
OUT_DIR   = os.path.join(REPO_ROOT, "phase3_internals", "ml_classifier")

JUDGED_FILE   = os.path.join(DATA_DIR, "llama8b_judged.parquet")
GEOMETRY_FILE = os.path.join(DATA_DIR, "llama8b_geometry.json")
CACHE_DIR     = os.path.join(OUT_DIR, "cache")
RESULTS_FILE  = os.path.join(OUT_DIR, "probe_adversarial_results.csv")

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


def cv_eval(X, y, groups, label, nonlinear=False):
    cv = GroupKFold(n_splits=N_FOLDS)
    aucs, f1s = [], []
    for tr, te in cv.split(X, y, groups):
        m = (HistGradientBoostingClassifier(max_iter=200,
                                            random_state=RANDOM_SEED)
             if nonlinear else make_model())
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(safe_auc(y[te], p))
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int), zero_division=0))
    print(f"    {label:<36} AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return np.mean(aucs), np.std(aucs), np.mean(f1s)


def build_sets(df, mask, layer_indices):
    def surface(cols):
        avail = [c for c in cols if c in df.columns]
        X = df[avail].apply(pd.to_numeric, errors="coerce").values
        return SimpleImputer(strategy="median").fit_transform(X)

    sets = {"surface": surface(SURFACE_FEATURES),
            "behavioural": surface(BEHAVIOURAL_FEATURES)}
    if "embedding_native" in df.columns:
        sets["prompt_embedding"] = np.vstack(
            df["embedding_native"].values).astype(np.float32)
    for layer in layer_indices:
        full = np.load(os.path.join(CACHE_DIR, f"layer_{layer}.npy"))
        sets[f"hidden_layer_{layer}"] = full[mask]
    return sets


def main():
    with open(GEOMETRY_FILE) as f:
        geom = json.load(f)
    n_layers = geom["n_layers"]
    layer_indices = geom["layer_indices"]

    full = pd.read_parquet(JUDGED_FILE)
    full = full[~full["is_safe_control"]]
    full = full[full["score"].notna()].reset_index(drop=True)

    # A row is adversarial if the model took a safety-relevant position:
    # it refused, hedged (score 1), or complied (score 2). Rows where it simply
    # answered a harmless question are excluded.
    adv_mask = (full["score"].isin([1, 2]) | full["is_refusal"]).values
    df = full[adv_mask].reset_index(drop=True)

    print(f"Full pool:          {len(full)} rows")
    print(f"Adversarial subset: {len(df)} rows "
          f"({len(df)/len(full)*100:.1f}%)")
    print(f"  refused:           {df['is_refusal'].sum()}")
    print(f"  partial (score 1): {(df['score'] == 1).sum()}")
    print(f"  compliance (2):    {(df['score'] == 2).sum()}")
    print(f"\nExcluded: {len(full) - len(df)} rows the model answered without "
          f"a safety-relevant decision")

    y = (df["score"] == 2).astype(int).values
    groups = df["source_prompt_id"].values

    print(f"\n{'=' * 70}")
    print(f"TARGET: compliance within adversarial subset "
          f"({y.sum()} positives, {y.mean()*100:.1f}%)")
    print(f"{'=' * 70}\n")

    sets = build_sets(df, adv_mask, layer_indices)
    rows = []

    for name, X in sets.items():
        auc, std, f1 = cv_eval(X, y, groups, name)
        layer = (int(name.split("_")[-1])
                 if name.startswith("hidden_layer_") else None)
        rows.append({"analysis": "adversarial_subset", "feature_set": name,
                     "layer": layer,
                     "layer_pct": (layer / (n_layers - 1) * 100
                                   if layer is not None else None),
                     "auc": auc, "auc_std": std, "f1": f1, "n": len(y)})

    best = max((k for k in sets if k.startswith("hidden_layer_")),
               key=lambda k: [r["auc"] for r in rows
                              if r["feature_set"] == k][0])
    X_best = sets[best]

    print(f"\n  non-linear control on {best}:")
    auc_nl, std_nl, f1_nl = cv_eval(X_best, y, groups,
                                    f"{best} (gradient boosting)",
                                    nonlinear=True)
    rows.append({"analysis": "adversarial_nonlinear", "feature_set": best,
                 "layer": int(best.split("_")[-1]), "layer_pct": None,
                 "auc": auc_nl, "auc_std": std_nl, "f1": f1_nl, "n": len(y)})

    # ── Generalisation checks on the sharper task ────────────────────────────
    print(f"\nPer language ({best}):")
    cv = GroupKFold(n_splits=N_FOLDS)
    proba = np.zeros(len(y))
    for tr, te in cv.split(X_best, y, groups):
        m = make_model().fit(X_best[tr], y[tr])
        proba[te] = m.predict_proba(X_best[te])[:, 1]

    for lang, idx in df.groupby("language").groups.items():
        idx = np.array(idx)
        if len(idx) < MIN_GROUP_N:
            continue
        auc = safe_auc(y[idx], proba[idx])
        print(f"    {lang:<12} n={len(idx):>5}  pos={y[idx].mean()*100:>5.1f}%"
              f"  AUC={auc:.3f}")
        rows.append({"analysis": "adv_per_language", "feature_set": best,
                     "layer": None, "layer_pct": None, "auc": auc,
                     "auc_std": np.nan, "f1": np.nan, "n": len(idx),
                     "stratum": lang})

    print(f"\nLeave-one-source-out ({best}):")
    for src in sorted(df["source"].unique()):
        te = np.where(df["source"] == src)[0]
        tr = np.where(df["source"] != src)[0]
        if len(te) < MIN_GROUP_N or len(np.unique(y[te])) < 2:
            continue
        m = make_model().fit(X_best[tr], y[tr])
        auc = safe_auc(y[te], m.predict_proba(X_best[te])[:, 1])
        print(f"    held out {src:<30} n={len(te):>5}  AUC={auc:.3f}")
        rows.append({"analysis": "adv_loso", "feature_set": best,
                     "layer": None, "layer_pct": None, "auc": auc,
                     "auc_std": np.nan, "f1": np.nan, "n": len(te),
                     "stratum": src})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)

    print(f"\n{'=' * 70}")
    print("COMPARISON — full pool vs adversarial subset")
    print(f"{'=' * 70}")
    print(f"{'feature set':<22} {'full pool':>12} {'adversarial':>14}")
    reference = {"surface": 0.671, "behavioural": 0.606,
                 "prompt_embedding": 0.741, "hidden_layer_8": 0.719,
                 "hidden_layer_16": 0.797, "hidden_layer_24": 0.794,
                 "hidden_layer_31": 0.792}
    for name in reference:
        adv = results[(results["analysis"] == "adversarial_subset") &
                      (results["feature_set"] == name)]["auc"]
        if len(adv):
            print(f"{name:<22} {reference[name]:>12.3f} {adv.iloc[0]:>14.3f}")

    print(f"\nResults: {RESULTS_FILE}")


if __name__ == "__main__":
    main()