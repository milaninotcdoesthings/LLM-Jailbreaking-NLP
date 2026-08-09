"""
17_train_probes.py
==================
Trains linear probes on Llama 3.1 8B internal representations and compares
them against surface features.

Chapter 1 found surface features do not predict attack success and concluded
the information lives in the model's representation rather than the text.
This tests that claim directly.

Output: probe_results.csv
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "extraction", "Data")
OUT_DIR   = os.path.join(REPO_ROOT, "phase3_internals", "ml_classifier")

JUDGED_FILE   = os.path.join(DATA_DIR, "llama8b_judged.parquet")
HIDDEN_FILE   = os.path.join(DATA_DIR, "llama8b_hidden_states.h5")
GEOMETRY_FILE = os.path.join(DATA_DIR, "llama8b_geometry.json")

RESULTS_FILE = os.path.join(OUT_DIR, "probe_results.csv")
CACHE_DIR    = os.path.join(OUT_DIR, "cache")

# ── Configuration ─────────────────────────────────────────────────────────────

N_FOLDS = 5
RANDOM_SEED = 42

# score==2 is a property of the prompt-model pair; is_refusal is a property of
# the model's disposition. They are not interchangeable — see the judge/detector
# cross-check, where score==0 covered ~12k rows the model simply answered.
TARGETS = {
    "compliance": lambda df: (df["score"] == 2).astype(int),
    "refusal":    lambda df: df["is_refusal"].astype(int),
}

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

RUN_NONLINEAR_CONTROL = True


# ── Loading ───────────────────────────────────────────────────────────────────

def load_labels() -> pd.DataFrame:
    df = pd.read_parquet(JUDGED_FILE)
    print(f"Rows loaded: {len(df)}")

    before = len(df)
    df = df[~df["is_safe_control"]]
    print(f"  dropped {before - len(df)} XSTest safe controls")

    before = len(df)
    df = df[df["score"].notna()]
    if before != len(df):
        print(f"  dropped {before - len(df)} without a verdict")

    df = df.reset_index(drop=True)

    print(f"\nUsable rows: {len(df)}")
    for name, fn in TARGETS.items():
        y = fn(df)
        print(f"  {name:<12} positives: {y.sum()} ({y.mean()*100:.1f}%)")

    print(f"\nGroups (source_prompt_id): {df['source_prompt_id'].nunique()}")
    return df


def load_layer(prompt_ids: list[str], layer: int) -> np.ndarray:
    """Reads one layer from HDF5, cached as .npy — gzip reads are slow."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"layer_{layer}.npy")

    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        if len(arr) == len(prompt_ids):
            return arr
        print(f"  cache for layer {layer} is stale, rebuilding")

    print(f"  reading layer {layer} from HDF5...")
    vectors = []
    with h5py.File(HIDDEN_FILE, "r") as f:
        for pid in tqdm(prompt_ids, desc=f"    layer {layer}", leave=False):
            vectors.append(f[pid][f"layer_{layer}"][:])

    arr = np.vstack(vectors).astype(np.float32)
    np.save(cache_path, arr)
    return arr


def build_surface_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        print(f"  missing columns, ignored: {missing}")
    X = df[available].apply(pd.to_numeric, errors="coerce").values
    return SimpleImputer(strategy="median").fit_transform(X)


def build_embedding_matrix(df: pd.DataFrame) -> np.ndarray | None:
    if "embedding_native" not in df.columns:
        return None
    return np.vstack(df["embedding_native"].values).astype(np.float32)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             label: str, nonlinear: bool = False) -> list[dict]:
    """
    Grouped CV on source_prompt_id: language variants of one prompt are near
    duplicates, so a row-level split would leak them across train and test.
    """
    cv = GroupKFold(n_splits=N_FOLDS)
    rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        if nonlinear:
            model = HistGradientBoostingClassifier(
                max_iter=200, random_state=RANDOM_SEED)
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2", C=1.0, max_iter=2000,
                    class_weight="balanced", random_state=RANDOM_SEED),
            )

        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)

        rows.append({
            "fold": fold,
            "auc": roc_auc_score(y[te], proba),
            "f1": f1_score(y[te], pred, zero_division=0),
            "accuracy": accuracy_score(y[te], pred),
            "n_train": len(tr),
            "n_test": len(te),
            "positive_rate_test": float(y[te].mean()),
        })

    aucs = [r["auc"] for r in rows]
    print(f"    {label:<34} AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GEOMETRY_FILE) as f:
        geom = json.load(f)
    n_layers = geom["n_layers"]
    layer_indices = geom["layer_indices"]
    print(f"Model: {geom['model']}")
    print(f"  {n_layers} layers, hidden dim {geom['hidden_dim']}")
    print(f"  extracted layers: {layer_indices}\n")

    df = load_labels()
    prompt_ids = df["prompt_id"].tolist()
    groups = df["source_prompt_id"].values

    print(f"\nBuilding feature matrices...")
    feature_sets = {}

    X_surf = build_surface_matrix(df, SURFACE_FEATURES)
    feature_sets["surface"] = X_surf
    print(f"  surface: {X_surf.shape}")

    X_behav = build_surface_matrix(df, BEHAVIOURAL_FEATURES)
    feature_sets["behavioural"] = X_behav
    print(f"  behavioural: {X_behav.shape}")

    X_emb = build_embedding_matrix(df)
    if X_emb is not None:
        feature_sets["prompt_embedding"] = X_emb
        print(f"  prompt_embedding: {X_emb.shape}")

    for layer in layer_indices:
        X = load_layer(prompt_ids, layer)
        feature_sets[f"hidden_layer_{layer}"] = X
        print(f"  hidden_layer_{layer}: {X.shape}")

    all_results = []
    t_start = time.time()

    for target_name, target_fn in TARGETS.items():
        y = target_fn(df).values
        print(f"\n{'=' * 66}")
        print(f"TARGET: {target_name}  "
              f"({y.sum()} positives of {len(y)}, {y.mean()*100:.1f}%)")
        print(f"{'=' * 66}")

        for fs_name, X in feature_sets.items():
            rows = evaluate(X, y, groups, fs_name)
            layer = (int(fs_name.split("_")[-1])
                     if fs_name.startswith("hidden_layer_") else None)
            for r in rows:
                all_results.append({
                    "target": target_name,
                    "feature_set": fs_name,
                    "layer": layer,
                    # Relative depth makes the curve comparable against a model
                    # with a different layer count (Qwen has 36, not 32).
                    "layer_pct": (layer / (n_layers - 1) * 100
                                  if layer is not None else None),
                    "model_type": "logistic",
                    **r,
                })

        if RUN_NONLINEAR_CONTROL:
            best = max(
                (fs for fs in feature_sets if fs.startswith("hidden_layer_")),
                key=lambda fs: np.mean([
                    r["auc"] for r in all_results
                    if r["target"] == target_name and r["feature_set"] == fs]),
            )
            print(f"\n  non-linear control on {best}:")
            rows = evaluate(feature_sets[best], y, groups,
                            f"{best} (gradient boosting)", nonlinear=True)
            layer = int(best.split("_")[-1])
            for r in rows:
                all_results.append({
                    "target": target_name, "feature_set": best,
                    "layer": layer,
                    "layer_pct": layer / (n_layers - 1) * 100,
                    "model_type": "gradient_boosting",
                    **r,
                })

    results = pd.DataFrame(all_results)
    results.to_csv(RESULTS_FILE, index=False)

    print(f"\n{'=' * 66}")
    print(f"SUMMARY  (mean AUC over {N_FOLDS} folds)")
    print(f"{'=' * 66}")
    summary = (results[results["model_type"] == "logistic"]
               .groupby(["target", "feature_set"])["auc"]
               .agg(["mean", "std"]).round(3))
    print(summary.to_string())

    print(f"\nDone in {(time.time() - t_start)/60:.1f} minutes")
    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()