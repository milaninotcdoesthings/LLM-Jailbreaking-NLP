"""
remap_categories.py - map benchmark categories onto the 10 synthetic ones.

The 10 target categories (same as the synthetic dataset):
  cyber, fraud, weapons, drugs, harassment, hate, privacy,
  misinfo, selfharm, extremism
Plus:
  hard_negative  (XSTest safe: benign but scary-sounding)
  other          (doesn't fit any of the above)

Rules where the mapping is clear; zero-shot (same mDeBERTa as phase 3)
for AdvBench prompts and anything ambiguous.
"""

import pandas as pd
import torch
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas()
BASE = Path("/Users/tommasomilanino/Developer/THESIS")
F = BASE / "benchmarks_metrics.parquet"
device = "mps" if torch.backends.mps.is_available() else "cpu"

TARGETS = ["cyber", "fraud", "weapons", "drugs", "harassment",
           "hate", "privacy", "misinfo", "selfharm", "extremism"]

# human-readable hypotheses for zero-shot (better than bare keywords)
ZS_LABELS = {
    "cyber": "hacking, malware, or breaking into computer systems",
    "fraud": "fraud, scams, phishing, or forgery",
    "weapons": "weapons, firearms, or explosives",
    "drugs": "illegal drugs or controlled substances",
    "harassment": "harassment, threats, stalking, or doxxing",
    "hate": "hate speech or discrimination against a group",
    "privacy": "surveilling or extracting private data about a person",
    "misinfo": "fabricating news, fake studies, or disinformation",
    "selfharm": "self-harm, suicide, or eating disorders",
    "extremism": "violent extremism or terrorist propaganda",
}
HYP = list(ZS_LABELS.values())
HYP2CAT = {v: k for k, v in ZS_LABELS.items()}

df = pd.read_parquet(F)
if "category_raw" not in df.columns:
    df = df.rename(columns={"category": "category_raw"})

# --- step 1: rules where it's unambiguous ------------------------------
def rule_map(row):
    src, cat = row["source"], str(row["category_raw"])
    # XSTest safe = hard negative; XSTest unsafe left to zero-shot
    if src == "xstest":
        return "hard_negative" if not cat.startswith("contrast") else None
    # Do-Not-Answer: only the clearly-mappable risk areas
    if src == "do_not_answer":
        return {
            "Discrimination, Exclusion, Toxicity, Hateful, Offensive": "hate",
            "Misinformation Harms": "misinfo",
        }.get(cat, None)   # the rest (Info Hazards, Malicious Uses...) too broad -> zero-shot
    return None            # advbench + everything else -> zero-shot

df["risk_category"] = df.apply(rule_map, axis=1)
n_todo = df["risk_category"].isna().sum()
print(f"[+] {(~df['risk_category'].isna()).sum()} mapped by rules, {n_todo} need zero-shot")

# --- step 2: zero-shot for the rest ------------------------------------
clf = pipeline("zero-shot-classification",
               model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=device)

def zshot(text):
    try:
        r = clf(str(text)[:512], HYP, multi_label=False)
        top_label, top_score = r["labels"][0], r["scores"][0]
        # if the model is unsure, call it "other" instead of forcing a bin
        return HYP2CAT[top_label] if top_score >= 0.40 else "other"
    except Exception:
        return "other"

mask = df["risk_category"].isna()
df.loc[mask, "risk_category"] = df.loc[mask, "raw_prompt"].progress_apply(zshot)

# --- report ------------------------------------------------------------
print("\nfinal distribution:")
print(df["risk_category"].value_counts().to_string())
print(f"\n'other' share: {(df.risk_category=='other').mean():.1%}")

df.to_parquet(F)
print(f"\n[+] saved -> {F.name}")