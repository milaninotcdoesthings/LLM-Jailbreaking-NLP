"""
04_translate_nllb.py
====================
Builds the parallel multilingual arm by translating a stratified sample of the
English corpus into German, Russian, and Arabic.

Rationale. Natively generating in each language, as Chapter 2 did, confounds two
quantities: the target model's vulnerability in that language, and the
generator's fluency in it. The Chapter 2 ordering (German > Arabic > Russian)
inverts the Chapter 1 ordering (Russian > German > Arabic) for exactly this
reason. A parallel corpus holds request content fixed, so any cross-lingual
difference is attributable to the target model.

The Aya experiment already established that translation is not itself the source
of the effect: native-speaker prompts reproduce the same ordering as translated
ones. The Chapter 2 native corpus is retained as the contrasting arm.

Translation runs locally with NLLB-200-distilled-600M. No external service sees
the corpus, and nothing is silently filtered or rewritten in transit.

Output: ai_regeneration/data/corpus_multilingual.csv
"""

import os
import json
import hashlib

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "facebook/nllb-200-distilled-600M"

DATA_DIR    = "data"
INPUT_FILE  = os.path.join(DATA_DIR, "corpus_english.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "corpus_multilingual.csv")
STATS_FILE  = os.path.join(DATA_DIR, "translation_stats.json")

# FLORES-200 codes. NLLB will not accept plain ISO codes.
LANGUAGES = {
    "german":  "deu_Latn",
    "russian": "rus_Cyrl",
    "arabic":  "arb_Arab",
}
SRC_LANG = "eng_Latn"

PER_CATEGORY   = 100      # stratified sample size per category
BATCH_SIZE     = 16
MAX_LENGTH     = 400
CHECKPOINT_EVERY = 500    # rows

RANDOM_SEED = 42

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


# ── Sampling ──────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balanced across category, and within category across generator, so no
    language ends up over-representing one arm of the corpus.
    """
    picked = []

    for category, group in df.groupby("category"):
        sources = group["source"].unique()
        per_source = max(1, PER_CATEGORY // len(sources))

        for source in sources:
            subset = group[group["source"] == source]
            n = min(per_source, len(subset))
            picked.append(subset.sample(n=n, random_state=RANDOM_SEED))

    out = pd.concat(picked, ignore_index=True)
    print(f"Stratified sample: {len(out)} prompts")
    print(f"  categories: {out['category'].nunique()}")
    print(f"  per category: {out.groupby('category').size().min()}"
          f"–{out.groupby('category').size().max()}")
    return out


# ── Translation ───────────────────────────────────────────────────────────────

def translate_batch(texts, tokenizer, model, target_code) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    bos_id = tokenizer.convert_tokens_to_ids(target_code)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
            max_length=MAX_LENGTH,
            num_beams=4,
        )

    return tokenizer.batch_decode(output, skip_special_tokens=True)


def flag_suspicious(source: str, translation: str) -> str | None:
    """
    NLLB occasionally degenerates: empty output, or a collapse to a few tokens
    on long input. Cheap length heuristics catch most of it. Flagged rows are
    kept, not dropped, so they can be inspected.
    """
    if not translation or not translation.strip():
        return "empty"

    src_words = len(source.split())
    tgt_words = len(translation.split())
    if src_words >= 10 and tgt_words < src_words * 0.3:
        return "too_short"
    if tgt_words > src_words * 3:
        return "too_long"

    # A single token repeated is the classic degeneration mode
    tokens = translation.split()
    if len(tokens) > 5 and len(set(tokens)) <= 2:
        return "degenerate"

    return None


def run(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print(f"\nDevice: {device}")
    print(f"Loading {MODEL_ID} — first run downloads ~2.5GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    records = []
    stats = {"translated": 0, "flagged": {}}

    for lang_name, lang_code in LANGUAGES.items():
        print(f"\nTranslating to {lang_name} ({lang_code})...")
        flagged_here = 0

        for start in tqdm(range(0, len(sample), BATCH_SIZE),
                          desc=f"  {lang_name}", unit="batch"):
            chunk = sample.iloc[start:start + BATCH_SIZE]
            texts = chunk["prompt_text"].astype(str).tolist()

            try:
                translations = translate_batch(
                    texts, tokenizer, model, lang_code)
            except Exception as e:
                tqdm.write(f"    batch failed at row {start}: {e}")
                translations = [""] * len(texts)

            for (_, row), translated in zip(chunk.iterrows(), translations):
                flag = flag_suspicious(row["prompt_text"], translated)
                if flag:
                    flagged_here += 1

                records.append({
                    # New id for the translated text; source_prompt_id links back
                    # to the English original so the train/test split can keep
                    # all translations of one prompt on the same side.
                    "prompt_id":        hashlib.sha1(
                        translated.encode("utf-8")).hexdigest()[:16],
                    "source_prompt_id": row["prompt_id"],
                    "prompt_text":      translated,
                    "prompt_text_en":   row["prompt_text"],
                    "source":           f"{row['source']}_translated",
                    "language":         lang_name,
                    "category":         row["category"],
                    "generator":        row.get("generator"),
                    "translator":       MODEL_ID,
                    "quality_flag":     flag,
                })
                stats["translated"] += 1

            if len(records) % CHECKPOINT_EVERY < BATCH_SIZE:
                pd.DataFrame(records).to_csv(
                    OUTPUT_FILE.replace(".csv", "_checkpoint.csv"), index=False)

        stats["flagged"][lang_name] = flagged_here
        print(f"  flagged for review: {flagged_here}")

    return pd.DataFrame(records), stats


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, stats: dict, sample_size: int) -> None:
    print("\n" + "=" * 62)
    print("TRANSLATION REPORT")
    print("=" * 62)
    print(f"English prompts sampled    {sample_size}")
    print(f"Translations produced      {stats['translated']}")
    print(f"Languages                  {', '.join(LANGUAGES)}")

    print(f"\nFlagged for review:")
    for lang, n in stats["flagged"].items():
        pct = n / sample_size * 100 if sample_size else 0
        print(f"  {lang:<10} {n:>4}  ({pct:.1f}%)")

    if "quality_flag" in df.columns:
        reasons = df["quality_flag"].dropna().value_counts()
        if len(reasons):
            print(f"\nFlag reasons:")
            for reason, n in reasons.items():
                print(f"  {n:>4}  {reason}")

    print(f"\nSpot-check these before proceeding. Print a few rows per language:")
    print(f"  df[df.language=='arabic'].sample(5)[['prompt_text_en','prompt_text']]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"{INPUT_FILE} not found — run 03_deduplicate_merge.py first.")

    corpus = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(corpus)} English prompts")

    sample = stratified_sample(corpus)
    translated, stats = run(sample)

    translated.to_csv(OUTPUT_FILE, index=False)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    checkpoint = OUTPUT_FILE.replace(".csv", "_checkpoint.csv")
    if os.path.exists(checkpoint):
        os.remove(checkpoint)

    report(translated, stats, len(sample))
    print(f"\nSaved {len(translated)} translations to {OUTPUT_FILE}")