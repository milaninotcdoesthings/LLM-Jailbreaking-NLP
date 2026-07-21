import pandas as pd
import torch
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

device = "mps" if torch.backends.mps.is_available() else "cpu"

def free_memory():
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

def save_checkpoint(df, phase):
    path = f"checkpoint_phase_{phase}.parquet"
    df.to_parquet(path, engine='pyarrow')
    print(f"💾 Checkpoint saved: {path}")

def _apply_sentiment_loop(df, sentiment_models, device):
    """Shared sentiment logic used by both pipeline functions."""
    df['sentiment_score'] = None
    df['sentiment_label'] = None

    for lang, model_name in sentiment_models.items():
        print(f"  -> Loading sentiment model for: {lang.upper()} ({model_name})")
        mask = df['language'].str.lower() == lang

        if mask.sum() > 0:
            try:
                sent_pipe = pipeline("text-classification", model=model_name, device=device)

                def apply_sentiment(text):
                    try:
                        res = sent_pipe(str(text)[:512])
                        return pd.Series([res[0]['score'], res[0]['label'].lower()])
                    except Exception as e:
                        print(f"[SENTIMENT ERROR] {e}")
                        return pd.Series([None, None])

                results = df[mask]['raw_prompt'].progress_apply(apply_sentiment)
                print(f"  Results shape: {results.shape}, head:\n{results.head(2)}")
                df.loc[mask, 'sentiment_score'] = results.iloc[:, 0].values
                df.loc[mask, 'sentiment_label'] = results.iloc[:, 1].values

                del sent_pipe
                free_memory()
                save_checkpoint(df, f"5_{lang}")
            except Exception as e:
                print(f"  ❌ Sentiment failed for {lang}: {e}")

    return df

def run_metrics_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🚀 Device in use: {device}")

    df = df.dropna(subset=['score_8B', 'score_70B']).copy()
    print(f"Rows retained after filtering: {len(df)}")

    # ── PHASE 1: Token Length ─────────────────────────────────────────────────
    try:
        print("\n=== PHASE 1: Token Length Computation ===")
        tokenizer_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        df['prompt_token_length'] = df['raw_prompt'].progress_apply(
            lambda x: len(tokenizer_qwen.encode(str(x)))
        )
        del tokenizer_qwen
        free_memory()
        save_checkpoint(df, 1)
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        save_checkpoint(df, "1_failed")

    # ── PHASE 2: Semantic Embeddings ──────────────────────────────────────────
    try:
        print("\n=== PHASE 2: Embedding Extraction ===")
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
        embeddings = embedder.encode(
            df['raw_prompt'].tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        df['semantic_embedding'] = list(embeddings)
        del embedder
        free_memory()
        save_checkpoint(df, 2)
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        save_checkpoint(df, "2_failed")

    # ── PHASE 3: Zero-Shot Intent Classification ──────────────────────────────
    try:
        print("\n=== PHASE 3: Zero-Shot Intent Classification ===")
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=device
        )
        candidate_labels = ["factual", "procedural", "opinion", "causal"]

        def get_intent(text):
            try:
                res = classifier(str(text), candidate_labels, multi_label=False)
                return res['labels'][0]
            except Exception:
                return None

        df['intent_category'] = df['raw_prompt'].progress_apply(get_intent)
        del classifier
        free_memory()
        save_checkpoint(df, 3)
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        save_checkpoint(df, "3_failed")

    # ── PHASE 4: Toxicity Score ───────────────────────────────────────────────
    try:
        print("\n=== PHASE 4: Toxicity Score Computation ===")
        toxicity_pipe = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            device=device
        )

        def get_toxicity(text):
            try:
                res = toxicity_pipe(str(text)[:512], top_k=None)
                for label_dict in res:
                    if label_dict['label'].lower() == 'toxic':
                        return label_dict['score']
                return 0.0
            except Exception as e:
                print(f"[TOXICITY ERROR] {e}")
                return None

        df['toxicity_score'] = df['raw_prompt'].progress_apply(get_toxicity)
        del toxicity_pipe
        free_memory()
        save_checkpoint(df, 4)
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        save_checkpoint(df, "4_failed")

    # ── PHASE 5: Language-Specific Sentiment Analysis ─────────────────────────
    try:
        print("\n=== PHASE 5: Language-Specific Sentiment Analysis ===")
        sentiment_models = {
            'arabic': "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
            'german': "oliverguhr/german-sentiment-bert",
            'russian': "BlancheFort/rubert-base-cased-sentiment"
        }
        df = _apply_sentiment_loop(df, sentiment_models, device)
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        save_checkpoint(df, "5_failed")

    # ── PHASE 6: Perplexity Score ─────────────────────────────────────────────
    try:
        print("\n=== PHASE 6: Perplexity Score Computation ===")
        perp_model_id = "Qwen/Qwen2.5-0.5B"
        perp_tokenizer = AutoTokenizer.from_pretrained(perp_model_id)
        perp_model = AutoModelForCausalLM.from_pretrained(
            perp_model_id,
            torch_dtype=torch.float16
        ).to(device)

        def calculate_perplexity(text):
            try:
                inputs = perp_tokenizer(str(text), return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = perp_model(**inputs, labels=inputs["input_ids"])
                    return torch.exp(outputs.loss).item()
            except Exception as e:
                print(f"[PERPLEXITY ERROR] {e}")
                return None

        df['perplexity_score'] = df['raw_prompt'].progress_apply(calculate_perplexity)
        del perp_model, perp_tokenizer
        free_memory()
        save_checkpoint(df, 6)
    except Exception as e:
        print(f"❌ Phase 6 failed: {e}")
        save_checkpoint(df, "6_failed")

    output_file = "native_multi_metrics.parquet"
    df.to_parquet(output_file, engine='pyarrow')
    print(f"\n✅ Pipeline completed! Data saved to: {output_file}")

    return df


def run_missing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🚀 Device in use: {device}")

    # ── PHASE 4: Toxicity Score ───────────────────────────────────────────────
    try:
        print("\n=== PHASE 4: Toxicity Score Computation ===")
        toxicity_pipe = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            device=device
        )

        def get_toxicity(text):
            try:
                res = toxicity_pipe(str(text)[:512], top_k=None)
                for label_dict in res:
                    if label_dict['label'].lower() == 'toxic':
                        return label_dict['score']
                return 0.0
            except Exception as e:
                print(f"[TOXICITY ERROR] {e}")
                return None

        df['toxicity_score'] = df['raw_prompt'].progress_apply(get_toxicity)
        del toxicity_pipe
        free_memory()
        save_checkpoint(df, 4)
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        save_checkpoint(df, "4_failed")

    # ── PHASE 5: Language-Specific Sentiment Analysis ─────────────────────────
    try:
        print("\n=== PHASE 5: Language-Specific Sentiment Analysis ===")
        sentiment_models = {
            'arabic': "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
            'german': "oliverguhr/german-sentiment-bert",
            'russian': "BlancheFort/rubert-base-cased-sentiment"
        }
        df = _apply_sentiment_loop(df, sentiment_models, device)
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        save_checkpoint(df, "5_failed")

    output_file = "native_multi_metrics.parquet"
    df.to_parquet(output_file, engine='pyarrow')
    print(f"\n✅ Done! Data saved to: {output_file}")

    return df

def run_english_missing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🚀 Device in use: {device}")

    # ── Semantic Embeddings ───────────────────────────────────────────────────
    try:
        print("\n=== Semantic Embeddings ===")
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
        embeddings = embedder.encode(
            df['raw_prompt'].tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        df['semantic_embedding'] = list(embeddings)
        del embedder
        free_memory()
        save_checkpoint(df, "english_embeddings")
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        save_checkpoint(df, "english_embeddings_failed")

    # ── Sentiment (English) ───────────────────────────────────────────────────
    try:
        print("\n=== Sentiment Analysis (English) ===")
        sent_pipe = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

        def apply_sentiment_en(text):
            try:
                res = sent_pipe(str(text)[:512])
                return pd.Series([res[0]['score'], res[0]['label'].lower()])
            except Exception as e:
                print(f"[SENTIMENT ERROR] {e}")
                return pd.Series([None, None])

        results = df['raw_prompt'].progress_apply(apply_sentiment_en)
        df['sentiment_score'] = results.iloc[:, 0].values
        df['sentiment_label'] = results.iloc[:, 1].values

        del sent_pipe
        free_memory()
        save_checkpoint(df, "english_sentiment")
    except Exception as e:
        print(f"❌ Sentiment failed: {e}")
        save_checkpoint(df, "english_sentiment_failed")

    output_file = "redteaming_results_2000_metrics.parquet"
    df.to_parquet(output_file, engine='pyarrow')
    print(f"\n✅ Done! Data saved to: {output_file}")

    return df