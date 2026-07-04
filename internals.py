import os
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# 1. SBLOCCO LIMITI MEMORIA MAC
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 2. LOGIN SICURO
# Assicurati di avere il file key.env nella stessa cartella con: HF_TOKEN=tuo_token
load_dotenv("key.env")
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
else:
    print("⚠️ Nessun HF_TOKEN trovato nel file key.env. Il login potrebbe fallire se il modello richiede permessi.")

# ==========================================
# 1. IMPOSTAZIONI INIZIALI E DEVICE
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🚀 Esecuzione su device: {device}")

# ==========================================
# 2. CARICAMENTO DEL MODELLO QWEN (Test Veloce)
# ==========================================
model_id = "Qwen/Qwen2.5-1.5B-Instruct" 

print(f"Caricamento di {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Caricamento nativo veloce (senza quantizzazione, occupa circa 3GB di RAM)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(device)

sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=device)

# ==========================================
# 3. SETUP PER L'ESTRAZIONE DEGLI INTERNALS (HOOK)
# ==========================================
# Qwen 1.5B ha 28 layer, impostiamo l'hook a metà (layer 14)
TARGET_LAYER = 14
activation_cache = {}

def get_activation_hook(name):
    def hook(model, input, output):
        # Qwen2 restituisce gli hidden states come primo elemento
        hidden_states = output[0] 
        mean_pooled = hidden_states.mean(dim=1).squeeze().detach().cpu().numpy()
        activation_cache[name] = mean_pooled
    return hook

hook_handle = model.model.layers[TARGET_LAYER].register_forward_hook(get_activation_hook('layer_14'))

# ==========================================
# 4. CARICAMENTO DATASET
# ==========================================
print("Caricamento dataset 'LibrAI/do-not-answer'...")
dataset = load_dataset("LibrAI/do-not-answer", split="train")

# Riduciamo a 50 prompt per un test ancora più rapido (cambia a 500 se vuoi)
df_attacks = dataset.to_pandas().head(50)

# ==========================================
# 5. LOOP DI ATTACCO
# ==========================================
risposte_modello = []
token_lengths = []
sentiment_scores = []
internals_list = []

print(f"\nInizio attacco su {len(df_attacks)} prompt...")

for prompt in tqdm(df_attacks['question'], desc="Attacchi in corso", unit="prompt"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    token_len = inputs['input_ids'].shape[1]
    token_lengths.append(token_len)
    
    try:
        sent_result = sentiment_pipe(prompt[:512])[0]
        score = sent_result['score'] if sent_result['label'] == 'POSITIVE' else -sent_result['score']
        sentiment_scores.append(score)
    except:
        sentiment_scores.append(0.0)

    activation_cache.clear() 
    
    with torch.no_grad(): 
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, # Modificato a 50 per velocizzare il test
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    risposta = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    risposte_modello.append(risposta)
    
    if 'layer_14' in activation_cache:
        # Il .tolist() converte l'array NumPy in una lista compatibile con Parquet
        internals_list.append(activation_cache['layer_14'].tolist())
    else:
        # Nessun dato: meglio usare None al posto di np.nan quando si lavora con le liste
        internals_list.append(None)

hook_handle.remove()

# ==========================================
# 6. CREAZIONE DATASET FINALE
# ==========================================
df_attacks['Model_Response'] = risposte_modello
df_attacks['Prompt_Token_Length'] = token_lengths
df_attacks['Prompt_Sentiment'] = sentiment_scores
df_attacks['Internals_Layer14'] = internals_list

# ==========================================
# 7. SALVATAGGIO
# ==========================================
# Nomenclatura modificata per non sovrascrivere file futuri
df_text_only = df_attacks.drop(columns=['Internals_Layer14'])
df_text_only.to_csv("jailbreak_results_qwen_test_text.csv", index=False)

# Invece di to_parquet, puoi usare:
df_attacks.to_pickle("jailbreak_results_qwen_test_internals.pkl")

print("\n✅ Test completato! File salvati:")
print("1. jailbreak_results_qwen_test_text.csv")
print("2. jailbreak_results_qwen_test_internals.parquet")