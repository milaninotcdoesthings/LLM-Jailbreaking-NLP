import os
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from huggingface_hub import login

# 1. SBLOCCO LIMITI MEMORIA MAC
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 2. LOGIN SICURO
load_dotenv("key.env")
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

# ==========================================
# 1. SETUP DEVICE
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🚀 Esecuzione su device: {device}")

# ==========================================
# 2. CARICAMENTO MODELLO
# ==========================================
model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
print(f"Caricamento di {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(device)

sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=device)

# ==========================================
# 3. CARICAMENTO DATASET LOCALI
# ==========================================
print("Lettura e unione dei dataset locali...")

# 🔴 DA MODIFICARE: Inserisci i percorsi esatti ai file .csv o .json che hai scaricato
PATH_DATASET_1 = "/Users/tommasomilanino/Developer/THESIS/phase3_internals/datasets/advbench/data/advbench/harmful_behaviors.csv"
PATH_DATASET_2 = "/Users/tommasomilanino/Developer/THESIS/phase3_internals/datasets/donotanswer/datasets/data_en.csv"
PATH_DATASET_3 = "/Users/tommasomilanino/Developer/THESIS/phase3_internals/datasets/xstest/xstest_prompts.csv"

# Caricamento Dataset 1 (es. AdvBench)
# Sostituisci 'NOME_COLONNA_1' con il vero nome della colonna che contiene l'attacco (es. 'goal')
df1 = pd.read_csv(PATH_DATASET_1)
df1 = df1.rename(columns={'NOME_COLONNA_1': 'goal'})
df1['dataset_source'] = 'Dataset_1' # Etichetta utile per le analisi future

# Caricamento Dataset 2 (es. Do-Not-Answer)
# Sostituisci 'NOME_COLONNA_2' con il vero nome della colonna (es. 'question')
df2 = pd.read_csv(PATH_DATASET_2)
df2 = df2.rename(columns={'NOME_COLONNA_2': 'question'})
df2['dataset_source'] = 'Dataset_2'

# Caricamento Dataset 3 (es. XSTest)
# Sostituisci 'NOME_COLONNA_3' con il vero nome della colonna (es. 'text' o 'prompt')
df3 = pd.read_csv(PATH_DATASET_3)
df3 = df3.rename(columns={'NOME_COLONNA_3': 'prompt'})
df3['dataset_source'] = 'Dataset_3'

# Unione di tutti i dataset
# Manteniamo solo le colonne 'prompt' e 'dataset_source' per pulizia, ma puoi tenerle tutte
df_attacks = pd.concat([
    df1[['goal', 'dataset_source']], 
    df2[['question', 'dataset_source']], 
    df3[['prompt', 'dataset_source']]
], ignore_index=True)

# Rimuovi eventuali righe vuote
df_attacks = df_attacks.dropna(subset=['prompt'])

# Riduciamo il dataset per il test veloce locale a 50 prompt
# (Quando sarai pronto per l'estrazione massiva, commenta la riga qui sotto)
df_attacks = df_attacks.head(50) 
print(f"Totale prompt pronti per l'attacco: {len(df_attacks)}")

# ==========================================
# 4. LOOP DI ATTACCO
# ==========================================
TARGET_LAYER = 14
risposte_modello = []
token_lengths = []
sentiment_scores = []
internals_list = []

print("\nInizio estrazione attivazioni...")

for text_prompt in tqdm(df_attacks['prompt'], desc="Attacchi in corso", unit="prompt"):
    inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
    token_len = inputs['input_ids'].shape[1]
    token_lengths.append(token_len)
    
    try:
        sent_result = sentiment_pipe(text_prompt[:512])[0]
        score = sent_result['score'] if sent_result['label'] == 'POSITIVE' else -sent_result['score']
        sentiment_scores.append(score)
    except:
        sentiment_scores.append(0.0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    
    generated_tokens = outputs.sequences[0][token_len:]
    risposta = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    risposte_modello.append(risposta)
    
    prefill_hidden_states = outputs.hidden_states[0]
    prompt_final_token_vector = prefill_hidden_states[TARGET_LAYER][0, -1, :].detach().cpu().numpy()
    
    internals_list.append(prompt_final_token_vector.tolist())

# ==========================================
# 5. CREAZIONE E SALVATAGGIO
# ==========================================
df_attacks['Model_Response'] = risposte_modello
df_attacks['Prompt_Token_Length'] = token_lengths
df_attacks['Prompt_Sentiment'] = sentiment_scores
df_attacks['Internals_Layer14'] = internals_list

df_text_only = df_attacks.drop(columns=['Internals_Layer14'])
df_text_only.to_csv("jailbreak_results_qwen_test_text.csv", index=False)
df_attacks.to_pickle("jailbreak_results_qwen_test_internals.pkl")

print("\n✅ Test completato! File salvati.")