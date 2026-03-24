import pandas as pd
import time
from groq import Groq

# API Configuration
API_KEY = "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"
MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=API_KEY)
input_file = "wildjailbreak_full.csv"
output_file = "attack_results_wildjail_dataset.csv"

# 1. Load dataset (Resume logic)
try:
    df = pd.read_csv(output_file)
    print("File di output trovato. Riprendo l'attacco da dove si era interrotto...")
except FileNotFoundError:
    df = pd.read_csv(input_file)
    df['model_response'] = None
    print("Inizio nuovo attacco completo...")

totale_prompt = len(df)
print(f"Totale prompt da processare: {totale_prompt}\n")

# 2. Main execution loop (SENZA TQDM)
for index, row in df.iterrows():
    
    # Salta se abbiamo già una risposta
    if pd.notna(row['model_response']) and row['model_response'] != "API_ERROR":
        continue
    adversarial_text = row['adversarial']
    full_message = f"Task:\n{adversarial_text}"
    
    # Print custom per tenere traccia (sostituisce la barra di caricamento)
    print(f"[Progresso: {index + 1} / {totale_prompt}] Invio prompt...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": full_message}],
                model=MODEL_NAME,
                temperature=0.1, 
                max_tokens=300
            )
            
            response = chat_completion.choices[0].message.content
            df.at[index, 'model_response'] = response
            print(" -> Successo!")
            break 
            
        except Exception as e:
            if attempt < max_retries - 1:
                # QUESTA È LA RIGA MODIFICATA DA INCOLLARE:
                print(f" -> DETTAGLIO ERRORE: {e}")
                print(f" -> Attendo 10 secondi (Tentativo {attempt+1})...")
                time.sleep(10)
            else:
                print(" -> Fallito dopo 3 tentativi. Salto al prossimo.")
                df.at[index, 'model_response'] = "API_ERROR"
    
    time.sleep(3) # Rate limit base
    
    # Autosave ogni 10 prompt
    if (index + 1) % 10 == 0:
        df.to_csv(output_file, index=False)
        print(" -> [Salvataggio automatico effettuato]")

# Final Save
df.to_csv(output_file, index=False)
print(f"\nFinito! Tutti i risultati salvati in '{output_file}'.")

