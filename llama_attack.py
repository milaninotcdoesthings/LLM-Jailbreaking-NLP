import pandas as pd
import time
from groq import Groq
from dotenv import load_dotenv
import os
from tqdm import tqdm

# ── Configurazione ─────────────────────────────────────────────────────────────
load_dotenv(dotenv_path="key.env")
api_key = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=api_key)
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

#Attaccking 8B and 70B llama models with ai generated prompts:

# Inizializziamo il client ASINCRONO

import asyncio
import random
from groq import AsyncGroq
from tqdm.asyncio import tqdm

aclient = AsyncGroq(api_key=api_key)

MODEL_8B = "llama-3.1-8b-instant"
MODEL_70B = "llama-3.3-70b-versatile"
MAX_TOKENS_GENERATI = 200

# --- PARAMETRI DI OTTIMIZZAZIONE ASINCRONA ---
CONCURRENCY_LIMIT = 20 
# Numero di prompt da elaborare prima di fare un salvataggio fisico sul file CSV
CHUNK_SIZE = 50 

# ── Funzione Asincrona Helper ────────────────────────────────────────────────
async def async_richiedi_a_groq(messaggio, nome_modello, max_retries=6):
    """Gestisce la chiamata asincrona con Exponential Backoff per i Rate Limit."""
    for attempt in range(max_retries):
        try:
            chat_completion = await aclient.chat.completions.create(
                messages=[{"role": "user", "content": messaggio}],
                model=nome_modello,
                temperature=0.1, 
                max_tokens=MAX_TOKENS_GENERATI
            )
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            messaggio_errore = str(e).lower()
            
            # Se è un Rate Limit (Errore 429), calcoliamo un'attesa esponenziale
            if "429" in messaggio_errore or "rate limit" in messaggio_errore:
                wait_time = (2 ** attempt) + random.uniform(1.0, 3.0) # Esponenziale + Jitter
                tqdm.write(f" ⏳ Rate limit su {nome_modello}. Attendo {wait_time:.1f}s (Tentativo {attempt+1})")
                await asyncio.sleep(wait_time)
            
            # Altri errori (es. timeout di rete)
            elif attempt < max_retries - 1:
                tqdm.write(f" ⚠️ Errore {nome_modello}: {e}. Riprovo tra 3s...")
                await asyncio.sleep(3)
            
            else:
                tqdm.write(f" ❌ Fallito definitivamente su {nome_modello}.")
                return "API_ERROR"

# ── Worker per la singola riga ───────────────────────────────────────────────
async def processa_singola_riga(index, row, df, colonna_prompt, semaphore, pbar):
    testo_attacco = str(row[colonna_prompt])
    full_message = f"Task:\n{testo_attacco}"
    
    async with semaphore:
        risultati = await asyncio.gather(
            async_richiedi_a_groq(full_message, MODEL_8B),
            async_richiedi_a_groq(full_message, MODEL_70B)
        )
        df.at[index, 'response_8B'] = risultati[0]
        df.at[index, 'response_70B'] = risultati[1]
        
    pbar.update(1)


# ── Secondo Sviluppo (Orchestratore Asincrono) ───────────────────────────────
async def secondo_sviluppo_attacco_doppio_async(file_input, colonna_prompt):
    print(f"\n🚀 AVVIO SECONDO SVILUPPO (ASINCRONO): {MODEL_8B} e {MODEL_70B}")
    
    try:
        df = pd.read_csv(file_input)
    except FileNotFoundError:
        print(f"ERRORE CRITICO: Il file '{file_input}' non esiste!")
        return

    # Inizializzazione colonne
    for col in ['response_8B', 'response_70B']:
        if col not in df.columns:
            df[col] = None

    # Isoliamo solo le righe che devono effettivamente essere processate
    da_processare = [(index, row) for index, row in df.iterrows()]

    totale_rimanente = len(da_processare)
    print(f"Prompt totali: {len(df)} | Prompt ancora da attaccare: {totale_rimanente}\n")
    
    if totale_rimanente == 0:
        print("🎉 Tutti gli attacchi sono già stati completati! Nessuna azione richiesta.")
        return

    # Inizializziamo Semaforo e Barra di Progresso
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    with tqdm(total=totale_rimanente, desc="Attacchi Asincroni") as pbar:
        
        # Processiamo i dati a blocchi (Chunks) per poter salvare i progressi in sicurezza
        for i in range(0, totale_rimanente, CHUNK_SIZE):
            chunk = da_processare[i : i + CHUNK_SIZE]
            
            # Creiamo i task per il blocco corrente
            coroutines = [
                processa_singola_riga(idx, row, df, colonna_prompt, semaphore, pbar) 
                for idx, row in chunk
            ]
            
            # Eseguiamo il blocco in modo concorrente
            await asyncio.gather(*coroutines)
            
            # Salvataggio fisico dopo ogni blocco completato
            df.to_csv(file_input, index=False)
            tqdm.write(f"💾 [Autosave] Salvati i progressi sul file {file_input}")

    # Salvataggio Finale di sicurezza
    df.to_csv(file_input, index=False)
    print(f"\n✅ Elaborazione Asincrona Completata! File '{file_input}' aggiornato.")


# ── Area di Esecuzione ───────────────────────────────────────────────────────
if __name__ == "__main__":
    
    NOME_FILE_INPUT = "redteaming_results_2000_metrics.csv"
    NOME_COLONNA_PROMPT = "raw_prompt" 
    
    # NOTA BENE: Se stai eseguendo questo codice su Jupyter Notebook, 
    # usa questa riga (commentando l'altra):
    # await secondo_sviluppo_attacco_doppio_async(NOME_FILE_INPUT, NOME_COLONNA_PROMPT)
    
    # Se lo stai eseguendo come file .py nel terminale:
    asyncio.run(secondo_sviluppo_attacco_doppio_async(NOME_FILE_INPUT, NOME_COLONNA_PROMPT))