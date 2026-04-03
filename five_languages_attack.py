import pandas as pd
from groq import AsyncGroq
from tqdm.auto import tqdm
import os
import asyncio
from dotenv import load_dotenv


load_dotenv(dotenv_path="key.env")  # carica il .env
api_key = os.environ.get("GROQ_API_KEY")
print(f"API key caricata: {api_key[:10]}..." if api_key else "❌ API key NON trovata!")
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

dataset_files = [
    "arabic_processed.csv",
    "french_processed.csv",
    "spanish_processed.csv",
    "german_processed_new.csv",
    "russian_processed.csv"
]

dataset_files_traduzione = {
    "arabic_processed_risultati_groq.csv": "Arabic",
    "french_processed_risultati_groq.csv": "French",
    "spanish_processed_risultati_groq.csv": "Spanish",
    "german_processed_new_risultati_groq.csv": "German",
    "russian_processed_risultati_groq.csv": "Russian"
}

MODELLO = "llama-3.3-70b-versatile"
CONCURRENCY = 20  # richieste simultanee — aumenta se il rate limit lo permette

def tronca_risposta(testo: str, n_parole: int = 15) -> str:
    return " ".join(testo.strip().split()[:n_parole])

async def processa_prompt(prompt: str, semaphore: asyncio.Semaphore) -> str:
    if not prompt.strip():
        return ""
    async with semaphore:  # controlla la concorrenza massima
        for attempt in range(3):  # retry automatico
            try:
                response = await client.chat.completions.create(
                    model=MODELLO,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.7
                )
                return tronca_risposta(response.choices[0].message.content)
            except Exception as e:
                if "429" in str(e):  # rate limit
                    wait = 2 ** attempt  # backoff: 1s, 2s, 4s
                    await asyncio.sleep(wait)
                else:
                    return f"ERRORE: {str(e)}"
        return "ERRORE: max retry raggiunto"

async def processa_dataset(file_path: str):
    print(f"\n📂 Caricamento: {file_path}")
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=True)
    df["Prompt"] = df["Prompt"].fillna("").astype(str)
    prompts = df["Prompt"].tolist()

    semaphore = asyncio.Semaphore(CONCURRENCY)
    risposte = [None] * len(prompts)

    # Progress bar aggiornata in tempo reale
    with tqdm(total=len(prompts), desc=f"{file_path}", unit="prompt") as pbar:
        async def task(i, prompt):
            risultato = await processa_prompt(prompt, semaphore)
            risposte[i] = risultato
            pbar.update(1)

        await asyncio.gather(*[task(i, p) for i, p in enumerate(prompts)])

    df["Groq_Response"] = pd.Series(risposte, index=df.index)
    output_path = file_path.replace(".csv", "_risultati_groq.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Salvato: {output_path}")

async def traduci_testo(testo: str, lingua: str, semaphore: asyncio.Semaphore) -> str:
    if not str(testo).strip() or str(testo).startswith("ERRORE"):
        return testo
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODELLO,
                    messages=[{
                        "role": "user",
                        "content": f"Translate this {lingua} text to English. Reply with ONLY the translation, no explanations:\n{testo}"
                    }],
                    max_tokens=50,
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(2 ** attempt)
                else:
                    return f"ERRORE_TRADUZIONE: {str(e)}"
        return "ERRORE: max retry raggiunto"

async def traduci_dataset(file_path: str, lingua: str):
    print(f"\n📂 Caricamento: {file_path} ({lingua})")
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=True)
    risposte = df["Groq_Response"].tolist()

    semaphore = asyncio.Semaphore(CONCURRENCY)
    traduzioni = [None] * len(risposte)

    with tqdm(total=len(risposte), desc=f"Traduzioni {lingua}", unit="prompt") as pbar:
        async def task(i, testo):
            traduzioni[i] = await traduci_testo(str(testo), lingua, semaphore)
            pbar.update(1)
        await asyncio.gather(*[task(i, t) for i, t in enumerate(risposte)])

    df["Groq_Response_EN"] = pd.Series(traduzioni, index=df.index)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")  # sovrascrive il file esistente
    print(f"✅ Colonna 'Groq_Response_EN' aggiunta a: {file_path}")


async def main():
    # Fase 1 — commentata perché già eseguita
    # for file_path in dataset_files:
    #     if not os.path.exists(file_path):
    #         print(f"⚠️ File non trovato: {file_path}")
    #         continue
    #     await processa_dataset(file_path)

    # Fase 2 — solo traduzioni
    for file_path, lingua in dataset_files_traduzione.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File non trovato: {file_path}")
            continue
        await traduci_dataset(file_path, lingua)

    print("\n🎉 Traduzioni completate!")
asyncio.run(main())