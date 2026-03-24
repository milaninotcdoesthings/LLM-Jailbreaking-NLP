import pandas as pd
from datasets import load_dataset

LOAD_HARMBENCH = False       # Impostato su False: salta questa parte
LOAD_WILDJAILBREAK = True

my_token = "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"

if LOAD_HARMBENCH:
    print("Scaricamento di HarmBench in corso...")
    dataset = load_dataset("walledai/HarmBench", "contextual", token=my_token)
    df = dataset['train'].to_pandas()
    print(df.head())
    df.to_csv("harmbench_dataset.csv", index=False)
    print("Fatto! File salvati nella tua cartella.")
else:
    print("Loading WildJailBreak instead")

if LOAD_WILDJAILBREAK: 
   print("\nScaricamento di WildJailbreak in corso (potrebbe volerci qualche minuto)...")
   dataset_wj = load_dataset("allenai/WildJailbreak", "eval", token=my_token)
   print(f"Le sezioni disponibili in questo dataset sono: {list(dataset_wj.keys())}")
   df_wj = dataset_wj['train'].to_pandas()
   df_wj.to_csv("wildjailbreak_full.csv", index=False)
   print(f"✅ WildJailbreak completo salvato! Totale righe: {len(df_wj)}")
else:
    print("Loading HarmBench instead")