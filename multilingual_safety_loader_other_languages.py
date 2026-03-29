import pandas as pd
from pathlib import Path

root = Path(".")
github_folder = root / "Multilingual_safety_benchmark"  # <-- cambia questo

# Itera su ogni sottocartella
for subfolder in github_folder.iterdir():
    if not subfolder.is_dir():
        continue  # salta se non è una cartella
    
    # Trova tutti i CSV dentro questa sottocartella
    csv_files = list(subfolder.rglob("*.csv"))
    
    if not csv_files:
        print(f"Nessun CSV in: {subfolder.name}")
        continue  # salta se non ci sono CSV
    
    print(f"\n{subfolder.name}: trovati {len(csv_files)} CSV")
    
    # Concat dei CSV di questa sottocartella
    dfs = [pd.read_csv(f) for f in csv_files]
    result = pd.concat(dfs, ignore_index=True)
    
    # Salva con il nome della sottocartella
    output_path = root / f"merged_{subfolder.name}.csv"
    result.to_csv(output_path, index=False)
    print(f"Salvato: {output_path}")