import pandas as pd
from pathlib import Path

root = Path(".")
github_folder = root / "Multilingual_safety_benchmark"  

# Iteration on each sub-folder
for subfolder in github_folder.iterdir():
    if not subfolder.is_dir():
        continue  
    
    
    csv_files = list(subfolder.rglob("*.csv"))
    
    if not csv_files:
        print(f"Nessun CSV in: {subfolder.name}")
        continue  
    
    print(f"\n{subfolder.name}: trovati {len(csv_files)} CSV")
    
    # CSV Files concatenation
    dfs = [pd.read_csv(f) for f in csv_files]
    result = pd.concat(dfs, ignore_index=True)
    
    # Saving with sub-folder name
    output_path = root / f"merged_{subfolder.name}.csv"
    result.to_csv(output_path, index=False)
    print(f"Salvato: {output_path}")