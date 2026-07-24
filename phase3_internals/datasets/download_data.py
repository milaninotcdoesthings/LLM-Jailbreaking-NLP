import subprocess
import os

# ── Dataset repositories ───────────────────────────────────────────────────────
datasets = {
    "advbench": "https://github.com/llm-attacks/llm-attacks.git",
    "donotanswer": "https://github.com/Libr-AI/do-not-answer.git",
    "xstest": "https://github.com/paul-rottger/exaggerated-safety.git"
}

output_dir = "phase3_internals/datasets"
os.makedirs(output_dir, exist_ok=True)

for name, url in datasets.items():
    dest = os.path.join(output_dir, name)
    if os.path.exists(dest):
        print(f"✅ {name} già presente, skip.")
        continue
    print(f"📥 Clonando {name}...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, dest],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✅ {name} clonato con successo.")
    else:
        print(f"❌ Errore clonando {name}: {result.stderr}")

print("\nDataset disponibili:")
for name in datasets.keys():
    dest = os.path.join(output_dir, name)
    if os.path.exists(dest):
        files = []
        for root, dirs, filenames in os.walk(dest):
            dirs[:] = [d for d in dirs if d != '.git']
            for f in filenames:
                files.append(os.path.relpath(os.path.join(root, f), dest))
        print(f"\n{name}/")
        for f in files:
            print(f"  {f}")