import os
import subprocess
import pandas as pd
import glob

# ==========================================
# 1. CLONE THE ENTIRE REPOSITORY
# ==========================================
repo_url = "https://github.com/Jarviswang94/Multilingual_safety_benchmark.git"
repo_dir = "Multilingual_safety_benchmark" # Name of the directory to be created

# Check if the repository has already been cloned to avoid downloading it twice
if not os.path.exists(repo_dir):
    print("Cloning the entire repository (this might take a few moments)...")
    # This executes the 'git clone' command in your terminal automatically
    subprocess.run(["git", "clone", repo_url])
    print("✅ Repository successfully cloned to your local machine!")
else:
    print("✅ The repository already exists on your machine. Skipping download.")


# ==========================================
# 2. FILTER AND EXTRACT THE 'en' FOLDER
# ==========================================
print("\nStarting the search for files in the English ('en') folder...")

# Point directly to the 'en' folder inside the downloaded repo
english_folder = os.path.join(repo_dir, "en")

# Search for all .csv files, automatically exploring subfolders
csv_file_paths = glob.glob(os.path.join(english_folder, "**", "*.csv"), recursive=True)

dataframes_list = []
for file_path in csv_file_paths:
    try:
        # 1. IL TRUCCO: header=None dice a Pandas di non usare la prima riga come titolo.
        temp_df = pd.read_csv(file_path, header=None)
        
        # 2. Rinominiamo la colonna 0 (la prima) in 'prompt'
        temp_df.rename(columns={0: 'prompt'}, inplace=True)
        
        # 3. Salviamo la categoria originale (il nome della cartella)
        folder_name = os.path.basename(os.path.dirname(file_path))
        temp_df['original_category'] = folder_name
        
        # Se per caso c'è una seconda colonna misteriosa nel CSV, teniamo solo 'prompt' e 'original_category'
        temp_df = temp_df[['prompt', 'original_category']]
        
        dataframes_list.append(temp_df)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


# ==========================================
# 3. MERGE ALL ENGLISH DATA
# ==========================================
if len(dataframes_list) > 0:
    df_multilingual_en = pd.concat(dataframes_list, ignore_index=True)
    
    # Save the final merged dataset to a CSV file
    output_file = "multilingual_benchmark_en_full.csv"
    df_multilingual_en.to_csv(output_file, index=False)
    
    print(f"\n🎉 Job completed!")
    print(f"Found {len(csv_file_paths)} CSV files.")
    print(f"Created a single dataset with {len(df_multilingual_en)} total prompts ({output_file}).")
    print("\nColumns of the new dataset:")
    print(df_multilingual_en.columns.tolist())
else:
    print("⚠️ No CSV files found in the 'en' folder.")


# ==========================================
# 4. ATTACKING LLAMA WITH PROCESSED DATA
# ==========================================
import os
import time
import pandas as pd
from tqdm.auto import tqdm 
from groq import Groq
from llama_attack import API_KEY 

MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=API_KEY)
input_file_new = "multi_processed.csv"
output_file_new = "multi_processed_responses.csv"

# 1. Load data and handle resume logic
if os.path.exists(output_file_new):
    df = pd.read_csv(output_file_new)
    print(f"Found existing output file. Resuming attack...")
else:
    df = pd.read_csv(input_file_new)
    df['model_response'] = None
    print(f"Starting new attack from {input_file_new}...")

def get_llama_response(prompt_text):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=MODEL_NAME,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e).lower()
        if "rate_limit" in error_msg or "429" in error_msg:
            time.sleep(10)
            return "ERROR: RATE_LIMIT"
        return f"ERROR: {str(e)}"

# 2. Identify rows that still need to be processed
# This ensures we skip prompts that already have a valid response
missing_indices = df[df['model_response'].isna() | (df['model_response'] == "ERROR: RATE_LIMIT")].index
print(f"Prompts remaining to be processed: {len(missing_indices)}")

# 3. Execution loop
for idx in tqdm(missing_indices, desc="Querying LLaMA 3"):
    prompt = str(df.loc[idx, "prompt"])
    res = get_llama_response(prompt)
    
    df.loc[idx, "model_response"] = res
    
    # Auto-save every 50 iterations to prevent data loss on crash
    if idx % 50 == 0:
        df.to_csv(output_file_new, index=False)
        
    time.sleep(2.1)

# Final save
df.to_csv(output_file_new, index=False)
print("\n🎉 Attack completed! All responses saved.")