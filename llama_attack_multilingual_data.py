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