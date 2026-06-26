Ecco la traduzione in inglese del file Markdown:

---

# LLM Jailbreaking & Safety — Thesis Research Project

**Master's Degree Thesis · NLP & AI Safety**

**Repository:** `milaninotcdoesthings/LLM-Jailbreaking-NLP`

## Project Overview

This repository contains the code, datasets, and experimental results of a Master's thesis focused on jailbreaking open-source Large Language Models (LLMs). The core objective of the research is to study the vulnerability of language models to adversarial attacks and to develop a system capable of predicting the probability of a jailbreak attack succeeding given an input prompt, leveraging the internal states of the target model.

The project is structured in three progressive phases, each building on the foundations of the previous one.

## 3-Phase Project Structure

### Phase 1 — Attacks on Open-Source Models via Groq

**Goal:** Run systematic jailbreaking attack campaigns against open-source LLMs, collecting model responses to build a labeled dataset of successful and failed attacks.

**Target models:**

* LLaMA 3.1 8B Instant (`llama-3.1-8b-instant`)
* LLaMA 3.3 70B Versatile (`llama-3.3-70b-versatile`)

Both are queried via the Groq API, which provides very low inference latency thanks to LPU hardware acceleration. Adversarial prompts are sourced from public red-teaming datasets including:

* **WildJailbreak** (`wildjailbreak_full.csv`) — heterogeneous attack dataset covering roleplay scenarios, social engineering, and prompt injection.
* **HarmBench** (`harmbench_dataset.csv`) — standard benchmark for evaluating model robustness against harmful behaviors.
* **Crimes and Illegal Activities** (`Crimes_And_Illegal_Activities_en.csv`) — dataset focused on requests for illicit content.
* **Multilingual Safety Benchmark** (`Multilingual_safety_benchmark/`) — multilingual extension of the attacks in Arabic, French, German, Russian, Spanish, and other languages, designed to test the consistency of cross-lingual safety guardrails.

**Key scripts:**

* `llama_attack.py` — main attack loop targeting LLaMA 8B via Groq, with rate limit handling, autosave every 10 prompts, and automatic resume logic in case of interruption.
* `llama_attack_multilingual_data.py` — variant for multilingual datasets.
* `five_languages_attack.py` — parallelized attacks across five languages simultaneously.
* `multilingual_safety.py` / `multilingual_safety.ipynb` — multilingual safety analysis pipeline.

**Produced outputs:**

* `attack_results_first_test.csv` — results from the first test on LLaMA 8B.
* `attack_results_full_dataset.csv` — results on the full dataset.
* `attack_results_llama70b_test.csv` — results on LLaMA 70B.
* `attack_results_wildjail_dataset.csv` — results on WildJailbreak.

**Automatic Evaluation with LLM-as-a-Judge**
To automatically label the success or failure of each attack, an LLM-as-a-Judge pattern is implemented in `llm_as_a_judge.py`. A second model (LLaMA 3.3 70B) evaluates each (prompt, response) pair and determines whether the attacked model provided substantive, actionable content (score = 1) or refused/deflected the request (score = 0). Evaluation is asynchronous with controlled concurrency (asyncio, semaphore with 3 workers) and outputs structured JSON.

---

### Phase 2 — Automated Adversarial Prompt Generation

**Goal:** Expand the attack dataset by generating new malicious prompts using unguardrailed models, in order to stress-test the target models with novel attack vectors.

Two complementary strategies are used:
**A) Local generation with Ollama**
Via `prompt_generation.py`, a model running locally through Ollama (`qwen2.5:latest`) is instructed with a system prompt framing it as an "AI Security Researcher specialized in Red-Teaming". The model generates prompts in batches of 10 at a time, using varied framing techniques:

* Diverse personas (student, businessperson, fictional character)
* Narrative approaches (educational inquiry, storytelling, technical debugging)
* High temperature (0.8) to maximize prompt diversity

**B) Unaligned models on Hugging Face**
As an additional source of adversarial prompts, unaligned models available on Hugging Face are used — models that have not undergone RLHF or safety fine-tuning and can therefore generate content without restrictions. These models are queried via the transformers library, producing prompts that challenge the target models' guardrails more directly than static datasets.

---

### Phase 3 — Predictive Model for Attack Success Estimation

**Goal:** Build a machine learning classifier that, given an input prompt, estimates the probability that the jailbreak attack will succeed against the target model.

**Feature Collection: Internal State Extraction via Hugging Face**
To build a sufficiently rich training dataset, internal states are extracted via the Hugging Face Transformers APIs. While the model generates a response, the following are captured:

* Hidden states (intermediate layer-by-layer representations)
* Attention values (attention weights per attention head)
* Logit distributions of the output in the early stages of generation

These vectors constitute the features of the predictive model, alongside textual metrics (length, lexical complexity, semantic category, language, jailbreak technique).

**ML Model (in development)**
The predictive classifier is currently under development. The planned architecture combines internal features extracted from hidden states with surface-level textual features of the prompt, training a model that returns a probability score $p(\text{success} \mid \text{prompt})$.

---

## File Structure

*(The file structure remains the same as your original provided text.)*

---

## Tech Stack

| Component | Technology |
| --- | --- |
| **Target models** | LLaMA 3.1 8B, LLaMA 3.3 70B Versatile |
| **Fast inference (attacks)** | Groq API |
| **Adversarial prompt generation** | Ollama + qwen2.5:latest |
| **Unaligned models** | Hugging Face Transformers |
| **Internal state extraction** | Hugging Face transformers (hidden states, attention) |
| **Automatic evaluation** | LLM-as-a-Judge (LLaMA 3.3 70B async) |
| **Data manipulation** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Async processing** | asyncio, asyncGroq |
| **Local environment** | Python 3.10+, Jupyter Notebook |

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/milaninotcdoesthings/LLM-Jailbreaking-NLP.git
cd LLM-Jailbreaking-NLP

# 2. Install dependencies
pip install groq pandas numpy matplotlib seaborn tqdm python-dotenv ollama transformers

# 3. Set up your Groq API key
echo "GROQ_API_KEY=your_key_here" > key.env

# 4. (Optional) Install Ollama for local prompt generation
ollama pull qwen2.5

```

## Usage

* **Phase 1:** `python llama_attack.py`
* **Phase 2:** `python prompt_generation.py`
* **Evaluation:** `python llm_as_a_judge.py`
* **EDA:** `jupyter notebook datasets_eda.ipynb`

## Academic Context

This research sits at the intersection of AI Safety and Adversarial NLP. Key references include:

* Many-Shot Jailbreaking (Anthropic, 2024)
* Saiem et al. — studies on prompt injection and safety filter bypass
* Tsmindashvili et al. — cross-lingual analysis of LLM safety
* Multilingual Safety Benchmark — evaluation of guardrail consistency

## Ethical Note

This project is conducted exclusively for academic research purposes on the security of language models. All experiments are run on open-source models in a controlled environment. The datasets and results are intended to contribute to the improvement of LLM defense mechanisms, not to facilitate malicious use.
