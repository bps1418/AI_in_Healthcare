import os
import pandas as pd
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
gpt4_path = "data/gpt4_shock_reasoning.csv"
gemini_path = "data/gemini_shock_reasoning.csv"
mistral_path = "data/mistral_shock_reasoning.csv"

# Load Reasonings
gpt4_df = pd.read_csv(gpt4_path)
gpt4_df["model_name"] = "gpt4"
gpt4_df = gpt4_df.rename(columns={"gpt4_reasoning": "reasoning_text"})

gemini_df = pd.read_csv(gemini_path)
gemini_df["model_name"] = "gemini"
gemini_df = gemini_df.rename(columns={"gemini_reasoning": "reasoning_text"})

mistral_df = pd.read_csv(mistral_path)
mistral_df["model_name"] = "mistral"
mistral_df = mistral_df.rename(columns={"mistral_reasoning": "reasoning_text"})

# Combine all
combined_df = pd.concat([gpt4_df, gemini_df, mistral_df], ignore_index=True)

print(f"Total explanations to score: {len(combined_df)}")

# DeepSeekR1 Local API Setup
DEEPSEEK_URL = "http://localhost:11434/api/generate"  # Adjust if needed

# Evaluation Function
def evaluate_explanation(reasoning_text):
    prompt = f"""You are a clinical reasoning evaluator.
Here is the explanation:

--- Reasoning ---
{reasoning_text}
-------------------

Evaluate and score:
- Completeness (1-5)
- Clarity (1-5)
- Transparency (1-5)
- Consistency (1-5)

Respond ONLY in JSON format like:
{{
    "Completeness": X,
    "Clarity": Y,
    "Transparency": Z,
    "Consistency": W
}}
"""
    payload = {
        "model": "deepseek-r1",  # or whichever you loaded locally
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(DEEPSEEK_URL, json=payload, timeout=60)
        result = response.json()["response"]
        # Sometimes model returns text + junk — extract JSON cleanly
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        clean_json = result[json_start:json_end]
        scores = json.loads(clean_json)
        return scores
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"Completeness": 0, "Clarity": 0, "Transparency": 0, "Consistency": 0}

# Now Evaluate
scored_rows = []

for idx, row in combined_df.iterrows():
    patient_id = row["patient_id"]
    model_name = row["model_name"]
    reasoning_text = row["reasoning_text"]

    print(f"Evaluating Patient {patient_id}, Model {model_name}...")

    scores = evaluate_explanation(reasoning_text)
    scored_rows.append({
        "patient_id": patient_id,
        "model_name": model_name,
        **scores
    })

    time.sleep(1)  # To be polite if hitting an API

# Final Scorecard
scorecard_df = pd.DataFrame(scored_rows)
scorecard_df.to_csv("data/shock_reasoning_scorecard.csv", index=False)

print("\n✅ Scorecard Generated Successfully!")