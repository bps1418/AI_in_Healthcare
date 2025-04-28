import os
import pandas as pd
import openai
import sys
import google.generativeai as genai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load patient summary
patient_df = pd.read_csv("data/patient_summary.csv")

# Configure API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Connect to Mistral locally if running Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# Clean text before saving to CSV
def clean_text(text):
    if isinstance(text, str):
        return text.replace("\n", " ").replace("\r", " ").strip()
    return text

# Initialize Output Lists
gpt4_outputs = []
gemini_outputs = []
mistral_outputs = []

# Process first 20 patients
n_patients = min(20, len(patient_df))

for idx, row in patient_df.head(n_patients).iterrows():
    patient_id = row["patient_id"]
    top_features = eval(row["top_features"])  # safely convert back
    model_prediction = row["model_prediction"]
    model_probability = row["model_probability"]

    # Build the prompt
    prompt = f"""You are a medical expert. A patient's top vital/lab features are:\n"""
    for k, (v, label) in top_features.items():
        v_str = "Missing" if v == 0 else f"{v:.2f}"
        prompt += f"- {k}: {v_str} ({label})\n"
    prompt += f"\nThe model predicts: {'Shock' if model_prediction == 1 else 'No Shock'}.\n"
    prompt += f"Model's probability of Shock: {model_probability:.2f}\n"
    prompt += "Explain the clinical reasoning behind this prediction clearly and concisely."

    # ---------------- GPT-4 ----------------
    try:
        gpt4_resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a medical expert."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        gpt4_outputs.append((patient_id, clean_text(gpt4_resp.choices[0].message.content)))
    except Exception as e:
        gpt4_outputs.append((patient_id, clean_text(str(e))))

    # ---------------- Gemini ----------------
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")  # safer
        gemini_resp = model.generate_content(prompt)
        gemini_outputs.append((patient_id, clean_text(gemini_resp.text)))
    except Exception as e:
        gemini_outputs.append((patient_id, clean_text(str(e))))

    # ---------------- Mistral (Ollama) ----------------
    try:
        mistral_payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=mistral_payload)
        mistral_text = response.json()["response"]
        mistral_outputs.append((patient_id, clean_text(mistral_text)))
    except Exception as e:
        mistral_outputs.append((patient_id, clean_text(str(e))))

# Save all outputs
os.makedirs("data", exist_ok=True)

pd.DataFrame(gpt4_outputs, columns=["patient_id", "gpt4_reasoning"]).to_csv("data/gpt4_shock_reasoning.csv", index=False)
pd.DataFrame(gemini_outputs, columns=["patient_id", "gemini_reasoning"]).to_csv("data/gemini_shock_reasoning.csv", index=False)
pd.DataFrame(mistral_outputs, columns=["patient_id", "mistral_reasoning"]).to_csv("data/mistral_shock_reasoning.csv", index=False)

print("Explanations generated and saved cleanly for GPT-4, Gemini, and Mistral!")