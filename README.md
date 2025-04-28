# Project Plan: Orchestrating LLMs to Explain Shock Predictions for ICU Care

Build a shock prediction system for ICU patients using a Transformer model, generate reasoning with GPT-4, Gemini, and Mistral, and evaluate the reasoning using a Scorecard framework powered by DeepSeek R1.

## Description:
This project builds a full Explainable AI system for shock prediction in ICU patients, blending deep learning with cutting-edge LLM-based reasoning — all benchmarked through an automated scoring system.

| Phase | Task | Deliverables |
|:------|:-----|:-------------|
| **1. Data Preparation** | - Extract ICU patients from MIMIC-III<br> - Filter incomplete records | Cleaned dataset (~4957 patients) |
| **2. DL Model Development** | - Build Transformer model<br> - Train/validate model | AUC 0.8226, confusion matrix, SHAP feature importance |
| **3. Reasoning Generation** | - Summarize features<br> - Prompt GPT-4, Gemini, Mistral | Reasoning CSVs for each model |
| **4. Scorecard Evaluation** | - Use DeepSeek R1 to score<br> - Transparency, Consistency, Clarity, Completeness | Shock Reasoning Scorecard CSV |
| **5. Visualization** | - Plot metrics<br> - Radar + Bar Charts | Visualizations for scorecard analysis |
| **6. Paper Writing** | - Draft ACM paper<br> - Compile main.tex + refs.bib | Final main.pdf ready |
| **7. Final Packaging** | - Create zipped folder<br> - Prepare for sharing or publishing | Project archive |

## Tools and Framework

| Area | Tools/Frameworks |
|:-----|:-----------------|
| **Data Processing** | Python (Pandas, NumPy) |
| **Model Training** | PyTorch |
| **LLM APIs** | OpenAI (GPT-4), Google AI Studio (Gemini 1.5 Pro), Ollama (Mistral 7B local) |
| **Scoring** | DeepSeek R1 (local Ollama) |
| **Visualization** | Matplotlib |
| **Paper Writing** | LaTeX (ACM style), VS Code, MacTeX |

## Key Metrics

| Metric | Target |
|:-------|:-------|
| **Model AUC** | 0.8226 |
| **Explainability Scores** | Transparency, Consistency, Clarity, Completeness |
| **Reasoning Dataset** | Reasoned explanations for 20 patients (expandable) |
| **Paper Quality** | Compiling without errors, ACM-ready |

## Timeline Summary

| Milestone | Duration |
|:----------|:---------|
| Data Preparation | 1 day |
| Model Training | 2–3 days |
| Reasoning Generation | 1–2 days |
| Scorecard Orchestration | 1 day |
| Plotting and Visualization | 0.5 day |
| Paper Writing | 1–2 days |
| Final Packaging | 0.5 day |

## Final Deliverables

Github: https://github.com/bps1418/AI_in_Healthcare

Reports: [https://github.com/bps1418/AI_in_Healthcare/](https://github.com/bps1418/AI_in_Healthcare/blob/master/HighRiskProject/report/shock_prediction_llm_reasoning_score_card.pdf)

| Deliverable | File/Folder |
|:------------|:------------|
| Report | `main.tex`, `refs.bib`, `main.pdf` (Available in Github) |
| Code | Notebook/Local Python script for model generation and reasoning score card (Available at Github) |
| Trained Model | `best_transformer_shock_model.pt` (Available in Github) |
| Reasoning Outputs | `gpt4_reasoning.csv`, `gemini_reasoning.csv`, `mistral_reasoning.csv` (Available at Github) |
| Scorecard | `shock_reasoning_scorecard.csv`(Available in Github) |
| PPT & Presentation video | Available in Github |

## How to run

Follow below steps to walkthrough project:
1. Open ShockPrediction_LLM_Explaination_scorecard.ipynb and run all cells. Assumption you have access to Google BigQuery, GPUs (consider Colab).
    a. This will fetch Data from MIMIC III using google bigquery - 40000 Patients data.
    b. Remove patients whose Vital and Labs data is not available.
    c. Prepare the model to train based on 80/20 training and validation.
    d. Create TransformerClassifier, and use AdamW Optimizer. 
    e. Train models with 15 Epochs and very low learning rate (5e-5)
    f. Plot validation Accuracy and Losses.
    g. Use SHAP to find Top Clinical Contributors to decisions
    h. Find Global Feature Importance, Graph it to visualize
    i. Create LLM prompt generator, Used GPT4 to see explainations. Score Card is done locally though.
    j. Identify 20 random patients from dataset with Patient Id and Lab+Vitals and save it as patient_summary.csv.
2. fetch_model_explain.py: Download patient_summary.csv from Colab and process for reasoning:
    a. Open file data/patient_summary.csv and read all patients feature(labs+vitals), Prediction from Transformer trained, and Probabillity of Shock predicted by model.
    a. Create a common prompt to send to all LLMs which will be sent to GPT-4, Gemini 1.5 Pro, and Local Mistral(Ollama)
    2. Open Connectivity with GPT-4, Gemini Model, and Mistral.
    3. Send Prompt to each models and save response to gpt4_shock_reasoning.csv, gemini_shock_reasoning.csv, and mistral_shock_reasoning.csv
3. Read resoning - <modelname like gpt4, gemini, mistral>_shock_reasoning.csv via llm_reasoning_score_card.py, gemini_shock_reasoning.csv, and mistral_shock_reasoning.csv
    a. Create prompt for Orchestration to find Clarity, Transparency, Consistency and Completeness.
    a. Read Reasoning from each model file, and send the consolidated prompt(See below) to DeepSeek-R1 running on Local Ollama.
    b. Save the response to csv data/shock_reasoning_scorecard.csv

#PROMPT for Score card / Rating reasoning

```
You are a clinical reasoning evaluator.
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
```
4. Plot score card csv data: 
    refer from High Risk Project/report
5. Refer full pdf reports at
  HighRiskProject/report/shock_prediction_llm_reasoning_score_card.pdf
