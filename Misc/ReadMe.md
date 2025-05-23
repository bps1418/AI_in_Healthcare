AI-Powered Early Detection of Sepsis Using Real-Time Patient Data

Project Objective:

Develop an AI model that predicts the onset of sepsis in hospitalized patients before clinical symptoms become severe. Sepsis is a life-threatening condition caused by the body’s response to infection, and early detection can save lives.

Why This Project?

Sepsis is deadly: It accounts for 1 in 3 hospital deaths.
Early detection can reduce mortality rates significantly.
ML can help by identifying hidden patterns in patient vitals that doctors might miss.
Approach:

Data Collection: Using MIMIC III dataset (BIGQUERY)
Data processing:
Handling Missing values (imputation using KNN)
Feature Engineering
Time series trends Vital Signs
Lab results changes over time
Patient age, or pre-existing conditions.
Normalize Data (For Standardizing data)
Model Development & Training: Basic or baseline Models
Random Forest / XGBoost to classification
Logistic Regression for explainability Deep Learning Approach
LSTM / GRU (for time-series patient vitals)
Transformer-based models (e.g., Time-Series BERT) for sequence modeling
Graph Neural Networks (GNNs) for relationships between patient parameters
Model Evaluation
Performance Metrics: * Precision, Recall, F1-Score (since false negatives are critical) * AUROC (Area Under ROC Curve) for model discrimination
Explainability
Use SHAP or LIME to explain AI decisions to doctors.