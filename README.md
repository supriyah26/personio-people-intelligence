# personio-people-intelligence

# Personio People Intelligence

Predictive attrition risk engine built as a Personio integration demo.

## What it does
- Predicts which employees are at risk of leaving in the next 90 days
- Explains WHY using SHAP values
- Generates plain-English HR insights using Claude (Anthropic API)
- Live Streamlit dashboard for HR teams

## Tech Stack
Python, scikit-learn, SHAP, Streamlit, Anthropic Claude API

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
IBM HR Analytics Attrition Dataset via Kaggle
