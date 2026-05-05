import pandas as pd
import pickle
import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

model = pickle.load(open("model.pkl", "rb"))
explainer = pickle.load(open("explainer.pkl", "rb"))
X_test = pd.read_csv("data/X_test.csv")

probs = model.predict_proba(X_test)[:,1]
shap_values = explainer.shap_values(X_test)

# Top 10 high risk
top10_idx = probs.argsort()[-10:][::-1]

client = anthropic.Anthropic()
results = []

for i in top10_idx:
    row = X_test.iloc[i]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values                                                
    top_factors = pd.Series(sv[i][:,1], index=X_test.columns).abs().nlargest(3).index.tolist()
    
    prompt = f"""Employee data: {row[top_factors].to_dict()}
Top attrition risk factors: {top_factors}
Risk score: {probs[i]:.0%}

Write a 2-sentence plain-English HR insight about why this employee is at risk. Be specific."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    
    results.append({
        "employee_index": int(i),
        "risk_score": round(float(probs[i]), 3),
        "top_factors": top_factors,
        "insight": response.content[0].text
    })
    print(f"Done: Employee {i} — {probs[i]:.0%} risk")

json.dump(results, open("insights.json", "w"), indent=2)
print("All insights saved.")
