import streamlit as st
import json
import pandas as pd
import pickle

st.title("🔴 Personio — Attrition Risk Dashboard")

model = pickle.load(open("model.pkl", "rb"))
X_test = pd.read_csv("data/X_test.csv")
insights = json.load(open("insights.json"))

probs = model.predict_proba(X_test)[:,1]

st.subheader("Top 10 At-Risk Employees")
for emp in insights:
    with st.expander(f"Employee {emp['employee_index']} — {emp['risk_score']*100:.0f}% risk"):
        st.write("**Top factors:**", ", ".join(emp['top_factors']))
        st.write("**AI Insight:**", emp['insight'])

st.subheader("Risk Score Distribution")
st.bar_chart(pd.Series(probs).value_counts(bins=10).sort_index())
