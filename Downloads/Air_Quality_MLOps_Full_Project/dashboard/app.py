import streamlit as st
import joblib
import pandas as pd

st.title("🌍 Air Quality Forecasting Dashboard")

model = joblib.load("models/best_model.pkl")

df = pd.read_csv("data/processed/air_quality_processed.csv")
prediction = model.predict(df.drop("AQI", axis=1)).mean()

st.metric("Predicted AQI", round(prediction, 2))

if prediction <= 50:
    st.success("Air Quality: GOOD")
elif prediction <= 100:
    st.warning("Air Quality: MODERATE")
else:
    st.error("Air Quality: UNHEALTHY / HAZARDOUS")
