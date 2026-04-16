import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
le = joblib.load("encoder.pkl")
columns = joblib.load("columns.pkl")

st.title("Loan Approval System")

st.write("Click button to predict")

if st.button("Predict"):
    sample = pd.DataFrame([[0]*len(columns)], columns=columns)
    pred = model.predict(sample)[0]
    result = le.inverse_transform([pred])[0]
    st.success(f"Prediction: {result}")
