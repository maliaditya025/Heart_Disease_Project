import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)
st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's health information below.")
st.divider()
age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=100,
    value=50
)
sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)
if sex == "Female":
    sex = 0
else:
    sex = 1
cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)
trestbps = st.number_input(
    "Resting Blood Pressure (mmHg)",
    min_value=80,
    max_value=250,
    value=120
)
chol = st.number_input(
    "Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=200
)
fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)
restecg = st.selectbox(
    "Resting ECG Result",
    [0, 1, 2]
)
thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)
exang = st.selectbox(
    "Exercise-Induced Angina",
    [0, 1]
)
oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)
slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)
ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)
thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)
st.divider()
if st.button("❤️ Predict Heart Disease", use_container_width=True):
    patient = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])
    patient_scaled = scaler.transform(patient)
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
        confidence = probability[0][1] * 100
        st.write(
            f"Prediction probability: **{confidence:.2f}%**"
        )
    else:
        st.success("✅ No Heart Disease")
        confidence = probability[0][0] * 100
        st.write(
            f"Prediction probability: **{confidence:.2f}%**"
        )