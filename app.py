import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

saved = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
model = saved["model"]
features = saved["features"]

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's health information below.")
st.info("This application is for educational purposes and is not a medical diagnostic tool.")

# Input fields and their dataset feature names
options = {
    "sex": ("Sex", [0, 1]),
    "cp": ("Chest Pain Type", [0, 1, 2, 3]),
    "fbs": ("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
    "restecg": ("Resting ECG", [0, 1, 2]),
    "exang": ("Exercise-Induced Angina", [0, 1]),
    "slope": ("Slope", [0, 1, 2]),
    "ca": ("Number of Major Vessels", [0, 1, 2, 3, 4]),
    "thal": ("Thal", [0, 1, 2, 3])
}

patient = {}

for feature in features:
    if feature in options:
        label, choices = options[feature]
        patient[feature] = st.selectbox(label, choices)
    elif feature == "age":
        patient[feature] = st.number_input(
            "Age (years)", 1, 100, 50
        )
    elif feature == "trestbps":
        patient[feature] = st.number_input(
            "Resting Blood Pressure", 80, 250, 120
        )
    elif feature == "chol":
        patient[feature] = st.number_input(
            "Cholesterol", 100, 600, 200
        )
    elif feature == "thalach":
        patient[feature] = st.number_input(
            "Maximum Heart Rate", 60, 220, 150
        )
    elif feature == "oldpeak":
        patient[feature] = st.number_input(
            "Oldpeak", 0.0, 10.0, 1.0, step=0.1
        )
    else:
        patient[feature] = st.number_input(feature, value=0.0)

if st.button("❤️ Predict Heart Disease", use_container_width=True):
    patient_df = pd.DataFrame([patient], columns=features)

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    if prediction == 1:
        st.error("⚠️ Model prediction: Positive")
    else:
        st.success("✅ Model prediction: Negative")

    st.write(f"Estimated positive-class probability: **{probability:.2%}**")
    st.caption(
        "This probability is a model estimate, not a confirmed "
        "clinical risk or diagnosis."
    )