import os
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")

# 1. Load trained model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

saved = joblib.load(MODEL_PATH)

model = saved["model"]
features = saved["features"]
model_name = saved.get("model_name", "Machine Learning Model")

# 2. Configure application
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")

st.write("Enter the patient's health information below.")

st.info(
    "This application is for educational purposes only "
    "and is not a medical diagnostic tool."
)

st.caption(f"Selected model: {model_name}")

# 3. Define input options
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

# 4. Collect patient information
patient = {}

for feature in features:

    if feature in options:
        label, choices = options[feature]

        patient[feature] = st.selectbox(
            label,
            choices
        )

    elif feature == "age":
        patient[feature] = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=100,
            value=50
        )

    elif feature == "trestbps":
        patient[feature] = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=250,
            value=120
        )

    elif feature == "chol":
        patient[feature] = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=600,
            value=200
        )

    elif feature == "thalach":
        patient[feature] = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

    elif feature == "oldpeak":
        patient[feature] = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

    else:
        patient[feature] = st.number_input(
            feature,
            value=0.0
        )

# 5. Make prediction
if st.button("❤️ Predict Heart Disease", use_container_width=True):

    patient_df = pd.DataFrame(
        [patient],
        columns=features
    )

    prediction = model.predict(patient_df)[0]

    if prediction == 1:
        st.error("⚠️ Model prediction: Positive")
    else:
        st.success("✅ Model prediction: Negative")

    # Display probability only when supported
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(patient_df)[0][1]

        st.write(
            f"Estimated positive-class probability: "
            f"**{probability:.2%}**"
        )

        st.caption(
            "This is a model estimate, not a confirmed "
            "clinical risk or diagnosis."
        )

    else:
        st.info(
            "Probability estimates are not available "
            "for the selected model."
        )