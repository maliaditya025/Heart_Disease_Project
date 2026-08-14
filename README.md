# ❤️ Heart Disease Prediction using Machine Learning

A Machine Learning web application built with **Streamlit** that predicts whether a person is likely to have heart disease based on various health parameters.

---

## 📌 Project Overview

This project uses a trained Machine Learning model to predict the presence of heart disease from patient health information.

Users enter medical details through an interactive Streamlit interface, and the application predicts whether heart disease is likely along with the prediction confidence.

---

## 🚀 Features

- Interactive Streamlit web application
- Real-time heart disease prediction
- Displays prediction confidence
- Machine Learning model trained using Scikit-learn
- Data preprocessing with StandardScaler
- Easy to retrain the model using `train_model.py`

---

## 📂 Project Structure

```
Heart_Disease_Project/
│
├── app.py                 # Streamlit application
├── train_model.py         # Model training script
├── heart.csv              # Dataset
├── heart_model.pkl        # Trained Machine Learning model
├── scaler.pkl             # StandardScaler object
├── requirements.txt       # Required Python packages
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 📊 Dataset

The project uses the Heart Disease dataset containing patient medical information such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thal

Target:

- 0 → No Heart Disease
- 1 → Heart Disease

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/maliaditya025/Heart_Disease_Project.git
```

```bash
cd Heart_Disease_Project
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start at:

```
http://localhost:8501
```

---

## 🧠 Train the Model

To retrain the model using the dataset:

```bash
python train_model.py
```

This script generates:

- heart_model.pkl
- scaler.pkl

---

## 📸 Application Preview

Add screenshots of your application here.

Example:

```
images/homepage.png
images/prediction.png
```

---

## 📈 Prediction Workflow

```
User Input
      │
      ▼
Data Preprocessing
      │
      ▼
StandardScaler
      │
      ▼
Machine Learning Model
      │
      ▼
Prediction
      │
      ▼
Result + Confidence Score
```

---

## 🤝 Contributing

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch
git push origin feature-name

5. Create a Pull Request

---

## 👨‍💻 Author

**Aditya Mali**

B.Tech Artificial Intelligence & Machine Learning

GitHub:
https://github.com/maliaditya025