# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("/Users/adii/Downloads/heart.csv")

df.head()

# %%
print(df.shape)
print(df.info())

# %%
print(df.isnull().sum())

# %%
print(df.describe())

# %%
for col in df.columns:
    print(col)
    print(df[col].unique())
    print("-"*40)

# %%
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %%
X = df.drop("target", axis=1)
y = df["target"]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# %%
from sklearn.metrics import accuracy_score

print("Accuracy :", accuracy_score(y_test, y_pred_lr))

# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression")
plt.show()

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_lr))

# %%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy :", accuracy_score(y_test, y_pred_dt))

# %%
cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree")
plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy :", accuracy_score(y_test, y_pred_rf))

# %%
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest")
plt.show()

# %%
from sklearn.metrics import roc_curve, auc

y_prob = lr.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="AUC = %0.2f"%roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %%
print("Logistic Regression :", accuracy_score(y_test, y_pred_lr))
print("Decision Tree       :", accuracy_score(y_test, y_pred_dt))
print("Random Forest       :", accuracy_score(y_test, y_pred_rf))

# %%
print("Enter Patient Details")
age = int(input("Enter Age (years): "))
sex = int(input("Enter Sex (0 = Female, 1 = Male): "))
cp = int(input("Enter Chest Pain Type (0-3): "))
trestbps = int(input("Enter Resting Blood Pressure (mmHg): "))
chol = int(input("Enter Cholesterol (mg/dl): "))
fbs = int(input("Fasting Blood Sugar > 120 mg/dl? (0 = No, 1 = Yes): "))
restecg = int(input("Enter Resting ECG Result (0-2): "))
thalach = int(input("Enter Maximum Heart Rate Achieved: "))
exang = int(input("Exercise-Induced Angina? (0 = No, 1 = Yes): "))
oldpeak = float(input("Enter Oldpeak (ST Depression): "))
slope = int(input("Enter Slope of Peak Exercise ST Segment (0-2): "))
ca = int(input("Enter Number of Major Vessels (0-4): "))
thal = int(input("Enter Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): "))

# %%
new_patient = scaler.transform(new_patient)

# %%
prediction = rf.predict(new_patient)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")

# %%

