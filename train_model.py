import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load and clean dataset
df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))

df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates()
df = df.dropna(subset=["target"])

X = df.drop(columns=["target"])
y = df["target"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Target must contain both classes: 0 and 1.")

if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
    raise ValueError("All input features must be numeric.")

# Remove constant features
X = X.loc[:, X.nunique(dropna=True) > 1]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

results = []
best_accuracy = -1
best_model = None
best_name = ""

# 4. Train and compare models
for name, estimator in models.items():

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        estimator
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2)
    })

    print(f"{name}: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_name = name

# 5. Display model comparison
results = pd.DataFrame(results).sort_values(
    "Accuracy (%)",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv(
    os.path.join(BASE_DIR, "model_comparison.csv"),
    index=False
)

# 6. Evaluate the selected model
predictions = best_model.predict(X_test)

print("\nBEST MODEL:", best_name)
print(f"BEST TEST ACCURACY: {best_accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 7. Save model and feature names
joblib.dump(
    {
        "model": best_model,
        "features": X.columns.tolist(),
        "model_name": best_name
    },
    os.path.join(BASE_DIR, "heart_model.pkl")
)

print("\nModel saved as heart_model.pkl")
print("Comparison saved as model_comparison.csv")