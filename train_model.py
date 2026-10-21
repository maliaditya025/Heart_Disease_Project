import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 1. Load and clean data
df = pd.read_csv("heart.csv")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates().dropna(subset=["target"])

X = df.drop(columns="target")
y = df["target"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Target must contain both classes: 0 and 1.")

if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
    raise ValueError("This version expects numeric input features.")

X = X.loc[:, X.nunique(dropna=True) > 1]

if X.isna().all().any():
    raise ValueError("A feature contains only missing values.")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models and parameters
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {"model__C": [0.1, 1, 10]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [3, 5, None]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_depth": [5, None]}
    ),
    "Extra Trees": (
        ExtraTreesClassifier(random_state=42),
        {"model__n_estimators": [100, 200]}
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]}
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [3, 5, 7, 9]}
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_score, best_model, best_name = -1, None, None

# 4. Compare models and feature subsets
for name, (estimator, params) in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("features", SelectKBest(score_func=f_classif)),
        ("model", estimator)
    ])

    search = GridSearchCV(
        pipeline,
        {
            **params,
            "features__k": list(range(1, X.shape[1] + 1))
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    results.append({
        "Model": name,
        "CV ROC-AUC": search.best_score_,
        "Features": search.best_params_["features__k"]
    })

    print(f"{name}: CV ROC-AUC = {search.best_score_:.4f}")

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_model = search.best_estimator_
        best_name = name

# 5. Compare results
results = pd.DataFrame(results).sort_values(
    "CV ROC-AUC", ascending=False
)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))
results.to_csv("model_comparison.csv", index=False)

# 6. Evaluate the selected model
pred = best_model.predict(X_test)

print("\nBEST MODEL:", best_name)
print("Test Accuracy:", accuracy_score(y_test, pred))
print("Test ROC-AUC:", roc_auc_score(
    y_test, best_model.predict_proba(X_test)[:, 1]
))
print("\nClassification Report:\n",
      classification_report(y_test, pred, zero_division=0))

# 7. Display selected features
mask = best_model.named_steps["features"].get_support()
print("Selected Features:", X.columns[mask].tolist())

# 8. Save complete pipeline and feature order
joblib.dump(
    {"model": best_model, "features": X.columns.tolist()},
    "heart_model.pkl"
)

print("\nBest model and preprocessing pipeline saved!")