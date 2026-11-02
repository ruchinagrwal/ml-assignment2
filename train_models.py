

import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from churn_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL, prepare_xy

SEED = 2026
DATA_PATH = "IT_customer_churn.csv"
Path("model").mkdir(exist_ok=True)

# ---------------------------------------------------------------- load
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows, {raw.shape[1]} columns")

n_dupes = raw.duplicated().sum()
raw = raw.drop_duplicates().reset_index(drop=True)
print(f"Dropped {n_dupes} duplicate rows -> {len(raw)} remain")

# ---------------------------------------------------------------- split
# Split the RAW frame so test_data.csv keeps the original columns/format.
raw_train, raw_test = train_test_split(
    raw, test_size=0.20, random_state=SEED, stratify=raw[TARGET_COL]
)
X_train, y_train = prepare_xy(raw_train)
X_test, y_test = prepare_xy(raw_test)
print(f"Train: {len(X_train)} | Test: {len(X_test)} | "
      f"churn rate train={y_train.mean():.3f}, test={y_test.mean():.3f}")

raw_test.to_csv("test_data.csv", index=False)


# ---------------------------------------------------------------- pipelines
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode categoricals; optionally standardise numerics."""
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num_step, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


# Scaling matters for distance/margin-based models; trees don't need it.
model_zoo = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=SEED), True),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                               random_state=SEED), False),
    "KNN": (
        KNeighborsClassifier(n_neighbors=15, weights="distance"), True),
    "Naive Bayes": (
        GaussianNB(), True),
    "Support Vector Machine": (
        SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED), True),
}

file_names = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# ---------------------------------------------------------------- train + eval
rows = []
for name, (clf, needs_scaling) in model_zoo.items():
    pipe = Pipeline([
        ("prep", build_preprocessor(needs_scaling)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    joblib.dump(pipe, f"model/{file_names[name]}")
    print(f"  trained + saved: {name}")

results = pd.DataFrame(rows).round(4)
results.to_csv("model/metrics_summary.csv", index=False)
print("\n=== Test-set metrics ===")
print(results.to_string(index=False))
