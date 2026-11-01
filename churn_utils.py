
import numpy as np
import pandas as pd

TARGET_COL = "Churn"

# Optional service columns -> used to build a "how many add-ons" feature
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "addon_count", "avg_monthly_spend", "charge_gap",
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Columns a user's CSV must contain (raw, before feature engineering)
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and add engineered features. Returns a new DataFrame."""
    out = df.copy()

    # TotalCharges arrives as text; blanks belong to brand-new customers (tenure 0)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    # SeniorCitizen is 0/1 -> treat as a category so it gets one-hot encoded
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(str)

    # --- engineered features ---
    out["addon_count"] = (out[ADDON_COLS] == "Yes").sum(axis=1)
    out["avg_monthly_spend"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], out["MonthlyCharges"]
    )
    # how far the current bill is from the customer's historical average
    out["charge_gap"] = out["MonthlyCharges"] - out["avg_monthly_spend"]
    return out


def encode_target(series: pd.Series) -> pd.Series:
    """'Yes'/'No' -> 1/0 (also accepts values that are already 0/1)."""
    mapped = series.astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0}
    )
    return mapped.astype(int)


def prepare_xy(df: pd.DataFrame):
    """Return (X, y) from a raw dataframe that includes the target column."""
    cleaned = clean_frame(df)
    X = cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = encode_target(df[TARGET_COL])
    return X, y


# ------------------------------------------------------------------
# Evaluation helper (used by the Streamlit app)
# ------------------------------------------------------------------
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def decide_labels(y_prob, model_labels, threshold: float = 0.5):
    """Hard labels for a threshold.

    At the default 0.5 we keep the model's own predict() labels, so the app
    matches the training script exactly. Any other threshold is applied to
    the predicted probabilities.
    """
    if abs(threshold - 0.5) < 1e-9:
        return np.asarray(model_labels).astype(int)
    return (np.asarray(y_prob) >= threshold).astype(int)


def compute_metrics(y_true, y_prob, threshold: float = 0.5, model_labels=None) -> dict:
    """Return the six assignment metrics for a given probability threshold.

    AUC is computed from the raw probabilities, so it does not change with
    the threshold. The other five use hard labels at the chosen threshold.
    """
    if model_labels is None:
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    else:
        y_pred = decide_labels(y_prob, model_labels, threshold)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
