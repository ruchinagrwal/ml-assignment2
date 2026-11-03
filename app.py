
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from churn_utils import (
    RAW_FEATURE_COLUMNS, TARGET_COL, clean_frame, compute_metrics,
    decide_labels, encode_target, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Churn Risk Lab", page_icon="📉", layout="wide")


# ------------------------------------------------------------ helpers
@st.cache_resource
def load_model(display_name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[display_name])


def score(df: pd.DataFrame, model_name: str):
    """Return (y_true, churn_probability, model_labels) for a raw dataframe."""
    cleaned = clean_frame(df)
    X = cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_true = encode_target(df[TARGET_COL])
    pipe = load_model(model_name)
    probs = pipe.predict_proba(X)[:, 1]
    labels = pipe.predict(X)
    return y_true.to_numpy(), probs, labels


# ------------------------------------------------------------ header
st.title("📉 Churn Risk Lab")
st.caption(
    "Upload held-out customer records and compare five pre-trained "
    "classifiers on whether each customer will leave."
)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    sample_path = Path(__file__).parent / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )

    st.header("2. Model")
    chosen = st.selectbox("Choose a classifier", list(MODEL_FILES))

    st.header("3. Decision threshold")
    threshold = st.slider(
        "Flag as churn when probability ≥", 0.10, 0.90, 0.50, 0.05,
        help="0.50 = each model's default predictions. Lower = catches more churners "
             "(higher recall) but more false alarms.",
    )

if uploaded is None:
    st.info("👈 Upload a CSV in the sidebar to begin. "
            "Use the sample file button if you need a ready-made test set.")
    st.stop()

# ------------------------------------------------------------ validate
try:
    data = pd.read_csv(uploaded)
except Exception as exc:  # unreadable / not a CSV
    st.error(f"Could not read the file: {exc}")
    st.stop()

missing = [c for c in RAW_FEATURE_COLUMNS + [TARGET_COL] if c not in data.columns]
if missing:
    st.error("The CSV is missing required columns: " + ", ".join(missing))
    st.stop()

try:
    y_true, probs, labels = score(data, chosen)
except Exception as exc:
    st.error(f"Could not score this file. Check the values in your columns. ({exc})")
    st.stop()

if len(np.unique(y_true)) < 2:
    st.error("The Churn column has only one class, so AUC cannot be computed.")
    st.stop()

y_pred = decide_labels(probs, labels, threshold)

st.success(f"Scored {len(data):,} customers with **{chosen}**.")

tab_metrics, tab_matrix, tab_compare, tab_rows = st.tabs(
    ["Metrics", "Confusion matrix & report", "All-model comparison", "Customer risk table"]
)

# ------------------------------------------------------------ tab 1: metrics
with tab_metrics:
    m = compute_metrics(y_true, probs, threshold, labels)
    cols = st.columns(6)
    for col, name in zip(cols, METRIC_ORDER):
        col.metric(name, f"{m[name]:.3f}")

    st.caption(f"Actual churn rate in this file: {y_true.mean():.1%} · "
               f"Predicted churn rate: {y_pred.mean():.1%}")

    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, linewidth=2, label=f"{chosen} (AUC = {m['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------ tab 2: matrix + report
with tab_matrix:
    left, right = st.columns(2)
    with left:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                    xticklabels=["Stayed", "Churned"],
                    yticklabels=["Stayed", "Churned"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix - {chosen}")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.write(f"**{tp}** churners caught · **{fn}** missed · "
                 f"**{fp}** false alarms · **{tn}** correctly kept")
    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_true, y_pred, target_names=["Stayed", "Churned"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

# ------------------------------------------------------------ tab 3: compare all
with tab_compare:
    st.write("Every model scored on the uploaded file at the chosen threshold.")
    rows = []
    for name in MODEL_FILES:
        yt, pr, lb = score(data, name)
        rows.append({"ML Model Name": name, **compute_metrics(yt, pr, threshold, lb)})
    table = pd.DataFrame(rows).set_index("ML Model Name")[METRIC_ORDER].round(4)
    st.dataframe(table)
    best = table["MCC"].idxmax()
    st.caption(f"Highest MCC on this file: **{best}**.")

# ------------------------------------------------------------ tab 4: risk table
with tab_rows:
    out = data.copy()
    out["churn_probability"] = probs.round(4)
    out["predicted_churn"] = np.where(y_pred == 1, "Yes", "No")
    out = out.sort_values("churn_probability", ascending=False)
    st.write("Customers ranked from highest to lowest churn risk.")
    st.dataframe(out)
    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    st.download_button("Download scored file", buffer.getvalue(),
                       file_name="scored_customers.csv", mime="text/csv")
