import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "Target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(
    page_title="Student Outcome Analytics",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------------
# Custom Styling
# ------------------------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

h1 {
    color: #1b4965;
}

[data-testid="stMetricValue"] {
    color: #0b6e4f;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------

@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


def score(model, X, y):
    y_pred = model.predict(X)

    try:
        auc = roc_auc_score(
            y,
            model.predict_proba(X),
            multi_class="ovr",
            average="macro",
            labels=model.classes_
        )
    except ValueError:
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "F1 Score": f1_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y, y_pred),
    }

    return metrics, y_pred


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("🚀 Prediction Dashboard")

uploaded = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type="csv"
)

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(MODEL_FILES)
)

sample = BASE_DIR / "test_data.csv"

if sample.exists():
    st.sidebar.download_button(
        "📥 Download Sample Dataset",
        sample.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------
# Main Page
# ------------------------------------------------------------------

st.title("🎯 Student Outcome Analytics")

st.caption(
    "Explore and compare machine learning models that predict whether "
    "students are likely to graduate, remain enrolled, or drop out."
)

with st.expander("📖 About the Dataset"):
    st.write("""
    This dataset contains student demographic, academic and enrolment
    information. The uploaded records are evaluated using multiple
    machine learning models to predict educational outcomes.
    """)

# ------------------------------------------------------------------
# No Upload State
# ------------------------------------------------------------------

if uploaded is None:

    st.info(
        "Please upload a CSV file containing all required feature columns "
        "and the Target column for model evaluation."
    )

    summary = MODEL_DIR / "metrics_summary.csv"

    if summary.exists():
        st.subheader("📈 Benchmark Performance on Test Dataset")

        ref = pd.read_csv(summary, index_col=0)
        st.dataframe(
            ref,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.4f")
                for c in ref.columns
            },
        )
    st.stop()

# ------------------------------------------------------------------
# Load Uploaded Data
# ------------------------------------------------------------------

data = pd.read_csv(uploaded, sep=None, engine="python")
data.columns = data.columns.str.strip()

model = load_model(MODEL_FILES[model_name])

required = list(model.feature_names_in_)

missing = [c for c in required + [TARGET] if c not in data.columns]

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

X = data[required]
y = data[TARGET]

st.success(
    f"Dataset loaded successfully with {len(data)} records | "
    + ", ".join(f"{k}: {v}" for k, v in y.value_counts().items())
)

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

tab_one, tab_all = st.tabs(
    [f"🔍 {model_name} Analysis", "🏆 Model Comparison"]
)

# ------------------------------------------------------------------
# Selected Model Analysis
# ------------------------------------------------------------------

with tab_one:

    metrics, y_pred = score(model, X, y)

    st.subheader("📊 Model Performance Metrics")

    for col, (label, value) in zip(st.columns(6), metrics.items()):
        col.metric(
            label,
            "n/a" if np.isnan(value) else f"{value:.3f}"
        )

    left, right = st.columns(2)

    with left:

        st.subheader("🎯 Confusion Matrix")

        labels = list(model.classes_)

        cm = confusion_matrix(
            y,
            y_pred,
            labels=labels
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

    with right:

        st.subheader("📋 Classification Report")

        report = classification_report(
            y,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).T.round(3),
            width="stretch"
        )

# ------------------------------------------------------------------
# Compare All Models
# ------------------------------------------------------------------

with tab_all:

    rows = {
        name: score(load_model(f), X, y)[0]
        for name, f in MODEL_FILES.items()
    }

    comparison = pd.DataFrame(rows).T.round(4)

    st.subheader("🏅 Model Comparison Table")

    st.dataframe(
            comparison,
            width="stretch",
            column_config={
            c: st.column_config.NumberColumn(format="%.4f")
            for c in comparison.columns
        },
    )
    best_model = comparison["MCC"].idxmax()

    st.success(
        f"Best performing model for this dataset (based on MCC): "
        f"{best_model}"
    )