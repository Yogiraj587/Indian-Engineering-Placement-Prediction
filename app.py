import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Placement Prediction Intelligence System",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.08);
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #2E86C1;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🎓 Student Placement Prediction System")
st.write("Upload your test dataset and evaluate trained ML models.")

# ---------------------------------------------------------
# Dataset Upload (MAIN PAGE)
# ---------------------------------------------------------
st.markdown("### 📂 Upload Test Dataset (CSV Only)")
uploaded_file = st.file_uploader("Drag and drop your test CSV here", type=["csv"])

# ---------------------------------------------------------
# Model Selection (MAIN PAGE)
# ---------------------------------------------------------
st.markdown("### 🤖 Select Model for Evaluation")
model_selected = st.selectbox(
    "Choose a trained model",
    [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "GaussianNB",
        "XGBoost"
    ]
)

# ---------------------------------------------------------
# Model Loader with Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_selected_model(model_name):
    model_files = {
        "Logistic Regression": "model/model_logistic_regression.pkl",
        "Random Forest": "model/model_random_forest.pkl",
        "Decision Tree": "model/model_decision_tree.pkl",
        "KNN": "model/model_knn.pkl",
        "GaussianNB": "model/model_gaussian_nb.pkl",
        "XGBoost": "model/model_xgboost.pkl"
    }
    return joblib.load(model_files[model_name])

# ---------------------------------------------------------
# If file uploaded
# ---------------------------------------------------------
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.markdown('<div class="section-title">🔎 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(data.head(), width='stretch')

    if "placement_status" not in data.columns:
        st.error("The dataset must contain 'placement_status' column.")
        st.stop()

    X_input = data.drop("placement_status", axis=1)
    y_true = data["placement_status"]

    # Load model safely
    try:
        model_pipeline = load_selected_model(model_selected)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Predict
    try:
        y_pred = model_pipeline.predict(X_input)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Probability check for AUC
    y_prob = None
    if hasattr(model_pipeline, "predict_proba"):
        try:
            y_prob = model_pipeline.predict_proba(X_input)[:, 1]
            auc_value = roc_auc_score(y_true, y_prob)
        except Exception:
            auc_value = None
    else:
        auc_value = None

    # ---------------------------------------------------------
    # Metrics Calculation
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # ---------------------------------------------------------
    # Display Metrics
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc_value:.4f}" if auc_value else "N/A")
    col3.metric("Precision", f"{precision:.4f}")
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">🧩 Confusion Matrix</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ---------------------------------------------------------
    # Classification Report
    # ---------------------------------------------------------
    st.markdown('<div class="section-title">📄 Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_true, y_pred)
    st.code(report)
