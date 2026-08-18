from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.dashboard.components import brand_header, inject_theme

st.set_page_config(page_title="About — CellScan", page_icon="🔬", layout="wide")
inject_theme()
brand_header()

st.markdown("## About CellScan")
st.markdown(
    """
    CellScan is a research and portfolio project exploring breast tissue classification
    from two angles at once: structured clinical measurements (the Wisconsin Diagnostic
    Breast Cancer dataset) and histopathology image patches. It combines supervised
    classifiers, unsupervised clustering, and explainability tooling (SHAP, LIME, Grad-CAM)
    behind a single dashboard.

    It was built to demonstrate a full ML system end to end — data cleaning through
    deployment-shaped UI — not to be used as a medical device.
    """
)

st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)

st.markdown("### Disclaimer")
st.markdown(
    """
    <div class="cs-card">
    <p><strong>This is not a diagnostic tool.</strong> CellScan is an educational and research
    prototype. It has not been validated on an independent clinical cohort, has not undergone
    any regulatory review, and carries no clearance from the FDA, CE, or any equivalent body.</p>
    <p>Predictions shown here must never be used to make, support, or influence an actual
    medical decision. Breast tissue diagnosis requires a licensed pathologist reviewing
    histology under a microscope, correlated with clinical and radiological findings — a
    single model score is not a substitute for that process, no matter how confident the
    displayed percentage looks.</p>
    <p>If you are looking at this dashboard because you or someone you know is concerned
    about a real diagnosis, please consult a qualified physician.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)

st.markdown("### What's under the hood")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        **Tabular pipeline**
        - Wisconsin Diagnostic Breast Cancer dataset, 30 features
        - Correlation-based feature pruning + Random Forest importance ranking
        - SMOTE for class balancing (see Model Performance for before/after)
        - Logistic Regression, Random Forest, XGBoost, SVM, MLP — each tuned via
          Grid/RandomizedSearchCV with stratified 5-fold CV
        - Voting + stacking ensembles over the top-performing models
        - SHAP (global + per-prediction) and LIME explanations
        """
    )
with col2:
    st.markdown(
        """
        **Image pipeline**
        - Histopathology patches (BreakHis / Kaggle IDC format)
        - Grayscale, CLAHE, denoising, augmentation
        - GLCM texture + Canny/Sobel edge features (classical route)
        - Custom CNN + transfer learning (ResNet50 / EfficientNetB0 / VGG16)
        - Grad-CAM for visual explanation
        - Conv-autoencoder reconstruction error as an unsupervised anomaly signal
        """
    )

st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
st.markdown(
    '<p class="cs-subtle">See README.md in the project root for setup, folder structure, '
    "and how to reproduce the training runs.</p>",
    unsafe_allow_html=True,
)
