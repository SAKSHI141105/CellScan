"""CellScan dashboard entry point.

    streamlit run src/dashboard/app.py

Streamlit auto-discovers everything under pages/ and builds the sidebar nav
from it — this file is just the landing page plus the one-time page config /
theme injection that every page needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# streamlit execs this file directly, so the project root isn't on sys.path
# by default — this has to happen before the `from src...` imports below,
# and since it runs once per server process it covers the page/ scripts too.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.dashboard.components import brand_header, disclaimer_banner, inject_theme

st.set_page_config(
    page_title="CellScan — Breast Tissue Decision Support",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
brand_header()

st.markdown("### Multi-modal breast tissue classification, with explanations")
st.markdown(
    """
    <p class="cs-subtle" style="max-width: 640px; font-size:0.98rem; line-height:1.55;">
    CellScan runs two independent pipelines over the Wisconsin Diagnostic Breast Cancer
    dataset and histopathology image patches — one on structured clinical measurements,
    one on tissue imagery — and pairs each prediction with a visual explanation of what
    drove it, rather than returning a bare label.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """<div class="cs-card"><h4>Clinical Data</h4>
        <p style="margin:0;">Enter the 30 diagnostic measurements (or upload a CSV row)
        and get a probability score backed by an SHAP explanation.</p></div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """<div class="cs-card"><h4>Histopathology Image</h4>
        <p style="margin:0;">Upload a tissue patch. The CNN pipeline preprocesses it
        automatically and returns a Grad-CAM heatmap alongside the prediction.</p></div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """<div class="cs-card"><h4>Model Insight</h4>
        <p style="margin:0;">Compare every trained model's metrics, inspect ROC curves,
        and explore how the unsupervised clustering separates the two classes.</p></div>""",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
st.markdown("Use the sidebar to jump to **Clinical Data**, **Upload Image**, **Model Performance**, or **Cluster Explorer**.")

st.markdown("<br>", unsafe_allow_html=True)
disclaimer_banner()
