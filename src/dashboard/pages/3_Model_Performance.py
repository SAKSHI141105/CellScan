from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.components import brand_header, inject_theme
from src.utils.config import PROJECT_ROOT, load_config

st.set_page_config(page_title="Model Performance — CellScan", page_icon="🔬", layout="wide")
inject_theme()
brand_header()

st.markdown("## Model Performance Comparison")

cfg = load_config()
figures_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]

comparison_path = figures_dir / "tabular_model_comparison.csv"
smote_path = figures_dir / "smote_comparison.csv"
roc_path = figures_dir / "tabular_roc_curves.png"

if not comparison_path.exists():
    st.warning(
        "No results yet. Run `python scripts/train_tabular.py` (and optionally "
        "`python scripts/evaluate_models.py` for the ROC plot) to populate this page."
    )
else:
    st.markdown("### Tabular models — held-out test set")
    comparison_df = pd.read_csv(comparison_path, index_col=0)
    st.dataframe(
        comparison_df.style.highlight_max(subset=["recall", "f1", "roc_auc"], color="#d8f3dc"),
        use_container_width=True,
    )
    st.caption("Sorted by recall — minimizing missed malignant cases is the priority metric here, not raw accuracy.")

    if roc_path.exists():
        st.image(str(roc_path), caption="ROC curves, held-out test set", use_container_width=False)

    if smote_path.exists():
        st.markdown("### SMOTE vs. baseline (Logistic Regression)")
        smote_df = pd.read_csv(smote_path, index_col=0)
        st.dataframe(smote_df, use_container_width=True)
        st.caption("Isolates the effect of class-balancing from the effect of model choice.")

clustering_path = figures_dir / "clustering_summary.csv"
if clustering_path.exists():
    st.markdown("### Clustering quality vs. true diagnosis labels")
    st.dataframe(pd.read_csv(clustering_path, index_col=0), use_container_width=True)
    st.caption("Silhouette measures cluster cohesion; ARI/NMI measure agreement with the true benign/malignant split.")
