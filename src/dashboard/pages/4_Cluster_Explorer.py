from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import brand_header, inject_theme
from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.models.tabular_clustering import project_2d, run_kmeans
from src.utils.config import load_config

st.set_page_config(page_title="Cluster Explorer — CellScan", page_icon="🔬", layout="wide")
inject_theme()
brand_header()

st.markdown("## Unsupervised Cluster Explorer")
st.markdown(
    '<p class="cs-subtle">How well does the feature space separate benign from malignant '
    "without ever using the diagnosis label? PCA and t-SNE project the 30-dimensional "
    "feature space down to 2D so the clustering can be inspected visually.</p>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _get_projection(method: str):
    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    X_scaled, _, _ = scale_features(X_train, X_test)
    coords = project_2d(X_scaled.values, method=method)
    _, km_labels = run_kmeans(X_scaled.values, k=2)
    out = pd.DataFrame(coords, columns=["dim_1", "dim_2"])
    out["true_diagnosis"] = y_train.map({0: "Benign", 1: "Malignant"}).values
    out["kmeans_cluster"] = km_labels.astype(str)
    return out


method = st.radio("Projection method", ["pca", "tsne"], horizontal=True, format_func=str.upper)
proj_df = _get_projection(method)

color_by = st.radio("Color by", ["true_diagnosis", "kmeans_cluster"], horizontal=True,
                     format_func=lambda x: "True diagnosis" if x == "true_diagnosis" else "KMeans cluster")

fig = px.scatter(
    proj_df, x="dim_1", y="dim_2", color=color_by,
    color_discrete_map={"Benign": "#2a9d8f", "Malignant": "#e76f51"} if color_by == "true_diagnosis" else None,
    title=f"{method.upper()} projection — colored by {'true diagnosis' if color_by == 'true_diagnosis' else 'KMeans cluster assignment'}",
    opacity=0.75,
)
fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=560)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "If the KMeans-colored plot roughly mirrors the true-diagnosis-colored plot, the "
    "clustering is recovering the diagnosis structure without ever seeing the labels — "
    "see the Model Performance page for the ARI/NMI scores that quantify this."
)
