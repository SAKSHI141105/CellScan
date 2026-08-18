from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.components import brand_header, disclaimer_banner, inject_theme, prediction_card
from src.dashboard.model_service import default_feature_values, predict_tabular
from src.dashboard.report import build_csv_report, build_pdf_report
from src.explainability.tabular_explain import plain_language_summary

st.set_page_config(page_title="Clinical Data — CellScan", page_icon="🔬", layout="wide")
inject_theme()
brand_header()

st.markdown("## Clinical Data Prediction")
st.markdown(
    '<p class="cs-subtle">Enter the 30 diagnostic measurements from a fine-needle aspirate, '
    "or upload a single-row CSV in the same format as the WDBC dataset.</p>",
    unsafe_allow_html=True,
)

FEATURE_GROUPS = {
    "Mean": ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
             "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"],
    "Standard Error": ["radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se"],
    "Worst": ["radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
              "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"],
}

defaults = default_feature_values()

input_tab, csv_tab = st.tabs(["Manual entry", "Upload CSV row"])

feature_values = {}
with input_tab:
    for group_name, cols in FEATURE_GROUPS.items():
        st.markdown(f"**{group_name} features**")
        grid = st.columns(5)
        for i, col_name in enumerate(cols):
            default_val = float(defaults.get(col_name, 0.0))
            feature_values[col_name] = grid[i % 5].number_input(
                col_name.replace("_", " "), value=round(default_val, 4), format="%.4f", key=f"manual_{col_name}"
            )

with csv_tab:
    uploaded = st.file_uploader("CSV with a single row of the 30 features", type=["csv"])
    if uploaded is not None:
        row_df = pd.read_csv(uploaded).iloc[0]
        feature_values = {c: float(row_df[c]) for c in sum(FEATURE_GROUPS.values(), []) if c in row_df}
        st.dataframe(row_df.to_frame().T, use_container_width=True)

st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)

if st.button("Run prediction", type="primary"):
    with st.spinner("Scoring against the tabular ensemble..."):
        result = predict_tabular(feature_values)

    left, right = st.columns([1, 1.3])
    with left:
        prediction_card(result["predicted_class"], result["probability_malignant"])
        st.caption(f"Model: {result['model_source']}")

        summary = plain_language_summary(result["top_contributors"], result["predicted_class"])
        st.markdown(f'<div class="cs-card"><h4>Explanation</h4><p style="margin:0;">{summary}</p></div>', unsafe_allow_html=True)

        csv_bytes = build_csv_report(result, source="manual clinical entry")
        pdf_bytes = build_pdf_report(result, source="manual clinical entry", top_contributors=result["top_contributors"])
        dl1, dl2 = st.columns(2)
        dl1.download_button("Download CSV report", csv_bytes, file_name="cellscan_report.csv", mime="text/csv")
        dl2.download_button("Download PDF report", pdf_bytes, file_name="cellscan_report.pdf", mime="application/pdf")

    with right:
        st.markdown('<div class="cs-card"><h4>Top contributing features (SHAP)</h4></div>', unsafe_allow_html=True)
        contrib = result["top_contributors"].copy()
        contrib["direction"] = contrib["shap_value"].apply(lambda v: "pushes toward malignant" if v > 0 else "pushes toward benign")
        st.bar_chart(contrib.set_index("feature")["shap_value"])
        st.dataframe(contrib[["feature", "value", "shap_value", "direction"]].round(4), use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)
disclaimer_banner()
