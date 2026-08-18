from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.components import brand_header, disclaimer_banner, inject_theme, prediction_card
from src.dashboard.model_service import default_feature_values, predict_tabular, predict_tabular_batch
from src.dashboard.report import build_csv_report, build_pdf_report
from src.explainability.tabular_explain import plain_language_summary

st.set_page_config(page_title="Clinical Data — CellScan", page_icon="🔬", layout="wide")
inject_theme()
brand_header()

st.markdown("## Clinical Data Prediction")
st.markdown(
    '<p class="cs-subtle">Score a single patient by hand, or upload a CSV of many rows '
    "(same 30 WDBC-style columns) for batch predictions.</p>",
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
ALL_FEATURES = sum(FEATURE_GROUPS.values(), [])

manual_tab, batch_tab = st.tabs(["Manual entry", "Batch CSV upload"])

# ---------------------------------------------------------------- manual entry
with manual_tab:
    defaults = default_feature_values()
    feature_values = {}
    for group_name, cols in FEATURE_GROUPS.items():
        st.markdown(f"**{group_name} features**")
        grid = st.columns(5)
        for i, col_name in enumerate(cols):
            default_val = float(defaults.get(col_name, 0.0))
            feature_values[col_name] = grid[i % 5].number_input(
                col_name.replace("_", " "), value=round(default_val, 4), format="%.4f", key=f"manual_{col_name}"
            )

    st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)

    if st.button("Run prediction", type="primary", key="run_manual"):
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
            dl1.download_button("Download CSV report", csv_bytes, file_name="cellscan_report.csv", mime="text/csv", key="dl_csv_manual")
            dl2.download_button("Download PDF report", pdf_bytes, file_name="cellscan_report.pdf", mime="application/pdf", key="dl_pdf_manual")

        with right:
            st.markdown('<div class="cs-card"><h4>Top contributing features (SHAP)</h4></div>', unsafe_allow_html=True)
            contrib = result["top_contributors"].copy()
            contrib["direction"] = contrib["shap_value"].apply(lambda v: "pushes toward malignant" if v > 0 else "pushes toward benign")
            st.bar_chart(contrib.set_index("feature")["shap_value"])
            st.dataframe(contrib[["feature", "value", "shap_value", "direction"]].round(4), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- batch upload
with batch_tab:
    st.markdown(
        '<p class="cs-subtle">CSV needs the 30 WDBC feature columns (radius_mean, texture_mean, ... '
        "fractal_dimension_worst). Extra columns like id or diagnosis are ignored.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("CSV of patient rows", type=["csv"], key="batch_csv")

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        missing = [c for c in ALL_FEATURES if c not in raw_df.columns]

        if missing:
            st.error(f"CSV is missing {len(missing)} required column(s): {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
        else:
            st.caption(f"{len(raw_df)} rows loaded.")
            if st.button("Run batch prediction", type="primary", key="run_batch"):
                with st.spinner(f"Scoring {len(raw_df)} rows..."):
                    predictions = predict_tabular_batch(raw_df)
                st.session_state["batch_results"] = pd.concat([raw_df.reset_index(drop=True), predictions], axis=1)

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        n_malignant = (results["predicted_class"] == "Malignant").sum()
        st.markdown(
            f'<div class="cs-card"><h4>Batch results</h4>'
            f'<p style="margin:0;">{len(results)} rows scored — '
            f'<strong>{n_malignant}</strong> flagged malignant, <strong>{len(results) - n_malignant}</strong> benign.</p></div>',
            unsafe_allow_html=True,
        )

        display_cols = [c for c in ["predicted_class", "probability_malignant", "risk_tier"] if c in results.columns]
        st.dataframe(
            results[display_cols + [c for c in ALL_FEATURES[:4]]].round(4),
            use_container_width=True,
        )

        full_csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download full results CSV", full_csv, file_name="cellscan_batch_results.csv", mime="text/csv", key="dl_batch_csv")

        st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
        st.markdown("**Inspect a single row's explanation**")
        row_idx = st.selectbox("Row", options=results.index.tolist(), format_func=lambda i: f"Row {i} — {results.loc[i, 'predicted_class']} ({results.loc[i, 'probability_malignant']:.1%})")

        if st.button("Explain this row", key="explain_batch_row"):
            with st.spinner("Computing SHAP explanation..."):
                row_dict = {c: float(results.loc[row_idx, c]) for c in ALL_FEATURES}
                detail = predict_tabular(row_dict)

            left, right = st.columns([1, 1.3])
            with left:
                prediction_card(detail["predicted_class"], detail["probability_malignant"])
                summary = plain_language_summary(detail["top_contributors"], detail["predicted_class"])
                st.markdown(f'<div class="cs-card"><h4>Explanation</h4><p style="margin:0;">{summary}</p></div>', unsafe_allow_html=True)
            with right:
                contrib = detail["top_contributors"].copy()
                st.bar_chart(contrib.set_index("feature")["shap_value"])

st.markdown("<br>", unsafe_allow_html=True)
disclaimer_banner()
