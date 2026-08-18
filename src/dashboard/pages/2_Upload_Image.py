from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.dashboard.components import brand_header, disclaimer_banner, inject_theme, prediction_card
from src.dashboard.model_service import predict_image
from src.dashboard.report import build_csv_report, build_pdf_report
from src.data_preprocessing.image_preprocessing import preprocess_single

st.set_page_config(page_title="Upload Image — CellScan", page_icon="🔬", layout="wide")
inject_theme()
brand_header()

st.markdown("## Histopathology Image Prediction")
st.markdown(
    '<p class="cs-subtle">Upload one or more stained tissue patches (BreakHis / IDC-style). '
    "CellScan converts each to grayscale, resizes to 224x224, applies CLAHE contrast "
    "enhancement and denoising, then scores it with the trained CNN.</p>",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Tissue patch image(s)", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True
)


def _load_and_preprocess(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")
    raw_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess_single(raw_array)
    return pil_img, preprocessed


if not uploaded_files:
    st.info("Upload one or more images to run predictions.")
else:
    if st.button("Run prediction", type="primary"):
        rows = []
        cache = {}
        with st.spinner(f"Running CNN inference on {len(uploaded_files)} image(s)..."):
            for f in uploaded_files:
                pil_img, preprocessed = _load_and_preprocess(f)
                result = predict_image(preprocessed)
                cache[f.name] = {"pil_img": pil_img, "preprocessed": preprocessed, "result": result}
                if result is not None:
                    rows.append({
                        "filename": f.name,
                        "predicted_class": result["predicted_class"],
                        "probability_malignant": round(result["probability_malignant"], 4),
                    })
        st.session_state["image_batch_cache"] = cache
        st.session_state["image_batch_results"] = pd.DataFrame(rows) if rows else None

    cache = st.session_state.get("image_batch_cache")
    results_df = st.session_state.get("image_batch_results")

    if cache and any(v["result"] is None for v in cache.values()):
        st.warning(
            "No trained image model found under data/models/image/. Run "
            "`python scripts/train_image.py` after downloading the histopathology "
            "dataset (see README) to enable predictions."
        )

    if results_df is not None and len(results_df):
        n_malignant = (results_df["predicted_class"] == "Malignant").sum()
        st.markdown(
            f'<div class="cs-card"><h4>Batch results</h4>'
            f'<p style="margin:0;">{len(results_df)} image(s) scored — '
            f'<strong>{n_malignant}</strong> flagged malignant, <strong>{len(results_df) - n_malignant}</strong> benign.</p></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results CSV", results_df.to_csv(index=False).encode("utf-8"),
            file_name="cellscan_image_batch_results.csv", mime="text/csv",
        )

        st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
        st.markdown("**Inspect a single image**")
        chosen = st.selectbox("Image", options=results_df["filename"].tolist())
        entry = cache[chosen]
        pil_img, preprocessed, result = entry["pil_img"], entry["preprocessed"], entry["result"]

        preview_left, preview_right = st.columns(2)
        preview_left.image(pil_img, caption="Original upload", use_container_width=True)
        preview_right.image(preprocessed, caption="Preprocessed (grayscale + CLAHE + denoise)", use_container_width=True, clamp=True)

        st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)
        left, right = st.columns([1, 1.2])
        with left:
            prediction_card(result["predicted_class"], result["probability_malignant"])
            st.caption(f"Model: {result['model_key']}")

            csv_bytes = build_csv_report(result, source=chosen)
            pdf_bytes = build_pdf_report(result, source=chosen)
            dl1, dl2 = st.columns(2)
            dl1.download_button("Download CSV report", csv_bytes, file_name="cellscan_image_report.csv", mime="text/csv", key=f"dl_csv_{chosen}")
            dl2.download_button("Download PDF report", pdf_bytes, file_name="cellscan_image_report.pdf", mime="application/pdf", key=f"dl_pdf_{chosen}")
        with right:
            st.markdown('<div class="cs-card"><h4>Grad-CAM — regions driving the prediction</h4></div>', unsafe_allow_html=True)
            overlay_rgb = cv2.cvtColor(result["gradcam_overlay"], cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, use_container_width=True)
            st.caption("Warmer regions contributed more strongly to the predicted class.")

st.markdown("<br>", unsafe_allow_html=True)
disclaimer_banner()
