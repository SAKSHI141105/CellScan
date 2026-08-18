from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
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
    '<p class="cs-subtle">Upload a stained tissue patch (BreakHis / IDC-style). CellScan '
    "converts it to grayscale, resizes to 224x224, applies CLAHE contrast enhancement and "
    "denoising, then scores it with the trained CNN.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Tissue patch image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    raw_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    preprocessed = preprocess_single(raw_array)

    preview_left, preview_right = st.columns(2)
    preview_left.image(pil_img, caption="Original upload", use_container_width=True)
    preview_right.image(preprocessed, caption="Preprocessed (grayscale + CLAHE + denoise)", use_container_width=True, clamp=True)

    st.markdown('<hr class="cs-divider">', unsafe_allow_html=True)

    if st.button("Run prediction", type="primary"):
        with st.spinner("Running CNN inference and Grad-CAM..."):
            result = predict_image(preprocessed)

        if result is None:
            st.warning(
                "No trained image model found under data/models/image/. Run "
                "`python scripts/train_image.py` after downloading the histopathology "
                "dataset (see README) to enable this page's predictions."
            )
        else:
            left, right = st.columns([1, 1.2])
            with left:
                prediction_card(result["predicted_class"], result["probability_malignant"])
                st.caption(f"Model: {result['model_key']}")

                csv_bytes = build_csv_report(result, source=uploaded_file.name)
                pdf_bytes = build_pdf_report(result, source=uploaded_file.name)
                dl1, dl2 = st.columns(2)
                dl1.download_button("Download CSV report", csv_bytes, file_name="cellscan_image_report.csv", mime="text/csv")
                dl2.download_button("Download PDF report", pdf_bytes, file_name="cellscan_image_report.pdf", mime="application/pdf")
            with right:
                st.markdown('<div class="cs-card"><h4>Grad-CAM — regions driving the prediction</h4></div>', unsafe_allow_html=True)
                overlay_rgb = cv2.cvtColor(result["gradcam_overlay"], cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, use_container_width=True)
                st.caption("Warmer regions contributed more strongly to the predicted class.")
else:
    st.info("Upload an image to run a prediction.")

st.markdown("<br>", unsafe_allow_html=True)
disclaimer_banner()
