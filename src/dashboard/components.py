"""Small render helpers so pages don't repeat the same st.markdown(f"<div...")
soup five times over. Everything here just returns/writes HTML that leans on
assets/style.css classes.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "assets"


def inject_theme():
    css = (ASSETS_DIR / "style.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def brand_header():
    st.markdown(
        """
        <div class="cs-brand">
            <div class="cs-brand-mark">CS</div>
            <div>
                <div class="cs-brand-name">CellScan</div>
                <div class="cs-brand-sub">Breast Tissue Decision Support — Research Build</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_tier(probability: float) -> tuple[str, str]:
    """probability = P(malignant), 0-1. Returns (css_class, label)."""
    if probability < 0.35:
        return "cs-risk-low", "Low Risk"
    if probability < 0.65:
        return "cs-risk-mid", "Moderate Risk"
    return "cs-risk-high", "High Risk"


def prediction_card(predicted_class: str, probability: float):
    css_class, label = risk_tier(probability)
    st.markdown(
        f"""
        <div class="cs-card">
            <h4>Prediction</h4>
            <div style="display:flex; align-items:baseline; gap:1.2rem; flex-wrap:wrap;">
                <div class="cs-score-number">{probability * 100:.1f}%</div>
                <div>
                    <div style="font-weight:600; font-size:1.05rem;">{predicted_class}</div>
                    <span class="cs-risk-badge {css_class}">{label}</span>
                </div>
            </div>
            <div class="cs-subtle" style="margin-top:0.6rem;">probability of malignancy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body_html: str):
    st.markdown(f'<div class="cs-card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def disclaimer_banner():
    st.markdown(
        """
        <div class="cs-disclaimer">
            <strong>Research use only.</strong> CellScan is an educational project, not a
            diagnostic device. It has not been reviewed by any regulatory body and must not
            inform real clinical decisions. See the About page for the full disclaimer.
        </div>
        """,
        unsafe_allow_html=True,
    )
