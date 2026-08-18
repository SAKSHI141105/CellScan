"""Everything the dashboard needs to go from raw input -> prediction ->
explanation, cached so Streamlit doesn't reload/retune on every rerun.

TODO: once train_tabular.py has been run for real, this should just load the
saved ensemble_voting.joblib unconditionally and drop the on-the-fly fallback
below. Left in for now because it makes `streamlit run` work immediately on a
fresh clone without forcing a training run first — genuinely useful for demos,
just not something you'd want in an actual clinical tool.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.explainability.tabular_explain import global_importance, make_shap_explainer, shap_values_for, top_contributors_for_sample
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
TABULAR_MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_dir"] / "tabular"
IMAGE_MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_dir"] / "image"


@st.cache_resource(show_spinner=False)
def load_tabular_assets():
    """Returns (model, scaler, feature_names, X_train_reference, model_source)."""
    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)

    ensemble_path = TABULAR_MODELS_DIR / "ensemble_voting.joblib"
    scaler_path = TABULAR_MODELS_DIR / "scaler.joblib"
    features_path = TABULAR_MODELS_DIR / "selected_features.joblib"

    if ensemble_path.exists() and scaler_path.exists() and features_path.exists():
        model = joblib.load(ensemble_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        X_train_scaled, _, _ = scale_features(X_train, X_test)
        return model, scaler, feature_names, X_train_scaled[feature_names], "trained (scripts/train_tabular.py)"

    # fallback: quick in-memory model so the dashboard is usable out of the box
    from sklearn.ensemble import RandomForestClassifier

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    feature_names = X_train.columns.tolist()
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model, scaler, feature_names, X_train_scaled, "fallback (untuned RandomForest — run scripts/train_tabular.py for the real ensemble)"


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model, model_name: str, _X_background: pd.DataFrame):
    return make_shap_explainer(_model, model_name, _X_background)


def predict_tabular(raw_feature_dict: dict) -> dict:
    model, scaler, feature_names, X_background, source = load_tabular_assets()

    full_row = pd.DataFrame([raw_feature_dict])
    scaled_full = pd.DataFrame(scaler.transform(full_row[scaler.feature_names_in_]), columns=scaler.feature_names_in_)
    X_row = scaled_full[feature_names]

    proba = float(model.predict_proba(X_row)[0, 1])
    predicted_class = "Malignant" if proba >= 0.5 else "Benign"

    explainer = get_shap_explainer(model, "ensemble_voting", X_background)
    shap_vals = shap_values_for(explainer, X_row)
    top_contrib = top_contributors_for_sample(shap_vals[0], feature_names, X_row.iloc[0].values, top_k=6)

    return {
        "predicted_class": predicted_class,
        "probability_malignant": proba,
        "top_contributors": top_contrib,
        "model_source": source,
        "feature_names": feature_names,
        "X_row": X_row,
    }


def default_feature_values() -> dict:
    """Median values from the reference dataset — used to pre-fill the clinical form."""
    df = clean_tabular(load_raw_tabular())
    return df.drop(columns=["diagnosis"]).median().to_dict()


@st.cache_resource(show_spinner=False)
def load_image_model(prefer: str = "transfer"):
    """Returns (model, model_key) or (None, None) if nothing's been trained yet —
    the image dataset is a few GB and isn't vendored, so this is expected on a
    fresh clone until scripts/train_image.py has been run.
    """
    import tensorflow as tf

    backbone = CFG["image"]["transfer_backbone"]
    candidates = [
        (IMAGE_MODELS_DIR / f"transfer_{backbone}_final.h5", f"transfer_{backbone}"),
        (IMAGE_MODELS_DIR / "custom_cnn_final.h5", "custom_cnn"),
    ]
    if prefer == "custom":
        candidates.reverse()

    for path, key in candidates:
        if path.exists():
            return tf.keras.models.load_model(path), key
    return None, None


def predict_image(preprocessed_img: np.ndarray) -> dict:
    from src.explainability.gradcam import explain_prediction

    model, model_key = load_image_model()
    if model is None:
        return None

    batch = preprocessed_img[np.newaxis, ..., np.newaxis] if preprocessed_img.ndim == 2 else preprocessed_img[np.newaxis, ...]
    proba = float(model.predict(batch, verbose=0)[0, 0])
    predicted_class = "Malignant" if proba >= 0.5 else "Benign"
    overlay = explain_prediction(model, model_key, preprocessed_img)

    return {
        "predicted_class": predicted_class,
        "probability_malignant": proba,
        "gradcam_overlay": overlay,
        "model_key": model_key,
    }
