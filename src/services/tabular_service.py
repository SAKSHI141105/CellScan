"""Tabular prediction + explanation logic, kept free of any web-framework
dependency so both the API layer and offline scripts can call the same code.

Model/scaler/explainer are loaded once per process and cached in module-level
globals — simple because this process only ever serves one active model, no
need for anything fancier than a dict.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.explainability.tabular_explain import lime_explain_instance, make_lime_explainer, make_shap_explainer, shap_values_for, top_contributors_for_sample
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CFG = load_config()
TABULAR_MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models_dir"] / "tabular"

FEATURE_GROUPS = {
    "mean": ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
             "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"],
    "se": ["radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
           "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se"],
    "worst": ["radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
              "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"],
}
ALL_FEATURES = sum(FEATURE_GROUPS.values(), [])

_cache: dict = {}


def _load_assets():
    """Returns (model, scaler, feature_names, X_train_reference, source_label)."""
    if "assets" in _cache:
        return _cache["assets"]

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
        result = (model, scaler, feature_names, X_train_scaled[feature_names], "trained (scripts/train_tabular.py)")
    else:
        # fallback: quick in-memory model so the API is usable before anyone's
        # run the real training script — see the README's design-decisions note
        from sklearn.ensemble import RandomForestClassifier

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        feature_names = X_train.columns.tolist()
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        result = (model, scaler, feature_names, X_train_scaled, "fallback (untuned RandomForest — run scripts/train_tabular.py for the real ensemble)")

    _cache["assets"] = result
    logger.info("Loaded tabular assets: %s", result[4])
    return result


def _get_explainer():
    if "explainer" in _cache:
        return _cache["explainer"]
    model, _, _, X_background, _ = _load_assets()
    explainer = make_shap_explainer(model, "ensemble_voting", X_background)
    _cache["explainer"] = explainer
    return explainer


def _get_lime_explainer():
    if "lime_explainer" in _cache:
        return _cache["lime_explainer"]
    _, _, _, X_background, _ = _load_assets()
    explainer = make_lime_explainer(X_background)
    _cache["lime_explainer"] = explainer
    return explainer


def explain_with_lime(raw_feature_dict: dict, num_features: int = 6) -> list[dict]:
    """Second, model-agnostic opinion alongside SHAP — see tabular_explain.py's
    module docstring for why both are worth having (SHAP's TreeExplainer path
    doesn't apply as cleanly to the SVM/MLP members of the voting ensemble).
    """
    model, scaler, names, _, _ = _load_assets()

    full_row = pd.DataFrame([raw_feature_dict])
    scaled_full = pd.DataFrame(scaler.transform(full_row[scaler.feature_names_in_]), columns=scaler.feature_names_in_)
    X_row = scaled_full[names]

    explainer = _get_lime_explainer()
    explanation = lime_explain_instance(explainer, model, X_row.iloc[0].values, num_features=num_features)
    return [{"feature": desc, "weight": round(float(weight), 4)} for desc, weight in explanation.as_list()]


def default_feature_values() -> dict:
    df = clean_tabular(load_raw_tabular())
    return df.drop(columns=["diagnosis"]).median().to_dict()


def feature_names() -> list[str]:
    _, _, names, _, _ = _load_assets()
    return names


def predict_single(raw_feature_dict: dict) -> dict:
    model, scaler, names, _, source = _load_assets()

    full_row = pd.DataFrame([raw_feature_dict])
    scaled_full = pd.DataFrame(scaler.transform(full_row[scaler.feature_names_in_]), columns=scaler.feature_names_in_)
    X_row = scaled_full[names]

    proba = float(model.predict_proba(X_row)[0, 1])
    predicted_class = "Malignant" if proba >= 0.5 else "Benign"

    explainer = _get_explainer()
    shap_vals = shap_values_for(explainer, X_row)
    top_contrib = top_contributors_for_sample(shap_vals[0], names, X_row.iloc[0].values, top_k=6)

    return {
        "predicted_class": predicted_class,
        "probability_malignant": proba,
        "top_contributors": top_contrib.to_dict(orient="records"),
        "model_source": source,
    }


def predict_batch(rows_df: pd.DataFrame) -> list[dict]:
    """Skips SHAP for speed — that's only computed when someone drills into a
    single row, not for every row in a large CSV.
    """
    model, scaler, names, _, _ = _load_assets()

    scaled = pd.DataFrame(scaler.transform(rows_df[scaler.feature_names_in_]), columns=scaler.feature_names_in_)
    X = scaled[names]
    proba = model.predict_proba(X)[:, 1]

    out = []
    for p in proba:
        tier = "Low" if p < 0.35 else ("Moderate" if p < 0.65 else "High")
        out.append({
            "predicted_class": "Malignant" if p >= 0.5 else "Benign",
            "probability_malignant": round(float(p), 4),
            "risk_tier": tier,
        })
    return out
