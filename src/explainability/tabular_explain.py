"""SHAP + LIME for the tabular models. SHAP does the heavy lifting (global
importance, summary plot, per-patient force plot); LIME is there mainly as a
second, model-agnostic opinion on individual predictions — useful for the
dashboard's "why did it say this" panel when the primary model is something
SHAP's TreeExplainer doesn't apply to as cleanly (SVM/MLP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

TREE_MODELS = {"random_forest", "xgboost", "ensemble_stacking"}


def make_shap_explainer(model, model_name: str, X_background: pd.DataFrame):
    """TreeExplainer for tree-based models (exact, fast); KernelExplainer as the
    fallback for everything else (slower — sampled, so we cap the background set).
    """
    if model_name in TREE_MODELS or hasattr(model, "estimators_"):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            logger.info("TreeExplainer unavailable for %s, falling back to KernelExplainer", model_name)

    background = shap.sample(X_background, min(100, len(X_background)))
    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    return shap.KernelExplainer(predict_fn, background)


def shap_values_for(explainer, X: pd.DataFrame):
    raw = explainer.shap_values(X)
    # binary classifiers report shap values per class, but the exact shape
    # depends on the shap version/explainer: older releases return a list of
    # two (n_samples, n_features) arrays, newer TreeExplainer returns one
    # (n_samples, n_features, n_classes) array. Either way we only want the
    # malignant (class-1) slice.
    if isinstance(raw, list):
        return raw[1]
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        return raw[..., 1]
    return raw


def global_importance(shap_vals: np.ndarray, feature_names: list[str]) -> pd.Series:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def top_contributors_for_sample(shap_vals_row: np.ndarray, feature_names: list[str], feature_values, top_k: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "value": np.asarray(feature_values),
        "shap_value": shap_vals_row,
    })
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).head(top_k).drop(columns="abs_shap")


def plain_language_summary(top_contributors, predicted_class: str) -> str:
    """Turns the top SHAP contributors into a sentence for non-technical users.
    Accepts either a DataFrame (feature/shap_value columns) or a list of
    {"feature", "shap_value", ...} dicts — the API layer works with plain
    dicts so it doesn't need pandas on its response path.
    """
    if isinstance(top_contributors, pd.DataFrame):
        records = top_contributors.to_dict(orient="records")
    else:
        records = top_contributors

    pushing_up = [r["feature"] for r in records if r["shap_value"] > 0]
    pushing_down = [r["feature"] for r in records if r["shap_value"] < 0]

    def _fmt(names):
        cleaned = [n.replace("_", " ") for n in names[:2]]
        return " and ".join(cleaned)

    if predicted_class.lower() == "malignant" and pushing_up:
        return f"This prediction is primarily driven by elevated {_fmt(pushing_up)} values."
    if predicted_class.lower() == "benign" and pushing_down:
        return f"This prediction is primarily driven by comparatively low {_fmt(pushing_down)} values."
    driver = pushing_up or pushing_down
    return f"This prediction is primarily driven by {_fmt(driver)}." if driver else "No single feature dominates this prediction."


def make_lime_explainer(X_train: pd.DataFrame, class_names=("Benign", "Malignant")):
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
    )


def lime_explain_instance(explainer: LimeTabularExplainer, model, instance_row: np.ndarray, num_features: int = 8):
    predict_fn = model.predict_proba
    return explainer.explain_instance(instance_row, predict_fn, num_features=num_features)
