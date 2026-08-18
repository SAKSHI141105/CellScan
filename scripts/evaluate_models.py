"""Regenerates the ROC comparison plot from saved tabular models. Split out
from train_tabular.py so we can re-plot without re-running the full tuning
sweep every time we just want a prettier chart.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.models.tabular_supervised import load_model
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger
from src.utils.metrics import roc_points

logger = get_logger(__name__)

COLORS = {
    "logistic_regression": "#457b9d",
    "random_forest": "#2a9d8f",
    "xgboost": "#e76f51",
    "svm": "#8338ec",
    "mlp": "#f4a261",
    "ensemble_voting": "#1d3557",
    "ensemble_stacking": "#264653",
}


def main():
    cfg = load_config()
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "tabular"
    figures_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]

    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    _, X_test_scaled, _ = scale_features(X_train, X_test)
    selected = load_model(models_dir / "selected_features.joblib")
    X_test_sel = X_test_scaled[selected]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")

    for name, color in COLORS.items():
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = load_model(model_path)
        y_proba = model.predict_proba(X_test_sel)[:, 1]
        fpr, tpr = roc_points(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — tabular models (held-out test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "tabular_roc_curves.png", dpi=150)
    logger.info("Saved ROC comparison to %s", figures_dir / "tabular_roc_curves.png")


if __name__ == "__main__":
    main()
