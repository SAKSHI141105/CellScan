"""End-to-end tabular training run: clean -> select features -> SMOTE
comparison -> clustering -> tune 5 models -> ensemble -> save everything.

Runs in a few minutes on a laptop CPU since WDBC is only 569 rows — the
RandomizedSearch on rf/xgb is the slow part, everything else is instant.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_preprocessing.imbalance import apply_smote
from src.data_preprocessing.tabular_preprocessing import (
    clean_tabular,
    load_raw_tabular,
    scale_features,
    train_test_split_tabular,
)
from src.feature_engineering.feature_selection import select_features
from src.models.tabular_clustering import run_all_clustering, summary_table
from src.models.tabular_supervised import build_ensemble, evaluate_fitted_models, save_models, train_all_models
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger
from src.utils.metrics import classification_report_dict

logger = get_logger(__name__)


def smote_vs_baseline(X_train, y_train, X_test, y_test, k_neighbors: int, random_state: int) -> pd.DataFrame:
    """A plain logistic regression trained with and without SMOTE — isolates
    the effect of resampling from the effect of a stronger model.
    """
    baseline = LogisticRegression(max_iter=2000, random_state=random_state).fit(X_train, y_train)
    baseline_metrics = classification_report_dict(
        y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]
    )

    X_res, y_res, before, after = apply_smote(X_train, y_train, k_neighbors, random_state)
    smoted = LogisticRegression(max_iter=2000, random_state=random_state).fit(X_res, y_res)
    smoted_metrics = classification_report_dict(
        y_test, smoted.predict(X_test), smoted.predict_proba(X_test)[:, 1]
    )

    comparison = pd.DataFrame({"baseline_no_smote": baseline_metrics, "with_smote": smoted_metrics}).T.round(4)
    logger.info("Class distribution before SMOTE: %s -> after: %s", before, after)
    return comparison, X_res, y_res


def main():
    cfg = load_config()
    tab_cfg = cfg["tabular"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"] / "tabular"
    figures_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    selected = select_features(X_train_scaled, y_train)
    logger.info("Selected %d/%d features after correlation pruning: %s", len(selected), X_train.shape[1], selected)
    X_train_sel, X_test_sel = X_train_scaled[selected], X_test_scaled[selected]

    # --- SMOTE comparison (its own reported section, per the spec) ---
    smote_comparison, X_train_res, y_train_res = smote_vs_baseline(
        X_train_sel, y_train, X_test_sel, y_test, tab_cfg["smote"]["k_neighbors"], tab_cfg["random_state"]
    )
    print("\n=== SMOTE vs baseline (Logistic Regression) ===")
    print(smote_comparison)
    smote_comparison.to_csv(figures_dir / "smote_comparison.csv")

    # --- unsupervised exploration on the full (unresampled) training data ---
    cluster_output = run_all_clustering(X_train_sel.values, y_train.values, cfg["clustering"])
    cluster_summary = summary_table(cluster_output["cluster_results"])
    print("\n=== Clustering metrics (silhouette / ARI / NMI vs true diagnosis) ===")
    print(cluster_summary)
    cluster_summary.to_csv(figures_dir / "clustering_summary.csv")

    # --- supervised: tune all 5 models on the SMOTE-resampled training set ---
    tuned = train_all_models(
        X_train_res, y_train_res, tab_cfg["models"], cv_folds=tab_cfg["cv_folds"],
        n_iter=tab_cfg["search_n_iter"], random_state=tab_cfg["random_state"],
    )
    for name, res in tuned.items():
        logger.info("[%s] cv recall: mean=%.4f std=%.4f", name, res["cv_scores"].mean(), res["cv_scores"].std())

    fitted = {name: res["best_estimator"] for name, res in tuned.items()}
    voting = build_ensemble(tuned, kind="voting", top_n=3).fit(X_train_res, y_train_res)
    stacking = build_ensemble(tuned, kind="stacking", top_n=3).fit(X_train_res, y_train_res)
    fitted["ensemble_voting"] = voting
    fitted["ensemble_stacking"] = stacking

    comparison_table = evaluate_fitted_models(fitted, X_test_sel, y_test)
    print("\n=== Final model comparison (held-out test set) ===")
    print(comparison_table)
    comparison_table.to_csv(figures_dir / "tabular_model_comparison.csv")

    best_params = {name: res["search"].best_params_ for name, res in tuned.items()}
    with open(models_dir / "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    save_models(fitted, models_dir)
    import joblib
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(selected, models_dir / "selected_features.joblib")

    logger.info("Tabular training run complete. Artifacts in %s", models_dir)


if __name__ == "__main__":
    main()
