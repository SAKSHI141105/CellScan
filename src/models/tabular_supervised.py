"""Supervised model zoo for the tabular pipeline.

Five algorithms, one tuning routine. Rather than writing five near-identical
"define grid -> GridSearchCV -> fit -> report" blocks, everything funnels
through `tune_model()` and a registry dict — the kind of thing that's obvious
in hindsight but only after you've written the copy-pasted version once and
gotten sick of it.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.utils.logging_setup import get_logger
from src.utils.metrics import classification_report_dict

logger = get_logger(__name__)

# n_estimators/max_depth grids for rf & xgb are wide enough that a full grid
# search takes a while — those two go through RandomizedSearchCV, the rest
# through plain GridSearchCV. Small enough grids that it doesn't matter much either way.
_RANDOMIZED_MODELS = {"random_forest", "xgboost"}


def _estimator_registry(random_state: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1),
        "svm": SVC(probability=True, random_state=random_state),
        "mlp": MLPClassifier(random_state=random_state, max_iter=1000, early_stopping=True),
    }


def tune_model(name: str, estimator, param_grid: dict, X_train, y_train, cv_folds: int, n_iter: int, random_state: int):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search_cls = RandomizedSearchCV if name in _RANDOMIZED_MODELS else GridSearchCV
    search_kwargs = dict(estimator=estimator, cv=cv, scoring="recall", n_jobs=-1, refit=True)
    if search_cls is RandomizedSearchCV:
        search_kwargs.update(param_distributions=param_grid, n_iter=n_iter, random_state=random_state)
    else:
        search_kwargs.update(param_grid=param_grid)

    search = search_cls(**search_kwargs)
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info("[%s] tuned in %.1fs -- best recall (cv)=%.4f -- params=%s", name, elapsed, search.best_score_, search.best_params_)
    return search


def train_all_models(X_train, y_train, model_cfgs: dict, cv_folds: int = 5, n_iter: int = 25, random_state: int = 42) -> dict:
    """Returns {name: {'search': fitted GridSearchCV/RandomizedSearchCV, 'cv_scores': array}}"""
    registry = _estimator_registry(random_state)
    results = {}
    for name, param_grid in model_cfgs.items():
        estimator = registry[name]
        search = tune_model(name, estimator, param_grid, X_train, y_train, cv_folds, n_iter, random_state)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(search.best_estimator_, X_train, y_train, cv=cv, scoring="recall")
        results[name] = {"search": search, "best_estimator": search.best_estimator_, "cv_scores": cv_scores}
    return results


def build_ensemble(tuned_results: dict, kind: str = "voting", top_n: int = 3):
    """Picks the top_n models by mean CV recall and combines them.
    kind='voting' -> soft VotingClassifier, kind='stacking' -> StackingClassifier
    with logistic regression as the meta-learner (kept simple on purpose —
    a fancier meta-learner wasn't worth the extra tuning surface here).
    """
    ranked = sorted(tuned_results.items(), key=lambda kv: kv[1]["cv_scores"].mean(), reverse=True)
    chosen = ranked[:top_n]
    estimators = [(name, res["best_estimator"]) for name, res in chosen]
    logger.info("Ensemble (%s) built from: %s", kind, [name for name, _ in chosen])

    if kind == "voting":
        return VotingClassifier(estimators=estimators, voting="soft")
    if kind == "stacking":
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5)
    raise ValueError(f"unknown ensemble kind: {kind}")


def evaluate_fitted_models(fitted: dict, X_test, y_test) -> pd.DataFrame:
    """fitted: {name: fitted_estimator}. Returns a metrics comparison table."""
    rows = {}
    for name, estimator in fitted.items():
        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
        rows[name] = classification_report_dict(y_test, y_pred, y_proba)
    return pd.DataFrame(rows).T.round(4).sort_values("recall", ascending=False)


def save_models(fitted: dict, out_dir):
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in fitted.items():
        joblib.dump(estimator, out_dir / f"{name}.joblib")
    logger.info("Saved %d models to %s", len(fitted), out_dir)


def load_model(path):
    return joblib.load(path)
