"""Trim the 30 WDBC features down before throwing them at slower models (SVM/MLP grid search
gets noticeably slower with redundant columns, and the *_worst / *_mean pairs are highly
correlated by construction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CORR_DROP_THRESHOLD = 0.95


def drop_correlated_features(X: pd.DataFrame, threshold: float = CORR_DROP_THRESHOLD) -> list[str]:
    """Greedy drop: for each pair above threshold, drop the second column encountered.
    Good enough here — we're not optimizing this, just cutting obvious redundancy
    (e.g. radius_mean vs perimeter_mean are ~0.99 correlated, no point keeping both).
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info("Dropping %d correlated features (threshold=%.2f): %s", len(to_drop), threshold, to_drop)
    return [c for c in X.columns if c not in to_drop]


def rank_by_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances


def select_features(X: pd.DataFrame, y: pd.Series, top_k: int | None = None) -> list[str]:
    kept_after_corr = drop_correlated_features(X)
    importances = rank_by_importance(X[kept_after_corr], y)
    if top_k:
        return importances.head(top_k).index.tolist()
    return kept_after_corr
