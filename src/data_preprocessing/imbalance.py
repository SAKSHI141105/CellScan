"""SMOTE handling, plus the baseline-vs-SMOTE comparison the report calls out
as its own section. WDBC isn't wildly imbalanced (~63/37), but the malignant
class is still the minority, and recall on exactly that class is what we
care about most — so it's worth showing whether SMOTE actually buys anything
here rather than just applying it because "that's what you do."
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def class_distribution(y: pd.Series) -> dict:
    return y.value_counts(normalize=True).round(3).to_dict()


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, k_neighbors: int = 5, random_state: int = 42):
    before = class_distribution(y_train)
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after = class_distribution(y_res)
    logger.info("Class balance before SMOTE: %s | after: %s", before, after)
    return X_res, y_res, before, after
