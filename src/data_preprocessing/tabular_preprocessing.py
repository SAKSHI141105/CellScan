"""Loading and cleaning for the Wisconsin Diagnostic Breast Cancer dataset.

We support two sources on purpose: if someone drops the Kaggle CSV
(data/raw/wdbc.csv) in, we use that; otherwise we fall back to the identical
dataset bundled in scikit-learn so the pipeline runs with zero setup. Same
569 rows, same 30 features either way — just saves a "go download this file
first" step for anyone cloning the repo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

KAGGLE_CSV_NAME = "wdbc.csv"

# sklearn's target is 0=malignant, 1=benign — we standardize on the more
# intuitive 1=malignant everywhere downstream (recall on malignant is what
# actually matters clinically).
_SKLEARN_LABEL_MAP = {0: 1, 1: 0}


def _from_kaggle_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    return df


def _sklearn_to_kaggle_colname(name: str) -> str:
    """sklearn names these 'mean radius' / 'worst concave points' / 'radius error';
    everything downstream (feature grouping, the dashboard form, _mean-suffix
    filtering in the EDA script) assumes the Kaggle CSV convention instead:
    'radius_mean' / 'concave_points_worst' / 'radius_se'.
    """
    if name.startswith("mean "):
        return name[len("mean "):].replace(" ", "_") + "_mean"
    if name.startswith("worst "):
        return name[len("worst "):].replace(" ", "_") + "_worst"
    if name.endswith(" error"):
        return name[: -len(" error")].replace(" ", "_") + "_se"
    return name.replace(" ", "_")


def _from_sklearn() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df.columns = [_sklearn_to_kaggle_colname(c) if c != "target" else c for c in df.columns]
    df = df.rename(columns={"target": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].map(_SKLEARN_LABEL_MAP)
    return df


def load_raw_tabular() -> pd.DataFrame:
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    csv_path = raw_dir / KAGGLE_CSV_NAME

    if csv_path.exists():
        logger.info("Loading tabular data from %s", csv_path)
        return _from_kaggle_csv(csv_path)

    logger.info("No local CSV found at %s, falling back to sklearn's bundled WDBC", csv_path)
    return _from_sklearn()


def basic_quality_report(df: pd.DataFrame) -> dict:
    """Cheap sanity checks we log/print before doing anything else with the data."""
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_by_col": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_balance": df["diagnosis"].value_counts(normalize=True).round(3).to_dict(),
    }
    return report


def clean_tabular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = df.columns.drop("diagnosis")
    if df[numeric_cols].isna().any().any():
        imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logger.info("Imputed missing values with column medians")

    return df


def train_test_split_tabular(df: pd.DataFrame, feature_cols: list[str] | None = None):
    cfg = load_config()["tabular"]
    target_col = cfg["target_col"]
    features = feature_cols or [c for c in df.columns if c != target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[target_col],
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df[target_col],
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    raw = load_raw_tabular()
    print(basic_quality_report(raw))
