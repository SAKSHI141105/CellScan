import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing.imbalance import apply_smote, class_distribution
from src.data_preprocessing.tabular_preprocessing import (
    _sklearn_to_kaggle_colname,
    basic_quality_report,
    clean_tabular,
    load_raw_tabular,
    scale_features,
    train_test_split_tabular,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_tabular()


def test_column_naming_matches_kaggle_convention():
    assert _sklearn_to_kaggle_colname("mean radius") == "radius_mean"
    assert _sklearn_to_kaggle_colname("worst concave points") == "concave_points_worst"
    assert _sklearn_to_kaggle_colname("radius error") == "radius_se"


def test_load_raw_tabular_shape(raw_df):
    assert raw_df.shape == (569, 31)
    assert "diagnosis" in raw_df.columns
    assert set(raw_df["diagnosis"].unique()) == {0, 1}


def test_quality_report_has_no_missing_on_clean_source(raw_df):
    report = basic_quality_report(raw_df)
    assert report["missing_by_col"] == {}
    assert report["duplicate_rows"] == 0


def test_clean_tabular_drops_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "diagnosis": [0, 0, 1]})
    cleaned = clean_tabular(df)
    assert len(cleaned) == 2


def test_clean_tabular_imputes_missing_with_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0], "diagnosis": [0, 1, 0]})
    cleaned = clean_tabular(df)
    assert cleaned["a"].isna().sum() == 0
    assert cleaned.loc[1, "a"] == pytest.approx(2.0)


def test_train_test_split_is_stratified(raw_df):
    df = clean_tabular(raw_df)
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    assert len(X_train) + len(X_test) == len(df)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_scale_features_zero_mean_unit_variance(raw_df):
    df = clean_tabular(raw_df)
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.mean().abs().max() < 1e-6
    assert abs(X_train_scaled.std(ddof=0).mean() - 1.0) < 1e-6


def test_smote_balances_classes(raw_df):
    df = clean_tabular(raw_df)
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    before = class_distribution(y_train)
    assert before[0] != before[1]  # confirms the training split really is imbalanced first

    X_res, y_res, before_dist, after_dist = apply_smote(X_train, y_train, k_neighbors=5, random_state=42)
    assert len(X_res) == len(y_res)
    assert y_res.value_counts()[0] == y_res.value_counts()[1]
