"""Shared metric helpers so the tabular and image pipelines report numbers the
same way — otherwise the "compare image vs tabular performance" section in the
report ends up comparing apples to oranges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)


def classification_report_dict(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # recall on the malignant class is the number that matters most here —
        # a missed malignant case (false negative) is far costlier than a false alarm
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["Actual: Benign", "Actual: Malignant"], columns=["Pred: Benign", "Pred: Malignant"])


def roc_points(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {model_name: metrics_dict}"""
    df = pd.DataFrame(results).T
    return df.sort_values("recall", ascending=False)


def clustering_metrics(X, cluster_labels, true_labels=None) -> dict:
    metrics = {}
    valid_mask = cluster_labels != -1  # DBSCAN noise points don't count toward silhouette
    if valid_mask.sum() > 1 and len(set(cluster_labels[valid_mask])) > 1:
        metrics["silhouette"] = float(silhouette_score(X[valid_mask], cluster_labels[valid_mask]))
    else:
        metrics["silhouette"] = float("nan")

    if true_labels is not None:
        metrics["ari"] = float(adjusted_rand_score(true_labels, cluster_labels))
        metrics["nmi"] = float(normalized_mutual_info_score(true_labels, cluster_labels))
    return metrics
