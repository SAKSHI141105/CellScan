"""Unsupervised exploration of the WDBC feature space.

The point of this module isn't to outperform the supervised models — it's to
check whether malignant/benign separate out *without* using the labels at
all, which is the realistic situation for a hospital sitting on unlabelled
historical scans. If KMeans with k=2 roughly recovers the diagnosis split,
that's a strong signal the features themselves carry the discriminative
information (see ARI/NMI in the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.logging_setup import get_logger
from src.utils.metrics import clustering_metrics

logger = get_logger(__name__)


def run_kmeans(X: np.ndarray, k: int = 2, random_state: int = 42):
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def run_hierarchical(X: np.ndarray, k: int = 2):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    return model, labels


def run_dbscan(X: np.ndarray, eps: float = 1.5, min_samples: int = 5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_noise = int((labels == -1).sum())
    logger.info("DBSCAN found %d clusters, %d noise points", len(set(labels)) - (1 if n_noise else 0), n_noise)
    return model, labels


def project_2d(X: np.ndarray, method: str = "pca", random_state: int = 42, perplexity: int = 30) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
    if method == "tsne":
        return TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca").fit_transform(X)
    raise ValueError(f"unknown projection method: {method}")


def run_all_clustering(X_scaled: np.ndarray, y_true: np.ndarray, cfg: dict) -> dict:
    """Runs KMeans/Hierarchical/DBSCAN and scores each against the true labels.
    Returns everything the dashboard's cluster-viz page and the report need.
    """
    results = {}

    _, km_labels = run_kmeans(X_scaled, k=cfg["kmeans_k"])
    results["kmeans"] = {"labels": km_labels, **clustering_metrics(X_scaled, km_labels, y_true)}

    _, hc_labels = run_hierarchical(X_scaled, k=cfg["kmeans_k"])
    results["hierarchical"] = {"labels": hc_labels, **clustering_metrics(X_scaled, hc_labels, y_true)}

    _, db_labels = run_dbscan(X_scaled, eps=cfg["dbscan_eps"], min_samples=cfg["dbscan_min_samples"])
    results["dbscan"] = {"labels": db_labels, **clustering_metrics(X_scaled, db_labels, y_true)}

    pca_2d = project_2d(X_scaled, "pca")
    tsne_2d = project_2d(X_scaled, "tsne", perplexity=cfg["tsne_perplexity"])

    return {"cluster_results": results, "pca_2d": pca_2d, "tsne_2d": tsne_2d}


def summary_table(cluster_results: dict) -> pd.DataFrame:
    rows = {name: {k: v for k, v in res.items() if k != "labels"} for name, res in cluster_results.items()}
    return pd.DataFrame(rows).T.round(3)
