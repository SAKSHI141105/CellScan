from __future__ import annotations

from fastapi import APIRouter, Query

from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.models.tabular_clustering import project_2d, run_kmeans

router = APIRouter(prefix="/api/clusters", tags=["clusters"])

_cache: dict = {}


@router.get("/projection")
def projection(method: str = Query("pca", pattern="^(pca|tsne)$")):
    cache_key = f"proj_{method}"
    if cache_key in _cache:
        return _cache[cache_key]

    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    X_scaled, _, _ = scale_features(X_train, X_test)

    coords = project_2d(X_scaled.values, method=method)
    _, km_labels = run_kmeans(X_scaled.values, k=2)

    points = [
        {
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "trueDiagnosis": "Malignant" if y_train.values[i] == 1 else "Benign",
            "kmeansCluster": str(km_labels[i]),
        }
        for i in range(len(coords))
    ]
    result = {"points": points}
    _cache[cache_key] = result
    return result
