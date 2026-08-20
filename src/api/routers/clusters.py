from __future__ import annotations

from fastapi import APIRouter, Query

from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.models.tabular_clustering import project, run_kmeans

router = APIRouter(prefix="/api/clusters", tags=["clusters"])

_cache: dict = {}

# UMAP's first call on a fresh process is dominated by numba JIT-compiling its
# internals (~90s measured locally) — every call after that is near-instant
# since numba caches the compiled code for the process lifetime. Rather than
# make whoever clicks "UMAP" first eat that cold start, main.py fires this
# off in a background thread at server boot.
WARM_METHODS = ["umap"]
WARM_DIMENSIONS = [2, 3]


def warm_projection_cache():
    for method in WARM_METHODS:
        for dims in WARM_DIMENSIONS:
            try:
                projection(method=method, dimensions=dims)
            except Exception:
                pass  # best-effort warmup; a real request will surface the actual error


@router.get("/projection")
def projection(
    method: str = Query("pca", pattern="^(pca|tsne|umap)$"),
    dimensions: int = Query(2, ge=2, le=3),
):
    cache_key = f"proj_{method}_{dimensions}d"
    if cache_key in _cache:
        return _cache[cache_key]

    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    X_scaled, _, _ = scale_features(X_train, X_test)

    coords = project(X_scaled.values, method=method, n_components=dimensions)
    _, km_labels = run_kmeans(X_scaled.values, k=2)

    points = []
    for i in range(len(coords)):
        point = {
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "trueDiagnosis": "Malignant" if y_train.values[i] == 1 else "Benign",
            "kmeansCluster": str(km_labels[i]),
        }
        if dimensions == 3:
            point["z"] = round(float(coords[i, 2]), 4)
        points.append(point)

    result = {"points": points}
    _cache[cache_key] = result
    return result
