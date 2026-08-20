from __future__ import annotations

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from sklearn.metrics import auc, confusion_matrix

from src.api.schemas import ReportRequest
from src.data_preprocessing.tabular_preprocessing import clean_tabular, load_raw_tabular, scale_features, train_test_split_tabular
from src.services import report_service
from src.services.tabular_service import TABULAR_MODELS_DIR
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.metrics import roc_points

router = APIRouter(prefix="/api", tags=["reports"])

CFG = load_config()
FIGURES_DIR = PROJECT_ROOT / CFG["paths"]["figures_dir"]

_MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "svm", "mlp", "ensemble_voting", "ensemble_stacking"]


def _read_csv_or_404(path, name: str):
    if not path.exists():
        raise HTTPException(404, f"{name} not found — run scripts/train_tabular.py first.")
    return pd.read_csv(path, index_col=0)


def _load_test_set_and_models() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Shared by roc_curves/confusion_matrices — both need the same held-out
    split plus whichever trained models are on disk, just scored differently.
    """
    features_path = TABULAR_MODELS_DIR / "selected_features.joblib"
    if not features_path.exists():
        raise HTTPException(404, "No trained models found — run scripts/train_tabular.py first.")

    df = clean_tabular(load_raw_tabular())
    X_train, X_test, y_train, y_test = train_test_split_tabular(df)
    _, X_test_scaled, _ = scale_features(X_train, X_test)

    selected = joblib.load(features_path)
    X_test_sel = X_test_scaled[selected]

    models = {}
    for name in _MODEL_ORDER:
        model_path = TABULAR_MODELS_DIR / f"{name}.joblib"
        if model_path.exists():
            models[name] = joblib.load(model_path)

    return X_test_sel, y_test, models


@router.get("/reports/tabular-comparison")
def tabular_comparison():
    df = _read_csv_or_404(FIGURES_DIR / "tabular_model_comparison.csv", "Model comparison")
    return {"models": [{"name": idx, **row} for idx, row in df.round(4).iterrows()]}


@router.get("/reports/smote-comparison")
def smote_comparison():
    df = _read_csv_or_404(FIGURES_DIR / "smote_comparison.csv", "SMOTE comparison")
    return {"rows": [{"name": idx, **row} for idx, row in df.round(4).iterrows()]}


@router.get("/reports/clustering-summary")
def clustering_summary():
    df = _read_csv_or_404(FIGURES_DIR / "clustering_summary.csv", "Clustering summary")
    return {"methods": [{"name": idx, **row} for idx, row in df.round(4).iterrows()]}


@router.get("/reports/roc-curves")
def roc_curves():
    X_test_sel, y_test, models = _load_test_set_and_models()

    curves = []
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_sel)[:, 1]
        fpr, tpr = roc_points(y_test, y_proba)
        # downsample to keep the payload small — the curve is smooth enough that
        # every point isn't needed for a legible chart
        step = max(1, len(fpr) // 60)
        curves.append({
            "name": name,
            "auc": round(float(auc(fpr, tpr)), 4),
            "points": [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])],
        })

    return {"curves": curves}


@router.get("/reports/confusion-matrices")
def confusion_matrices():
    X_test_sel, y_test, models = _load_test_set_and_models()

    matrices = []
    for name, model in models.items():
        y_pred = model.predict(X_test_sel)
        # ravel order for a binary confusion_matrix is [[tn, fp], [fn, tp]]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        matrices.append({
            "name": name,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    return {"matrices": matrices}


@router.post("/report/csv")
def report_csv(payload: ReportRequest):
    csv_bytes = report_service.build_csv_report(payload.model_dump(), payload.source)
    return Response(content=csv_bytes, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=cellscan_report.csv"})


@router.post("/report/pdf")
def report_pdf(payload: ReportRequest):
    pdf_bytes = report_service.build_pdf_report(
        payload.model_dump(), payload.source, payload.top_contributors, payload.gradcam_png_base64
    )
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=cellscan_report.pdf"})
