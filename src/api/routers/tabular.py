from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from src.api.schemas import TabularPredictRequest
from src.explainability.tabular_explain import plain_language_summary
from src.services import tabular_service

router = APIRouter(prefix="/api/tabular", tags=["tabular"])


@router.get("/defaults")
def get_defaults():
    return {
        "feature_groups": tabular_service.FEATURE_GROUPS,
        "values": tabular_service.default_feature_values(),
    }


@router.post("/predict")
def predict(payload: TabularPredictRequest):
    missing = [f for f in tabular_service.ALL_FEATURES if f not in payload.features]
    if missing:
        raise HTTPException(422, f"Missing features: {', '.join(missing)}")

    result = tabular_service.predict_single(payload.features)
    result["explanation"] = plain_language_summary(result["top_contributors"], result["predicted_class"])
    return result


@router.post("/predict-batch")
async def predict_batch(file: UploadFile):
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    missing = [c for c in tabular_service.ALL_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(422, f"CSV is missing {len(missing)} required column(s): {', '.join(missing[:8])}")

    predictions = tabular_service.predict_batch(df)
    rows = df.to_dict(orient="records")
    for row, pred in zip(rows, predictions):
        row.update(pred)

    n_malignant = sum(1 for p in predictions if p["predicted_class"] == "Malignant")
    return {
        "rows": rows,
        "n_rows": len(rows),
        "n_malignant": n_malignant,
        "n_benign": len(rows) - n_malignant,
    }


@router.post("/explain-row")
def explain_row(payload: TabularPredictRequest):
    """Same as /predict but named separately in the frontend for clarity when
    drilling into one row from a batch result — behaviorally identical.
    """
    return predict(payload)
