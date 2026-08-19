from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from src.data_preprocessing.image_preprocessing import preprocess_single
from src.services import image_service

router = APIRouter(prefix="/api/image", tags=["image"])


@router.get("/status")
def status():
    model, model_key = image_service.load_image_model()
    return {"available": model is not None, "model_key": model_key}


async def _read_and_preprocess(file: UploadFile) -> np.ndarray:
    import cv2

    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"Could not decode image: {file.filename}")
    return preprocess_single(img)


@router.post("/predict")
async def predict(file: UploadFile):
    preprocessed = await _read_and_preprocess(file)
    result = image_service.predict(preprocessed)
    if result is None:
        raise HTTPException(
            503,
            "No trained image model found under data/models/image/. Run "
            "scripts/train_image.py after downloading the histopathology dataset.",
        )
    result["filename"] = file.filename
    return result


@router.post("/predict-batch")
async def predict_batch(files: list[UploadFile]):
    results = []
    for file in files:
        preprocessed = await _read_and_preprocess(file)
        result = image_service.predict(preprocessed)
        if result is None:
            raise HTTPException(
                503,
                "No trained image model found under data/models/image/. Run "
                "scripts/train_image.py after downloading the histopathology dataset.",
            )
        result["filename"] = file.filename
        results.append(result)

    n_malignant = sum(1 for r in results if r["predicted_class"] == "Malignant")
    return {"results": results, "n_malignant": n_malignant, "n_benign": len(results) - n_malignant}
