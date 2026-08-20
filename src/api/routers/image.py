from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from src.data_preprocessing.image_preprocessing import preprocess_single
from src.services import image_service
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/image", tags=["image"])


@router.get("/status")
def status():
    model, model_key = image_service.load_image_model()
    return {"available": model is not None, "model_key": model_key}


async def _read_and_preprocess(file: UploadFile) -> np.ndarray:
    import cv2

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(400, detail={"error": "Empty upload", "details": f"{file.filename} contained no data"})

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            400,
            detail={"error": "Could not decode image", "details": f"{file.filename} isn't a readable PNG/JPG/TIF file"},
        )

    try:
        return preprocess_single(img)
    except Exception as exc:
        # grayscale/CLAHE/denoise chain — a genuinely corrupt or degenerate
        # image (e.g. all-zero pixels) can trip OpenCV here even after
        # decoding succeeded, so this is a separate failure mode from the
        # decode check above
        logger.exception("Preprocessing failed for %s", file.filename)
        raise HTTPException(422, detail={"error": "Image preprocessing failed", "details": str(exc)}) from exc


def _run_inference(preprocessed: np.ndarray, filename: str) -> dict:
    try:
        result = image_service.predict(preprocessed)
    except Exception as exc:
        logger.exception("Model inference failed for %s", filename)
        raise HTTPException(500, detail={"error": "Model inference failed", "details": str(exc)}) from exc

    if result is None:
        raise HTTPException(
            503,
            detail={
                "error": "No trained image model available",
                "details": "Run scripts/train_image.py after downloading the histopathology dataset (see README).",
            },
        )
    result["filename"] = filename
    return result


@router.post("/predict")
async def predict(file: UploadFile):
    preprocessed = await _read_and_preprocess(file)
    return _run_inference(preprocessed, file.filename)


@router.post("/predict-batch")
async def predict_batch(files: list[UploadFile]):
    results = [_run_inference(await _read_and_preprocess(f), f.filename) for f in files]
    n_malignant = sum(1 for r in results if r["predicted_class"] == "Malignant")
    return {"results": results, "n_malignant": n_malignant, "n_benign": len(results) - n_malignant}
