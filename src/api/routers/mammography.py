from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

from src.data_preprocessing.mammography_preprocessing import preprocess_upload
from src.services import mammography_service
from src.utils.config import load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/mammography", tags=["mammography"])


@router.get("/status")
def status():
    model, is_demo = mammography_service.load_mammography_model()
    return {"available": model is not None, "is_demo": is_demo}


async def _read_and_preprocess(file: UploadFile):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(400, detail={"error": "Empty upload", "details": f"{file.filename} contained no data"})

    cfg = load_config()["mammography"]
    try:
        return preprocess_upload(raw_bytes, file.filename, cfg)
    except Exception as exc:
        logger.exception("Preprocessing failed for %s", file.filename)
        raise HTTPException(
            422,
            detail={"error": "Mammogram preprocessing failed", "details": str(exc)},
        ) from exc


def _run_inference(preprocessed, filename: str) -> dict:
    try:
        result = mammography_service.predict(preprocessed)
    except Exception as exc:
        logger.exception("Model inference failed for %s", filename)
        raise HTTPException(500, detail={"error": "Inference Engine Failure", "details": str(exc)}) from exc

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
