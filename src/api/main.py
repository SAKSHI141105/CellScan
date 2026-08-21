"""CellScan API — wraps the tabular/image ML pipelines for the React frontend.

    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import clusters, image, mammography, reports, tabular
from src.services import image_service, mammography_service
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _warm_image_models():
    # TensorFlow's import alone takes 15-20s on a cold process — lazily
    # eating that cost on someone's *first* prediction request looks
    # indistinguishable from a hung/broken server from the browser side.
    # Loading both model services here means that cost lands during server
    # boot (visible in the startup logs) instead of during a user's first
    # upload.
    try:
        image_service.load_image_model()
    except Exception:
        logger.exception("Histopathology model warmup failed — first real request will retry and surface the error")
    try:
        mammography_service.load_mammography_model()
    except Exception:
        logger.exception("Mammography model warmup failed — first real request will retry and surface the error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # daemon threads so a slow/failed warmup never blocks shutdown
    threading.Thread(target=clusters.warm_projection_cache, daemon=True).start()
    threading.Thread(target=_warm_image_models, daemon=True).start()
    yield


app = FastAPI(title="CellScan API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Vite's default dev port is 5173; 4173 is its preview-build port.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tabular.router)
app.include_router(image.router)
app.include_router(mammography.router)
app.include_router(reports.router)
app.include_router(clusters.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
