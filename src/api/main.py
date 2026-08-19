"""CellScan API — wraps the tabular/image ML pipelines for the React frontend.

    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import clusters, image, reports, tabular
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

app = FastAPI(title="CellScan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Vite's default dev port is 5173; 4173 is its preview-build port.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tabular.router)
app.include_router(image.router)
app.include_router(reports.router)
app.include_router(clusters.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
