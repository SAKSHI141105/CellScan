from __future__ import annotations

from pydantic import BaseModel


class TabularPredictRequest(BaseModel):
    features: dict[str, float]


class ReportRequest(BaseModel):
    predicted_class: str
    probability_malignant: float
    source: str
    top_contributors: list[dict] | None = None
    gradcam_png_base64: str | None = None
