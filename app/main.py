"""FastAPI service serving the churn model and the demo frontend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.data import PROJECT_ROOT
from src.features import FEATURE_ORDER

from .schemas import CustomerFeatures, PredictionResponse

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Telco Churn Predictor",
    description=(
        "Serves a scikit-learn churn pipeline trained on the IBM Telco "
        "Customer Churn dataset. See `/docs` for the interactive API."
    ),
    version="1.0.0",
)

_bundle: dict[str, Any] | None = None


def get_bundle() -> dict[str, Any]:
    """Load the saved pipeline once, with a helpful error if it is missing."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model artifact not found. Train it first with "
                    "`python -m src.train`."
                ),
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def risk_band(probability: float, threshold: float) -> tuple[str, str]:
    if probability < min(0.33, threshold):
        return "low", "Low risk"
    if probability < max(0.66, threshold):
        return "medium", "Moderate risk"
    return "high", "High risk"


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    loaded = MODEL_PATH.exists()
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }


@app.get("/model-info", tags=["model"])
def model_info() -> dict[str, Any]:
    bundle = get_bundle()
    metrics = load_metrics()
    test = metrics.get("test", {})
    return {
        "model_name": bundle.get("model_name"),
        "threshold": bundle.get("threshold"),
        "trained_at": bundle.get("trained_at"),
        "features": bundle.get("features", FEATURE_ORDER),
        "metrics": {
            "roc_auc": test.get("roc_auc"),
            "f1": test.get("f1"),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "accuracy": test.get("accuracy"),
        },
        "comparison": metrics.get("comparison", {}),
        "top_features": metrics.get("top_features", []),
        "train_rows": metrics.get("train_rows"),
        "test_rows": metrics.get("test_rows"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    bundle = get_bundle()
    pipeline = bundle["pipeline"]

    row = customer.model_dump()
    frame = pd.DataFrame([[row[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)

    try:
        probability = float(pipeline.predict_proba(frame)[0, 1])
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Scoring failed: {exc}"
        ) from exc

    threshold = float(bundle.get("threshold", 0.5))
    level, label = risk_band(probability, threshold)

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=probability >= threshold,
        threshold=round(threshold, 4),
        risk_level=level,
        risk_label=label,
        model_name=bundle.get("model_name", "unknown"),
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
