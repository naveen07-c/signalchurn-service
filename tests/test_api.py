"""Focused tests for the cleaning layer and the prediction API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.data import clean_dataframe
from src.features import CATEGORICAL_OPTIONS, FEATURE_ORDER

STATIC_INDEX = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"

VALID_CUSTOMER = {
    "tenure": 8,
    "MonthlyCharges": 84.55,
    "TotalCharges": 676.4,
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

client = TestClient(app)


def test_clean_handles_blank_total_charges():
    raw = pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "tenure": [0, 12, 24],
            "MonthlyCharges": [29.85, 56.0, 70.0],
            "TotalCharges": ["", "672.0", " 1,680.0 "],
            "SeniorCitizen": [0, 0, 1],
            **{col: [CATEGORICAL_OPTIONS[col][0]] * 3 for col in CATEGORICAL_OPTIONS},
            "Churn": ["No", "Yes", "No"],
        }
    )
    X, y, report = clean_dataframe(raw)

    assert list(X.columns) == FEATURE_ORDER
    assert list(y) == [0, 1, 0]
    assert report["blank_total_charges"] == 1
    # Blank becomes NaN, "1,680" becomes NaN (not numeric) -> both reported missing.
    assert report["missing_values"]["TotalCharges"] == 2


def test_health_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_has_metrics():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert 0.5 < body["metrics"]["roc_auc"] <= 1.0
    assert body["model_name"]
    assert body["top_features"]


def test_predict_returns_probability():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_name"]


def test_predict_rejects_unknown_category():
    payload = {**VALID_CUSTOMER, "Contract": "Three year"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Contract" in response.text


def test_predict_rejects_out_of_range_numeric():
    response = client.post("/predict", json={**VALID_CUSTOMER, "tenure": 500})
    assert response.status_code == 422


def test_form_covers_every_model_feature():
    html = STATIC_INDEX.read_text()
    for feature in FEATURE_ORDER:
        assert f'name="{feature}"' in html, f"{feature} missing from the HTML form"


def test_form_covers_every_category_option():
    html = STATIC_INDEX.read_text()
    for field, options in CATEGORICAL_OPTIONS.items():
        select_start = html.index(f'name="{field}"')
        select_end = html.index("</select>", select_start)
        section = html[select_start:select_end]
        for option in options:
            assert f">{option}<" in section, f"{field} option {option!r} missing"


@pytest.mark.parametrize("path", ["/", "/docs"])
def test_pages_render(path):
    assert client.get(path).status_code == 200
