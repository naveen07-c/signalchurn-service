"""Pydantic request/response models for the churn API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.features import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    NUMERIC_BOUNDS,
)


class CustomerFeatures(BaseModel):
    """One customer record, validated against the dataset's real vocabulary."""

    tenure: int = Field(
        ..., ge=NUMERIC_BOUNDS["tenure"][0], le=NUMERIC_BOUNDS["tenure"][1],
        description="Months the customer has been with the company",
        json_schema_extra={"example": 8},
    )
    MonthlyCharges: float = Field(
        ..., ge=NUMERIC_BOUNDS["MonthlyCharges"][0], le=NUMERIC_BOUNDS["MonthlyCharges"][1],
        description="Current monthly bill in USD",
        json_schema_extra={"example": 84.55},
    )
    TotalCharges: float = Field(
        ..., ge=NUMERIC_BOUNDS["TotalCharges"][0], le=NUMERIC_BOUNDS["TotalCharges"][1],
        description="Lifetime charges in USD",
        json_schema_extra={"example": 676.4},
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="1 if the customer is a senior citizen, else 0",
        json_schema_extra={"example": 0},
    )

    gender: str = Field(..., json_schema_extra={"example": "Female"})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"})
    InternetService: str = Field(..., json_schema_extra={"example": "Fiber optic"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "No"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "Yes"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "Yes"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(
        ..., json_schema_extra={"example": "Electronic check"}
    )

    @field_validator(*CATEGORICAL_FEATURES)
    @classmethod
    def validate_option(cls, value: Any, info) -> str:
        field = info.field_name
        value = str(value).strip()
        allowed = CATEGORICAL_OPTIONS[field]
        if value not in allowed:
            raise ValueError(
                f"{field} must be one of: {', '.join(allowed)}"
            )
        return value


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    threshold: float
    risk_level: str
    risk_label: str
    model_name: str
