"""Single source of truth for the churn model's input schema.

Both the training pipeline and the FastAPI layer import from here so the
model, the API validation and the HTML form can never drift apart.
"""

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

YES_NO = ["Yes", "No"]
YES_NO_INTERNET = ["Yes", "No", "No internet service"]
YES_NO_PHONE = ["Yes", "No", "No phone service"]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": YES_NO_PHONE,
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": YES_NO_INTERNET,
    "OnlineBackup": YES_NO_INTERNET,
    "DeviceProtection": YES_NO_INTERNET,
    "TechSupport": YES_NO_INTERNET,
    "StreamingTV": YES_NO_INTERNET,
    "StreamingMovies": YES_NO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# (min, max) used by the API for sanity checks and by the form for inputs.
NUMERIC_BOUNDS = {
    "tenure": (0, 100),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 100_000),
    "SeniorCitizen": (0, 1),
}
