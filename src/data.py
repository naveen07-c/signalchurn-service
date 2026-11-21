"""Dataset download and cleaning.

The IBM Telco Customer Churn dataset is the canonical public example of a
"messy" tabular problem: TotalCharges arrives as blank strings for brand-new
customers, several columns are free-text categoricals, and the target is a
Yes/No label that has to become 0/1.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, ID_COLUMN, NUMERIC_FEATURES, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def ensure_dataset(path: Path = RAW_DATA_PATH) -> Path:
    """Return the local CSV path, downloading it once if necessary."""
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset -> {path}")
    try:
        urllib.request.urlretrieve(DATASET_URL, path)
    except OSError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(
            f"Could not download the dataset from {DATASET_URL}.\n"
            f"Download it manually and save it to {path}."
        ) from exc
    return path


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(ensure_dataset(path))


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Clean the raw frame and split it into features and target.

    Returns (X, y, report) where the report captures what was fixed, so the
    training log shows the data cleaning rather than hiding it.
    """
    report: dict = {
        "rows_in": int(len(df)),
        "columns_in": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    df = df.drop_duplicates().copy()
    report["rows_after_dedupe"] = int(len(df))

    # TotalCharges is a string column with blanks for tenure == 0 customers.
    if df["TotalCharges"].dtype == object or pd.api.types.is_string_dtype(
        df["TotalCharges"]
    ):
        stripped = df["TotalCharges"].astype(str).str.strip()
        report["blank_total_charges"] = int((stripped == "").sum())
        df["TotalCharges"] = pd.to_numeric(stripped, errors="coerce")
    else:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    report["missing_values"] = {
        col: int(count)
        for col, count in df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        .isna()
        .sum()
        .items()
        if count
    }

    # Normalise the categorical strings (defensive: trailing spaces exist).
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    df[TARGET] = df[TARGET].astype(str).str.strip()
    report["target_counts"] = df[TARGET].value_counts().to_dict()
    report["churn_rate"] = round(float((df[TARGET] == "Yes").mean()), 4)

    y = df[TARGET].map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad = sorted(df.loc[y.isna(), TARGET].unique().tolist())
        raise ValueError(f"Unexpected values in target column {TARGET}: {bad}")
    y = y.astype(int)

    X = df.drop(columns=[ID_COLUMN, TARGET])
    X = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    return X, y, report
