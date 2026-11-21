# SignalChurn — tabular churn predictor

An end-to-end demo of the classic tabular ML workflow: clean messy data in Pandas,
compare three scikit-learn models, persist the best pipeline with `joblib`, and serve
it behind a FastAPI endpoint with a hand-built HTML/CSS frontend.

Dataset: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
(7,043 rows, 19 features, 26.5% churn). The CSV is downloaded automatically on first run.

## Results

5-fold stratified cross-validation on the training split (80%):

| Model | F1 (mean ± sd) | ROC-AUC (mean ± sd) |
|---|---|---|
| **LogisticRegression** (selected) | 0.5923 ± 0.0296 | **0.8461** ± 0.0125 |
| RandomForest | 0.5727 ± 0.0231 | 0.8447 ± 0.0090 |
| HistGradientBoosting | 0.5575 ± 0.0343 | 0.8311 ± 0.0061 |

Held-out test set (1,409 rows), best model, decision threshold 0.32 tuned on
out-of-fold predictions to maximise F1:

| ROC-AUC | F1 | Precision | Recall | Accuracy | KS |
|---|---|---|---|---|---|
| 0.8419 | 0.6147 | 0.5323 | 0.7273 | 0.7580 | 0.5188 |

Confusion matrix: TN 796 · FP 239 · FN 102 · TP 272.

## What the pipeline does

**Cleaning** (`src/data.py`)
- Drops duplicate rows and the `customerID` identifier.
- `TotalCharges` arrives as text with 11 blank values (new customers with
  `tenure == 0`); it is coerced with `pd.to_numeric(errors="coerce")` and the
  resulting NaNs are imputed with the training median inside the pipeline.
- Trims whitespace from categorical strings and maps `Churn` Yes/No → 1/0.
- Prints a cleaning report (blanks coerced, missing counts, churn rate).

**Preprocessing** — a single `ColumnTransformer` inside every candidate pipeline,
so imputation/encoding/scaling is fit only on training folds (no leakage):
- Numeric: median imputation → `StandardScaler`.
- Categorical: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`.

**Model selection** (`src/train.py`)
- Compares LogisticRegression, RandomForest and HistGradientBoosting on mean CV
  ROC-AUC (F1 is reported alongside; accuracy alone would be misleading at 26.5%
  positives).
- Tunes the decision threshold on out-of-fold probabilities, refits on the full
  training split, and evaluates once on the held-out test split.
- Saves the complete pipeline + metadata to `models/churn_pipeline.joblib` and a
  human-readable report to `models/metrics.json`.

## Project layout

```
app/
  main.py              FastAPI app: /predict, /model-info, /health, static UI
  schemas.py           Pydantic request/response models with real category vocab
  static/index.html    form UI
  static/styles.css    styling
  static/app.js        fetch /model-info, POST /predict, render result
src/
  features.py          single source of truth for features/options/bounds
  data.py              download + clean the CSV
  train.py             compare models, tune threshold, save artifact
tests/test_api.py      cleaning + API + form/feature consistency tests
data/raw/              dataset (auto-downloaded)
models/                trained artifact + metrics (generated)
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train          # trains, compares, saves models/churn_pipeline.joblib
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> for the UI, or <http://127.0.0.1:8000/docs> for
the interactive OpenAPI docs.

> If your shell has an unrelated `PYTHONPATH` (e.g. ROS on Linux), run commands
> with `env -u PYTHONPATH` to avoid plugin/import conflicts.

## API

`POST /predict` expects one customer record; every categorical value is validated
against the vocabulary seen in training (invalid values return HTTP 422).

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "tenure": 2, "MonthlyCharges": 95.5, "TotalCharges": 191.0, "SeniorCitizen": 0,
    "gender": "Female", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "Yes", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
  }'
```

```json
{
  "churn_probability": 0.825,
  "churn_prediction": true,
  "threshold": 0.32,
  "risk_level": "high",
  "risk_label": "High risk",
  "model_name": "LogisticRegression"
}
```

Other endpoints:
- `GET /health` — service/model status (503 from `/predict` if the model is missing).
- `GET /model-info` — model name, metrics, CV comparison, top global drivers.
- `GET /docs` — Swagger UI.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers blank-`TotalCharges` cleaning, live prediction, 422 validation for unknown
categories and out-of-range numbers, and checks that the HTML form contains every
feature and every allowed category option used by the model.

## Notes and limitations

- This is a demo on a public, somewhat dated dataset; `models/metrics.json` is the
  source of truth for the shipped numbers rather than a claim of production accuracy.
- The 0.32 threshold favours recall (catching churners) at the cost of precision.
  For a different business trade-off, change `threshold` in the saved bundle or
  tune it in `src/train.py`.
- No authentication, rate limiting or persistence — add those before exposing the
  service publicly. `/predict` is synchronous and scores one row per request.
- Feature importance shown in the UI is global (coefficients / impurity importance),
  not a per-customer explanation.
