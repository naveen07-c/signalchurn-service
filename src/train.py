"""Train, compare and persist churn models.

Run:  python -m src.train

Steps
1. Load + clean the Telco churn CSV (see src/data.py).
2. Build leakage-free sklearn Pipelines (impute -> scale/one-hot -> model).
3. Compare LogisticRegression, RandomForest and HistGradientBoosting with
   stratified 5-fold CV on F1 and ROC-AUC.
4. Pick the best by mean CV ROC-AUC, tune the decision threshold on
   out-of-fold predictions, refit on the training split and report the
   held-out test metrics.
5. Save the winning pipeline (with metadata) to models/churn_pipeline.joblib.
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import PROJECT_ROOT, load_raw, clean_dataframe
from .features import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    TARGET,
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            random_state=RANDOM_STATE,
        ),
    }


def humanize(name: str) -> str:
    """Turn 'categorical__Contract_Two year' into 'Contract = Two year'."""
    label = name.split("__", 1)[-1]
    for column in CATEGORICAL_OPTIONS:
        prefix = f"{column}_"
        if label.startswith(prefix):
            return f"{column} = {label[len(prefix):]}"
    return label


def feature_importances(pipeline: Pipeline) -> list[dict]:
    """Global importance, signed for linear models."""
    pre: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()

    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_)
    else:  # pragma: no cover - none of our candidates hit this
        return []

    order = np.argsort(np.abs(weights))[::-1][:10]
    return [
        {
            "feature": humanize(str(names[i])),
            "importance": round(float(abs(weights[i])), 4),
            "direction": (
                "increases churn" if weights[i] > 0 else "decreases churn"
            )
            if hasattr(model, "coef_")
            else None,
        }
        for i in order
    ]


def evaluate_at_threshold(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Sweep thresholds and return the one maximising F1 on OOF predictions."""
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)
    scores = [f1_score(y_true, proba >= t) for t in candidates]
    best_idx = int(np.argmax(scores))
    return float(candidates[best_idx]), float(scores[best_idx])


def main() -> None:
    raw = load_raw()
    X, y, cleaning_report = clean_dataframe(raw)

    print("=" * 72)
    print("DATA")
    print("=" * 72)
    print(f"rows={cleaning_report['rows_after_dedupe']}  "
          f"features={X.shape[1]}  churn_rate={cleaning_report['churn_rate']:.1%}")
    print(f"blank TotalCharges coerced to NaN: "
          f"{cleaning_report['blank_total_charges']}")
    print(f"missing after coercion: {cleaning_report['missing_values']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 72)
    print(f"MODEL COMPARISON ({CV_FOLDS}-fold stratified CV on training split)")
    print("=" * 72)
    header = f"{'model':<24}{'F1 (mean +- sd)':>20}{'ROC-AUC (mean +- sd)':>22}{'fit s':>9}"
    print(header)
    print("-" * len(header))

    comparison: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        started = time.perf_counter()
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        elapsed = time.perf_counter() - started
        comparison[name] = {
            "cv_f1_mean": round(float(scores["test_f1"].mean()), 4),
            "cv_f1_std": round(float(scores["test_f1"].std()), 4),
            "cv_roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "cv_roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
            "cv_seconds": round(elapsed, 2),
        }
        f1_cell = (
            f"{comparison[name]['cv_f1_mean']:.4f} +- "
            f"{comparison[name]['cv_f1_std']:.4f}"
        )
        auc_cell = (
            f"{comparison[name]['cv_roc_auc_mean']:.4f} +- "
            f"{comparison[name]['cv_roc_auc_std']:.4f}"
        )
        print(f"{name:<24}{f1_cell:>20}{auc_cell:>22}{elapsed:>8.1f}s")
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

    best_name = max(comparison, key=lambda m: comparison[m]["cv_roc_auc_mean"])
    best_pipe = fitted[best_name]

    # Tune the decision threshold on out-of-fold predictions (never the test set).
    oof_proba = cross_val_predict(
        Pipeline([("preprocessor", build_preprocessor()), ("model", candidate_models()[best_name])]),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, oof_f1 = best_f1_threshold(y_train, oof_proba)

    best_pipe.fit(X_train, y_train)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_at_threshold(y_test, test_proba, threshold)
    test_metrics_at_50 = evaluate_at_threshold(y_test, test_proba, 0.50)
    test_metrics["classification_report"] = classification_report(
        y_test, test_proba >= threshold, target_names=["No churn", "Churn"], output_dict=True
    )
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    ks = float(np.max(tpr - fpr))

    print("\n" + "=" * 72)
    print(f"BEST MODEL: {best_name}  (selected on mean CV ROC-AUC)")
    print("=" * 72)
    print(f"decision threshold tuned on OOF predictions: {threshold:.2f} "
          f"(OOF F1 {oof_f1:.4f}; 0.50 default F1 "
          f"{test_metrics_at_50['f1']:.4f} on test)")
    print(f"held-out test  ROC-AUC={test_metrics['roc_auc']:.4f}  "
          f"F1={test_metrics['f1']:.4f}  "
          f"precision={test_metrics['precision']:.4f}  "
          f"recall={test_metrics['recall']:.4f}  "
          f"accuracy={test_metrics['accuracy']:.4f}  KS={ks:.4f}")
    print(f"confusion matrix: {test_metrics['confusion_matrix']}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "threshold": threshold,
        "features": FEATURE_ORDER,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nsaved pipeline -> {MODEL_PATH}")

    metrics = {
        "model_name": best_name,
        "threshold": threshold,
        "selection_metric": "cv_roc_auc_mean",
        "comparison": comparison,
        "test": test_metrics,
        "test_at_default_threshold": test_metrics_at_50,
        "test_ks": round(ks, 4),
        "cleaning": cleaning_report,
        "top_features": feature_importances(best_pipe),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURE_ORDER,
        "trained_at": artifact["trained_at"],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved metrics  -> {METRICS_PATH}")

    print("\nTop global drivers:")
    for row in metrics["top_features"][:5]:
        direction = f" ({row['direction']})" if row["direction"] else ""
        print(f"  - {row['feature']}: {row['importance']:.4f}{direction}")


if __name__ == "__main__":
    main()
