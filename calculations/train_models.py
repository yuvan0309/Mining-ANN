"""Train and compare several machine learning models for FoS prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from .data_ingestion import load_dataset

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency in this environment
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - optional dependency in this environment
    HAS_LIGHTGBM = False


def _build_preprocessor(feature_frame: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Construct the preprocessing pipeline and return supporting metadata."""

    drop_columns = {"fos", "point_label", "source_file"}
    numeric_features: List[str] = [
        str(col) for col, dtype in feature_frame.dtypes.items() if dtype.kind in {"i", "f"}
    ]
    categorical_features: List[str] = [
        str(col) for col in feature_frame.columns if col not in numeric_features and col not in drop_columns
    ]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def _prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features/target and capture columns that will be dropped."""

    drop_cols = ["fos", "source_file", "point_label"]
    existing_drop_cols = [col for col in drop_cols if col in data.columns]
    X = data.drop(columns=existing_drop_cols)
    y = data["fos"].astype(float)
    return X, y


def _collect_models(preprocessor: ColumnTransformer) -> List[Tuple[str, Pipeline]]:
    """Instantiate the estimators we want to compare."""

    models: List[Tuple[str, Pipeline]] = []

    rf = Pipeline([
        ("preprocessor", clone(preprocessor)),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])
    models.append(("random_forest", rf))

    if HAS_XGBOOST:
        xgb = Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    reg_lambda=1.0,
                ),
            ),
        ])
        models.append(("xgboost", xgb))

    ann = Pipeline([
        ("preprocessor", clone(preprocessor)),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=5000,
                random_state=42,
                early_stopping=True,
            ),
        ),
    ])
    models.append(("ann_mlp", ann))

    # SVM model
    svm = Pipeline([
        ("preprocessor", clone(preprocessor)),
        (
            "model",
            SVR(
                kernel="rbf",
                C=100.0,
                gamma="scale",
                epsilon=0.01,
                cache_size=500,
            ),
        ),
    ])
    models.append(("svm", svm))

    # LightGBM model
    if HAS_LIGHTGBM:
        lgbm = Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ])
        models.append(("lightgbm", lgbm))

    return models


def _safe_kfold(n_samples: int) -> Optional[KFold]:
    if n_samples < 3:
        return None
    n_splits = min(5, n_samples)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def train_and_evaluate(base_dir: Path) -> List[Dict[str, object]]:
    data = load_dataset(base_dir)
    X, y = _prepare_features(data)

    if not HAS_XGBOOST:
        raise ImportError(
            "XGBoost is required for the requested comparison. Install it via 'pip install xgboost'."
        )

    preprocessor, _, _ = _build_preprocessor(X)
    models = _collect_models(preprocessor)

    if len(X) < 3:
        raise ValueError(
            "FoS model training requires at least three samples. "
            "Please add more survey points before running the trainer."
        )

    test_size = max(0.2, 1 / len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    reports: List[Dict[str, object]] = []
    metrics_path = base_dir / "models" / "model_performance.json"

    cv = _safe_kfold(len(X))

    for name, model_pipeline in models:
        pipeline = clone(model_pipeline)

        cv_mean = None
        cv_std = None
        if cv is not None:
            cv_scores = cross_val_score(clone(model_pipeline), X, y, cv=cv, scoring="r2")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std(ddof=0))

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        # Fit once more on the entire dataset so we persist the most accurate model snapshot.
        final_pipeline = clone(model_pipeline)
        final_pipeline.fit(X, y)
        model_path = base_dir / "models" / f"{name}.joblib"
        joblib.dump(final_pipeline, model_path)

        report = {
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "saved_model": str(model_path.relative_to(base_dir)),
        }
        reports.append(report)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    return reports


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    reports = train_and_evaluate(base_dir)
    print("Model comparison (lower RMSE/MAE and higher R2 are better):")
    for record in reports:
        line = (
            f"- {record['model']}: RMSE={record['rmse']:.4f}, MAE={record['mae']:.4f}, "
            f"R2={record['r2']:.4f}"
        )
        if record.get("cv_r2_mean") is not None:
            line += f", CV R2 mean={record['cv_r2_mean']:.4f}"
        print(line)
    print("Artifacts saved in the 'models' directory.")


if __name__ == "__main__":
    main()
