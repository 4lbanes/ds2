from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline import HousePricesCleaner, ID_COLUMN, TARGET


BASE_DIR = Path(__file__).resolve().parents[2]
ACOMP_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "treino.csv"
PUBLIC_TEST_PATH = DATA_DIR / "teste_publico.csv"
MODEL_PATH = ACOMP_DIR / "modelo_acomp2.joblib"
METRICS_PATH = ACOMP_DIR / "metricas_preliminares.csv"
PUBLIC_PREDICTIONS_PATH = ACOMP_DIR / "predicoes_teste_publico.csv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo obrigatoria nao encontrada: {TARGET}")

    y = df[TARGET].copy()
    x = df.drop(columns=[TARGET])
    if ID_COLUMN in x.columns:
        x = x.drop(columns=[ID_COLUMN])
    return x, y


def infer_columns(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = x.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: Iterable[str], categorical_cols: Iterable[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Ausente")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_cols)),
            ("cat", categorical_pipeline, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.clip(np.asarray(y_pred), a_min=0, a_max=None)
    
    return float(np.sqrt(mean_squared_error(np.log1p(y_true_arr), np.log1p(y_pred_arr))))


def evaluate_model(name: str, pipeline: Pipeline, x_train, x_valid, y_train, y_valid) -> dict[str, float | str]:
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    predictions = np.clip(pipeline.predict(x_valid), a_min=0, a_max=None)
    return {
        "modelo": name,
        "rmsle": rmsle(y_valid, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, predictions))),
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "r2": float(r2_score(y_valid, predictions)),
        "tempo_treino_s": float(train_seconds),
    }


def candidate_regressors() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_select_model() -> tuple[pd.DataFrame, Pipeline, list[str]]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bruto de treino nao encontrado em {TRAIN_PATH}. "
            "O pipeline deve ser treinado a partir do CSV sujo fornecido pelo projeto."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    x, y = split_features_target(train_df)
    feature_columns = x.columns.tolist()
    numeric_cols, categorical_cols = infer_columns(x)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    results = []
    fitted_models: dict[str, Pipeline] = {}
    for name, regressor in candidate_regressors().items():
        pipeline = Pipeline(
            steps=[
                ("cleaner", HousePricesCleaner()),
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", build_model(regressor)),
            ]
        )
        results.append(evaluate_model(name, pipeline, x_train, x_valid, y_train, y_valid))
        fitted_models[name] = pipeline

    metrics = pd.DataFrame(results).sort_values("rmsle", ascending=True).reset_index(drop=True)
    best_model_name = str(metrics.loc[0, "modelo"])

    best_pipeline = Pipeline(
        steps=[
            ("cleaner", HousePricesCleaner()),
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", build_model(candidate_regressors()[best_model_name])),
        ]
    )
    best_pipeline.fit(x, y)

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_columns": feature_columns,
            "target": TARGET,
            "id_column": ID_COLUMN,
            "metrics": metrics.to_dict(orient="records"),
            "selected_model": best_model_name,
            "train_path": str(TRAIN_PATH.relative_to(BASE_DIR)),
            "public_test_path": str(PUBLIC_TEST_PATH.relative_to(BASE_DIR)),
            "cleaning": "HousePricesCleaner dentro do sklearn Pipeline",
        },
        MODEL_PATH,
    )

    return metrics, best_pipeline, feature_columns


def predict_public_test(best_pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    if not PUBLIC_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bruto de teste publico nao encontrado em {PUBLIC_TEST_PATH}. "
            "O pipeline deve receber e limpar o CSV sujo antes de gerar predicoes."
        )

    public_test = pd.read_csv(PUBLIC_TEST_PATH)
    ids = public_test[ID_COLUMN].copy() if ID_COLUMN in public_test.columns else pd.Series(range(len(public_test)))
    x_public = public_test.drop(columns=[TARGET], errors="ignore")
    if ID_COLUMN in x_public.columns:
        x_public = x_public.drop(columns=[ID_COLUMN])
    x_public = x_public.reindex(columns=feature_columns)

    predictions = np.clip(best_pipeline.predict(x_public), a_min=0, a_max=None)
    return pd.DataFrame({ID_COLUMN: ids, "SalePrice_predito": predictions})


def main() -> pd.DataFrame:
    metrics, best_pipeline, feature_columns = train_and_select_model()
    metrics.to_csv(METRICS_PATH, index=False)

    public_predictions = predict_public_test(best_pipeline, feature_columns)
    public_predictions.to_csv(PUBLIC_PREDICTIONS_PATH, index=False)

    print("Metricas preliminares:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nModelo selecionado: {metrics.loc[0, 'modelo']}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Predicoes publicas salvas em: {PUBLIC_PREDICTIONS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
