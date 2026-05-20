from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ACOMP_DIR = Path(__file__).resolve().parent
MODEL_PATH = ACOMP_DIR / "modelo_acomp2.joblib"
TARGET = "SalePrice"
ID_COLUMN = "Id"

NONE_CATEGORY_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ZERO_NUMERIC_COLS = [
    "MasVnrArea",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]


class HousePricesCleaner(BaseEstimator, TransformerMixin):
    """Replica a limpeza documentada no Acompanhamento 1 dentro do pipeline final."""

    def fit(self, x: pd.DataFrame, y=None):
        data = self._as_dataframe(x)
        self.feature_names_in_ = data.columns.to_numpy()
        self.none_category_cols_ = [col for col in NONE_CATEGORY_COLS if col in data.columns]
        self.zero_numeric_cols_ = [col for col in ZERO_NUMERIC_COLS if col in data.columns]

        if {"Neighborhood", "LotFrontage"}.issubset(data.columns):
            self.lot_by_neighborhood_ = data.groupby("Neighborhood")["LotFrontage"].median()
            self.lot_global_median_ = data["LotFrontage"].median()
        else:
            self.lot_by_neighborhood_ = pd.Series(dtype=float)
            self.lot_global_median_ = np.nan

        categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

        self.categorical_modes_ = {
            col: data[col].mode(dropna=True).iloc[0]
            for col in categorical_cols
            if data[col].notna().any()
        }
        self.numeric_medians_ = {
            col: data[col].median()
            for col in numeric_cols
            if col != TARGET and data[col].notna().any()
        }
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        cleaned = self._as_dataframe(x).copy()

        for col in self.none_category_cols_:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna("Ausente")

        for col in self.zero_numeric_cols_:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna(0)

        if "LotFrontage" in cleaned.columns:
            if "Neighborhood" in cleaned.columns:
                by_neighborhood = cleaned["Neighborhood"].map(self.lot_by_neighborhood_)
                cleaned["LotFrontage"] = cleaned["LotFrontage"].fillna(by_neighborhood)
            cleaned["LotFrontage"] = cleaned["LotFrontage"].fillna(self.lot_global_median_)

        for col, mode in self.categorical_modes_.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna(mode)

        for col, median in self.numeric_medians_.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna(median)

        return cleaned

    @staticmethod
    def _as_dataframe(x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x
        return pd.DataFrame(x)


def _load_artifact(model_path: str | Path = MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {model_path}. "
            "Execute acompanhamentos/acomp2/treino_modelos.py antes de chamar predict()."
        )
    if str(ACOMP_DIR) not in sys.path:
        sys.path.insert(0, str(ACOMP_DIR))
    return joblib.load(model_path)


def _prepare_features(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    id_column = artifact.get("id_column", ID_COLUMN)
    target = artifact.get("target", TARGET)
    feature_columns = artifact["feature_columns"]

    x = df.drop(columns=[target], errors="ignore")
    if id_column in x.columns:
        x = x.drop(columns=[id_column])
    return x.reindex(columns=feature_columns)


def predict(input_csv: str | Path) -> np.ndarray:
    artifact = _load_artifact()
    data = pd.read_csv(input_csv)
    x = _prepare_features(data, artifact)
    predictions = artifact["pipeline"].predict(x)
    return np.clip(np.asarray(predictions, dtype=float), a_min=0, a_max=None)


def prever_precos(input_csv: str | Path) -> np.ndarray:
    return predict(input_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executa predicoes de precos de imoveis.")
    parser.add_argument("input_csv", help="Caminho para o CSV de entrada sem SalePrice.")
    args = parser.parse_args()

    preds = predict(args.input_csv)
    print(preds)
