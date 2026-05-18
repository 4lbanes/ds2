from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ACOMP_DIR = Path(__file__).resolve().parent
MODEL_PATH = ACOMP_DIR / "modelo_acomp2.joblib"


def _load_artifact(model_path: str | Path = MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {model_path}. "
            "Execute acompanhamentos/acomp2/treino_modelos.py antes de chamar predict()."
        )
    return joblib.load(model_path)


def _prepare_features(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    id_column = artifact.get("id_column", "Id")
    target = artifact.get("target", "SalePrice")
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
