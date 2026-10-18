# pylint: disable=too-many-locals

import sys

import great_expectations.expectations as gxe
import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def main(raw_path: str, preprocessed_path: str, report_path: str) -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    target = params["global"]["target"]
    missing_drop_threshold = params["preprocess"]["missing_drop_threshold"]
    z_threshold = params["preprocess"]["z_threshold"]
    outlier_cols = params["preprocess"]["outlier_cols"]
    cols_to_drop = params["preprocess"]["cols_to_drop"]
    missing_drop_threshold = params.get("preprocess", {}).get("missing_drop_threshold", 0.5)

    # Cargar datasets
    out = pd.read_csv(preprocessed_path)
    df_raw = pd.read_csv(raw_path)

    # 1. Validar que no existan filas totalmente duplicadas
    if out.duplicated().sum() > 0:
        raise AssertionError("El dataset preprocesado contiene filas duplicadas.")

    expectations = [
        # El target nunca debe ser nulo
        gxe.ExpectColumnValuesToNotBeNull(column=target),
    ]

    # 2. Validar que las columnas no informativas fueron eliminadas
    for col in cols_to_drop:
        if col in out.columns:
            raise AssertionError(f"La columna '{col}' debería haber sido eliminada durante el preprocesamiento.")

    # 3. Validar columnas remanentes respecto al porcentaje máximo de nulos permitidos
    max_missing_allowed = missing_drop_threshold
    min_not_null_ratio = 1.0 - max_missing_allowed
    for col in out.columns:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=min_not_null_ratio))

    # 4. Validar columna categórica/binaria 'explicit'
    if "explicit" in out.columns:
        expectations += [
            gxe.ExpectColumnValuesToBeOfType(column="explicit", type_="int64"),
            gxe.ExpectColumnValuesToBeInSet(column="explicit", value_set=[0, 1]),
        ]

    # 5. Validación de Outliers basada en el dataset original (raw)
    margin = float(z_threshold) * (1 + 1e-9)
    mean = df_raw[outlier_cols].mean()
    std = df_raw[outlier_cols].std()

    for c in outlier_cols:
        if c in out.columns and std[c] > 0:
            expectations.append(
                gxe.ExpectColumnValuesToBeBetween(
                    column=c,
                    min_value=float(mean[c] - margin * std[c]),
                    max_value=float(mean[c] + margin * std[c]),
                )
            )

    results = {"preprocess": run_validation(out, "preprocess", expectations)}
    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
