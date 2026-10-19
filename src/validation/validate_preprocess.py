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

    # Load datasets
    out = pd.read_csv(preprocessed_path)
    df_raw = pd.read_csv(raw_path)

    # 1. Validate that there are no fully duplicate rows
    if out.duplicated().sum() > 0:
        raise AssertionError("El dataset preprocesado contiene filas duplicadas.")

    expectations = [
        # The target should never be null.
        gxe.ExpectColumnValuesToNotBeNull(column=target),
    ]

    # 2. Verify that the non-informative columns were removed
    for col in cols_to_drop:
        if col in out.columns:
            raise AssertionError(f"La columna '{col}' debería haber sido eliminada durante el preprocesamiento.")

    # 3. Validate remaining columns against the maximum percentage of null values allowed
    max_missing_allowed = missing_drop_threshold
    min_not_null_ratio = 1.0 - max_missing_allowed
    for col in out.columns:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=min_not_null_ratio))

    # 4. Validate categorical/binary column 'explicit'
    if "explicit" in out.columns:
        expectations += [
            gxe.ExpectColumnValuesToBeOfType(column="explicit", type_="int64"),
            gxe.ExpectColumnValuesToBeInSet(column="explicit", value_set=[0, 1]),
        ]

    # 5. Outlier validation based on the original (raw) dataset
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
