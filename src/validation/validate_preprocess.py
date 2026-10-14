import sys

import great_expectations.expectations as gxe
import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def main(raw_path: str, preprocessed_path: str, report_path: str) -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    target = params["global"]["target"]
    z_threshold = params["preprocess"]["z_threshold"]
    outlier_cols = params["preprocess"]["outlier_cols"]
    cols_to_drop = params["preprocess"]["cols_to_drop"]

    out = pd.read_csv(preprocessed_path)

    # Target column must never be null
    expectations = [
        gxe.ExpectColumnValuesToNotBeNull(column=target),
    ]

    # Non-informative columns were dropped
    for col in cols_to_drop:
        if col in out.columns:
            raise AssertionError(f"Column '{col}' should have been dropped during preprocessing.")

    # Validate categorical/binary column 'explicit'
    if "explicit" in out.columns:
        expectations += [
            gxe.ExpectColumnValuesToBeOfType(column="explicit", type_="int64"),
            gxe.ExpectColumnValuesToBeInSet(column="explicit", value_set=[0, 1]),
        ]
        
    # Outlier validation
    margin = float(z_threshold) * (1 + 1e-9)
    mean = out[outlier_cols].mean()
    std = out[outlier_cols].std()

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
