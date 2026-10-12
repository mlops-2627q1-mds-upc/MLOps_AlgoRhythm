import sys

import great_expectations.expectations as gxe
import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def expected_state(raw: pd.DataFrame, params: dict) -> dict:
    """Independently recompute what preprocess.py should have produced."""
    target = params["global"]["target"]
    p = params["preprocess"]

    # 1. Duplicates are removed on the raw rows, before any column is dropped
    df = raw.drop_duplicates().reset_index(drop=True)

    # 2. Columns with more missing values than the threshold
    missing_pct = df.isna().mean()
    sparse_cols = missing_pct.index[missing_pct > p["missing_drop_threshold"]].tolist()
    df = df.drop(columns=sparse_cols)

    # 3. Rows without target (the target is never imputed)
    if target in df.columns:
        df = df.dropna(subset=[target])

    # 4. Non-informative columns
    df = df.drop(columns=p["cols_to_drop"], errors="ignore")

    # 5. Duplicates created by dropping the columns above
    df = df.drop_duplicates().reset_index(drop=True)

    # 6. Z-score outliers: mean/std are computed at this point of the pipeline
    cols = p["outlier_cols"]
    mean, std = df[cols].mean(), df[cols].std()
    z = (df[cols] - mean) / std
    keep = ~(z.abs() > p["z_threshold"]).any(axis=1)

    return {
        "columns": df.columns.tolist(),
        "n_rows": int(keep.sum()),
        "mean": mean,
        "std": std,
    }


def main(raw_path: str, preprocessed_path: str, report_path: str) -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    target = params["global"]["target"]
    z_threshold = params["preprocess"]["z_threshold"]
    outlier_cols = params["preprocess"]["outlier_cols"]

    raw = pd.read_csv(raw_path)
    out = pd.read_csv(preprocessed_path)
    exp = expected_state(raw, params)

    expectations = [
        # Columns: sparse + cols_to_drop removed, nothing else lost, no stray index column
        gxe.ExpectTableColumnsToMatchSet(column_set=exp["columns"], exact_match=True),
        # Rows: duplicates, missing-target rows and outliers all removed (and only those)
        gxe.ExpectTableRowCountToEqual(value=exp["n_rows"]),
        # Target never imputed and never null
        gxe.ExpectColumnValuesToNotBeNull(column=target),
    ]

    # change_type(): 'explicit' must be an integer 0/1
    if "explicit" in exp["columns"]:
        expectations += [
            gxe.ExpectColumnValuesToBeOfType(column="explicit", type_="int64"),
            gxe.ExpectColumnValuesToBeInSet(column="explicit", value_set=[0, 1]),
        ]

    # remove_outliers(): every remaining value lies within mean +/- z_threshold * std
    # (using the statistics at the moment preprocess.py computed them)
    margin = z_threshold * (1 + 1e-9)  # float tolerance at the boundary
    for c in outlier_cols:
        expectations.append(
            gxe.ExpectColumnValuesToBeBetween(
                column=c,
                min_value=float(exp["mean"][c] - margin * exp["std"][c]),
                max_value=float(exp["mean"][c] + margin * exp["std"][c]),
            )
        )

    results = {"preprocess": run_validation(out, "preprocess", expectations)}
    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
