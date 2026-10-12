import sys

import great_expectations.expectations as gxe
import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def check_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Check that no row appears in both train and test (data leakage)."""
    shared_cols = [c for c in train.columns if c in test.columns]
    n_overlap = len(
        train[shared_cols]
        .drop_duplicates()
        .merge(test[shared_cols].drop_duplicates(), how="inner", on=shared_cols)
    )
    return {
        "success": n_overlap == 0,
        "details": {"overlapping_rows": n_overlap},
    }


def main(preprocessed_path: str, split_dir: str, report_path: str) -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    test_split = params["split-data"]["test_split"]

    full = pd.read_csv(preprocessed_path)
    train = pd.read_csv(f"{split_dir}/train.csv")
    test = pd.read_csv(f"{split_dir}/test.csv")

    n = len(full)
    expected_test = round(n * test_split)
    tol = max(1, int(0.01 * n))  # 1% tolerance for rounding/stratification

    def size_check(expected):
        return gxe.ExpectTableRowCountToBeBetween(
            min_value=expected - tol, max_value=expected + tol
        )

    results = {
        "train": run_validation(
            train,
            "train",
            [
                size_check(n - expected_test),
                gxe.ExpectTableColumnsToMatchSet(column_set=list(full.columns), exact_match=True),
            ],
        ),
        "test": run_validation(
            test,
            "test",
            [
                size_check(expected_test),
                gxe.ExpectTableColumnsToMatchSet(column_set=list(full.columns), exact_match=True),
            ],
        ),
    }

    results["no_overlap"] = check_no_overlap(train, test)

    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
