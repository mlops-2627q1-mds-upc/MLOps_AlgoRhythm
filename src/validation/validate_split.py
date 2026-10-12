import sys

import great_expectations.expectations as gxe
import pandas as pd
import yaml

from common import finish, run_validation


def main(preprocessed_path: str, split_dir: str, report_path: str) -> None:
    params = yaml.safe_load(open("params.yaml"))
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
                gxe.ExpectTableColumnsToMatchSet(
                    column_set=list(full.columns), exact_match=True
                ),
            ],
        ),
        "test": run_validation(
            test,
            "test",
            [
                size_check(expected_test),
                gxe.ExpectTableColumnsToMatchSet(
                    column_set=list(full.columns), exact_match=True
                ),
            ],
        ),
    }

    # No row can appear in both train and test (data leakage).
    # Inner merge on all shared columns: any match is an overlapping row.
    shared_cols = [c for c in train.columns if c in test.columns]
    n_overlap = len(train[shared_cols].drop_duplicates().merge(
        test[shared_cols].drop_duplicates(), how="inner", on=shared_cols
    ))
    results["no_overlap"] = {
        "success": n_overlap == 0,
        "details": {"overlapping_rows": n_overlap},
    }

    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])