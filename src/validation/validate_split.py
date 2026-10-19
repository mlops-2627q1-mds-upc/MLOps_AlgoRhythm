# pylint: disable=too-many-locals

import sys

import pandas as pd
import yaml

from src.validation.common import finish


def row_hashes(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


def main(preprocessed_path: str, split_dir: str, report_path: str) -> None:
    # 1. Load parameters from params.yaml
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_split_target = params["split-data"]["train_split"]
    test_split_target = params["split-data"]["test_split"]

    # 2. Load datasets
    full = pd.read_csv(preprocessed_path)
    train = pd.read_csv(f"{split_dir}/train.csv")
    test = pd.read_csv(f"{split_dir}/test.csv")

    n_total = len(full)
    n_train = len(train)
    n_test = len(test)

    # 3. Calculate actual proportions
    actual_train_ratio = n_train / n_total if n_total > 0 else 0
    actual_test_ratio = n_test / n_total if n_total > 0 else 0

    # 4. Define a strict tolerance (e.g., 1% variation allowed for rounding)
    tolerance = 0.01

    # 5. Customized validations for sizes/proportions
    train_proportional_success = abs(actual_train_ratio - train_split_target) <= tolerance
    test_proportional_success = abs(actual_test_ratio - test_split_target) <= tolerance

    overlap = set(row_hashes(train)) & set(row_hashes(test))

    results = {
        "train_proportion": {
            "success": train_proportional_success,
            "details": {
                "expected_ratio": train_split_target,
                "actual_ratio": actual_train_ratio,
                "expected_rows": round(n_total * train_split_target),
                "actual_rows": n_train,
            },
        },
        "test_proportion": {
            "success": test_proportional_success,
            "details": {
                "expected_ratio": test_split_target,
                "actual_ratio": actual_test_ratio,
                "expected_rows": round(n_total * test_split_target),
                "actual_rows": n_test,
            },
        },
        "row_conservation": {
            "success": n_train + n_test == n_total,
            "details": {
                "n_total": n_total,
                "n_train": n_train,
                "n_test": n_test,
                "difference": n_total - (n_train + n_test),
            },
        },
        "train_test_overlap": {
            "success": len(overlap) == 0,
            "details": {"overlapping_rows": len(overlap)},
        },
    }

    # 6. Finalize and generate report
    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
