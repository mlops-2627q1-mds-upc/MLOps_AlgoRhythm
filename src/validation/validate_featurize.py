# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals

import sys

import great_expectations.expectations as gxe
import numpy as np
import pandas as pd
import yaml

from src.validation.common import finish, run_validation


def expected_fill_values(train: pd.DataFrame, target: str, max_unique: int, skew_threshold: float) -> dict:
    """Independently recompute the imputation values featurize.py should learn on train."""
    numeric = train.select_dtypes(include=np.number).columns.tolist()
    categorical = train.select_dtypes(exclude=np.number).columns.tolist()

    # Low-cardinality numeric columns are treated as categorical (target excluded)
    low_card = [c for c in numeric if train[c].nunique() <= max_unique and c != target]
    numeric = [c for c in numeric if c not in low_card]
    categorical = categorical + low_card

    def is_integer_valued(s: pd.Series) -> bool:
        s = s.dropna()
        return len(s) > 0 and np.allclose(s, s.round())

    integer_cols = [c for c in numeric if is_integer_valued(train[c])]

    fill = {}
    for col in numeric + categorical:
        if train[col].isna().all():
            continue  # featurize.py skips these columns (they stay NaN)
        if col in numeric:
            value = (
                train[col].median()
                if abs(train[col].skew()) > skew_threshold
                else train[col].mean()
            )
            if col in integer_cols:
                value = round(value)
        else:
            value = train[col].mode().iloc[0]
        fill[col] = value
    return fill


def expected_one_hot(df: pd.DataFrame, col: str, prefix: str, categories: list) -> pd.DataFrame:
    """One-hot encode `col` with a fixed category list; unseen categories become all zeros."""
    dummies = pd.get_dummies(
        pd.Categorical(df[col], categories=categories), prefix=prefix, dtype=int
    )
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


def mismatched_columns(actual: pd.DataFrame, expected: pd.DataFrame, cols: list) -> list:
    """Columns whose values differ between actual and expected (NaN == NaN)."""
    bad = []
    for c in cols:
        a, e = actual[c], expected[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(e):
            same = np.isclose(a.to_numpy(float), e.to_numpy(float), equal_nan=True)
        else:
            same = ((a == e) | (a.isna() & e.isna())).to_numpy()
        if not same.all():
            bad.append(c)
    return bad


def check(success: bool, **details) -> dict:
    return {"success": bool(success), "details": details}


def validate_split(name, out, expected, n_rows_in, fill, cat_col, dummy_cols, is_train):
    """Build all validations for one split (train or test)."""
    other_cols = [c for c in expected.columns if c not in dummy_cols]
    imputed_cols = [c for c in fill if c != cat_col and c in out.columns]

    expectations = [
        # Exact columns, in the exact order: original columns first, dummies last
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=expected.columns.tolist()),
        # featurize.py never adds or removes rows
        gxe.ExpectTableRowCountToEqual(value=n_rows_in),
        *[gxe.ExpectColumnValuesToNotBeNull(column=c) for c in imputed_cols + dummy_cols],
        *[gxe.ExpectColumnValuesToBeInSet(column=c, value_set=[0, 1]) for c in dummy_cols],
    ]
    results = {name: run_validation(out, f"{name}_processed", expectations)}

    # Value-level checks only make sense if the shapes match
    if len(out) != len(expected) or list(out.columns) != list(expected.columns):
        results[f"{name}_values"] = check(
            False, reason="shape or columns differ from expected; see GX results"
        )
        return results

    # Imputation: every value equals interim value, or the train-learned fill if it was NaN
    bad_imp = mismatched_columns(out, expected, [other_cols])
    results[f"{name}_imputation"] = check(not bad_imp, mismatched_columns=bad_imp)

    # One-hot: values equal the expected encoding, and row sums are consistent
    bad_oh = mismatched_columns(out, expected, dummy_cols)
    row_sums = out[dummy_cols].sum(axis=1)
    rows_ok = (row_sums == 1).all() if is_train else (row_sums <= 1).all()
    results[f"{name}_one_hot"] = check(
        not bad_oh and rows_ok,
        mismatched_columns=bad_oh,
        rows_with_invalid_sum=int((row_sums != 1).sum() if is_train else (row_sums > 1).sum()),
    )

    # Infinite values are not caught by the null checks
    numeric = out.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    results[f"{name}_no_inf"] = check(not np.isinf(numeric).any())
    return results


def main(interim_dir: str, processed_dir: str, report_path: str) -> None:
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    target = params["global"]["target"]
    f = params["featurize"]
    cat_col, prefix = f["categorical_col"], f["one_hot_prefix"]

    train_in = pd.read_csv(f"{interim_dir}/train.csv")
    test_in = pd.read_csv(f"{interim_dir}/test.csv")
    train_out = pd.read_csv(f"{processed_dir}/train.csv")
    test_out = pd.read_csv(f"{processed_dir}/test.csv")

    # Everything is learned on train only, then applied to both splits
    fill = expected_fill_values(train_in, target, f["categorical_max_unique"], f["skew_threshold"])
    train_imp = train_in.fillna(value=fill)
    test_imp = test_in.fillna(value=fill)
    categories = sorted(train_imp[cat_col].dropna().unique())

    exp_train = expected_one_hot(train_imp, cat_col, prefix, categories)
    exp_test = expected_one_hot(test_imp, cat_col, prefix, categories)
    dummy_cols = [c for c in exp_train.columns if c not in train_imp.columns]

    results = {}
    results.update(
        validate_split(
            "train", train_out, exp_train, len(train_in), fill, cat_col, dummy_cols, is_train=True
        )
    )
    results.update(
        validate_split(
            "test", test_out, exp_test, len(test_in), fill, cat_col, dummy_cols, is_train=False
        )
    )
    finish(results, report_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
