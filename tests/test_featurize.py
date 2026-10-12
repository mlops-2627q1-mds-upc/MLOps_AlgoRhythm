import numpy as np
import pandas as pd
import pytest

from src.data.featurize import (
    apply_imputer,
    classify_variables,
    fit_imputer,
    one_hot_encode,
)


# ---------------------------------------------------------- classify_variables
def test_classify_variables_splits_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "duration": [100, 200, 300, 400, 500],  # 5 unique -> numeric
            "flag": [0, 1, 0, 1, 0],  # 2 unique -> low cardinality -> categorical
            "genre": ["a", "b", "a", "b", "a"],  # strings -> categorical
            "target": [1, 1, 2, 2, 1],  # 2 unique, but the target is never reclassified
        }
    )
    numeric, categorical, _ = classify_variables(df, "target", categorical_max_unique=2)
    assert sorted(numeric) == ["duration", "target"]
    assert sorted(categorical) == ["flag", "genre"]


def test_classify_variables_integer_cols_only_include_integer_valued_numerics():
    df = pd.DataFrame(
        {
            "whole": [1.0, 2.0, 3.0, 4.0, np.nan],  # integer-valued (NaN is ignored)
            "fractional": [1.5, 2.5, 3.5, 4.5, 5.5],
            "target": [1, 2, 3, 4, 5],
        }
    )
    _, _, integer_cols = classify_variables(df, "target", categorical_max_unique=2)
    assert "whole" in integer_cols
    assert "fractional" not in integer_cols

# ---------------------------------------------------------------- fit_imputer
def test_fit_imputer_uses_mean_for_symmetric_columns():
    train = pd.DataFrame(
        {"a": [10.5, 20.5, 30.5, np.nan], "target": [1.0, 2.0, 3.0, 4.0]}
    )
    fill = fit_imputer(train, "target", categorical_max_unique=2, skew_threshold=1.0)
    assert fill["a"] == pytest.approx(20.5)  # mean, not integer-valued -> not rounded


def test_fit_imputer_uses_median_for_skewed_columns():
    train = pd.DataFrame(
        {"a": [1.5, 1.5, 1.5, 1.5, 100.5, np.nan], "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    fill = fit_imputer(train, "target", categorical_max_unique=1, skew_threshold=1.0)
    assert fill["a"] == pytest.approx(1.5)  # median; the mean would be ~21


def test_fit_imputer_rounds_integer_valued_columns():
    train = pd.DataFrame({"a": [1, 3, 3, 5, 5], "target": [1.0, 2.0, 3.0, 4.0, 5.0]})
    fill = fit_imputer(train, "target", categorical_max_unique=2, skew_threshold=1.0)
    assert fill["a"] == 3  # mean is 3.4, rounded because the column is integer-valued


def test_fit_imputer_uses_mode_for_categorical_columns():
    train = pd.DataFrame(
        {
            "genre": ["x", "x", "y", None],
            "a": [1.5, 2.5, 3.5, 4.5],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fill = fit_imputer(train, "target", categorical_max_unique=2, skew_threshold=1.0)
    assert fill["genre"] == "x"


def test_fit_imputer_skips_columns_entirely_missing_in_train():
    train = pd.DataFrame(
        {"empty": [np.nan] * 4, "a": [1.5, 2.5, 3.5, 4.5], "target": [1.0, 2.0, 3.0, 4.0]}
    )
    fill = fit_imputer(train, "target", categorical_max_unique=2, skew_threshold=1.0)
    assert "empty" not in fill
    assert "a" in fill

# --------------------------------------------------------------- apply_imputer
def test_apply_imputer_uses_train_values_not_test_statistics():
    test = pd.DataFrame({"a": [100.0, np.nan], "genre": [None, "y"]})
    out = apply_imputer(test, {"a": 2.0, "genre": "x"}, "test")
    assert out["a"].tolist() == [100.0, 2.0]  # 2.0 comes from train, not from test's 100.0
    assert out["genre"].tolist() == ["x", "y"]


def test_apply_imputer_does_not_modify_the_input():
    test = pd.DataFrame({"a": [1.0, np.nan]})
    apply_imputer(test, {"a": 5.0}, "test")
    assert test["a"].isna().sum() == 1


def test_apply_imputer_leaves_nans_in_columns_without_fill_value():
    test = pd.DataFrame({"empty_in_train": [np.nan, np.nan], "a": [np.nan, 1.0]})
    out = apply_imputer(test, {"a": 9.0}, "test")
    assert out["empty_in_train"].isna().all()
    assert out["a"].tolist() == [9.0, 1.0]

# -------------------------------------------------------------- one_hot_encode
def test_one_hot_encode_creates_prefixed_int_columns_at_the_end():
    df = pd.DataFrame({"genre": ["a", "b", "a"], "v": [1, 2, 3]})
    out = one_hot_encode(df, "genre", "g", ["a", "b"], "train")
    assert list(out.columns) == ["v", "g_a", "g_b"]
    assert out["g_a"].tolist() == [1, 0, 1]
    assert out["g_b"].tolist() == [0, 1, 0]
    assert pd.api.types.is_integer_dtype(out["g_a"])


def test_one_hot_encode_follows_the_order_of_the_given_categories():
    df = pd.DataFrame({"genre": ["a", "b"]})
    out = one_hot_encode(df, "genre", "g", ["b", "a"], "train")
    assert list(out.columns) == ["g_b", "g_a"]


def test_one_hot_encode_unseen_category_becomes_all_zeros():
    test = pd.DataFrame({"genre": ["b", "zzz"], "v": [4, 5]})  # "zzz" is not in train
    out = one_hot_encode(test, "genre", "g", ["a", "b"], "test")
    assert list(out.columns) == ["v", "g_a", "g_b"]  # same columns as train
    assert out.loc[1, ["g_a", "g_b"]].tolist() == [0, 0]


def test_one_hot_encode_creates_missing_categories_as_zero_columns():
    # A category present in train but absent in this split must still produce its column
    test = pd.DataFrame({"genre": ["a", "a"]})
    out = one_hot_encode(test, "genre", "g", ["a", "b"], "test")
    assert out["g_b"].tolist() == [0, 0]


def test_one_hot_encode_each_known_row_sums_to_one():
    df = pd.DataFrame({"genre": ["a", "b", "b", "a"]})
    out = one_hot_encode(df, "genre", "g", ["a", "b"], "train")
    assert (out[["g_a", "g_b"]].sum(axis=1) == 1).all()


def test_one_hot_encode_raises_if_column_is_missing():
    with pytest.raises(KeyError):
        one_hot_encode(pd.DataFrame({"v": [1]}), "genre", "g", ["a"], "train")