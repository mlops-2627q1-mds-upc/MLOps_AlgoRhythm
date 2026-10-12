import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    change_type,
    drop_columns_empty,
    drop_columns_not_informative,
    drop_duplicates,
    drop_rows_no_target,
    remove_outliers,
)


# ---------------------------------------------------------------- change_type
def test_change_type_casts_bool_to_int():
    df = pd.DataFrame({"explicit": [True, False, True]})
    out = change_type(df)
    assert out["explicit"].tolist() == [1, 0, 1]
    assert pd.api.types.is_integer_dtype(out["explicit"])

# ------------------------------------------------------------ drop_duplicates
def test_drop_duplicates_removes_repeated_rows_and_resets_index():
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "y"]})
    out = drop_duplicates(df)
    assert len(out) == 2
    assert out.index.tolist() == [0, 1]


def test_drop_duplicates_keeps_unique_rows_untouched():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = drop_duplicates(df)
    assert out.equals(df)


def test_duplicates_created_by_dropping_columns_are_removed_by_second_pass():
    # Rows differ only in 'track_id'; they become identical once it is dropped
    df = pd.DataFrame({"track_id": ["a", "b"], "x": [1, 1]})
    out = drop_duplicates(df)
    assert out.equals(df)
    out = drop_columns_not_informative(out, ["track_id"])
    out = drop_duplicates(out)
    assert len(out) == 1
    assert out.index.tolist() == [0]

# --------------------------------------------------------- drop_columns_empty
def test_drop_columns_empty_threshold_is_strict():
    df = pd.DataFrame(
        {
            "full": [1, 2, 3, 4],
            "half_empty": [1, 2, None, None],  # 50% missing
            "mostly_empty": [1, None, None, None],  # 75% missing
        }
    )
    out = drop_columns_empty(df, missing_drop_threshold=0.5)
    assert list(out.columns) == ["full", "half_empty"]  # 50% is not > 50%


def test_drop_columns_empty_keeps_everything_when_no_nans():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = drop_columns_empty(df, missing_drop_threshold=0.1)
    assert list(out.columns) == ["a", "b"]

# ------------------------------------------------------- drop_rows_no_target
def test_drop_rows_no_target_drops_nan_rows_and_never_imputes():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [10.0, np.nan, 30.0]})
    out = drop_rows_no_target(df, "target")
    assert out["target"].tolist() == [10.0, 30.0]
    assert out["x"].tolist() == [1, 3]
    assert out.index.tolist() == [0, 1]


def test_no_drop_rows_no_target_missing():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [10.0, 20.0, 30.0]})
    out = drop_rows_no_target(df, "target")
    assert out.equals(df)


def test_no_drop_rows_no_target_missing_other_col_missing():
    df = pd.DataFrame({"x": [1, np.nan, 3], "target": [10.0, 20.0, 30.0]})
    out = drop_rows_no_target(df, "target")
    assert out.equals(df)

# ------------------------------------------- drop_columns_not_informative
def test_drop_columns_not_informative_drops_listed_columns():
    df = pd.DataFrame({"id": [1, 2], "keep": [3, 4], "name": ["a", "b"]})
    out = drop_columns_not_informative(df, ["id", "name"])
    assert list(out.columns) == ["keep"]


def test_drop_columns_not_informative_ignores_absent_columns():
    df = pd.DataFrame({"keep": [1, 2]})
    out = drop_columns_not_informative(df, ["does_not_exist"])
    assert out.equals(df)

# ----------------------------------------------------------- remove_outliers
def test_remove_outliers_drops_extreme_row():
    df = pd.DataFrame({"v": [1.0] * 20 + [100.0]})
    out = remove_outliers(df, ["v"], z_threshold=3)
    assert len(out) == 20
    assert out["v"].max() == 1.0


def test_remove_outliers_keeps_all_rows_when_there_are_none():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = remove_outliers(df, ["v"], z_threshold=3)
    assert len(out) == 5


def test_remove_outliers_row_is_removed_if_outlier_in_any_column():
    df = pd.DataFrame(
        {
            "v": [float(i) for i in range(21)],  # no outliers here
            "w": [1.0] * 20 + [100.0],  # last row is an outlier here
        }
    )
    out = remove_outliers(df, ["v", "w"], z_threshold=3)
    assert len(out) == 20
    assert 20.0 not in out["v"].tolist()


def test_remove_outliers_only_looks_at_given_columns():
    df = pd.DataFrame({"v": [1.0] * 20 + [100.0], "ignored": range(21)})
    out = remove_outliers(df, ["ignored"], z_threshold=3)
    assert len(out) == 21  # the extreme value is in 'v', which was not checked


def test_remove_outliers_resets_index():
    df = pd.DataFrame({"v": [100.0] + [1.0] * 20})
    out = remove_outliers(df, ["v"], z_threshold=3)
    assert out.index.tolist() == list(range(20))


def test_remove_outliers_raises_on_missing_column():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        remove_outliers(df, ["does_not_exist"], z_threshold=3)