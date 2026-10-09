#!/usr/bin/env python3
"""
Stage 1 - EDA cleaning
    1. Remove fully duplicated rows.
    2. Drop columns with more than MISSING_DROP_THRESHOLD missing values.
    3. Drop rows with a missing target (the target is never imputed).
    4. Impute remaining missing values:
         - numeric: median if |skew| > 1, mean otherwise (rounded if integer-valued)
         - categorical: mode
    5. (Optional) save the intermediate dataset (dataset_eda.csv).

Stage 2 - Preprocessing
    6. Remove rows that are Z-score outliers in at least one of OUTLIER_COLS.
    7. Drop unnecessary columns.
    8. One-hot encode the categorical column.
    9. Final sanity checks and save.

Usage:
    python preprocessing_pipeline.py
    python preprocessing_pipeline.py --raw-path data/raw/dataset.csv \
        --output-path data/processed/dataset_preprocessed.csv
"""

from __future__ import annotations
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


TARGET = "popularity"
MISSING_DROP_THRESHOLD = 0.50   # drop columns with more than 50% missing values
CATEGORICAL_MAX_UNIQUE = 15     # numeric columns with <= this many unique values are categorical
SKEW_THRESHOLD = 1.0            # |skew| above this -> impute with median, otherwise mean
Z_THRESHOLD = 3.0
OUTLIER_COLS = ["duration_ms", "loudness", "speechiness"]
COLS_TO_DROP = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]
CATEGORICAL_COL = "track_genre"
ONE_HOT_PREFIX = "genre"

logger = logging.getLogger("preprocessing_pipeline")

def classify_variables(df, target:str=TARGET, categorical_max_unique:int=CATEGORICAL_MAX_UNIQUE) -> tuple[list[str], list[str], list[str]]:
    """
    Classifies the dataframe columns in numerical and categorical.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    low_card_numeric = [
        c for c in numeric_cols
        if df[c].nunique() <= categorical_max_unique and c != target
    ]
    numeric_cols = [c for c in numeric_cols if c not in low_card_numeric]
    categorical_cols = categorical_cols + low_card_numeric

    def is_integer_valued(s: pd.Series) -> bool:
        s = s.dropna()
        return len(s) > 0 and np.allclose(s, s.round())

    integer_cols = [c for c in numeric_cols if is_integer_valued(df[c])]
    return numeric_cols, categorical_cols, integer_cols

def drop_duplicates(df) -> pd.DataFrame:
    """
    Given a dataframe, drops duplicated rows.
    """
    # Duplicates (before imputing, so they are detected on the raw values)
    n_dup = int(df.duplicated().sum())
    logger.info("Fully duplicated rows: %d (%.2f%%)", n_dup, 100 * n_dup / max(len(df), 1))
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Shape after removing duplicates: %s", df.shape)
    
    return df

def drop_columns_not_informative(df: pd.DataFrame, cols:list[str] = COLS_TO_DROP) -> pd.DataFrame:
    """
    Given a dataframe and a list of columns with information not important for our problem to drop, drops those columns.
    """
    cols_present = [c for c in cols if c in df.columns]
    df = df.drop(columns=cols, errors="ignore")
    logger.info("Dropped columns: %s | shape: %s", cols_present, df.shape)
    return df

def drop_columns_empty(df, missing_drop_threshold:float=MISSING_DROP_THRESHOLD) -> pd.DataFrame:
    """
    Given a dataframe and a threshold, drops columns that have more missing values than the threshold.
    """
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct.index[missing_pct > missing_drop_threshold].tolist()
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped columns (too many NaN): %s", cols_to_drop or "none")
    return df

def drop_rows_no_target(df, target:str=TARGET) -> pd.DataFrame:
    """
    Given a dataframe and the target, drops rows that have no value in target variable.
    """
    # Never impute the target: rows without target are dropped
    if target and target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target])
        logger.info("Rows dropped for missing target: %d", before - len(df))
    return df
    

def clean_missing(df, target:str=TARGET, categorical_max_unique:int = CATEGORICAL_MAX_UNIQUE, skew_threshold:float = SKEW_THRESHOLD) -> pd.DataFrame:
    """
    Given a dataframe, imputes missing data
    """
    numeric_cols, categorical_cols, integer_cols = classify_variables(df, target, categorical_max_unique)

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    for col in df.columns:
        if not df[col].isna().any(): # if there is no missing data
            continue
        if col in numeric_cols:
            if abs(df[col].skew()) > skew_threshold:
                fill = df[col].median()
            else:
                fill = df[col].mean()
            if col in integer_cols:
                fill = round(fill)
            strategy = "median/mean"
        else:
            fill = df[col].mode().iloc[0]
            strategy = "mode"
        df[col] = df[col].fillna(fill)
        logger.info("  %-20s imputed with %s: %s", col, strategy, fill)

    logger.info("Remaining missing values: %d", int(df.isna().sum().sum()))
    return df

def remove_outliers(df, cols: list[str] = OUTLIER_COLS, z_threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    """
    Given a dataframe removes rows that are outliers
    """
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Outlier columns not found in the dataset: {missing_cols}")

    z_scores = (df[cols] - df[cols].mean()) / df[cols].std()
    is_outlier = z_scores.abs() > z_threshold

    logger.info("Outliers per variable:\n%s", is_outlier.sum().to_string())

    rows_to_remove = is_outlier.any(axis=1)
    rows_before = len(df)
    df = df.loc[~rows_to_remove].reset_index(drop=True)
    logger.info(
        "Rows before: %s | removed: %s (%.2f%%) | after: %s",
        f"{rows_before:,}", f"{int(rows_to_remove.sum()):,}",
        100 * rows_to_remove.mean(), f"{len(df):,}",
    )
    return df

def one_hot_encode(df: pd.DataFrame, col: str = CATEGORICAL_COL, prefix: str = ONE_HOT_PREFIX) -> pd.DataFrame:
    
    if col not in df.columns:
        raise KeyError(f"Categorical column '{col}' not found in the dataset")
    n_categories = df[col].nunique()
    df = pd.get_dummies(df, columns=[col], prefix=prefix, dtype=int)
    logger.info("Categories encoded: %d | new shape: %s", n_categories, df.shape)
    return df

def final_check(df: pd.DataFrame) -> None:
    
    non_numeric = df.select_dtypes(exclude=np.number).columns.tolist()
    logger.info("Final shape: %s", df.shape)
    logger.info("Missing values: %d", int(df.isna().sum().sum()))
    logger.info("Non-numeric columns left: %s", non_numeric or "none")


# Pipeline
def run_pipeline() -> pd.DataFrame:
    
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")
    logger.info("Loaded dataset: %s rows x %d columns", f"{df.shape[0]:,}", df.shape[1])

    df = drop_duplicates(df)
    
    df = drop_columns_empty(df)
    df = drop_rows_no_target(df)
    df = clean_missing(df)
    df = remove_outliers(df)
    
    df = drop_columns_not_informative(df)
    df = one_hot_encode(df)
    final_check(df)

    df.to_csv(INTERIM_DATA_DIR / "dataset_transformed.csv", index=False)
    logger.info("Preprocessed dataset saved to %s", INTERIM_DATA_DIR / "dataset_transformed.csv")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()