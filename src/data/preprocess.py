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

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "../../data/raw/dataset.csv"
OUTPUT_PATH = BASE_DIR / "../../data/interim/dataset_transformed.csv"

# Stage 1 - EDA cleaning
TARGET = "popularity"
MISSING_DROP_THRESHOLD = 0.50   # drop columns with more than 50% missing values
CATEGORICAL_MAX_UNIQUE = 15     # numeric columns with <= this many unique values are categorical
SKEW_THRESHOLD = 1.0            # |skew| above this -> impute with median, otherwise mean

# Stage 2 - Preprocessing
Z_THRESHOLD = 3.0
OUTLIER_COLS = ["duration_ms", "loudness", "speechiness"]
COLS_TO_DROP = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]
CATEGORICAL_COL = "track_genre"
ONE_HOT_PREFIX = "genre"

logger = logging.getLogger("preprocessing_pipeline")


# Stage 1 - EDA cleaning
def classify_variables(df: pd.DataFrame, target: str | None = TARGET, categorical_max_unique: int = CATEGORICAL_MAX_UNIQUE) -> tuple[list[str], list[str], list[str]]:
    
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


def clean_missing_and_duplicates(df: pd.DataFrame, target: str | None = TARGET,  missing_drop_threshold: float = MISSING_DROP_THRESHOLD,
    categorical_max_unique: int = CATEGORICAL_MAX_UNIQUE, skew_threshold: float = SKEW_THRESHOLD) -> pd.DataFrame:

    numeric_cols, categorical_cols, integer_cols = classify_variables(df, target, categorical_max_unique)
    missing_pct = df.isna().mean()

    # Duplicates (before imputing, so they are detected on the raw values)
    n_dup = int(df.duplicated().sum())
    logger.info("Fully duplicated rows: %d (%.2f%%)", n_dup, 100 * n_dup / max(len(df), 1))
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Shape after removing duplicates: %s", df.shape)

    # Drop columns that are mostly empty
    cols_to_drop = missing_pct.index[missing_pct > missing_drop_threshold].tolist()
    df_clean = df.drop(columns=cols_to_drop)
    logger.info("Dropped columns (too many NaN): %s", cols_to_drop or "none")

    # Never impute the target: rows without target are dropped
    if target and target in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=[target])
        logger.info("Rows dropped for missing target: %d", before - len(df_clean))

    # Impute the remaining columns
    numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
    categorical_cols = [c for c in categorical_cols if c in df_clean.columns]

    for col in df_clean.columns:
        if not df_clean[col].isna().any():
            continue
        if col in numeric_cols:
            if abs(df_clean[col].skew()) > skew_threshold:
                fill = df_clean[col].median()
            else:
                fill = df_clean[col].mean()
            if col in integer_cols:
                fill = round(fill)
            strategy = "median/mean"
        else:
            fill = df_clean[col].mode().iloc[0]
            strategy = "mode"
        df_clean[col] = df_clean[col].fillna(fill)
        logger.info("  %-20s imputed with %s: %s", col, strategy, fill)

    logger.info("Remaining missing values: %d", int(df_clean.isna().sum().sum()))
    return df_clean


# Stage 2 - Preprocessing
def remove_outliers(df: pd.DataFrame, cols: list[str] = OUTLIER_COLS, z_threshold: float = Z_THRESHOLD) -> pd.DataFrame:

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


def drop_columns(df: pd.DataFrame, cols: list[str] = COLS_TO_DROP) -> pd.DataFrame:

    cols_present = [c for c in cols if c in df.columns]
    df = df.drop(columns=cols, errors="ignore")
    logger.info("Dropped columns: %s | shape: %s", cols_present, df.shape)
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
    
    df = pd.read_csv(RAW_PATH)
    logger.info("Loaded dataset: %s rows x %d columns", f"{df.shape[0]:,}", df.shape[1])

    # --- Stage 1: EDA cleaning
    df = clean_missing_and_duplicates(df)

    # --- Stage 2: preprocessing
    df = remove_outliers(df)
    df = drop_columns(df)
    df = one_hot_encode(df)
    final_check(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Preprocessed dataset saved to %s", OUTPUT_PATH)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()