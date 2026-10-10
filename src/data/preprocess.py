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

from loguru import logger
from pathlib import Path
import pandas as pd
import yaml


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

def drop_columns_empty(df, missing_drop_threshold:float) -> pd.DataFrame:
    """
    Given a dataframe and a threshold, drops columns that have more missing values than the threshold.
    """
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct.index[missing_pct > missing_drop_threshold].tolist()
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped columns (too many NaN): %s", cols_to_drop or "none")
    return df

def drop_rows_no_target(df, target:str) -> pd.DataFrame:
    """
    Given a dataframe and the target, drops rows that have no value in target variable.
    """
    # Never impute the target: rows without target are dropped
    if target and target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target])
        logger.info("Rows dropped for missing target: %d", before - len(df))
    return df

def drop_columns_not_informative(df: pd.DataFrame, cols:list[str]) -> pd.DataFrame:
    """
    Given a dataframe and a list of columns with information not important for our problem to drop, drops those columns.
    """
    cols_present = [c for c in cols if c in df.columns]
    df = df.drop(columns=cols, errors="ignore")
    logger.info("Dropped columns: %s | shape: %s", cols_present, df.shape)
    return df


def main():
    
    params = yaml.safe_load(open("params.yaml"))
    target = params["global"]["target"]
    missing_drop_threshold = params["clean-data"]["missing_drop_threshold"]
    cols_to_drop = params["clean-data"]["cols_to_drop"]
    
    
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")
    logger.info("Loaded dataset: %s rows x %d columns", f"{df.shape[0]:,}", df.shape[1])

    df = drop_duplicates(df)
    df = drop_columns_empty(df, missing_drop_threshold)
    df = drop_rows_no_target(df, target)
    df = drop_columns_not_informative(df, cols_to_drop)

    df.to_csv(INTERIM_DATA_DIR / "dataset_transformed.csv", index=False)
    logger.info("Preprocessed dataset saved to %s", INTERIM_DATA_DIR / "dataset_transformed.csv")


if __name__ == "__main__":
    main()