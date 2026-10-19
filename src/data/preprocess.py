import sys

import pandas as pd
import yaml
from loguru import logger


def change_type(df) -> pd.DataFrame:
    df["explicit"] = df["explicit"].astype(int)
    return df


def drop_duplicates(df) -> pd.DataFrame:
    """
    Given a dataframe, drops duplicated rows.
    """
    # Duplicates (before imputing, so they are detected on the raw values)
    n_dup = int(df.duplicated().sum())
    logger.info("Fully duplicated rows: {}", n_dup, 100 * n_dup / max(len(df), 1))
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Shape after removing duplicates: {}", df.shape)

    return df


def drop_columns_empty(df, missing_drop_threshold: float) -> pd.DataFrame:
    """
    Given a dataframe and a threshold, drops columns that have more missing values than the threshold.
    """
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct.index[missing_pct > missing_drop_threshold].tolist()
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped columns (too many NaN): {}", cols_to_drop or "none")
    return df


def drop_rows_no_target(df, target: str) -> pd.DataFrame:
    """
    Given a dataframe and the target, drops rows that have no value in target variable.
    """
    # Never impute the target: rows without target are dropped
    if target and target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        logger.info("Rows dropped for missing target: {}", before - len(df))
    return df


def drop_columns_not_informative(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Given a dataframe and a list of columns with information not important for our problem to drop, drops those columns.
    """
    cols_present = [c for c in cols if c in df.columns]
    df = df.drop(columns=cols, errors="ignore")
    logger.info("Dropped columns: {} | shape: {}", cols_present, df.shape)
    return df


def remove_outliers(df, cols: list[str], z_threshold: float) -> pd.DataFrame:
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
        "Rows before: {} | removed: {} ({:.2f}%) | after: {}",
        f"{rows_before:,}",
        f"{int(rows_to_remove.sum()):,}",
        100 * rows_to_remove.mean(),
        f"{len(df):,}",
    )
    return df


def main():
    # Validate arguments
    if len(sys.argv) != 4:
        logger.error("Arguments error. Usage: \tpython preprocess.py <input_path> <output_path>\n")
        sys.exit(1)

    input_path = sys.argv[1]
    inter_path = sys.argv[2]
    output_path = sys.argv[3]

    # Load parameters
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    target = params["global"]["target"]
    outlier_cols = params["preprocess"]["outlier_cols"]
    z_threshold = params["preprocess"]["z_threshold"]
    missing_drop_threshold = params["preprocess"]["missing_drop_threshold"]
    cols_to_drop = params["preprocess"]["cols_to_drop"]

    # Load dataset
    df = pd.read_csv(input_path)
    logger.info("Loaded dataset from {}: {} rows x {} columns", input_path, f"{df.shape[0]:,}", df.shape[1])

    # Apply transformations
    df = change_type(df)
    df = drop_duplicates(df)
    df = drop_columns_empty(df, missing_drop_threshold)
    df = drop_rows_no_target(df, target)
    df = drop_columns_not_informative(df, cols_to_drop)
    df = drop_duplicates(df)

    # Save dataset for outlier validation
    df.to_csv(inter_path, index=False)
    logger.info("Intermediate dataset saved to {}", inter_path)

    df = remove_outliers(df, outlier_cols, z_threshold)

    # Save dataset
    df.to_csv(output_path, index=False)
    logger.info("Preprocessed dataset saved to {}", output_path)


if __name__ == "__main__":
    main()
