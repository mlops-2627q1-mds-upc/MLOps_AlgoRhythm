import pandas as pd
import numpy as np
import yaml
from loguru import logger
from pathlib import Path
import sys
from src.config import TRAIN_FILE, VAL_FILE, TEST_FILE



def classify_variables(df, target:str, categorical_max_unique:int) -> tuple[list[str], list[str], list[str]]:
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


def clean_missing(df, target:str, categorical_max_unique:int, skew_threshold:float) -> pd.DataFrame:
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
        "Rows before: %s | removed: %s (%.2f%%) | after: %s",
        f"{rows_before:,}", f"{int(rows_to_remove.sum()):,}",
        100 * rows_to_remove.mean(), f"{len(df):,}",
    )
    return df

def one_hot_encode(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    
    if col not in df.columns:
        raise KeyError(f"Categorical column '{col}' not found in the dataset")
    n_categories = df[col].nunique()
    df = pd.get_dummies(df, columns=[col], prefix=prefix, dtype=int)
    logger.info("Categories encoded: %d | new shape: %s", n_categories, df.shape)
    return df


def main():
    # Validate arguments
    if len(sys.argv) != 3:
        logger.error("Arguments error. Usage: \tpython featurize.py <input_folder_path> <output_folder_path>\n")
        sys.exit(1)
            
    input_folder_path = Path(sys.argv[1])
    output_folder_path = Path(sys.argv[2])
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Load parameters
    params = yaml.safe_load(open("params.yaml"))
    target = params["global"]["target"]
    categorical_max_unique = params["featurize"]["categorical_max_unique"]
    skew_threshold = params["featurize"]["skew_threshold"]
    outlier_cols = params["featurize"]["outlier_cols"]
    z_threshold = params["featurize"]["z_threshold"]
    categorical_col = params["featurize"]["categorical_col"]
    one_hot_prefix = params["featurize"]["one_hot_prefix"]
    
    # Load dataset
    train_df = pd.read_csv(input_folder_path / TRAIN_FILE)
    val_df = pd.read_csv(input_folder_path / VAL_FILE)
    test_df = pd.read_csv(input_folder_path / TEST_FILE)
    
    df = clean_missing(df, target, categorical_max_unique, skew_threshold)
    df = remove_outliers(df, outlier_cols, z_threshold)
    df = one_hot_encode(df, categorical_col, one_hot_prefix)
    
    
if __name__ == "__main__":
    main() 