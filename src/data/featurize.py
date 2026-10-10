import pandas as pd
import numpy as np
import yaml
from loguru import logger
from pathlib import Path
import sys
from src.config import TRAIN_FILE, TEST_FILE


def classify_variables(df, target: str, categorical_max_unique: int) -> tuple[list[str], list[str], list[str]]:
    """
    Classifies the dataframe columns in numerical and categorical.
    Must be called on the training set only.
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


def fit_imputer(train_df: pd.DataFrame, target: str, categorical_max_unique: int,
                skew_threshold: float) -> dict:
    """
    Learns the fill value of every column using the training set only
    (median/mean for numeric columns, mode for categorical ones).
    Returns a {column: fill_value} dict.
    """
    numeric_cols, categorical_cols, integer_cols = classify_variables(
        train_df, target, categorical_max_unique
    )

    fill_values = {}
    for col in numeric_cols + categorical_cols:
        if train_df[col].isna().all():
            logger.warning("Column {} is entirely missing in train; skipping", col)
            continue

        if col in numeric_cols:
            if abs(train_df[col].skew()) > skew_threshold:
                fill, strategy = train_df[col].median(), "median"
            else:
                fill, strategy = train_df[col].mean(), "mean"
            if col in integer_cols:
                fill = round(fill)
        else:
            fill, strategy = train_df[col].mode().iloc[0], "mode"

        fill_values[col] = fill
        if train_df[col].isna().any():
            logger.info("{:<20} will be imputed with {}: {}", col, strategy, fill)

    return fill_values


def apply_imputer(df: pd.DataFrame, fill_values: dict, name: str) -> pd.DataFrame:
    """
    Imputes missing values using the fill values learned on train.
    """
    df = df.copy().fillna(value=fill_values)
    logger.info("[{}] Remaining missing values: {}", name, int(df.isna().sum().sum()))
    return df


def one_hot_encode(df: pd.DataFrame, col: str, prefix: str, categories: list, name: str) -> pd.DataFrame:
    """
    One-hot encodes `col` using a fixed list of categories (learned on train),
    so train and test end up with exactly the same columns in the same order.
    """
    if col not in df.columns:
        raise KeyError(f"Categorical column '{col}' not found in the dataset")

    n_unseen = int((~df[col].isin(categories) & df[col].notna()).sum())
    if n_unseen:
        logger.warning("[{}] {} rows with categories not present in train (encoded as all zeros)", name, n_unseen)

    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=categories)
    df = pd.get_dummies(df, columns=[col], prefix=prefix, dtype=int)
    logger.info("[{}] Categories encoded: {} | new shape: {}", name, len(categories), df.shape)
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
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    target = params["global"]["target"]
    categorical_max_unique = params["featurize"]["categorical_max_unique"]
    skew_threshold = params["featurize"]["skew_threshold"]
    categorical_col = params["featurize"]["categorical_col"]
    one_hot_prefix = params["featurize"]["one_hot_prefix"]

    # Load datasets
    train_df = pd.read_csv(input_folder_path / TRAIN_FILE)
    test_df = pd.read_csv(input_folder_path / TEST_FILE)

    # 1. Missing values: fit on train, apply to train and test
    fill_values = fit_imputer(train_df, target, categorical_max_unique, skew_threshold)
    train_df = apply_imputer(train_df, fill_values, "train")
    test_df = apply_imputer(test_df, fill_values, "test")

    # 2. One-hot encoding of categorical_col in train and test (categories taken from train)
    categories = sorted(train_df[categorical_col].dropna().unique())
    train_df = one_hot_encode(train_df, categorical_col, one_hot_prefix, categories, "train")
    test_df = one_hot_encode(test_df, categorical_col, one_hot_prefix, categories, "test")

    # Save
    train_df.to_csv(output_folder_path / TRAIN_FILE, index=False)
    test_df.to_csv(output_folder_path / TEST_FILE, index=False)
    logger.info("Featurized datasets saved to {}", output_folder_path)


if __name__ == "__main__":
    main()