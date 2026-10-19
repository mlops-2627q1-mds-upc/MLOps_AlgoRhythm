import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import TEST_FILE, TRAIN_FILE


def main():
    # Validate arguments
    if len(sys.argv) != 3:
        logger.error("Arguments error. Usage: \tpython split_data.py <input_path> <output_folder_path>\n")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_folder_path = Path(sys.argv[2])
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Load parameters
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    seed = params["global"]["seed"]
    train_split = params["split-data"]["train_split"]
    test_split = params["split-data"]["test_split"]

    # Load dataset
    df = pd.read_csv(input_path)

    # Split data
    train_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)

    # Save datasets
    train_df.to_csv(output_folder_path / TRAIN_FILE, index=False)
    test_df.to_csv(output_folder_path / TEST_FILE, index=False)

    logger.info(
        f"Data split completed -> Train: {len(train_df):,} ({train_split})| "
        f" Test: {len(test_df):,} ({test_split})"
        f"(Total: {len(df):,} samples)"
    )


if __name__ == "__main__":
    main()
