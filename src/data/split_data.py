import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from src.config import INTERIM_DATA_DIR, SEED, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT 

def split_data():
    """
    Splits preprocesed data into train, validation and test sets and save them to the 'data/interim' directory.
    """
    df = df = pd.read_csv(INTERIM_DATA_DIR / "dataset_preprocessed.csv")
    
    train_val_df, test_df = train_test_split(df, test_size=TEST_SPLIT, random_state=SEED)
    
    val_ratio_relative = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio_relative,random_state=SEED)
    
    train_df.to_csv(INTERIM_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(INTERIM_DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(INTERIM_DATA_DIR / "test.csv", index=False)
    
    logger.info(
    f"Data split completed -> Train: {len(train_df):,} ({TRAIN_SPLIT})| "
    f"Val: {len(val_df):,} ({VALIDATION_SPLIT}) | Test: {len(test_df):,} ({TEST_SPLIT})"
    f"(Total: {len(df):,} samples)")
    
if __name__ == "__main__":
    split_data()
    
    