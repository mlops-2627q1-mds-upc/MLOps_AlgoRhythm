from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR




def missing_value_imputation(df):
    return df


def outlier_removal(df):
    """
    Remove outliers from the dataframe using the IQR method and the z-score method.
    """

    def detect_outliers(df):
        return df

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return df

    # an individual is an outlier if it gets flagged by both the IQR method and the z-score method

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the dataframe.
    """
    COLS_TO_DROP = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]
    df.drop(columns=COLS_TO_DROP, errors="ignore")
    return df

def preprocess_data(df):
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    # TODO put preprocessing steps here

    df.to_csv(PROCESSED_DATA_DIR / "dataset_preprocessed.csv", index=False)

if __name__ == "__main__":
    preprocess_data()