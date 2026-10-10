from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import joblib
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.modeling.train import AUDIO_FEATURES


MODEL_PATH = MODELS_DIR / "random_forest.pkl"
OUTPUT_PATH = REPORTS_DIR / "predictions.csv"

# We need to specify the path of our file of INPUT
def predict(input_path: Path, model_path: Path = MODEL_PATH, output_path: Path = OUTPUT_PATH,):
    """Load final trained model and predict Spotify track popularity"""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    #Load input data
    logger.info(f"Loading input data from {input_path}")
    df = pd.read_csv(input_path)

    # Check that all features are present
    missing_features = [feature for feature in AUDIO_FEATURES if feature not in df.columns]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # features in the same order training
    X = df[AUDIO_FEATURES].copy()
    X["explicit"] = X["explicit"].astype(int)

    # Predictions
    predictions = model.predict(X)

    # Add predictions to the original dataset
    df["predicted_popularity"] = predictions

    # Save the results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Predictions saved to {output_path}")

    return df


if __name__ == "__main__":
    typer.run(predict)