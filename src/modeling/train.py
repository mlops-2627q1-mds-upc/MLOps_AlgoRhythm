import joblib
import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

DATA_PATH = PROCESSED_DATA_DIR / "dataset_transformed.csv"
MODEL_PATH = MODELS_DIR / "random_forest.pkl"

TARGET_COL = "popularity"
TEST_SIZE = 0.2
RANDOM_STATE = 42

AUDIO_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "explicit",
]

N_ESTIMATORS = 100


def main(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
):
    # dagshub / MLflow
    dagshub.init(
        repo_owner="sergi.gonzalez.martos",
        repo_name="MLOps_AlgoRhythm",
        mlflow=True,
    )

    mlflow.set_experiment("spotify-song-popularity")

    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    df["explicit"] = df["explicit"].astype(int)

    logger.info(f"Dataset shape: {df.shape}")

    # Select features and target
    X = df[AUDIO_FEATURES]
    y = df[TARGET_COL]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,random_state=RANDOM_STATE)

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Train and evaluate the final selected model
    with mlflow.start_run(run_name="random-forest-final") as run:

        mlflow.set_tag("model_stage", "final")

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        logger.info("Training Random Forest...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("features", ", ".join(AUDIO_FEATURES))

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log trained model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        logger.info(f"MLflow run ID: {run.info.run_id}")

    # Save model locally for later predictions
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Results
    logger.success(f"Model saved to {model_path}")
    logger.info(f"MAE:  {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R²:   {r2:.2f}")

    print("\nFinal model evaluation")
    print("----------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")

    logger.success("Final model training completed.")


if __name__ == "__main__":
    import typer

    typer.run(main)
