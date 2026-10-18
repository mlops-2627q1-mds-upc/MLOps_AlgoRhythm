# pylint: disable=too-many-locals,too-many-statements

from pathlib import Path

import dagshub
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TEST_FILE, TRAIN_FILE

DATA_PATH = PROCESSED_DATA_DIR
MODEL_PATH = MODELS_DIR / "random_forest.pkl"


def main(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
):

    # Load parameters
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        target = params["global"]["target"]
        seed = params["global"]["seed"]
        n_estimator = params["train"]["n_estimators"]
        test_split = params["split-data"]["test_split"]

    # dagshub / MLflow
    dagshub.init(
        repo_owner="sergi.gonzalez.martos",
        repo_name="MLOps_AlgoRhythm",
        mlflow=True,
    )

    mlflow.set_experiment("spotify-song-popularity")

    # Load data
    logger.info(f"Loading dataset from {data_path}")
    train_df = pd.read_csv(data_path / TRAIN_FILE)
    test_df = pd.read_csv(data_path / TEST_FILE)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    tracker = EmissionsTracker(project_name="spotify_rf_training", save_to_file=False)
    tracker.start()

    # Train and evaluate the final selected model
    with mlflow.start_run(run_name="random-forest-final") as run:

        mlflow.set_tag("model_stage", "final")

        model = RandomForestRegressor(
            n_estimators=n_estimator,
            random_state=seed,
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
        mlflow.log_param("n_estimators", n_estimator)
        mlflow.log_param("random_state", seed)
        mlflow.log_param("test_size", test_split)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # greenAI data
        tracker.stop()
        data = tracker.final_emissions_data
        emissions_kg = getattr(data, "emissions", 0.0)
        energy_kwh = getattr(data, "energy_consumed", 0.0)
        duration_sec = getattr(data, "duration", 0.0)
        country = getattr(data, "country_name", "Unknown")
        region = getattr(data, "region", "Unknown")
        cpu_model = getattr(data, "cpu_model", "Unknown")
        cpu_count = getattr(data, "cpu_count", 0)
        gpu_model = getattr(data, "gpu_model", "None")
        gpu_count = getattr(data, "gpu_count", 0)

        mlflow.log_metric("co2_emissions_kg", emissions_kg if emissions_kg else 0.0)
        mlflow.log_metric("energy_consumed_kwh", energy_kwh if energy_kwh else 0.0)
        mlflow.log_metric("compute_duration_seconds", duration_sec if duration_sec else 0.0)

        mlflow.log_param("cpu_model", str(cpu_model))
        mlflow.log_param("cpu_count", int(cpu_count))
        mlflow.log_param("gpu_model", str(gpu_model))
        mlflow.log_param("gpu_count", int(gpu_count))
        mlflow.log_param("compute_country", str(country))
        mlflow.log_param("compute_region", str(region))

        # Log trained model to MLflow and register it in the Model Registry
        # description
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="spotify_popularity_rf",
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
    logger.info(f"Emissions CO2: {emissions_kg:.2f}")

    print("\nFinal model evaluation")
    print("----------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    print(f"Emissions CO2: {emissions_kg:.2f}")

    logger.success("Final model training completed.")


if __name__ == "__main__":
    import typer

    typer.run(main)
