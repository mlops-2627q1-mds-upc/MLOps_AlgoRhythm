---
language:
- en
license: mit
tags:
- tabular
- regression
- spotify
- MLOps
- dagshub
- mlflow
metrics:
- mae
- rmse
- r2
datasets:
- spotify-tracks-dataset
pipeline_tag: tabular-regression
---

# Model Card for Spotify Song Popularity Predictor

## Model Details

### Model Description
This model is part of an MLOps pipeline designed to predict the Spotify popularity score of a track (ranging from 0 to 100) based strictly on its audio characteristics and selected metadata, excluding artist identity and artist popularity to test the predictive power of track features.

- **Developed by:** Ana Martinez, Enikö Beke, Marta Sunyer, Ruddy Cuellar, and Sergi González.
- **Model type:** Supervised Regression 
- **Language(s):** English
- **License:** MIT
- **Repository / Tracking:** Integrated with DagsHub, DVC, and MLflow for dataset versioning and experiment tracking.

## Uses

### Direct Use
- Predicting the popularity score (0–100) of new or unreleased songs using audio features and track metadata.
- Serving predictions via an interactive UI/API.
- Comparing performance across different model versions stored in the Model Registry.

### Out-of-Scope Use
- Predicting popularity on platforms other than Spotify.
- Evaluating artist relevance or success as artist metrics are explicitly excluded from the feature set.

## Bias, Risks, and Limitations

- **Exclusion of Artist Metadata:** By omitting artist and historical popularity, the model may underestimate songs whose success is heavily based by artist fame.
- **Historical Bias:** Popularity scores reflect Spotify's algorithmic preferences and habits of listeners ONLY at the time of data collection.
- **Data Drift:** Changes in musical trends over time can affect model performance. The system includes data drift monitoring to flag models for further investigation or retraining.

## How to Get Started with the Model

```python
import joblib
import pandas as pd

# Load trained model artifact
model = joblib.load("model.pkl")

# Example audio features and track metadata
sample_track = pd.DataFrame([{
    'danceability': 0.75,
    'energy': 0.80,
    'loudness': -5.5,
    'speechiness': 0.05,
    'acousticness': 0.12,
    'instrumentalness': 0.00,
    'liveness': 0.10,
    'valence': 0.65,
    'tempo': 120.0,
    'duration_ms': 210000,
    'explicit': 0,
    'track_genre': 'pop'
}])

# Predict popularity score
predicted_score = model.predict(sample_track)
print(f"Predicted Spotify Popularity Score: {predicted_score[0]:.2f}")
````

## Training Details

### Training Data
- **Dataset:** Spotify Tracks Dataset.
- **Target Variable:** `popularity` (Integer score from 0 to 100 representing play count and recency).
- **Input Dimension:** 128 total features:
  * **14 Base Features:** `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `time_signature`
  * **114 Categorical Dummy Features:** One-hot encoded columns derived from `track_genre`
* **Omitted Features:** `artists`, `album_name`, and `track_name`

#### Training Procedure
- **Preprocessing:** Handling missing values and outliers, one-hot encoding for the categorical genre variable, and scaling numeric audio features.
- **Pipeline:** Automated with DVC (`dvc.yaml`) for data cleaning, feature engineering, and training execution.
- **Experiment Tracking:** Hyperparameters, parameters, and metrics are logged automatically to MLflow / DagsHub.

---

## Evaluation

#### Evaluation Metrics
* **MAE (Mean Absolute Error):** Measures the average absolute prediction error in popularity points.
* **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.
* **$R^2$ Score:** Quantifies the proportion of target variance explained by the model features.

### Model Results & Efficiency

| Model | Features | MAE | RMSE | R² | Training Time | CO₂ Emissions (kg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (RF Baseline - 128 feat.)** | 128 | 11.03 | 15.89 | 0.441 | 188.81 s | ~0.000209 |
| **Best Absolute MAE (`rf-n200` - 128 feat.)** | 128 | **10.99** | 15.83 | 0.445 | 341.30 s | ~0.000379 |
| **Best Variance (`rf-maxfeat05` - 14 feat.)** | 14 | 11.23 | **15.35** | **0.478** | **45.22 s** | **~0.000041** |

> **Notes:** 
> - **`rf-n200` (128 features):** Lowest absolute error ($MAE = 10.99$), but requires around 8x more training time and generates a larger carbon footprint.
> - **`rf-maxfeat05` (14 features):** The most efficient model with the best explanatory power ($R^2 = 0.478$, $RMSE = 15.35$), low training time (~45 s) and low $CO_2$ emissions.

---

### Operational Metrics

- **Inference Latency:** TBD
- **Training Duration (Selected Model):** 341.3 seconds
- **Training CO₂ Emissions:** ~3.79 × 10⁻4 kg (tracked via CodeCarbon)
- **Feature Set:** 14 numeric features and 114 genre-encoded features.
---

## Monitoring and Maintenance
- **Data Drift Detection:** Simulated using split dataset evaluation. Features exceeding defined drift thresholds trigger an automated flag for investigation or retraining.
