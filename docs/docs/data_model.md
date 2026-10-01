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
This model is part of an MLOps pipeline designed to predict the Spotify popularity score of a track (ranging from 0 to 100) based strictly on its audio characteristics and selected metadata, intentionally excluding artist identity and artist popularity to test the predictive power of track features.

- **Developed by:** 
- **Model type:** Supervised Regression 
- **Language(s):** English
- **License:** MIT
- **Repository / Tracking:** Integrated with DagsHub, DVC, and MLflow for experiment tracking and versioning.

## Uses

### Direct Use
- Predicting the popularity score (0–100) of new or unreleased songs using audio features and track metadata.
- Serving predictions via an interactive UI/API.
- Comparing performance across different model versions stored in the Model Registry.

### Out-of-Scope Use
- Predicting popularity on platforms other than Spotify.
- Evaluating artist relevance or success, as artist metrics are explicitly excluded from the feature set.

## Bias, Risks, and Limitations

- **Exclusion of Artist Metadata:** By omitting artist and historical popularity, the model may underestimate songs whose success is heavily driven by artist fame.
- **Historical Bias:** Popularity scores reflect Spotify's algorithmic preferences and listener habits at the time of data collection.
- **Data Drift:** Changes in musical trends over time can degrade model performance. The system includes data drift monitoring to flag models for further investigation or retraining.

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
