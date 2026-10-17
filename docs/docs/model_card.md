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
- **Data Drift:** Changes in musical trends over time can affect model performance. The system includes data drift monitoring to flag models for further investigation or retraining (to be implemented).
- Since outliers were removed predictions for songs with extreme values are less reliable.

## How to Get Started with the Model


### Requirements
Use the same library versions the model was trained with (see `requirements.txt`).

### Option 1: Load the model from the repository
The model file is tracked with DVC:
```bash
dvc pull   # downloads models/random_forest.pkl
```

```python
import joblib
import pandas as pd

model = joblib.load("models/random_forest.pkl")

# The model expects exactly 128 features, in this order:
features = list(model.feature_names_in_)

# Build one song: start from all zeros (every genre off) and fill in the values
song = pd.DataFrame([dict.fromkeys(features, 0.0)])
song.loc[0, ["duration_ms", "explicit", "danceability", "energy", "key",
             "loudness", "mode", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence", "tempo",
             "time_signature"]] = [210000, 0, 0.7, 0.8, 5, -5.5, 1,
                                   0.05, 0.1, 0.0, 0.12, 0.6, 120.0, 4]
song.loc[0, "genre_pop"] = 1  # set exactly one genre column to 1

popularity = model.predict(song)[0]
print(f"Predicted popularity: {popularity:.1f} (scale 0-100)")
```

### Option 2: Load the registered model from MLflow (DagsHub)
```python
import dagshub, mlflow.sklearn

dagshub.init(repo_owner="sergi.gonzalez.martos", repo_name="MLOps_AlgoRhythm", mlflow=True)
model = mlflow.sklearn.load_model("models:/spotify_popularity_rf/latest")
```

### Input format
- One row per song, with the 128 columns listed in `model.feature_names_in_`.
- Audio features use Spotify's original scales (e.g. `danceability` in 0-1, `loudness` in dB).
- Genre is one-hot encoded: set exactly one `genre_*` column to 1 and the rest to 0.
- Output: a predicted popularity score between 0 and 100.



## Training Details

### Training Data
- **Dataset:** Spotify Tracks Dataset.
- **Target Variable:** `popularity` (Integer score from 0 to 100 representing play count and recency).
- **Input Dimension:** 128 total features:
  * **14 Base Features:** `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `time_signature`
  * **114 Categorical Dummy Features:** One-hot encoded columns derived from `track_genre`
* **Omitted Features:** `artists`, `album_name`, and `track_name`

### Training Procedure

* **Model:** `RandomForestRegressor` (scikit-learn); hyperparameters are defined in `params.yaml`.
* **Preprocessing** (`preprocessing_pipeline.py`):
   * Fully duplicated rows are removed.
   * Columns with more than 50% missing values are dropped, and rows with a missing target are dropped (the target is never imputed).
   * Rows that are Z-score outliers (|z| > 3.0) in at least one of `duration_ms`, `loudness` or `speechiness` are removed.
   * Non-feature columns are dropped: `Unnamed: 0`, `track_id`, `artists`, `album_name`, `track_name`.
   * Remaining missing values are imputed: numeric features with the median or mean and categorical features with the mode.
   * `track_genre` is one-hot encoded with the prefix `genre_` (114 genres).
   * No scaling is applied (Random Forests are insensitive to feature scaling).
   * *Scaling:* none (Random Forests are insensitive to feature scaling).
* **Pipeline:** automated with DVC (`dvc.yaml`) for data cleaning, feature engineering, and training execution.
* **Experiment tracking:** hyperparameters, parameters, metrics, and carbon emissions are logged automatically to MLflow / DagsHub.

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
- **Data Drift Detection:** Simulated using split dataset evaluation. Features exceeding defined drift thresholds trigger an automated flag for investigation or retraining (to be implemented).
