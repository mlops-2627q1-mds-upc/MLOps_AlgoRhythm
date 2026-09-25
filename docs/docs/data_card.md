---
language:
- en
license: bsd
task_categories:
- tabular-classification
tags:
- music
- arts and entertainment
- audio
- tabular
- categorical
pretty_name: Spotify Tracks Dataset
---

# Dataset Card: Spotify Tracks Dataset

## 1. Dataset Description
- **Original Source / Repository:** https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Primary Modality:** Tabular (`.csv`)
- **Total Records:** 114,000 tracks
- **License:** BSD License

### Summary
This dataset consists of 114,000 Spotify tracks across 125 different musical genres. Each entry contains acoustic features computed via the Spotify Web API (such as `danceability`, `energy`, `valence`, `tempo`, etc.) along with metadata such as `track_name`, `artists`, `album_name`, and `popularity`.

---

## 2. Dataset Structure

### Features / Fields
The dataset contains 20 columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `track_id` | String | The Spotify ID for the track. |
| `artists` | String | Artists' names. Multiple artists are separated by `;`. |
| `album_name` | String | Name of the album in which the track appears. |
| `track_name` | String | Name of the track. |
| `popularity` | Integer (0-100) | Metric based on total plays and recent streams. |
| `duration_ms` | Integer | Track length in milliseconds. |
| `explicit` | Boolean | Whether the track has explicit lyrics (`true`/`false`). |
| `danceability` | Float (0.0-1.0) | Describes how suitable a track is for dancing. |
| `energy` | Float (0.0-1.0) | Perceptual measure of intensity and activity. |
| `key` | Integer (-1 to 11)| Pitch class notation (`0 = C`, `1 = C♯/D♭`, etc.). |
| `loudness` | Float (dB) | Overall loudness in decibels. |
| `mode` | Integer (0 or 1) | Modality (`1 = Major`, `0 = Minor`). |
| `speechiness` | Float (0.0-1.0) | Presence of spoken words in the track. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, and values below 0.33 represent non-speech-like tracks. |
| `acousticness` | Float (0.0-1.0) | Confidence measure of whether the track is acoustic. |
| `instrumentalness` | Float (0.0-1.0) | Predicts whether a track contains no vocals. |
| `liveness` | Float (0.0-1.0) | Probability that the track was performed live. |
| `valence` | Float (0.0-1.0) | Musical positiveness (happy/cheerful vs. sad/angry). |
| `tempo` | Float (BPM) | Estimated tempo in beats per minute. |
| `time_signature` | Integer (3-7) | Estimated time signature (e.g., `4 = 4/4`). |
| `track_genre` | String | Musical genre assigned to the track (125 genres total). |

---

## 3. Data Collection & Preprocessing
* **Source:** Collected using the **Spotify Web API**.
* **Cleaning:** Tracks were fetched and parsed into tabular format, standardizing numerical acoustic measures and categorical tags.

---

## 4. Considerations & Limitations
* **Popularity Decay:** The `popularity` metric is dynamic and reflects Spotify stream activity at the time of data collection.
* **Sampling Bias:** Genre labels are based on Spotify's internal categorization.