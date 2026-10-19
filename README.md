# MLOps_AlgoRhythm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# MLOps_AlgoRhythm

A reproducible Machine Learning project for predicting the popularity score of Spotify tracks using audio features and selected track metadata.

## Project overview

This project studies whether audio features and track metadata can predict a Spotify popularity score from 0 to 100. We train a regression model using features such as danceability, energy, loudness, tempo, and genre.

The model does not use artist names, album names, or track names as input features. This lets us study predictions based on track characteristics without directly including artist identity. It does not remove every possible bias from the data or from Spotify’s popularity measure.

The project focuses on a reproducible Machine Learning workflow. It uses DVC to version data and model artifacts, MLflow to track experiments, Great Expectations to validate pipeline data, Pytest to test code and model behaviour, and CodeCarbon to estimate training energy use and emissions.

## Team Members

| Name | Surname | GitHub Profile | UPC Email |
| --- | --- | --- | --- |
| **Ana** | Martinez Barbosa | [@anamtzbarbosa](https://www.google.com/search?q=https://github.com/anamtzbarbosa&utm_source=gemini) | `ana.martinez.barbosa@estudiantat.upc.edu` |
| **Enikő** | Beke | [@enikoandrea123](https://www.google.com/search?q=https://github.com/enikoandrea123&utm_source=gemini) | `eniko.andrea.beke@estudiantat.upc.edu` |
| **Ruddy** | Cuellar | [@ruddycuellarm](https://www.google.com/search?q=https://github.com/ruddycuellarm&utm_source=gemini) | `ruddy.alex.cuellar@estudiantat.upc.edu` |
| **Marta** | Sunyer Giménez | [@MartaSunyer](https://www.google.com/search?q=https://github.com/MartaSunyer&utm_source=gemini) | `marta.sunyer@estudiantat.upc.edu` |
| **Sergi** | González Martos | [@sergiGM0](https://www.google.com/search?q=https://github.com/sergiGM0&utm_source=gemini) | `sergi.gonzalez.martos@estudiantat.upc.edu` |

## Limitations
The model predicts Spotify’s popularity score, not general music quality or an artist’s success. Popularity reflects Spotify’s data at the time of collection and may change over time. The model also excludes artist identity, and its performance may change as music trends and Spotify’s user behaviour change.

## Setup and Execution

1. **Create a virtual environment** inside the project folder:

   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install the dependencies** listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the entire DVC pipeline**:

   ```bash
   dvc repro
   ```

5. **Run static code analysis** using Pylint:

   ```bash
   pylint .
   ```

6. **Execute all tests** using pytest:

   ```bash
   pytest -v
   ```
