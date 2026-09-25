# MLOps_AlgoRhythm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Natural Language and Multimodal Music Recommendation System

An end-to-end Machine Learning project for personalized music recommendation using natural language queries and vector similarity matching.

## Project Overview

Finding tracks that match a specific mood, activity, or atmosphere often fails with traditional keyword search. This project leverages pre-trained Sentence Transformer models and visual embedding extractors to translate natural language prompts (e.g., *"upbeat music for a rainy evening"*) into tailored track recommendations.

We process free-text queries into a unified vector space, perform high-speed cosine similarity search against pre-computed Spotify track embeddings, and combine audio features (danceability, energy, valence) to return ranked top-10 track recommendations via API.

## Dataset

We use the **Spotify Dataset** sourced from Kaggle and enriched via the official **Spotify Developer Web API**.

* **Granularity:** Track-level metadata and audio attributes
* **Scope:** Multi-genre global music catalog
* **Volume:** $\approx 73,000+$ track observations
* **Source:** Spotify Developer API / Kaggle Music Datasets

## Team Members

| Name | Surname | GitHub Profile | UPC Email |
| --- | --- | --- | --- |
| **Ana** | Martinez Barbosa | [@anamtzbarbosa](https://www.google.com/search?q=https://github.com/anamtzbarbosa&utm_source=gemini) | `ana.martinez.barbosa@estudiantat.upc.edu` |
| **Enikő** | Beke | [@enikoandrea123](https://www.google.com/search?q=https://github.com/enikoandrea123&utm_source=gemini) | `eniko.andrea.beke@estudiantat.upc.edu` |
| **Ruddy** | Cuellar | [@ruddycuellarm](https://www.google.com/search?q=https://github.com/ruddycuellarm&utm_source=gemini) | `ruddy.alex.cuellar@estudiantat.upc.edu` |
| **Marta** | Sunyer Giménez | [@MartaSunyer](https://www.google.com/search?q=https://github.com/MartaSunyer&utm_source=gemini) | `marta.sunyer@estudiantat.upc.edu` |
| **Sergi** | González Martos | [@sergiGM0](https://www.google.com/search?q=https://github.com/sergiGM0&utm_source=gemini) | `sergi.gonzalez.martos@estudiantat.upc.edu` |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------