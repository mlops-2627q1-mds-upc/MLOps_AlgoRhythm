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

## Repository Structure

## Team Members

| Name | Surname | GitHub Profile | UPC Email |
| --- | --- | --- | --- |
| **Ana** | Martinez Barbosa | [@anamtzbarbosa](https://www.google.com/search?q=https://github.com/anamtzbarbosa&utm_source=gemini) | `ana.martinez.barbosa@estudiantat.upc.edu` |
| **Enikő** | Beke | [@enikoandrea123](https://www.google.com/search?q=https://github.com/enikoandrea123&utm_source=gemini) | `eniko.andrea.beke@estudiantat.upc.edu` |
| **Ruddy** | Cuellar | [@ruddycuellarm](https://www.google.com/search?q=https://github.com/ruddycuellarm&utm_source=gemini) | `ruddy.alex.cuellar@estudiantat.upc.edu` |
| **Marta** | Sunyer Giménez | [@MartaSunyer](https://www.google.com/search?q=https://github.com/MartaSunyer&utm_source=gemini) | `marta.sunyer@estudiantat.upc.edu` |
| **Sergi** | González Martos | [@sergiGM0](https://www.google.com/search?q=https://github.com/sergiGM0&utm_source=gemini) | `sergi.gonzalez.martos@estudiantat.upc.edu` |
