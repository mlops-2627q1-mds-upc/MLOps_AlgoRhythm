# pylint: disable=redefined-outer-name

"""Behavioral tests for the Spotify popularity RandomForest"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TEST_FILE

DATA_PATH = PROCESSED_DATA_DIR
MODEL_PATH = MODELS_DIR / "random_forest.pkl"

# Only continuous features pushed to extremes.
# Binary (explicit, genre_*) are left alone
# otherwise we create inputs that can never exist (e.g. genre_pop = 1.0001).
CONTINUOUS_COLS = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

N_SAMPLE = 300
SEED = 42


@pytest.fixture(scope="module")
def trained_model():
    """Load trained model."""
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_df(trained_model):
    """Test data -> exactly the model's input features."""
    df = pd.read_csv(DATA_PATH / TEST_FILE)
    features = list(trained_model.feature_names_in_)

    missing = [col for col in features if col not in df.columns]
    assert not missing, f"Missing model features: {missing}"

    return df[features].copy()


@pytest.fixture(scope="module")
def sample_df(test_df):
    """Random subset of the test set for tests"""
    n = min(N_SAMPLE, len(test_df))
    return test_df.sample(n=n, random_state=SEED).reset_index(drop=True)


@pytest.fixture
def sample_row(test_df):
    """One test example"""
    return test_df.iloc[[0]].copy()


##### Minimum functionality
def test_min_functionality_output_range(trained_model, sample_df):
    """Predictions are inside the valid Spotify range (0-100)."""
    predictions = trained_model.predict(sample_df)

    assert predictions.shape == (len(sample_df),)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 100)).all()


##### Invariance tests (output should not change)
def test_model_invariance_to_small_noise(trained_model, sample_df):
    """Tiny noise (+-0.1%) on continuous features barely moves predictions"""
    rng = np.random.default_rng(SEED)
    original = trained_model.predict(sample_df)

    perturbed = sample_df.copy()
    noise = 1 + rng.uniform(-0.001, 0.001, size=(len(perturbed), len(CONTINUOUS_COLS)))
    perturbed[CONTINUOUS_COLS] = perturbed[CONTINUOUS_COLS].to_numpy() * noise

    new = trained_model.predict(perturbed)
    diff = np.abs(original - new)

    print(f"\n [Invariance small noise] mean diff: {diff.mean():.4f} | max diff: {diff.max():.4f}")

    assert diff.mean() < 0.5, f"Mean change too large: {diff.mean():.4f}"
    assert np.percentile(diff, 95) < 2.0, "Too many rows changed"


def test_prediction_invariant_to_batch_context(trained_model, test_df):
    """The same song gets the same prediction alone or predicted in a batch with others"""
    song = test_df.iloc[[0]].copy()
    another_song = test_df.iloc[[1]].copy()

    prediction_alone = trained_model.predict(song)[0]

    batch = pd.concat([another_song, song], ignore_index=True)
    prediction_in_batch = trained_model.predict(batch)[1]

    print(
        f"\n [Invariance batch] prediction_alone: {prediction_alone:.4f} "
        f"| prediction_in_batch: {prediction_in_batch:.4f}"
    )

    assert prediction_alone == pytest.approx(prediction_in_batch, abs=1e-8)


def test_prediction_invariant_to_row_order(trained_model, sample_df):
    """Shuffle rows only shuffles the predictions, predicitons should be the same"""
    original = trained_model.predict(sample_df)

    perm = np.random.default_rng(SEED).permutation(len(sample_df))
    shuffled = trained_model.predict(sample_df.iloc[perm])
    # print(f"\n [Invariance row order] original: {original[perm]} | shuffled: {shuffled}")

    np.testing.assert_allclose(shuffled, original[perm], atol=1e-8)


def test_prediction_is_the_same(trained_model, sample_df):
    """Same input twice -> same output."""
    first = trained_model.predict(sample_df)
    second = trained_model.predict(sample_df)

    np.testing.assert_allclose(first, second, rtol=0, atol=1e-8)


##### Directional expectation tests (output should move in a known direction)
# In random forest a single row can be noisy. We change one feature for many rows and compare the MEAN prediction.


def _mean_prediction(model, df, **overrides):
    modified = df.copy()
    for col, value in overrides.items():
        modified[col] = value
    return model.predict(modified).mean()


def test_directional_instrumentalness(trained_model, sample_df):
    """More instrumental tracks (no lyrics) tend to be less popular."""
    low = _mean_prediction(trained_model, sample_df, instrumentalness=0.0)
    high = _mean_prediction(trained_model, sample_df, instrumentalness=0.9)

    print(f"\n [Directional] instrumentalness 0.0 -> {low:.2f} | 0.9 -> {high:.2f}")
    assert high < low, "Higher instrumentalness should lower predicted popularity"


def test_directional_genre_switch(trained_model, sample_df):
    """Switching a niche genre to a mainstream one should raise predicted popularity."""
    genre_cols = [c for c in sample_df.columns if c.startswith("genre_")]

    def with_genre(genre):
        df = sample_df.copy()
        df[genre_cols] = 0
        df[f"genre_{genre}"] = 1
        return trained_model.predict(df).mean()

    niche, mainstream = with_genre("grindcore"), with_genre("pop")

    print(f"\n [Directional] grindcore -> {niche:.2f} | pop -> {mainstream:.2f}")
    assert mainstream > niche


#### Edge cases


@pytest.mark.parametrize("value", [0.0, -1.0, 9_999_999.0])
def test_model_edge_cases(trained_model, sample_row, value):
    """Extreme continuous inputs still give valid outputs in 0-100."""
    edge_row = sample_row.copy()
    edge_row[CONTINUOUS_COLS] = value

    prediction = trained_model.predict(edge_row)[0]
    print(f"\n [Edge case] value={value}: prediction={prediction:.4f}")

    assert np.isfinite(prediction)
    assert 0 <= prediction <= 100
