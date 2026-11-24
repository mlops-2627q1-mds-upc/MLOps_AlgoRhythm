"""Unit tests for the UML domain classes and add-song sequence."""

import pytest
from sklearn.dummy import DummyRegressor


from src.domain import Model, User, UserRole, UserService


def test_user_exposes_id_username_and_role():
    user = User(
        user_id=1,
        username="anna",
        hashed_password="encoded-hash",
        role=UserRole.USER,
        stored_songs=[],
    )

    assert user.get_id() == "1"
    assert user.get_username() == "anna"
    assert user.role == UserRole.USER


def test_admin_role_is_reported_by_is_admin():
    admin = User(
        username="root",
        hashed_password="encoded-hash",
        role=UserRole.ADMIN,
        stored_songs=[],
    )
    normal = User(
        username="anna",
        hashed_password="encoded-hash",
        role=UserRole.USER,
        stored_songs=[],
    )

    assert admin.is_admin() is True
    assert normal.is_admin() is False


def test_user_add_song_appends_song_record():
    user = User(
        username="anna",
        hashed_password="encoded-hash",
        role=UserRole.USER,
        stored_songs=[],
    )
    song = {"title": "Example track", "popularity": 70}

    user.add_song(song)

    assert user.stored_songs == [song]


def test_user_service_add_song_allows_account_owner():
    user = User(
        user_id=1,
        username="anna",
        hashed_password="encoded-hash",
        role=UserRole.USER,
        stored_songs=[],
    )
    song = {"title": "Example track"}

    UserService.add_song(user, song, user.get_id())

    assert user.stored_songs == [song]

def test_user_service_rejects_different_user_without_mutating_collection():
    user = User(
        user_id=1,
        username="anna",
        hashed_password="encoded-hash",
        role=UserRole.USER,
        stored_songs=[],
    )
    song = {"title": "Example track"}

    with pytest.raises(PermissionError, match="own account"):
        UserService.add_song(user, song, "not-the-owner")

    assert user.stored_songs == []


def test_model_getters_and_model_info():
    model = Model(name="Popularity Random Forest", params={"n_estimators": 100})

    assert model.get_id() == str(model.model_id)
    assert model.get_name() == "Popularity Random Forest"
    assert model.get_params() == {"n_estimators": 100}
    assert model.get_model_info() == {
        "model_id": str(model.model_id),
        "name": "Popularity Random Forest",
        "params": {"n_estimators": 100},
    }


def test_model_predict_delegates_to_fitted_estimator():
    estimator = DummyRegressor(strategy="constant", constant=64.0)
    estimator.fit([[0.0], [1.0]], [0.0, 1.0])
    model = Model(name="Popularity baseline")

    result = model.predict(estimator, [[2.0], [3.0]])

    assert result == {
        "model_id": str(model.model_id),
        "model_name": "Popularity baseline",
        "predictions": [64.0, 64.0],
    }
