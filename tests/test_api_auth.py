from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.auth import get_current_user
from src.domain.user import User, UserRole


def _make_user(user_id: int, username: str, role: UserRole) -> User:
    return User(
        user_id=user_id,
        username=username,
        hashed_password="encoded-hash",
        role=role,
        stored_songs=[],
    )


def test_public_endpoint_allows_anonymous_access():
    client = TestClient(app)

    response = client.get("/public-endpoint")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome! Anyone can access this endpoint."}


def test_signed_in_endpoint_returns_401_when_unauthenticated():
    def _unauthorized():
        raise HTTPException(status_code=401, detail="Not authenticated.")

    app.dependency_overrides[get_current_user] = _unauthorized
    client = TestClient(app)

    try:
        response = client.get("/signed-in-endpoint")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_signed_in_endpoint_allows_authenticated_user():
    current_user = _make_user(user_id=1, username="anna", role=UserRole.USER)
    app.dependency_overrides[get_current_user] = lambda: current_user
    client = TestClient(app)

    try:
        response = client.get("/signed-in-endpoint")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello anna! You have access to this signed endpoint."
    finally:
        app.dependency_overrides.clear()


def test_admin_endpoint_forbids_non_admin_user():
    current_user = _make_user(user_id=1, username="anna", role=UserRole.USER)
    app.dependency_overrides[get_current_user] = lambda: current_user
    client = TestClient(app)

    try:
        response = client.get("/admin-endpoint")
        assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_admin_endpoint_allows_admin_user():
    admin_user = _make_user(user_id=2, username="root", role=UserRole.ADMIN)
    app.dependency_overrides[get_current_user] = lambda: admin_user
    client = TestClient(app)

    try:
        response = client.get("/admin-endpoint")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello root! You have access to this admin endpoint."
    finally:
        app.dependency_overrides.clear()