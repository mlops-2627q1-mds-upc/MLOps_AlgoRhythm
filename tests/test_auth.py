from src.database import Base, get_db
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import pytest

# Setup ephemeral in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
from src.api.main import app
from src.domain.user import User, UserRole

# Override the get_db dependency to use the test database
def override_get_db():
  db = TestingSessionLocal()
  try:
    yield db
  finally:
    db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
  Base.metadata.create_all(bind=test_engine)
  yield
  Base.metadata.drop_all(bind=test_engine)


# TEST: Public Endpoint
def test_public_endpoint():
  response = client.get("/public-endpoint")
  assert response.status_code == 200
  assert response.json() == {
      "message": "Welcome! Anyone can access this endpoint."
  }

# TEST: Signed-in Endpoint without authentication
def test_signed_in_endpoint_unauthorized():
  response = client.get("/signed-in-endpoint")
  assert response.status_code == 401

# TEST: Signed-in Endpoint with authentication
def test_signed_in_endpoint_success():
  db = TestingSessionLocal()
  user = User(
      user_id=1,
      username="anna",
      hashed_password="fakehash",
      role=UserRole.USER,
  )
  db.add(user)
  db.commit()
  db.close()

  response = client.get("/signed-in-endpoint")
  assert response.status_code == 200
  assert "Hello anna" in response.json()["message"]

# TEST: Owner Endpoint with incorrect user
def test_owner_endpoint_forbidden():
  db = TestingSessionLocal()
  user = User(
      user_id=1, username="anna", hashed_password="fakehash", role=UserRole.USER
  )
  db.add(user)
  db.commit()
  db.close()

  response = client.get("/users/2/owner-endpoint")
  assert response.status_code == 403

# TEST: Owner Endpoint with correct user
def test_owner_endpoint_success():
  db = TestingSessionLocal()
  user = User(
      user_id=1, username="anna", hashed_password="fakehash", role=UserRole.USER
  )
  db.add(user)
  db.commit()
  db.close()

  response = client.get("/users/1/owner-endpoint")
  assert response.status_code == 200
  assert "You are the owner" in response.json()["message"]

# TEST: Admin Endpoint with regular user
def test_admin_endpoint_as_regular_user_forbidden():
  db = TestingSessionLocal()
  user = User(
      user_id=1, username="anna", hashed_password="fakehash", role=UserRole.USER
  )
  db.add(user)
  db.commit()
  db.close()

  response = client.get("/admin-endpoint")
  assert response.status_code == 403

# TEST: Admin Endpoint with admin user
def test_admin_endpoint_as_admin_success():
  db = TestingSessionLocal()
  admin_user = User(
      user_id=99, username="anna", hashed_password="fakehash", role=UserRole.ADMIN
  )
  db.add(admin_user)
  db.commit()
  db.close()

  response = client.get("/admin-endpoint")
  assert response.status_code == 200
  assert "Hello anna" in response.json()["message"]