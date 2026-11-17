from database import Base, get_db
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup ephemeral in-memory test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

from api.main import app, get_current_user
from domain.models import UserDB, Role


def override_get_db():
  db = TestingSessionLocal()
  try:
    yield db
  finally:
    db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def setup_function():
  Base.metadata.create_all(bind=test_engine)


def teardown_function():
  Base.metadata.drop_all(bind=test_engine)


def test_public_endpoint():
  response = client.get("/public")
  assert response.status_code == 200


def test_ownership_enforcement_forbidden():
  db = TestingSessionLocal()
  regular_user = UserDB(
      user_id=1, username="alice", hashed_password="pw", role=Role.USER
  )
  db.add(regular_user)
  db.commit()

  # Mock current user as Alice (ID 1)
  app.dependency_overrides[get_current_user] = lambda: regular_user

  # Alice tries to post to Bob's ID (2)
  response = client.post(
      "/users/2/songs", params={"title": "Song", "rhythm": "Fast", "popularity": 10}
  )
  assert response.status_code == 403
  db.close()