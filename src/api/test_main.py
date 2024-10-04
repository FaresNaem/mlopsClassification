import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app, get_db
from database import Base, User, create_user, get_user
from util_auth import get_password_hash, create_access_token

# Setup an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override the get_db dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create a TestClient
client = TestClient(app)

# Fixture to create tables and a test client
@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

# Utility to create a user in the test database
def create_test_user(db, username, password, role='user'):
    hashed_password = get_password_hash(password)
    return create_user(db, username=username, password_hash=hashed_password, role=role)

# Test for login endpoint
def test_login():
    # Create a user
    with TestingSessionLocal() as db:
        create_test_user(db, "testuser", "testpass")

    # Attempt to login with valid credentials
    response = client.post("/login", data={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    assert "access_token" in response.json()

# Test admin-only access
def test_admin_route():
    # Create an admin user
    with TestingSessionLocal() as db:
        create_test_user(db, "adminuser", "adminpass", role="admin")

    # Login as admin
    response = client.post("/login", data={"username": "adminuser", "password": "adminpass"})
    token = response.json()["access_token"]

    # Access admin-only route
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/admin-only", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome, admin!"

# Test non-admin access to admin route
def test_non_admin_access():
    # Create a non-admin user
    with TestingSessionLocal() as db:
        create_test_user(db, "testuser", "testpass")

    # Login as a non-admin
    response = client.post("/login", data={"username": "testuser", "password": "testpass"})
    token = response.json()["access_token"]

    # Attempt to access admin-only route
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/admin-only", headers=headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Not authorized"
