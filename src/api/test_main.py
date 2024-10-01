import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import MagicMock, patch
from api.main import app

# Set up TestClient for testing FastAPI app
client = TestClient(app)


# Mocking database dependency
@pytest.fixture
def db_mock():
    db = MagicMock(spec=Session)
    yield db


# ------------------ TESTS FOR NORMAL USER ------------------

# Test user registration (normal user, not admin)
@patch("app.get_user")
@patch("app.create_user")
def test_signup_normal_user(mock_create_user, mock_get_user, db_mock):
    # Simulate that the user doesn't already exist
    mock_get_user.return_value = None

    # Call the signup endpoint
    response = client.post("/signup", json={"username": "normal_user", "password": "password123"})

    # Verify that the create_user function was called
    mock_create_user.assert_called_once()

    assert response.status_code == 200
    assert response.json() == {"message": "User created successfully"}


# Test signup failure when username is already taken
@patch("app.get_user")
def test_signup_existing_user(mock_get_user, db_mock):
    # Simulate that the user already exists
    mock_get_user.return_value = {"username": "existing_user", "password_hash": "hashed_password"}

    # Call the signup endpoint
    response = client.post("/signup", json={"username": "existing_user", "password": "password123"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Username already taken"}


# Test login as a normal user
@patch("app.get_user")
@patch("app.verify_password")
def test_login_normal_user(mock_verify_password, mock_get_user):
    # Simulate user returned from the database
    mock_get_user.return_value = {"username": "normal_user", "password_hash": "hashed_password", "role": "user"}
    mock_verify_password.return_value = True

    # Call the login endpoint
    response = client.post("/login", data={"username": "normal_user", "password": "password123"})

    assert response.status_code == 200
    assert "access_token" in response.json()


# Test login failure with invalid password
@patch("app.get_user")
@patch("app.verify_password")
def test_login_invalid_password(mock_verify_password, mock_get_user):
    # Simulate user returned from the database but with an incorrect password
    mock_get_user.return_value = {"username": "normal_user", "password_hash": "hashed_password"}
    mock_verify_password.return_value = False

    # Call the login endpoint
    response = client.post("/login", data={"username": "normal_user", "password": "wrongpassword"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid credentials"}


# Test prediction functionality for normal user (authenticated)
@patch("app.verify_access_token")
@patch("app.predict_classification")
def test_predict_category_for_normal_user(mock_predict_classification, mock_verify_access_token):
    # Simulate valid user token (normal user, not admin)
    mock_verify_access_token.return_value = {"sub": "normal_user", "role": "user"}

    # Mock the classification prediction function
    mock_predict_classification.return_value = [1]  # Simulate a prediction of class '1'

    # Simulate a prediction request
    files = {"file": ("test.jpg", b"fake_image_data", "image/jpeg")}
    data = {"designation": "Test Product", "description": "This is a test"}

    response = client.post("/predict", headers={"Authorization": "Bearer valid_token"}, files=files, data=data)

    assert response.status_code == 200
    assert response.json() == {"predicted_class": 1}


# Test prediction functionality for unauthorized user (missing token)
def test_predict_category_without_token():
    files = {"file": ("test.jpg", b"fake_image_data", "image/jpeg")}
    data = {"designation": "Test Product", "description": "This is a test"}

    response = client.post("/predict", files=files, data=data)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token or user not authenticated"}


# ------------------ TESTS FOR ADMIN USER ------------------

# Test login as an admin user
@patch("app.get_user")
@patch("app.verify_password")
def test_login_as_admin(mock_verify_password, mock_get_user):
    # Simulate admin user returned from the database
    mock_get_user.return_value = {"username": "Admin", "password_hash": "hashed_password", "role": "admin"}
    mock_verify_password.return_value = True

    # Call the login endpoint
    response = client.post("/login", data={"username": "Admin", "password": "password"})

    assert response.status_code == 200
    assert "access_token" in response.json()


# Test login with invalid credentials for admin
@patch("app.get_user")
@patch("app.verify_password")
def test_login_invalid_credentials(mock_verify_password, mock_get_user):
    mock_get_user.return_value = None
    response = client.post("/login", data={"username": "Admin", "password": "wrongpassword"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid credentials"}


# Test admin-only access with valid admin token
@patch("app.verify_access_token")
def test_admin_route_with_admin_access(mock_verify_access_token, db_mock):
    # Simulate valid admin token
    mock_verify_access_token.return_value = {"sub": "Admin", "role": "admin"}

    response = client.get("/admin-only", headers={"Authorization": "Bearer valid_token"})

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome, admin!"}


# Test admin-only access with non-admin token
@patch("app.verify_access_token")
def test_admin_route_with_non_admin_access(mock_verify_access_token):
    # Simulate valid token but not admin
    mock_verify_access_token.return_value = {"sub": "User", "role": "user"}

    response = client.get("/admin-only", headers={"Authorization": "Bearer valid_token"})

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin access required"}


# Test product addition as admin
@patch("app.add_product")
@patch("app.verify_access_token")
def test_add_product_as_admin(mock_verify_access_token, mock_add_product, db_mock):
    # Simulate valid admin token
    mock_verify_access_token.return_value = {"sub": "Admin", "role": "admin"}

    # Mock the add_product function
    mock_add_product.return_value = None

    # Simulate product addition
    files = {"image": ("test.jpg", b"fake_image_data", "image/jpeg")}
    data = {"designation": "Test Product", "description": "This is a test", "category": "Electronics"}

    response = client.post("/add-product-data", headers={"Authorization": "Bearer valid_token"}, files=files, data=data)

    assert response.status_code == 200
    assert response.json() == {"message": "Product added successfully"}


# Test product addition without admin access
@patch("app.verify_access_token")
def test_add_product_non_admin(mock_verify_access_token):
    # Simulate valid token but not admin
    mock_verify_access_token.return_value = {"sub": "User", "role": "user"}

    files = {"image": ("test.jpg", b"fake_image_data", "image/jpeg")}
    data = {"designation": "Test Product", "description": "This is a test", "category": "Electronics"}

    response = client.post("/add-product-data", headers={"Authorization": "Bearer valid_token"}, files=files, data=data)

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin access required"}
