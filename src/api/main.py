# this is docker customized version
import time
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from joblib import load as joblib_load
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import os
import numpy as np
from sqlalchemy.orm import Session
from uuid import uuid4

from util_model import predict_classification, train_model_on_new_data, evaluate_model_on_untrained_data
from util_auth import create_access_token, verify_password, get_password_hash, verify_access_token, admin_required
from database import create_user, get_user, add_product, SessionLocal, User, create_tables, delete_user, log_event, get_all_logs, is_database_available

# Load vectorizer and model globally when the app starts
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Tfidf_Vectorizer.joblib')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'retrained_balanced_model.keras')

vectorizer = joblib_load(vectorizer_path)
model = load_model(model_path)

# Dependency to get a session from the database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize OAuth2PasswordBearer to extract access token from the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI()

# Initialize database when the app starts
@app.on_event("startup")
def on_startup():
    # Wait until the SQL Server is available
    while not is_database_available():
        print("Waiting for the database to become available...")
        time.sleep(5)  # Wait for 5 seconds before retrying
    create_tables()  # Create tables if they don't exist


# Updated signup function to assign admin role to the first user
@app.post("/signup")
async def signup(username: str, password: str, db: Session = Depends(get_db)):
    existing_user = get_user(db, username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    # Check if any users exist in the database
    users_exist = db.query(User).count() > 0

    # Assign 'admin' role to the first user, otherwise 'user' role
    role = 'admin' if not users_exist else 'user'

    hashed_password = get_password_hash(password)
    new_user =create_user(db, username, hashed_password, role)
     # Log the signup event
    log_event(db, new_user.id, f"User {username} signed up ")
    
    return {"message": f"User created successfully with role: {role}"}


# User authentication and token generation
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username})
    # Log the login event
    log_event(db, user.id, f"User {user.username} logged in")
    return {"access_token": token, "token_type": "bearer"}

# Product category prediction endpoint
@app.post("/predict")
async def predict_category(
    token: str = Depends(oauth2_scheme),
    designation: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...)
):
    user_info = verify_access_token(token)
    if not user_info:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or user not authenticated")
    
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = int(predicted_result['predicted_class'][0])
    confidence = float(predicted_result['confidence'][0])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

# Admin-only route
@app.get("/admin-only")
@admin_required()
async def admin_route(
    request: Request,
    session: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    return {"message": "Welcome, admin!"}

# Define a directory to store uploaded images
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

@app.post("/add-product-data")
@admin_required()
async def add_product_api(
    request: Request,
    session: Session = Depends(get_db),
    image_filename: str = Form(...),  # Expecting the image filename instead of upload
    designation: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    token: str = Depends(oauth2_scheme)
):
    # Assume the images are available in the /images directory
    local_image_path = os.path.join("/images", image_filename)  # From mounted Windows folder
    file_extension = os.path.splitext(image_filename)[1]
    new_image_filename = f"{uuid4()}{file_extension}"
    destination_image_path = os.path.join(UPLOAD_DIR, new_image_filename)

    try:
        # Check if the image exists in the mounted directory
        if not os.path.exists(local_image_path):
            raise HTTPException(status_code=404, detail=f"Image '{image_filename}' not found")

        # Create the uploads directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Copy the image from the mounted directory to the uploads directory
        with open(local_image_path, "rb") as src_file:
            with open(destination_image_path, "wb") as dest_file:
                dest_file.write(src_file.read())

        # Add product data to the database
        add_product(session, destination_image_path, designation, description, category)

        return {"message": "Product added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding product: {str(e)}")

# Endpoint to evaluate the model on untrained data
@app.get("/evaluate")
@admin_required()
async def evaluate_model_endpoint(
    request: Request,
    session: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    try:
        f1, report = evaluate_model_on_untrained_data(model, vectorizer, session)
        return {"f1_score": f1, "classification_report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Endpoint to train the model on new data
@app.get("/train")
@admin_required()
async def train_model_endpoint(
    request: Request,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    try:
        f1, report = train_model_on_new_data(model, vectorizer, db)
        return {"f1_score": f1, "classification_report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Admin-only route to return all logs
@app.get("/admin/logs")
@admin_required()
async def get_logs(
    request: Request,
    session: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    # Call the get_all_logs function to retrieve the logs
    logs = get_all_logs(session)
    
    # Convert logs to a list of dictionaries for JSON serialization
    log_list = [
        {
            "id": log.id,
            "timestamp": log.timestamp,
            "user_id": log.user_id,
            "event": log.event
        } for log in logs
    ]
    
    return {"logs": log_list}


# Admin-only route to delete a user by username
@app.delete("/delete-user/{username}")
@admin_required()
async def delete_user_by_admin(
    request: Request,  # Added Request as the first parameter
    username: str,
    session: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
):
    # Attempt to delete the user by username
    result = delete_user(session, username)

    # Check if the user was found and deleted
    if result:
        return {"message": f"User '{username}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="User not found")