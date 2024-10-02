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
from database import create_user, get_user, add_product, SessionLocal

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

# User authentication and token generation
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/signup")
async def signup(username: str, password: str, db: Session = Depends(get_db)):
    existing_user = get_user(db, username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(password)
    create_user(db, username, hashed_password)
    return {"message": "User created successfully"}

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

# Endpoint to add new product data
@app.post("/add-product-data")
@admin_required()
async def add_product_api(
    request: Request,
    session: Session = Depends(get_db),
    image: UploadFile = File(...),
    designation: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    token: str = Depends(oauth2_scheme)
):
    file_extension = os.path.splitext(image.filename)[1]
    image_filename = f"{uuid4()}{file_extension}"
    image_path = os.path.join(UPLOAD_DIR, image_filename)

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(await image.read())
        
        add_product(session, image_path, designation, description, category)
        return {"message": "Product added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving product: {str(e)}")

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
