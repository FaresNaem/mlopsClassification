from fastapi import FastAPI, UploadFile, File, Form
from fastapi import Depends, HTTPException
from joblib import load as joblib_load
from tensorflow.keras.models import load_model
import os
from util_model import predict_classification
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from database import create_user, get_user, add_product
from database import SessionLocal  # Import the session maker
from util_auth import create_access_token, verify_password, get_password_hash, verify_access_token, admin_required
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends, Request
from uuid import uuid4
from util_model import train_model_on_new_data, evaluate_model_on_untrained_data

# Load vectorizer and model globally when the app starts
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Tfidf_vectorizer.joblib')
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


@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/signup")
@app.post("/signup")
@app.post("/signup")
@app.post("/signup")
@app.post("/signup")
@app.post("/signup")
@app.post("/signup")
async def signup(username: str, password: str, db: Session = Depends(get_db)):
    # Check if the username already exists in the database
    existing_user = get_user(db, username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    # If the username doesn't exist, create the new user
    hashed_password = get_password_hash(password)
    create_user(db, username, hashed_password)

    return {"message": "User created successfully"}


@app.post("/predict")
async def predict_category(
        token: str = Depends(oauth2_scheme),  # Authentication dependency
        designation: str = Form(...),
        description: str = Form(...),
        file: UploadFile = File(...)
):
    # Verify the access token
    user_info = verify_access_token(token)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or user not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # If token is valid, proceed with prediction
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
        "predicted_class": predicted_class_int,  # Ensure predicted class is an integer
        "confidence": confidence_float  # Ensure confidence score is a float

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
        "predicted_class": predicted_class_int,  # Ensure predicted class is an integer
        "confidence": confidence_float  # Ensure confidence score is a float
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
        "predicted_class": predicted_class_int,  # Ensure predicted class is an integer
    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
        "predicted_class": predicted_class_int,  # Ensure predicted class is an integer
    image = Image.open(BytesIO(image_data))








    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']


    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])
    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)

    # Call the prediction function (returns both predicted class and confidence)
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)

    # Call the prediction function (returns both predicted class and confidence)
    image = Image.open(BytesIO(image_data))


    image = Image.open(BytesIO(image_data))


    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)

    # Call the prediction function (returns both predicted class and confidence)
    image = Image.open(BytesIO(image_data))

    image = Image.open(BytesIO(image_data))


    image = Image.open(BytesIO(image_data))
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)

    # Call the prediction function (returns both predicted class and confidence)
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)

    # Call the prediction function (returns both predicted class and confidence)
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)

    # Call the prediction function (returns both predicted class and confidence)
    image = Image.open(BytesIO(image_data))


    image = Image.open(BytesIO(image_data))
    image = Image.open(BytesIO(image_data))

    # Call the prediction function (returns both predicted class and confidence)
    predicted_result = predict_classification(model, vectorizer, designation, description, image)
    predicted_class = predicted_result['predicted_class']
    confidence = predicted_result['confidence']
    predicted_class_int = int(predicted_class[0])
    confidence_float = float(confidence[0])

    return {
        "predicted_class": predicted_class_int,  # Ensure predicted class is an integer
        "confidence": confidence_float  # Ensure confidence score is a float
    }


@app.get("/admin-only")
@admin_required()  # Decorator is still applied without modifying it
async def admin_route(
        request: Request,
        session: Session = Depends(get_db),  # Inject the Session here
        token: str = Depends(oauth2_scheme)  # Optionally inject the token using oauth2_scheme
):
    return {"message": "Welcome, admin!"}


# Define a directory to store uploaded images
##UPLOAD_DIR = r"C:\Users\user\Documents\DS_WB\images_uploaded"
# Get UPLOAD_DIR from environment variables, defaulting to './uploads' if not set
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")


@app.post("/add-product-data")
@admin_required()  # Decorator checks for admin privileges
async def add_product_api(
        request: Request,  # Add request parameter
        session: Session = Depends(get_db),  # Database session dependency
        image: UploadFile = File(...),
        designation: str = Form(...),  # 'designation' as a product title
        description: str = Form(...),
        category: str = Form(...),
        token: str = Depends(oauth2_scheme)
):
    # Generate a unique filename using UUID and preserve the original file extension
    file_extension = os.path.splitext(image.filename)[1]
    image_filename = f"{uuid4()}{file_extension}"
    image_path = os.path.join(UPLOAD_DIR, image_filename)

    try:
        # Ensure the upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save the image to the specified directory
        with open(image_path, "wb") as f:
            f.write(await image.read())

        # Call the function to add product data to the database
        add_product(session, image_path, designation, description, category)

        # Return a successful response
        return {"message": "Product added successfully"}
    except Exception as e:
        # If an error occurs, raise an HTTPException
        raise HTTPException(status_code=500, detail=f"Error saving product: {str(e)}")


# New endpoint to evaluate the model on untrained data
@app.get("/evaluate")
@admin_required()  # Only admins can access this endpoint
async def evaluate_model_endpoint(request: Request,
                                  session: Session = Depends(get_db),  # Inject the Session here
                                  token: str = Depends(oauth2_scheme)  # Optionally inject the token using oauth2_scheme
                                  ):
    try:
        # Call the function that evaluates the model
        f1, report = evaluate_model_on_untrained_data(model, vectorizer, session)

        # Return the F1 score and classification report
        return {
            "f1_score": f1,
            "classification_report": report
        }
    except Exception as e:
        # Handle exceptions and return an error message
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# New endpoint to train the model
@app.get("/train")
@admin_required()  # Only admins can access this endpoint
async def train_model_endpoint(request: Request, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    try:
        # Call the function that trains the model
        f1, report = train_model_on_new_data(model, vectorizer, db)

        # Return the F1 score and classification report
        return {
            "f1_score": f1,
            "classification_report": report
        }
    except Exception as e:
        # Handle exceptions and return an error message
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")