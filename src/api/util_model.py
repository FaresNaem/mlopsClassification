from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow import expand_dims, convert_to_tensor, float32
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.utils import to_categorical
import numpy as np
from database import get_untrained_products, Session, update_product_state


# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = np.array(image) / 255.0
    image = convert_to_tensor(image, dtype=float32)
    image = preprocess_input(image)
    return expand_dims(image, axis=0)


def predict_classification(model, vectorizer, designation: str, description: str, image: Image.Image):
    # Preprocess text data
    text_data = designation + ' ' + description
    processed_text = vectorizer.transform([text_data]).toarray()

    # Preprocess image data (no need to re-open the image)
    processed_image = preprocess_image(image)

    # Extract image features using EfficientNetB0
    image_features = EfficientNetB0(weights='imagenet', include_top=False)(processed_image)
    image_features = GlobalAveragePooling2D()(image_features).numpy()

    # Perform prediction
    prediction = model.predict([processed_text, image_features])
    predicted_class = np.argmax(prediction, axis=1)

    # Get confidence score (maximum probability)
    confidence = np.max(prediction, axis=1)

    return {'predicted_class': predicted_class, 'confidence': confidence}


def train_model_on_new_data(model, vectorizer, session: Session):
    """
    Function to train a pre-trained model using untrained products and return F1-score and classification report.

    Parameters:
        model: Pre-trained model
        vectorizer: Text vectorizer
        session: SQLAlchemy session to retrieve products

    Returns:
        f1: F1-score
        report: Classification report
    """
    # 1. Retrieve untrained products
    products = get_untrained_products(session)
    if not products:
        return "No new data available for training."

    # 2. Preprocess the data
    X_text = []
    X_image = []
    y = []
    product_ids = []

    for product in products:
        product_ids.append(product.id)
        text_data = product.designation + ' ' + product.description
        processed_text = vectorizer.transform([text_data]).toarray()[0]

        # Open image and preprocess it
        image = Image.open(product.image_path)
        processed_image = preprocess_image(image).reshape(-1)

        X_text.append(processed_text)
        X_image.append(processed_image)
        y.append(product.category)  # Assuming category is a class label

    X_text = np.array(X_text)
    X_image = np.array(X_image)
    y = np.array(y)

    # Assuming y is categorical; convert to integer labels if necessary
    num_classes = len(np.unique(y))
    y = to_categorical([int(label) for label in y], num_classes=num_classes)

    # 3. Train the model (fine-tuning on the new data)
    model.fit([X_text, X_image], y, epochs=5, batch_size=32, validation_split=0.2)

    # 4. Evaluate the model using F1 score and classification report
    y_pred = np.argmax(model.predict([X_text, X_image]), axis=1)
    y_true = np.argmax(y, axis=1)

    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)

    # 5. Mark products as trained (state = 1)
    update_product_state(session, product_ids)

    return f1, report


def evaluate_model_on_untrained_data(model, vectorizer, session: Session):
    """
    Function to evaluate a pre-trained model on untrained data (Product.state == 0).

    Parameters:
        model: Pre-trained model to be evaluated
        vectorizer: Text vectorizer for transforming text data
        session: SQLAlchemy session to fetch products

    Returns:
        f1: F1-score on the untrained data
        report: Classification report
    """
    # 1. Retrieve untrained products (Product.state == 0)
    products = get_untrained_products(session)

    if not products:
        return "No new data available for evaluation."

    # 2. Preprocess the data (both text and image)
    X_text = []
    X_image = []
    y_true = []

    for product in products:
        # Combine the designation and description as text data
        text_data = product.designation + ' ' + product.description
        processed_text = vectorizer.transform([text_data]).toarray()[0]

        # Open image and preprocess it
        image = Image.open(product.image_path)
        processed_image = preprocess_image(image).reshape(-1)  # Preprocess image and flatten it

        # Append to data lists
        X_text.append(processed_text)
        X_image.append(processed_image)
        y_true.append(product.category)  # Assuming `category` holds the actual label

    # Convert to numpy arrays
    X_text = np.array(X_text)
    X_image = np.array(X_image)
    y_true = np.array([int(label) for label in y_true])  # Convert true labels to integers

    # 3. Perform predictions using the model
    y_pred = np.argmax(model.predict([X_text, X_image]), axis=1)

    # 4. Calculate evaluation metrics
    f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)

    return f1, report
