from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow import expand_dims, convert_to_tensor, float32
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.utils import to_categorical
import numpy as np
from database import get_untrained_products, Session, update_product_state


number_to_cat = {
     0: "10: Specialized Literature: music, physics",
     1: "40: Videogames, PS, XBox, Nintendo, cables",
     2: "50: Tech, controllers, fans, cables, cameras",
     3: "60: Console videogames, vintage and modern",
     4: "1140: Figurines, funkos, collectionable mugs",
     5: "1160: Collection cards: Pokemon, FIFA, Yu-Gi-Oh",
     6: "1180: Figurines, table games, miscelaneous",
     7: "1280: Kids: Toys, dolls, puff bears",
     8: "1281: Toys, cards (Yu-Gi-Oh), babys",
     9: "1300: Drones",
    10: "1301: Kids and babys toys, clothes, shoes",
    11: "1302: Outdoor toys, trampolines, gym, sports",
    12: "1320: Babys: deco, carts, backpacks, diappers",
    13: "1560: Home deco and furniture",
    14: "1920: Cussions. ",
    15: "1940: Food, coffee, gums, chiclets, mermalade",
    16: "2060: Photography, Christmas, deco, illumination",
    17: "2220: Pet toys, collars, cussions, pots, brushes",
    18: "2280: Magazines, science, art, historical journals",
    19: "2403: Books, comics, mangas",
    20: "2462: Console Videogames, consoles and games",
    21: "2522: Papershop, A5 size, pencils, notebooks",
    22: "2582: Outdoor furniture: tables, deco, plants",
    23: "2583: Pools, pumps, water cleaning",
    24: "2585: Bricolage, house repair, cleaning",
    25: "2705: Books, novels",
    26: "2905: PC Videogames"
}


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

    predicted_class = int(predicted_class[0])
    confidence = float(confidence)
    cat = number_to_cat[predicted_class]

    return {'predicted_class': predicted_class , 'confidence': confidence, 'cat': cat }


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
