# MLOps Product Classification Application

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
- [API Endpoints](#api-endpoints)
- [Model Overview](#model-overview)
- [Database](#database)
- [Contributors](#contributors)

## Project Overview
The MLOps Product Classification Application is designed to classify products based on images and text descriptions. Built with FastAPI, this application exposes a REST API allowing both administrators and general users to interact with a machine learning model for efficient product categorization.

### Objectives
- Automate product classification to save time and improve accuracy.
- Enable both admin and general user functionalities through role-based access.

## Installation
To get started with this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd MLOps_classification
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Locally
1. Navigate to the API directory:
   ```bash
   cd src/api
   ```

2. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

3. (Optional) Adjust the database credentials by setting the environment variables:
   ```bash
   POSTGRES_USER=my_user POSTGRES_PASSWORD=my_password POSTGRES_HOST=db POSTGRES_PORT=5432 POSTGRES_DB=mydb uvicorn main:app --reload
   ```

### Running with Docker
1. Pull the Docker image:
   ```bash
   docker pull fnaem/mlops_dockercustom:v2
   ```

2. (Optional) Change database credentials in `docker-compose.yml`.

3. Run Docker Compose:
   ```bash
   docker-compose up
   ```

4. Access the API documentation: Open your browser at `http://localhost:8000/docs`.

## API Endpoints

### Authentication
- **POST** `/auth/login`: User login.
- **POST** `/auth/signup`: User registration.

### Prediction
- **POST** `/predict`: Classify a product based on image and text input.

### Data Management (Admin Only)
- **POST** `/admin/add-product-data`: Upload new product data.
- **POST** `/admin/retrain-model`: Retrain the model with updated data.
- **GET** `/admin/evaluate-model`: Retrieve model performance metrics.

### Monitoring and Logs (Admin Only)
- **GET** `/admin/logs`: Fetch logs of API operations.
- **DELETE** `/delete-user/{username}`: Remove a user by username.

## Model Overview
The application uses a multimodal machine learning model that processes both images and text for classification. The model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Database
The application uses PostgreSQL to store:
- Product images and descriptions.
- User information and roles.
- Logs of actions and events.

## Contributors
- Vitalij Merenics
- Fares Naem
- Sebastián Mantilla-Serrano
