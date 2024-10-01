# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/app/uploads

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for caching dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the upload directory
RUN mkdir -p $UPLOAD_DIR

# Copy the entire project into the container
COPY . .

# Change the working directory to the src/api directory
WORKDIR /app/src/api

# Expose port 8000 for FastAPI
EXPOSE 8000

# Define the entry point for the container (run the FastAPI application)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
