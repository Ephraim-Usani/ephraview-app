# Use official Python base image
FROM python:3.10-slim

# Install system dependencies needed for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Dash
ENV PORT=8080

# Expose the port Dash uses
EXPOSE 8080

# Run the app
CMD ["gunicorn", "dicom_dash_app:app", "--bind", "0.0.0.0:8080"]
