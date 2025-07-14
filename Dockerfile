# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variable for TensorFlow CPU
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port (Dash default)
EXPOSE 8050

# Run the Dash app
CMD ["python", "dicom_dash_app.py"]
