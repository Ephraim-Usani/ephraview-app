# Use official Python image
FROM python:3.10-slim

# Install OS-level dependencies required for OpenCV and others
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all app files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port
EXPOSE 8080
ENV PORT 8080

# Start Gunicorn server for Dash
CMD ["gunicorn", "dicom_dash_app:server", "--bind", "0.0.0.0:8080"]
