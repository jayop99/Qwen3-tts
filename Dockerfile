FROM python:3.10-slim

WORKDIR /app

# Install system dependencies FIRST (including git and build tools)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# RunPod serverless command
CMD ["python", "-u", "handler.py"]
