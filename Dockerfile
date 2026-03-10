FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Force install latest transformers (CRITICAL!)
RUN pip install --no-cache-dir transformers>=4.47.0
RUN pip install --no-cache-dir accelerate>=0.34.0
RUN pip install --no-cache-dir huggingface-hub>=0.25.0

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# Set environment variable
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# RunPod serverless command
CMD ["python", "-u", "handler.py"]
