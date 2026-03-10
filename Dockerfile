FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# CRITICAL: Prevent interactive apt-get prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies (all in one layer for speed)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages (combined for faster build)
RUN pip install --no-cache-dir \
    transformers>=4.47.0 \
    accelerate>=0.34.0 \
    huggingface-hub>=0.25.0 \
    torchaudio>=2.1.0 \
    sentencepiece \
    protobuf \
    soundfile \
    && rm -rf ~/.cache/pip

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy handler code
COPY handler.py .

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# RunPod serverless command
CMD ["python", "-u", "handler.py"]
