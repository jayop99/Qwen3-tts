FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
# We install qwen-tts directly from GitHub to get the latest version
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    scipy \
    huggingface-hub>=0.25.0 \
    git+https://github.com/QwenLM/Qwen3-TTS.git

COPY handler.py .

CMD ["python", "-u", "handler.py"]
