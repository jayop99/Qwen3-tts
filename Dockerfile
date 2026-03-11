FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install from GitHub directly (official source)
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /tmp/Qwen3-TTS && \
    pip install --no-cache-dir -e /tmp/Qwen3-TTS && \
    rm -rf ~/.cache/pip

RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    scipy \
    huggingface-hub>=0.25.0 \
    && rm -rf ~/.cache/pip

COPY handler.py .

CMD ["python", "-u", "handler.py"]
Also replace requirements.txt with just:
soundfile
scipy
