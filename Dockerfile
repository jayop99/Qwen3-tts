# Start with a GPU-ready image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1. Install system audio tools
RUN apt-get update && apt-get install -y ffmpeg sox libsox-dev git && rm -rf /var/lib/apt/lists/*

# 2. Clone the official Qwen3-TTS repo
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git .

# 3. Install the dependencies from their requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install the package in editable mode (standard for this repo)
RUN pip install -e .

# 5. Fast Attention is a must for your <1s goal
RUN pip install -U flash-attn --no-build-isolation

# Copy your specific app.py that calls the VoiceDesign model
COPY app.py .

EXPOSE 8000
CMD ["python", "app.py"]
