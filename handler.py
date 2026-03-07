import runpod
import torch
import base64
import soundfile as sf
from io import BytesIO
from transformers import pipeline

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

print("Loading TTS pipeline...")

tts = pipeline(
    "text-to-speech",
    model=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)

print("Model loaded.")

def generate_audio(text: str):
    with torch.no_grad():
        audio = tts(text)

    audio_array = audio["audio"]
    sampling_rate = audio["sampling_rate"]

    buffer = BytesIO()
    sf.write(buffer, audio_array, sampling_rate, format="WAV")
    audio_bytes = buffer.getvalue()

    encoded_audio = base64.b64encode(audio_bytes).decode()
    return encoded_audio

def handler(event):
    inputs = event.get("input", {})
    texts = inputs.get("texts")

    if not texts:
        text = inputs.get("text")
        if not text:
            return {"error": "Provide 'text' or 'texts'"}
        texts = [text]

    results = []
    for text in texts:
        results.append(generate_audio(text))

    return {"audios": results}

runpod.serverless.start({"handler": handler})
