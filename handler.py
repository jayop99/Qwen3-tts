import runpod
import torch
import base64
import soundfile as sf
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, pipeline

MODEL_NAME = "Qwen/Qwen3-TTS"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tts = pipeline(
    "text-to-speech",
    model=MODEL_NAME,
    device=0
)

print("Model loaded.")

def generate_audio(text):

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
        audio = generate_audio(text)
        results.append(audio)

    return {
        "audios": results
    }


runpod.serverless.start({
    "handler": handler
})
