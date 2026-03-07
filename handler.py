import runpod
import torch
import base64
import soundfile as sf
from io import BytesIO
from qwen_tts import Qwen3TTSModel

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

print("Loading Qwen3 TTS model...")

model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16
)

print("Model loaded.")

def generate_audio(text):
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="English"
    )

    buffer = BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    audio_bytes = buffer.getvalue()

    return base64.b64encode(audio_bytes).decode()


def handler(event):
    text = event["input"]["text"]
    audio = generate_audio(text)
    return {"audio": audio}


runpod.serverless.start({"handler": handler})
