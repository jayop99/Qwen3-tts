import os
import sys
import io
import base64
import traceback

print("[BOOT 1/5] Python started", flush=True)

try:
    import torch
    print(f"[BOOT 2/5] torch {torch.__version__} ✅  |  CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[BOOT 2/5] FATAL - torch: {e}", flush=True)
    sys.exit(1)

try:
    import soundfile as sf
    import numpy as np
    print("[BOOT 3/5] soundfile + numpy ✅", flush=True)
except Exception as e:
    print(f"[BOOT 3/5] FATAL - soundfile/numpy: {e}", flush=True)
    sys.exit(1)

try:
    from qwen_tts import Qwen3TTSModel
    print("[BOOT 4/5] qwen_tts ✅", flush=True)
except Exception as e:
    print(f"[BOOT 4/5] FATAL - qwen_tts: {e}", flush=True)
    sys.exit(1)

try:
    import runpod
    print("[BOOT 5/5] runpod ✅", flush=True)
except Exception as e:
    print(f"[BOOT 5/5] FATAL - runpod: {e}", flush=True)
    sys.exit(1)

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
device   = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"[CONFIG] Model  : {MODEL_ID}", flush=True)
print(f"[CONFIG] Device : {device}", flush=True)

model = None

def load_model():
    global model
    print(f"[LOAD] Downloading {MODEL_ID} ...", flush=True)
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=device,
            dtype=torch.bfloat16,
        )
        print("[LOAD] ✅ Model ready!", flush=True)
    except Exception as e:
        print(f"[LOAD] FATAL - model load failed: {e}", flush=True)
        traceback.print_exc()
        raise

def handler(event):
    global model
    try:
        if model is None:
            load_model()

        inp      = event.get("input", {})
        text     = inp.get("prompt", "").strip()
        language = inp.get("language", "English")
        speaker  = inp.get("speaker", "Ethan")
        instruct = inp.get("instruct", "")

        if not text:
            return {"error": "Missing 'prompt' in input", "statusCode": 400}

        print(f"[GEN] lang={language} | speaker={speaker} | text={text[:60]}", flush=True)

        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct if instruct else None,
        )

        audio_np = wavs[0]
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV")
        buf.seek(0)

        print(f"[GEN] ✅ Done — {len(audio_np)} samples @ {sr} Hz", flush=True)

        return {
            "audio":       base64.b64encode(buf.read()).decode("utf-8"),
            "format":      "wav",
            "sample_rate": sr,
            "speaker":     speaker,
            "language":    language,
            "statusCode":  200,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "statusCode": 500}

print("[START] Starting RunPod serverless worker...", flush=True)
runpod.serverless.start({"handler": handler
