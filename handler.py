import os
import io
import base64
import torch
import traceback
import runpod
from transformers import AutoProcessor, AutoModelForTextToSpeech
from huggingface_hub import login
import torchaudio

# 🔑 CRITICAL: Login with HF_TOKEN before loading model
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
    print("✅ HF_TOKEN login successful")
else:
    print("❌ HF_TOKEN not found!")

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Global model cache
model = None
processor = None
device = None

def load_model():
    """Load the Qwen3-TTS model (called once at startup)"""
    global model, processor, device
    
    print(f"🔄 Loading model: {MODEL_ID}")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTextToSpeech.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"✅ Model loaded successfully on {device}")
    return device

def handler(event):
    """RunPod serverless handler"""
    try:
        global model, processor, device
        
        # Load model if not already loaded
        if model is None or processor is None:
            device = load_model()
        
        # Extract input (RunPod uses "input" -> "prompt")
        input_data = event.get("input", {})
        text = input_data.get("prompt", "")
        
        if not text:
            return {
                "error": "Text input is required",
                "statusCode": 400
            }
        
        print(f"🎤 Generating speech for: {text}")
        
        # Prepare input
        inputs = processor(text, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            audio = model.generate(**inputs)
        
        # Convert to numpy
        audio_np = audio.cpu().numpy().flatten()
        
        # Save to WAV buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, torch.tensor(audio_np).unsqueeze(0), processor.sampling_rate, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "samplerate": processor.sampling_rate,
            "format": "wav"
        }
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "statusCode": 500
        }

# Initialize model on startup
if __name__ == "__main__":
    print("🚀 Starting Qwen3-TTS serverless endpoint...")
    load_model()
    print("✅ Server ready!")
