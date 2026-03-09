import runpod
import torch
import base64
import io
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from soundfile import write as soundfile_write

# Global model cache
model = None
processor = None

def load_model():
    """Load the Qwen3-TTS model (called once at startup)"""
    global model, processor
    
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    
    print(f"Loading model: {model_id}")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return device

def handler(event):
    """RunPod serverless handler"""
    try:
        global model, processor
        
        # Load model if not already loaded
        if model is None:
            device = load_model()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Extract input
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        
        if not text:
            return {
                "error": "Text input is required",
                "statusCode": 400
            }
        
        print(f"Generating speech for: {text}")
        
        # Prepare input
        inputs = processor(text, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            audio = model.generate(**inputs)
        
        # Convert to numpy
        audio_np = audio.cpu().numpy().flatten()
        
        # Save to WAV buffer
        buffer = io.BytesIO()
        soundfile_write(buffer, audio_np, samplerate=16000, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "samplerate": 16000,
            "format": "wav"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "statusCode": 500
        }

# Initialize model on startup
if __name__ == "__main__":
    print("Starting Qwen3-TTS serverless endpoint...")
    load_model()
    print("Server ready!")
