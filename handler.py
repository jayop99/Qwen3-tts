import runpod
import torch
import base64
import io
from transformers import AutoModelForTextToSpeech, AutoTokenizer
import soundfile as sf

# Global variables for model caching
model = None
tokenizer = None

def load_model():
    """Load Qwen3-TTS model once and cache it"""
    global model, tokenizer
    
    if model is None:
        print("Loading Qwen3-TTS model...")
        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTextToSpeech.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")
    
    return model, tokenizer

def handler(event):
    """
    RunPod serverless handler for Qwen3-TTS
    Expected input: {"input": {"text": "Hello this is a test"}}
    Returns: {"audio": "base64_encoded_wav"}
    """
    try:
        # Extract text from input
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        
        if not text:
            return {"error": "No text provided in input"}
        
        print(f"Generating speech for: {text[:50]}...")
        
        # Load model (cached after first call)
        model, tokenizer = load_model()
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        # Convert to audio array
        audio_array = outputs.cpu().numpy().squeeze()
        
        # Convert to WAV format in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, samplerate=12000, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print("Audio generation complete!")
        
        return {
            "audio": audio_base64,
            "format": "wav",
            "sample_rate": 12000
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

# Start RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
