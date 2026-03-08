import runpod
import torch
import soundfile as sf
import base64
import io
from qwen_tts import Qwen3TTSModel

# Global model instance (loaded once per worker)
model = None

def load_model():
    """Load Qwen3-TTS model once at startup"""
    global model
    
    if model is None:
        print("Loading Qwen3-TTS model...")
        
        # Determine device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        print(f"Using device: {device}, dtype: {dtype}")
        
        # Load model with flash attention if available
        try:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device,
                dtype=dtype,
                attn_implementation="flash_attention_2"
            )
            print("Model loaded with Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention failed ({e}), falling back to default")
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=device,
                dtype=dtype
            )
            print("Model loaded with default attention")
    
    return model

def handler(job):
    """
    RunPod serverless handler for Qwen3-TTS
    Input: { "input": { "text": "Hello world", "speaker": "Vivian", "language": "English", "instruct": "" } }
    Output: { "audio": "base64_encoded_wav", "sample_rate": 24000 }
    """
    job_input = job.get("input", {})
    
    # Extract parameters
    text = job_input.get("text")
    speaker = job_input.get("speaker", "Vivian")  # Default speaker
    language = job_input.get("language", "English")
    instruct = job_input.get("instruct", "")  # Voice style instruction
    
    if not text:
        return {"error": "No text provided in input"}
    
    try:
        # Load model (cached after first call)
        tts_model = load_model()
        
        print(f"Generating speech for: {text[:50]}...")
        
        # Generate audio using CustomVoice model
        if instruct:
            # With instruction control (1.7B model feature)
            wavs, sr = tts_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct
            )
        else:
            # Without instruction
            wavs, sr = tts_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker
            )
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, wavs[0], sr, format='WAV')
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()
        
        # Encode to base64
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        print(f"Generated audio: {len(wav_bytes)} bytes, sample rate: {sr}")
        
        return {
            "audio": audio_b64,
            "sample_rate": sr,
            "format": "wav",
            "speaker": speaker,
            "language": language
        }
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return {"error": str(e)}

# Start RunPod serverless worker
if __name__ == "__main__":
    print("Starting Qwen3-TTS RunPod Serverless Worker...")
    runpod.serverless.start({"handler": handler})
