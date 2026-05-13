import os
import base64
import torch
import runpod
import tempfile
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from faster_whisper import WhisperModel
import scipy.io.wavfile as wavfile
import numpy as np

# 1. LOAD MODELS AT STARTUP (This is your "Keep Warm" state)
print("⏳ Loading OmniVoice & ASR into VRAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=torch.float16,
    load_asr=True
)
print("✅ Models loaded and ready!")

def process_clone(job):
    """
    The main execution block. RunPod passes the payload inside job['input']
    """
    job_input = job["input"]
    text = job_input.get("text")
    ref_audio_b64 = job_input.get("ref_audio_b64")
    language = job_input.get("language", "Auto")
    
    # Security Check (Optional but recommended)
    expected_secret = os.getenv('TTS_SECRET_PASSWORD')
    if expected_secret and job_input.get("x_tts_secret") != expected_secret:
        return {"error": "Unauthorized"}

    try:
        # Decode the incoming Base64 reference audio
        audio_bytes = base64.b64decode(ref_audio_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
            temp_ref.write(audio_bytes)
            temp_ref_path = temp_ref.name

        # Extract Voice Prompt
        vc_prompt = model.create_voice_clone_prompt(ref_audio=temp_ref_path, ref_text=None)
        
        # Generate Audio
        gen_config = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)
        audio = model.generate(
            text=text, 
            language=None if language == "Auto" else language,
            voice_clone_prompt=vc_prompt,
            speed=1.0,
            generation_config=gen_config
        )
        
        # Convert output to Base64 to send back to your Python backend
        waveform = (audio[0] * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            wavfile.write(temp_out.name, model.sampling_rate, waveform)
            with open(temp_out.name, "rb") as f:
                out_b64 = base64.b64encode(f.read()).decode("utf-8")
                
        # Cleanup
        os.remove(temp_ref_path)
        os.remove(temp_out.name)
        
        # RunPod returns this exact JSON back to your server
        return {"audio_base64": out_b64}

    except Exception as e:
        return {"error": str(e)}

# 2. START THE SERVERLESS LISTENER
runpod.serverless.start({"handler": process_clone})