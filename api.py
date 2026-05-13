import os
import sys
import torch
import uuid
import re
import scipy.io.wavfile as wavfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form,Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
# OmniVoice Imports
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from google.colab import userdata
import tempfile
from faster_whisper import WhisperModel

app = FastAPI(title="OmniVoice API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🛡️ UNIVERSAL SECURITY CHECK ---
# --- 🛡️ STRICT SECURITY CHECK ---

# --- 🛡️ STRICT SECURITY CHECK ---
def verify_request(secret_header: str):
    # Fetch from the OS environment, NOT colab userdata
    expected_secret = os.getenv('TTS_SECRET_PASSWORD')

    # 1. Check if the environment variable is missing
    if not expected_secret:
        print("🚨 SERVER ERROR: TTS_SECRET_PASSWORD environment variable is missing!")
        raise HTTPException(status_code=500, detail="Server security is misconfigured.")
    
    # 2. Check the user's header against the expected secret
    if not secret_header or secret_header != expected_secret:
        print(f"🚨 Blocked Unauthorized Access. Header received: {secret_header}")
        raise HTTPException(status_code=401, detail="Unauthorized GPU access. Invalid TTS Password.")
# Initialize Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    load_asr=True,
)
sampling_rate = model.sampling_rate
TEMP_DIR = "./Omni_Audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Dialect mapping from original app.py
DIALECT_MAP = {
    "Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话",
    "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话",
    "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话",
    "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话",
}

def get_safe_filename(text, language):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip().replace(" ", "_")[:20] or "audio"
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    return f"{TEMP_DIR}/{clean_text}_{lang}_{uuid.uuid4().hex[:8]}.wav"

@app.post("/voice-design")
async def voice_design(
    text: str = Form(...),
    language: str = Form("Auto"),
    gender: str = Form("Female"),
    age: str = Form("Young Adult"),
    pitch: str = Form("Moderate Pitch"),
    style: str = Form("Auto"),
    accent: str = Form("Auto"),
    dialect: str = Form("Auto"),
    speed: float = Form(1.0),
    steps: int = Form(32),
    x_tts_secret: str = Header(None) # ✨ Add this parameter
):
    verify_request(x_tts_secret)
    # Construct instructions from attributes[cite: 1]
    attributes = [gender, age, pitch, style, accent, dialect]
    instruct = ", ".join([DIALECT_MAP.get(a, a) for a in attributes if a and a != "Auto"])
    
    gen_config = OmniVoiceGenerationConfig(num_step=steps, guidance_scale=2.0)
    
    try:
        audio = model.generate(
            text=text, 
            language=None if language == "Auto" else language,
            instruct=instruct if instruct else None,
            speed=speed,
            generation_config=gen_config
        )
        
        waveform = (audio[0] * 32767).astype(np.int16)
        output_path = get_safe_filename(text, language)
        wavfile.write(output_path, sampling_rate, waveform)
        
        return FileResponse(output_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import traceback

@app.post("/voice-clone")
async def voice_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(None),
    speed: float = Form(1.0),
    steps: int = Form(32),
    x_tts_secret: str = Header(None)
):
    try:
        verify_request(x_tts_secret)
        
        # 1. Save reference audio
        unique_id = uuid.uuid4().hex[:8]
        temp_ref_path = f"{TEMP_DIR}/ref_{unique_id}_{ref_audio.filename}"
        with open(temp_ref_path, "wb") as buffer:
            shutil.copyfileobj(ref_audio.file, buffer)
            
        print(f"📥 [COLAB] Received file: {temp_ref_path}")

        # 2. Extract Clone Prompt
        # Logic: Use provided text, otherwise let ASR (which must be loaded) handle it
        safe_ref_text = ref_text.strip() if (ref_text and ref_text.strip()) else None
        
        vc_prompt = model.create_voice_clone_prompt(
            ref_audio=temp_ref_path, 
            ref_text=safe_ref_text
        )
        
        # 3. Generate Cloned Voice
        gen_config = OmniVoiceGenerationConfig(num_step=steps, guidance_scale=2.0)
        audio = model.generate(
            text=text, 
            language=None if language == "Auto" else language,
            voice_clone_prompt=vc_prompt,
            speed=speed,
            generation_config=gen_config
        )
        
        # 4. Save and Return
        waveform = (audio[0] * 32767).astype(np.int16)
        output_path = get_safe_filename(text, language)
        wavfile.write(output_path, model.sampling_rate, waveform)
        
        return FileResponse(output_path, media_type="audio/wav")
        
    except Exception as e:
        # 🚀 THIS IS THE KEY: It prints the error in RED in Colab
        print("🔥 [COLAB CRASH DETECTED]")
        traceback.print_exc() 
        # Return the actual error to your Python backend
        raise HTTPException(status_code=500, detail=f"Colab Model Error: {str(e)}")
    finally:
        if 'temp_ref_path' in locals() and os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)

# 👇 Load the model into the GPU (device="cuda") using float16 for maximum speed
print("⏳ Loading Whisper Small model into Colab GPU...")
# whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ GPU Whisper Model Loaded!")

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...), 
    language: str = Form("en")
):
    # 1. Save the incoming audio to the Colab disk temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_path = temp_audio.name

    try:
        # 2. Transcribe using the GPU
        segments, _ = whisper_model.transcribe(temp_path, language=language, beam_size=5)
        full_text = "".join([segment.text for segment in segments])
        
        return {"text": full_text.strip()}

    except Exception as e:
        print(f"Colab Whisper Error: {e}")
        return {"text": ""}
        
    finally:
        # 3. Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)