# app/coqui_tts.py
from TTS.api import TTS
import sounddevice as sd
import numpy as np

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

def speak_text(text):
    if not text.strip():
        return
    
    # 🛑 Prevent kernel-size error for very short inputs
    if len(text.split()) < 3:
        print(f"[TTS] Skipping short text: {text}")
        return

    print(f"[TTS] Speaking: {text}")
    wav = tts.tts(text)
    wav = np.array(wav)
    sd.play(wav, samplerate=22050)
    sd.wait()