#audio_io.py
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper

ffmpeg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ffmpeg"))
os.environ["PATH"] += os.pathsep + ffmpeg_dir

model = whisper.load_model("base")

def listen_once(duration=5, fs=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write("input.wav", fs, audio)
    result = model.transcribe("input.wav")
    return result["text"]
