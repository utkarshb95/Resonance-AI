"""EXPERIMENTAL: unfinished audio enhancement debugger; not used by main.py."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import noisereduce as nr
import msvcrt 
import scipy
import librosa

# Configuration
SAMPLE_RATE = 16000
CHUNK = SAMPLE_RATE * 3  # 3-second chunks
USE_NOISE_REDUCTION = True  # Set to False to disable
MONITOR_RAW = False       # Toggle with R key
MONITOR_PROCESSED = False # Toggle with P key

# Initialize
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK, input_device_index=1)
model = WhisperModel("medium.en", device="cuda", compute_type="float16")

# Audio monitors
raw_monitor = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True, output_device_index=12)
proc_monitor = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True, output_device_index=12)
noise_profile = None

def capture_noise_profile():
    global noise_profile
    print("Recording noise profile (2s)...")
    frames = [stream.read(CHUNK) for _ in range(2)]
    noise_audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    noise_profile = noise_audio
    print("Noise profile captured!")

def enhance_mic_audio(audio_np, sr=16000):
    # Step 1: Dynamic Range Compression
    compressed = np.tanh(audio_np * 2)  # Reduces peakiness
    
    # Step 2: Speech Volume Normalization
    rms = np.sqrt(np.mean(compressed**2))
    if rms < 0.02:  # Only amplify quiet speech
        compressed *= 3.0
    
    # Step 3: High-Frequency Boost (2-4kHz range)
    sos = scipy.signal.butter(4, [2000, 4000], 'bandpass', fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, compressed)
    
    # Prevent clipping
    return np.clip(filtered, -1.0, 1.0)


print("""Controls:
 R - Toggle raw audio monitoring
 P - Toggle processed audio monitoring
 N - Capture new noise profile
 Q - Quit
""")
# Main loop
try:
    while True:
        # Windows-compatible keyboard input
        if msvcrt.kbhit():
            key = msvcrt.getch().decode().lower()
            if key == 'r' and raw_monitor:
                MONITOR_RAW = not MONITOR_RAW
            if key == 'p' and proc_monitor:
                MONITOR_PROCESSED = not MONITOR_PROCESSED
            if key == 'n':
                capture_noise_profile()
            if key == 'q':
                break
            print(f"Raw: {'ON' if MONITOR_RAW else 'OFF'} | Processed: {'ON' if MONITOR_PROCESSED else 'OFF'}")

        # Audio processing
        raw_bytes = stream.read(CHUNK)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        audio_16k = librosa.resample(audio, orig_sr=44100, target_sr=16000, res_type="soxr_hq")

        audio_en = enhance_mic_audio(audio_16k, SAMPLE_RATE)

        if USE_NOISE_REDUCTION and noise_profile is not None:
            processed = nr.reduce_noise(y=audio, y_noise=noise_profile, sr=SAMPLE_RATE)
            proc_bytes = (processed * 32767).astype(np.int16).tobytes()
        elif USE_NOISE_REDUCTION and noise_profile is None:
            print("Noise profile not set. Reducing noise without profile.")
            proc_bytes = processed = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)
            proc_bytes = (processed * 32767).astype(np.int16).tobytes()
        else:
            proc_bytes = raw_bytes

        # Audio monitoring
        if MONITOR_RAW:
            raw_monitor.write(raw_bytes)
        if MONITOR_PROCESSED:
            proc_monitor.write(proc_bytes)

        # Transcription
        segments, _ = model.transcribe(audio_en, vad_filter=True, language="en", 
                                       vad_parameters={"threshold": 0.45,    # Lower = more aggressive speech detection
                                                       "min_speech_duration_ms": 500, #300
                                                       "max_speech_duration_s": 15,
                                                       "speech_pad_ms": 300}, #200
                                       beam_size=5, patience=1.0, condition_on_previous_text=False)
        for seg in segments:
            print(seg.text)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.close()
    if raw_monitor: raw_monitor.close()
    if proc_monitor: proc_monitor.close()
    p.terminate()
