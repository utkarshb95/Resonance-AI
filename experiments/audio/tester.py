"""EXPERIMENTAL: standalone audio transcription prototype; not used by main.py."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import sounddevice as sd
import pyaudio
import numpy as np
import queue
from faster_whisper import WhisperModel

# noise_reduction.py
import noisereduce as nr

class AudioDebugger:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.monitor_stream = None
        self.noise_profile = None
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
    def enable_monitoring(self):
        self.monitor_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=12
        )
        
    def denoise_audio(self, audio_np):
        if self.noise_profile is None:
            return audio_np
        return nr.reduce_noise(
            y=audio_np,
            sr=self.sample_rate,
            y_noise=self.noise_profile,
            prop_decrease=1.0
        )
        
    def capture_noise_profile(self, duration=2):
        print(f"Recording noise profile for {duration} seconds...")
        # Implement noise profile capture if needed
        frames = []
        for _ in range(int(self.sample_rate / 1024 * duration)):
            data = self.input_stream.read(1024)
            frames.append(data)
        
        noise_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        self.noise_profile = noise_audio.astype(np.float32) / 32768.0
        print("Noise profile captured!")



# Parameters
# SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
# CHUNK_DURATION = 3   # seconds per chunk

# # Create a queue to hold audio chunks
# audio_queue = queue.Queue()

# # Callback function to collect audio data
# def audio_callback(indata, frames, time, status):
#     audio_queue.put(indata.copy())

# # Load the model (change model size as needed)
# model = WhisperModel("small.en", device="cuda", compute_type="float16")  # Minimal settings[3][1]

# print("Listening... Press Ctrl+C to stop.")

# try:
#     with sd.InputStream(device=2 ,samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
#         while True:
#             audio_chunk = audio_queue.get()
#             # Convert to float32 numpy array
#             audio_data = audio_chunk.flatten().astype(np.float32)
#             # Transcribe the audio chunk
#             segments, _ = model.transcribe(audio_data, vad_filter=True, language="en")
#             for segment in segments:
#                 print(segment.text)
# except KeyboardInterrupt:
#     print("\nStopped.")

#---------------comparing sounddevice and pyaudio------------------

SAMPLE_RATE = 16000
CHUNK = SAMPLE_RATE * 3  # 5 seconds of audio

def tester():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=1)  # Adjust device index as needed

    model = WhisperModel("medium.en", device="cuda", compute_type="float16")  # Adjust model size as needed
    print("Listening... Press Ctrl+C to stop.")

    debugger = AudioDebugger(sample_rate=SAMPLE_RATE)  # Create an instance of AudioDebugger

    try:
        while True:
            audio_bytes = stream.read(CHUNK)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_denoised = debugger.denoise_audio(audio_np)  # Use the instance to call denoise_audio
            segments, _ = model.transcribe(audio_denoised, vad_filter=True, language="en", vad_parameters=dict(
                                                threshold=0.5,
                                                min_speech_duration_ms=500,
                                                max_speech_duration_s=20))
            for segment in segments:
                print(segment.text)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    tester()
