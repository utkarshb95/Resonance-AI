#!/usr/bin/env python
# coding: utf-8

# ## Initial code to test audio

# In[ ]:


import sounddevice as sd
print(sd.query_devices())  # Find virtual cable's index


# In[ ]:


import sounddevice as sd
import numpy as np

# Define parameters
duration = 5  # seconds
sample_rate = 16000  # Hz

# Record audio
print("Recording...")
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.float32)
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Play the recorded audio
print("Playing back the recorded audio...")
sd.play(audio_data, samplerate=sample_rate)
sd.wait()  # Wait until playback is finished
print("Playback finished.")


# In[ ]:


print("Playing back the recorded audio...")
sd.play(audio_data, samplerate=sample_rate)
sd.wait()  # Wait until playback is finished
print("Playback finished.")


# ## Import dependencies

# In[21]:


import sounddevice as sd
import numpy as np
import queue
from collections import deque
from whispercpp import Whisper
from openai import OpenAI
import soundfile as sf
import os
from io import BytesIO
import keyboard
from groq import Groq
import time
import pyaudio
import wave
import tempfile
import pyperclip
import webrtcvad
import threading
from transformers import pipeline, logging
import tensorflow as tf


# ## Testing local transcription

# In[ ]:


from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


# In[ ]:


from whispercpp import Whisper
whisper_model = Whisper.from_pretrained("base.en")  # Note method change


# In[ ]:


# Initialize Whisper model (tiny.en for low latency)
whisper_model = Whisper("tiny.en")  # Replace with "base.en" or larger models if needed

# Configuration
SAMPLERATE = 16000  # Whisper works best at this rate
BLOCKSIZE = 1024    # 64ms chunks (tweak for lower latency)
DEVICE_ID = 1      # Replace with your virtual cable device index (use sd.query_devices())
audio_queue = queue.Queue(maxsize=10)  # Thread-safe buffer for audio chunks

# Callback function to capture audio chunks
def audio_callback(indata, frames, time, status):
    """Non-blocking audio chunk handler."""
    if status:
        print(f"Audio callback error: {status}")
    audio_queue.put(indata[:, 0].copy())  # Convert to mono and add to queue

# Process audio chunks in real-time using Whisper
def process_audio():
    """Continuously process audio chunks from the queue."""
    print("Starting transcription...")
    while True:
        try:
            # Get an audio chunk from the queue
            chunk = audio_queue.get()
            # Transcribe the chunk using Whisper.cpp
            text = whisper_model.transcribe(chunk.astype(np.float32))
            if text.strip():  # If transcription is not empty
                print(f"Transcribed Text: {text}")
                # Example: Trigger response for specific keywords
                if "memoization" in text.lower():
                    print("[Response] Memoization is an optimization technique...")
        except Exception as e:
            print(f"Error during transcription: {e}")

# Start the real-time audio stream and transcription loop
def start_real_time_transcription():
    """Start the audio stream and transcription."""
    try:
        print("Initializing audio stream...")
        with sd.InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                            device=DEVICE_ID, channels=1, callback=audio_callback):
            process_audio()  # Start processing audio chunks in real-time
    except Exception as e:
        print(f"Error initializing audio stream: {e}")

if __name__ == "__main__":
    print("Starting real-time meeting assistant...")
    start_real_time_transcription()


# ## Testing online API transcription

# In[2]:


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# In[6]:


# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "you are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
# )

# print(chat_completion.choices[0].message.content)


# In[3]:


# Audio recording parameters
CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
VAD_AGGRESSIVENESS = 3  # 1-3 (3=most aggressive noise filtering)
MIN_SPEECH_DURATION = 1.2  # Seconds of speech to trigger processing
PRE_SPEECH_BUFFER = 2.0  # Seconds to capture before speech starts
POST_SPEECH_BUFFER = 1.5  # Seconds to capture after speech ends
MAX_BUFFER_SECONDS = 5 # Maximum buffer size in seconds

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)  # Aggressive noise filtering
audio_queue = queue.Queue()
processing_lock = threading.Lock()
in_speech = False
speech_start = 0 


# In[4]:


processing_lock = threading.Lock()
audio_buffer = np.array([], dtype=np.int16)  # Declare and initialize audio_buffer globally
in_speech = False
speech_start = 0
buffer_start_idx = 0

def record_audio():
    """VAD based audio recording with threading."""
    global audio_buffer, in_speech, speech_start, buffer_start_idx
    
    def callback(in_data, frame_count, time_info, status):
        global audio_buffer, in_speech, speech_start, buffer_start_idx
        
        pcm = np.frombuffer(in_data, dtype=np.int16)

        with processing_lock:
            is_speech = vad.is_speech(pcm.tobytes(), RATE, len(pcm))
            
            if is_speech:
                if not in_speech:
                    # Speech start: track buffer position
                    speech_start = time.time()
                    buffer_start_idx = max(0, len(audio_buffer) - RATE//2)  # 0.5s lookback
                    in_speech = True
                audio_buffer = np.concatenate([audio_buffer, pcm])
            else:
                if in_speech:
                    # Speech end: calculate captured duration
                    captured_seconds = (len(audio_buffer) - buffer_start_idx) / RATE
                    if captured_seconds >= MIN_SPEECH_DURATION:
                        # Capture pre/post speech context
                        pre_samples = min(buffer_start_idx, int(RATE * PRE_SPEECH_BUFFER))
                        post_samples = int(RATE * POST_SPEECH_BUFFER)
                        
                        segment = audio_buffer[
                            buffer_start_idx - pre_samples : 
                            len(audio_buffer) + post_samples
                        ]
                        audio_queue.put(segment.copy())
                        
                    in_speech = False

        print(f"VAD: {'🗣️' if is_speech else '🔇'} | Buffer: {len(audio_buffer)/RATE:.1f}s", end='\r')
        return (None, pyaudio.paContinue)

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                    stream_callback=callback, start=False)
    stream.start_stream()
    return stream

def save_audio(audio_numpy):
    """Save numpy array directly"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f, audio_numpy, RATE, format='WAV', subtype='PCM_16')
        return f.name

def transcribe_audio(audio_numpy):
    """Direct memory-based transcription"""
    # Convert numpy array to bytes buffer
    buffer = BytesIO()
    audio_numpy = audio_numpy.astype(np.float32) / 32768.0 # Normalize to float32
    sf.write(buffer, audio_numpy, RATE, format='WAV', 
             subtype='FLOAT')
    buffer.seek(0)

    try:
        response = client.audio.transcriptions.create(
            file=("audio.wav", buffer.read()),
            model="whisper-large-v3-turbo",  # Verified correct model name
            response_format="text",
            language="en",
            temperature=0.2,
        )
        return response.strip()
    except Exception as e:
        print(f"⚠️ Transcription failed: {str(e)}")
        return ""

def play_audio(filename="recorded_audio.wav"):
    import pyaudio
    import wave

    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()

    # Open stream for playback
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play back data in chunks
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Playback finished.")

def process_transcription(text):
    """Process the transcribed text and generate a response."""
    cleaned = text.lower().strip()[:200] # Limit input length
    if not cleaned:
        return "No speech detected"
    keywords = {
        "memoization": "Memoization is an optimization technique that stores results of expensive function calls.",
        "recursion": "Recursion is a method where a function calls itself to solve smaller instances of a problem.",
        "dynamic programming": "Dynamic programming is a method for solving complex problems by breaking them down into simpler subproblems.",
        "algorithm": "An algorithm is a step-by-step procedure for calculations.",
        "data structure": "A data structure is a particular way of organizing and storing data in a computer.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
        "artificial intelligence": "Artificial intelligence is the simulation of human intelligence in machines.",
        "deep learning": "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "natural language processing": "Natural language processing is a field of AI that focuses on the interaction between computers and humans through natural language.",
        "computer vision": "Computer vision is a field of AI that enables computers to interpret and understand visual information from the world.",
        "reinforcement learning": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.",
        "supervised learning": "Supervised learning is a type of machine learning where the model is trained on labeled data.",
        "unsupervised learning": "Unsupervised learning is a type of machine learning where the model is trained on unlabeled data.",
        "transfer learning": "Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task.",
        "chini": "Chini to pagal hai."
    }
    
    for kw in sorted(keywords.keys(), key=len, reverse=True):
        if kw in cleaned:
            return keywords[kw]
    return f"Command not recognized: {cleaned[:30]}..."

def copy_to_clipboard(text):
    """Copy text to clipboard."""
    pyperclip.copy(text)
    print("Response copied to clipboard!")


# def main():
#     while True:
#         input("Press Enter to start recording...")
#         audio_frames = record_audio()
#         temp_file_path = save_audio(audio_frames)
#         transcription = transcribe_audio(temp_file_path)
#         print("Transcription:", transcription)
#         os.remove(temp_file_path)

# if __name__ == "__main__":
#     main()


# ## Question detector

# In[ ]:


logging.set_verbosity_error()  # Suppress warnings

class QuestionDetector:
    def __init__(self):
        # Lightweight model for real-time use (90MB)
        self.classifier = pipeline(
            "text-classification", 
            model="shahrukhx01/question-vs-statement-classifier"
        )

        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama3-70b-8192"
        
    def detect_question(self, text):
        """Returns True if input text is a question"""
        if not text.strip():
            return False
            
        result = self.classifier(text[:512])  # Trim to model's max length
        return result[0]['label'] == 'LABEL_1' and result[0]['score'] > 0.85
    
    def generate_answer(self, question, context=""):
        """Generate an answer to the question using Groq API"""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": f"""You are an expert interview coach. Generate concise, professional responses using:
                        - STAR method (Situation, Task, Action, Result)
                        - Max 3 sentences
                        - Incorporate keywords: {context}
                        - Maintain a professional tone"""},
                          {"role": "user", "content": f"Suggest response to: {question}"}],
                model=self.model,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer."


# In[ ]:


texts = [
    "What is your greatest strength?", 
    "Tell me about a time you solved a problem.",
    "I worked at XYZ Corp for 5 years.",
    "Why did you leave your previous job?",
    "It was great working with my team."
]
context = "AWS, Python, React, Team Leadership"

qd = QuestionDetector()
for text in texts:
    print(f"Text: {text}")
    if qd.detect_question(text):
        print("Detected as a question.")
        response = qd.generate_answer(text, context)
        print(f"Generated Answer: {response}\n")
    else:
        print("Detected as a statement.")


# ## Main Workflow

# In[ ]:


FULL_TRANSCRIPT = []  # Simple list to store all spoken phrases

def main():
    global FULL_TRANSCRIPT

    # Initialize detector
    qd = QuestionDetector()
    context = "AWS, Python, React, Team Leadership"
    
    stream = record_audio()
    print("🎤 Listening... (Press Ctrl+C to stop)")
    
    try:
        while True:
            try:
                audio_data = audio_queue.get(timeout=0.2)
                
                if audio_data.size >= int(RATE * 0.3):  # Minimal audio threshold
                    # Core pipeline
                    transcription = transcribe_audio(audio_data)
                    # response = process_transcription(transcription)
                    print(f"Text: {transcription}")
                    # if question detected
                    if qd.detect_question(transcription):
                        print("Detected as a question.")
                        response = qd.generate_answer(text, context)    
                        print(f"💬 Generated Response: {response}\n")
                    else:
                        print("Detected as a statement.")
                        response = "Sorry, I couldn't generate an answer."
                    
                    # Simple storage
                    FULL_TRANSCRIPT.append(transcription)
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        
    finally:
        stream.stop_stream()
        stream.close()
        
        # Simple verification
        print("\n=== COMPLETE TRANSCRIPT ===")
        for i, phrase in enumerate(FULL_TRANSCRIPT, 1):
            print(f"{i}. {phrase}")

if __name__ == "__main__":
    main()

