import os
import queue
import pyaudio
import webrtcvad
from groq import Groq
from transformers import logging
import faster_whisper

# Initialize the Groq client with the API key from environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Parameters for audio recording
FORMAT = pyaudio.paInt16      # 16-bit format
CHANNELS_RECORD_SYS = 2       # Stereo audio for system recording
CHANNELS_RECORD_MIC = 1       # Mono audio for recording
CHANNELS_PLAYBACK = 2         # Stereo audio for playback
RATE = 16000                  # Sampling rate (16 kHz)
CHUNK = 480                   # Buffer size
SYS_AUDIO_INDEX = 1           # System audio device index (virtual cable)
MIC_AUDIO_INDEX = 2           # Microphone device index
OUTPUT_DEVICE_INDEX = 12      # Output device index
PLAYBACK_ENABLED = False      # Set to True for debugging

# Additional parameters for transcription
MIN_SEGMENT_LENGTH = 3  # Seconds of audio per request
OVERLAP = 1.2           # Overlap in seconds for audio segments
REQUEST_INTERVAL = 1.5  # Seconds between requests (under 20/min)
LAST_REQUEST_TIME = {"system": 0, "microphone": 0}
VAD_AGGRESSIVENESS = 1  # WebRTC VAD filter
VAD_MODE = webrtcvad.Vad(VAD_AGGRESSIVENESS)  # Initialize VAD
DEDUPE_SIMILARITY = 0.65  # Adjusted for better balance

# Initialize models once (outside worker threads)
WHISPER_MODELS = {
    "system": faster_whisper.WhisperModel("small.en", device="cuda", compute_type="float16"),
    "microphone": faster_whisper.WhisperModel("small.en", device="cuda", compute_type="float16")
}

# Post processing parameters
SIMILARITY_THRESHOLD = 0.75  # 75% similarity considered duplicate
MAX_HISTORY_LENGTH = 5       # Keep last 5 transcriptions 
TRANSCRIPTION_HISTORY = {"system": [], "microphone": []}

# Queues for async processing
sys_queue = queue.Queue()
mic_queue = queue.Queue()
TRANSCRIPTION_QUEUE = queue.Queue(maxsize=200)
RESPONSE_QUEUE = queue.Queue(maxsize=20)

logging.set_verbosity_error()  # Suppress warnings



# import appendix
# import os
# import time
# import queue
# import threading
# import tempfile
# import wave
# import re
# from io import BytesIO
# from functools import lru_cache
# from collections import deque

# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import pyaudio
# import keyboard
# import pyperclip
# import webrtcvad
# import difflib

# from groq import Groq
# from openai import OpenAI
# from transformers import pipeline, logging, GPT2TokenizerFast
# from sentence_transformers import SentenceTransformer
# from rapidfuzz import fuzz
# import tensorflow as tf
# import torch
# import faster_whisper