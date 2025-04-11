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
import torch
import difflib
import faster_whisper
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# print(sd.query_devices())


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

# Additional parameters for transcription
MIN_SEGMENT_LENGTH = 3  # Seconds of audio per request
REQUEST_INTERVAL = 1.5 # Seconds between requests (under 20/min)
LAST_REQUEST_TIME = {"system": 0, "microphone": 0}
VAD_AGGRESSIVENESS = 1  # WebRTC VAD filter

# Post processing parameters
SIMILARITY_THRESHOLD = 0.75  # 75% similarity considered duplicate
MAX_HISTORY_LENGTH = 5       # Keep last 5 transcriptions 
TRANSCRIPTION_HISTORY = {"system": [], "microphone": []}

# Queues for async processing
sys_queue = queue.Queue()
mic_queue = queue.Queue()

# Deduplicate function
DEDUPE_WINDOW = 5  # Compare last 5 words for duplicates
DEDUPE_SIMILARITY = 0.65  # 65% similarity = duplicate
def deduplicate(new_text, history):
    """Remove redundant phrases using sliding window comparison"""
    if not new_text or not history:
        return new_text
    
    # Split into words
    new_words = new_text.split()
    last_words = history[-1].split()[-DEDUPE_WINDOW:]  # Last N words of previous transcript
    
    # Check for overlapping prefix
    overlap = 0
    for i in range(1, min(len(new_words), len(last_words)) + 1):
        if new_words[:i] == last_words[-i:]:
            overlap = i
    
    # Remove overlapping part
    return " ".join(new_words[overlap:]) if overlap > 0 else new_text


