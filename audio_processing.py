import numpy as np
import time
import difflib
from config import (
    WHISPER_MODELS, RATE, MIN_SEGMENT_LENGTH, REQUEST_INTERVAL, OVERLAP, VAD_MODE,
    TRANSCRIPTION_HISTORY, TRANSCRIPTION_QUEUE, DEDUPE_SIMILARITY, MAX_HISTORY_LENGTH, 
)

class AudioTranscriber:
    def __init__(self):
        """Initialize the audio transcriber with necessary parameters."""
        self.last_request_time = {"system": 0, "microphone": 0}
        self.audio_buffers = {"system": np.array([], dtype=np.int16), "microphone": np.array([], dtype=np.int16)}

    @staticmethod
    def deduplicate(new_text, history):
        """Deduplicate transcriptions based on similarity."""
        if not new_text or not history:
            return new_text
        
        last_text = history[-1]
        similarity = difflib.SequenceMatcher(None, new_text.lower(), last_text.lower()).ratio()
        return new_text if similarity < DEDUPE_SIMILARITY else ""
    
    @staticmethod
    def is_speech(audio_chunk):
        """Validate and check for speech in 30ms mono chunks."""
        if len(audio_chunk) != 480:  # 30ms @16kHz
            return False
        # audio_chunk = audio_chunk * 1.5  # 3.5dB gain
        return VAD_MODE.is_speech(
            audio_chunk.astype(np.int16).tobytes(),
            sample_rate=RATE,
            length=len(audio_chunk) # Explicitly state length
        )
    
    @staticmethod
    def transcribe_audio(audio_np, source_name):
        """Transcribe audio data using Whisper model."""
        audio_np = audio_np.astype(np.float32) / 32768.0 # Normalize to float32

        # Convert stereo to mono if needed
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Final validation before transcription to catch VAD false positives (RMS Energy Check) 
        def is_silent(audio_np, threshold=0.02):
            rms = np.sqrt(np.mean(audio_np**2))
            return rms < threshold
        if is_silent(audio_np):     # Skip silent segments
            return ""

        try:
            segments, _ = WHISPER_MODELS[source_name].transcribe(
                audio_np, 
                language="en",
                beam_size=5,
                temperature=0.3,
                word_timestamps=True,
                vad_filter=False,
                repetition_penalty=1.5,
                no_speech_threshold=0.25,
                condition_on_previous_text=True,
                patience=1.5,
                initial_prompt="Focus on computer science terms",
                prefix=TRANSCRIPTION_HISTORY[source_name][-1] if TRANSCRIPTION_HISTORY[source_name] else ""
            )
            return f"[{source_name.upper()}]: {' '.join(seg.text for seg in segments)}"

        except Exception as e:
            print(f"⚠️ {source_name} transcription failed: {str(e)}")
            return ""

    def process_audio(self, queue, source):
        """Process audio data from the queue and transcribe."""
        while True:
            data = queue.get()
            if data is None:  # Exit signal
                break
            
            try:
                self.audio_buffers[source] = np.concatenate([self.audio_buffers[source], data])
                required_samples = int(MIN_SEGMENT_LENGTH * RATE)  # Minimum segment length in samples

                current_time = time.time()
                if (len(self.audio_buffers[source]) >= required_samples and
                        current_time - self.last_request_time[source] >= REQUEST_INTERVAL):

                    segment = self.audio_buffers[source][:required_samples]
                    self.audio_buffers[source] = self.audio_buffers[source][-(required_samples + int(RATE * OVERLAP)):]

                    speech_detected = any(self.is_speech(segment[i:i+480]) for i in range(0, len(segment), 480))
                    if not speech_detected:
                        print(f"⚠️ No speech detected in {source} audio segment.")
                        continue

                    text = self.transcribe_audio(segment, source)
                    self.last_request_time[source] = current_time

                    if text:
                        clean_text = self.deduplicate(text, TRANSCRIPTION_HISTORY[source])
                        TRANSCRIPTION_HISTORY[source].append(clean_text)
                        TRANSCRIPTION_HISTORY[source] = TRANSCRIPTION_HISTORY[source][-MAX_HISTORY_LENGTH:]

                        print(clean_text)

                        # Add to transcription queue for response generation
                        if clean_text and not TRANSCRIPTION_QUEUE.full():
                            TRANSCRIPTION_QUEUE.put(f"[{source.upper()}]: {clean_text}")
                            print(f"✅ Added to queue: {clean_text}")
            finally:
                queue.task_done()