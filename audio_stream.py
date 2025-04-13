import pyaudio
import numpy as np
from config import *
from audio_processing import AudioTranscriber

class AudioStreamManager:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.streams = {}
        self.buffers = {
            "system": np.array([], dtype=np.int16),
            "microphone": np.array([], dtype=np.int16)
        }

    def start_streams(self):
        """Initialize all audio streams"""
        self.streams = {
            "system": self.p.open(
                format=FORMAT,
                channels=CHANNELS_RECORD_SYS,
                rate=RATE,
                input=True,
                input_device_index=SYS_AUDIO_INDEX,
                frames_per_buffer=CHUNK
            ),
            "mic": self.p.open(
                format=FORMAT,
                channels=CHANNELS_RECORD_MIC,
                rate=RATE,
                input=True,
                input_device_index=MIC_AUDIO_INDEX,
                frames_per_buffer=CHUNK
            )
        }
        if PLAYBACK_ENABLED:
            print("Playback enabled.")
            self.streams["playback"] = self.p.open(
                format=FORMAT,
                channels=CHANNELS_PLAYBACK,
                rate=RATE,
                output=True,
                output_device_index=OUTPUT_DEVICE_INDEX,
                frames_per_buffer=CHUNK
            )

    def capture_audio(self):
        """Main capture loop (to be run in a thread)"""
        try:
            while True:
                # Capture system audio
                try:
                    sys_data = self.streams["system"].read(CHUNK)
                except IOError as e:
                    print(f"Error capturing system audio: {e}")
                    continue
                sys_audio = np.frombuffer(sys_data, dtype=np.int16)
                # Convert to mono and add to buffer
                sys_audio_mono = sys_audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                self.buffers["system"] = np.concatenate([self.buffers["system"], sys_audio_mono])

                # Process complete 480-sample chunks from buffer
                while len(self.buffers["system"]) >= CHUNK:
                    chunk = self.buffers["system"][:CHUNK]
                    self.buffers["system"] = self.buffers["system"][CHUNK:]
                    if AudioTranscriber.is_speech(chunk):
                        sys_queue.put(chunk.copy())

                # Capture microphone audio
                try:
                    mic_data = self.streams["mic"].read(CHUNK)
                except IOError as e:
                    print(f"Error capturing microphone audio: {e}")
                    continue
                mic_audio = np.frombuffer(mic_data, dtype=np.int16)
                if len(mic_audio) == CHUNK and AudioTranscriber.is_speech(mic_audio):
                    mic_queue.put(mic_audio.copy())

                # Mix and Playback
                if PLAYBACK_ENABLED:
                    if len(sys_audio_mono) == len(mic_audio):
                        mixed_mono = sys_audio_mono + mic_audio
                        mixed_stereo = np.repeat(mixed_mono, 2)
                        mixed_stereo = np.clip(mixed_stereo, -32768, 32767).astype(np.int16)
                        self.streams["playback"].write(mixed_stereo.tobytes())
        except KeyboardInterrupt:
            print("\n🛑 Capture loop stopped")
            self.shutdown()

    def shutdown(self):
        """Cleanup resources"""
        for stream in self.streams.values():
            stream.stop_stream()
            stream.close()
        self.p.terminate()