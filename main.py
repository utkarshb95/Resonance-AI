import threading
from config import sys_queue, mic_queue
from audio_processing import AudioTranscriber

if __name__ == "__main__":
    transcriber = AudioTranscriber()

    # Start transcription threads for system and microphone queues
    threading.Thread(target=transcriber.process_audio, args=(sys_queue, "system"), daemon=True).start()
    threading.Thread(target=transcriber.process_audio, args=(mic_queue, "microphone"), daemon=True).start()

    print("Transcription threads started. Press Ctrl+C to stop.")