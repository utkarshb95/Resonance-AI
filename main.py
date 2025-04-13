import threading
import time
from config import sys_queue, mic_queue
from audio_processing import AudioTranscriber
from audio_stream import AudioStreamManager
from response import response_worker

class MainApp:
    def __init__(self):
        self.stream_manager = AudioStreamManager()
        self.transcriber = AudioTranscriber()
        self._run = True # Control flag for stopping the app

    def start(self):
        """Start audio streams and transcription."""
        try:
            self.stream_manager.start_streams()

            # Start transcription threads for system and microphone queues
            threading.Thread(target=self.transcriber.process_audio, args=(sys_queue, "system"), daemon=True).start()
            threading.Thread(target=self.transcriber.process_audio, args=(mic_queue, "microphone"), daemon=True).start()

            # Start audio capture in the main thread
            capture_thread = threading.Thread(target=self.stream_manager.capture_audio, daemon=True)
            capture_thread.start()

            print("Transcription threads started. Press Ctrl+C to stop.")

            # Start the response worker thread
            threading.Thread(target=response_worker, daemon=True).start()

            while self._run:
                # Main loop to keep the app running
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("Stopping transcription...")
            self.shutdown()
        
    def shutdown(self):
        """Cleanup resources."""
        print("\n🛑 Shutting down...")
        self._run = False
        self.stream_manager.shutdown()
        
        # Signal queues to stop workers
        sys_queue.put(None)
        mic_queue.put(None)

if __name__ == "__main__":
    app = MainApp()
    app.start()
    print("Program stopped.")


    