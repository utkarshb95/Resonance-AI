import time
from config import TRANSCRIPTION_QUEUE, RESPONSE_QUEUE
from response import ContextManager, QuestionDetector, response_worker
import threading

def test_context_deduplication():
    """Test context manager's deduplication logic"""
    cm = ContextManager(window_size=2)
    cm.add_to_context("Hello world")
    cm.add_to_context("Hello world")  # Duplicate
    assert len(cm.context_buffer) == 1
    cm.add_to_context("New entry")
    assert len(cm.context_buffer) == 2

def test_question_detection():
    """Test hybrid question detection"""
    detector = QuestionDetector()
    assert detector._is_question("What is Python?") is True
    assert detector._is_question("I like Python") is False
    assert detector._is_question("What is recursion?") is True, "Question not detected!"

def test_full_workflow():
    """End-to-end test with hardcoded input"""
    # Start worker thread
    worker_thread = threading.Thread(target=response_worker, daemon=True)
    worker_thread.start()

    # Simulate transcription input
    TRANSCRIPTION_QUEUE.put("[SYSTEM]: What is recursion?")
    time.sleep(5)  # Allow processing time

    # Verify response
    assert not RESPONSE_QUEUE.empty(), "RESPONSE_QUEUE is empty!"
    response = RESPONSE_QUEUE.get()
    assert "recursion" in response.lower(), f"Unexpected response: {response}"

    print(f"Generated response: {response}")
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_context_deduplication()
    test_question_detection()
    test_full_workflow()
    print("All tests completed successfully.")