import os
import re
import threading
from queue import Empty
from collections import OrderedDict
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from transformers import pipeline
from rapidfuzz import fuzz
from config import TRANSCRIPTION_QUEUE, RESPONSE_QUEUE

class ContextManager:
    def __init__(self, window_size=5):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.context_buffer = []
        self.window_size = window_size
        self.min_similarity = 0.85

    def add_to_context(self, text):
        """Add text to context with deduplication."""
        if not text.strip():
            return

        cleaned = self._clean_text(text)
        new_embed = self.encoder.encode(cleaned)

        # Batch similarity check for efficiency
        if self.context_buffer:
            embeddings = np.array([embed for _, embed in self.context_buffer])
            similarities = np.dot(embeddings, new_embed) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(new_embed)
            )
            if np.max(similarities) > self.min_similarity:
                return

        self.context_buffer.append((cleaned, new_embed))
        self.context_buffer = self.context_buffer[-self.window_size:]

    def get_recent_context(self, num=3):
        """Retrieve the most recent text only from the context buffer."""
        return [entry[0] for entry in self.context_buffer[-num:]] 

    @staticmethod
    def _clean_text(text):
        """Basic text normalization"""
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

class QuestionDetector:
    def __init__(self):
        self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.context = ContextManager()
        
        # Hybrid detection models
        self.local_classifier = pipeline(
            "text-classification",
            model="shahrukhx01/question-vs-statement-classifier"
        )
        self.fallback_model = "openai/gpt-oss-20b"
        self.main_model = "openai/gpt-oss-120b"
        
        # Response cache
        self.response_cache = OrderedDict()
        self.cache_threshold = 0.85  # Fuzzy match threshold
        self.cache_lock = threading.Lock() # Thread-safe cache access

    def process_input(self, text, speaker):
        """Main processing pipeline"""
        self.context.add_to_context(f"{speaker}: {text}")
        
        if self._is_question(text):
            return self.generate_answer(text)
        return None

    def _is_question(self, text):
        """Hybrid question detection"""
        # Fast local check
        local_result = self.local_classifier(text[:512])[0]
        print(f"🔍 Local classifier result: {local_result}")

        if local_result['score'] > 0.9:
            is_question = local_result['label'] == 'LABEL_1'
            print(f"🏷️ Detected as {'Question' if is_question else 'Statement'}")
            return local_result['label'] == 'LABEL_1'
            
        # LLM verification for edge cases
        print("🟡 Using LLM verification")
        return self._llm_question_verification(text)

    def _llm_question_verification(self, text):
        context_text = " ".join(self.context.get_recent_context())
        prompt = f"""Context: {context_text}
        Is this a question needing response? Text: {text}
        Answer only Yes/No:"""
        
        try:
            response = self.groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.fallback_model,
                max_tokens=1,
                temperature=0
            )
            return "yes" in response.choices[0].message.content.lower()
        except Exception as e:
            print(f"⚠️ Groq verification failed: {e}")
            return False

    def generate_answer(self, question):
        """Generate answer with caching and fallback"""
        # Thread-safe cache access
        with self.cache_lock:
            cached = self._get_cached_response(question)
            if cached:
                return cached
            
        try:
            print(f"🔎 Generating answer")
            answer = self._generate_groq_answer(question)
            print(f"✅ Generated answer")
            return answer
        except Exception as e:
            print(f"Groq error: {e}")
            return self._generate_fallback_answer(question)

    def _generate_groq_answer(self, question):
        """Main answer generation"""
        try:
            context_text = " ".join(self.context.get_recent_context())
            prompt = f"""Interview context: {context_text}
            Question: {question}
            Provide concise answer using STAR method (2-3 sentences):"""
            
            response = self.groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.main_model,
                temperature=0.3,
                max_tokens=150
            )
            answer = response.choices[0].message.content.strip()

            # Return the generated answer
            return answer
        except Exception as e:
            print(f"⚠️ Groq answer generation failed: {e}")
            raise e

    def _generate_fallback_answer(self, question):
        """Local fallback for reliability"""
        return "Could you please rephrase that question? I want to ensure I understand it completely."

    def _get_cached_response(self, query):
        """Check for similar cached queries"""

        for cached_query in self.response_cache:
            if fuzz.ratio(query.lower(), cached_query.lower()) > self.cache_threshold:
                return self.response_cache[cached_query]
        return None

    def _add_to_cache(self, question, answer):
        """Manage cache storage"""

        with self.cache_lock:
            if len(self.response_cache) > 100: # Limit cache size
                self.response_cache.popitem(last=False)
            self.response_cache[question] = answer

def response_worker():
    detector = QuestionDetector()
    while True:
        try:
            transcript_entry = TRANSCRIPTION_QUEUE.get(timeout=1)
            print(f"📥 Retrieved from TRANSCRIPTION_QUEUE")
            if transcript_entry is None:
                break
            
            # Parse "[SYSTEM]: ..." format
            speaker_part, _, text = transcript_entry.partition(']: ')
            speaker = speaker_part.lstrip('[').lower()
            
            if response := detector.process_input(text, speaker):
                print(f"❓ Detected question")
                RESPONSE_QUEUE.put(response)
                print(f"✅ Added to RESPONSE_QUEUE")
            else:
                print(f"🔄 No question detected")
            
            TRANSCRIPTION_QUEUE.task_done()
                
        except Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Response worker error: {e}")
            TRANSCRIPTION_QUEUE.task_done()

def response_consumer():
    """Consume responses from the RESPONSE_QUEUE."""
    while True:
        try:
            response = RESPONSE_QUEUE.get(timeout=1)
            if response is None:
                break
            
            # Simulate sending the response to a UI or another system
            print(f"💬 AI Response: {response}")
            RESPONSE_QUEUE.task_done()
            
        except Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Response consumer error: {e}")
            RESPONSE_QUEUE.task_done()
