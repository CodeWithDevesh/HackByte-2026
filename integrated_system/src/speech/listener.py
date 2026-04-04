import speech_recognition as sr
import threading
from src.core.event_bus import shared_event_bus

class VoiceListenerNode:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        # Adjust for ambient noise on startup
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self._running = True
        print("[VOICE] Listener initialized. I'm listening for 'read'...")

    def start(self):
        """Starts the background listening thread."""
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        while self._running:
            try:
                with self.microphone as source:
                    # phrase_time_limit keeps the listening windows short
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
                # Convert Speech to Text (using Google's free API for testing)
                transcript = self.recognizer.recognize_google(audio).lower()
                print(f"[VOICE] Heard: '{transcript}'")

                # Publish to the EventBus (OCR Node is subscribed to this!)
                shared_event_bus.publish("voice_command", transcript)

            except sr.WaitTimeoutError:
                continue # No speech detected, keep looping
            except sr.UnknownValueError:
                continue # Speech was muffled/not understood
            except Exception as e:
                print(f"[VOICE] Error: {e}")

    def stop(self):
        self._running = False