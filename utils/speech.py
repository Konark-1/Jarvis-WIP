import speech_recognition as sr
import pyttsx3
import threading
import queue
from typing import Optional, List

from pydantic import BaseModel, Field
from utils.logger import setup_logger

class SpeechRecognizer(BaseModel):
    """Speech recognition utility for Jarvis"""
    
    # Configuration
    energy_threshold: int = 300
    pause_threshold: float = 0.8
    timeout: Optional[float] = None
    phrase_time_limit: Optional[float] = 10
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("speech_recognizer")
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.energy_threshold
        self.recognizer.pause_threshold = self.pause_threshold
        
        # Set up audio source
        self.source = None
    
    def listen(self) -> Optional[str]:
        """Listen for speech and convert to text"""
        text = None
        
        try:
            with sr.Microphone() as source:
                self.logger.info("Listening...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit
                )
                
                self.logger.info("Processing speech...")
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                
                self.logger.info(f"Recognized: {text}")
        except sr.WaitTimeoutError:
            self.logger.info("No speech detected within timeout")
        except sr.UnknownValueError:
            self.logger.info("Could not understand audio")
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from Google Speech Recognition service: {e}")
        except Exception as e:
            self.logger.error(f"Error in speech recognition: {e}")
        
        return text

class TextToSpeech(BaseModel):
    """Text-to-speech utility for Jarvis"""
    
    # Configuration
    rate: int = 150
    volume: float = 1.0
    voice_id: Optional[str] = None
    
    # Note: queue.Queue is not serializable, so it can't be a model field
    # We'll initialize it in __init__ instead
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("text_to_speech")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
        
        # Set voice if specified
        if self.voice_id:
            self.engine.setProperty('voice', self.voice_id)
        
        # Initialize speech queue
        self.speech_queue = queue.Queue()
        
        # Start speech thread
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
    
    def _speech_worker(self):
        """Background worker to process speech queue"""
        while True:
            text = self.speech_queue.get()
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                self.logger.error(f"Error in text-to-speech: {e}")
            finally:
                self.speech_queue.task_done()
    
    def speak(self, text: str):
        """Speak the given text (non-blocking)"""
        self.logger.info(f"Speaking: {text}")
        self.speech_queue.put(text)
    
    def speak_sync(self, text: str):
        """Speak the given text (blocking)"""
        self.logger.info(f"Speaking (sync): {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {e}")
    
    def get_available_voices(self) -> List[str]:
        """Get a list of available voice IDs"""
        return [voice.id for voice in self.engine.getProperty('voices')]
    
    def set_voice(self, voice_id: str):
        """Set the voice by ID"""
        self.voice_id = voice_id
        self.engine.setProperty('voice', voice_id) 