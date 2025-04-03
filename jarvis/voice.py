"""
Voice interaction module for Jarvis.
Handles speech recognition and synthesis capabilities.
"""

import os
import logging
from typing import Optional, Callable
from datetime import datetime

# Initialize logger
logger = logging.getLogger("jarvis.voice")

# Check for speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("Speech recognition not available. Install speech_recognition package.")

# Check for speech synthesis
try:
    import pyttsx3
    SPEECH_SYNTHESIS_AVAILABLE = True
except ImportError:
    SPEECH_SYNTHESIS_AVAILABLE = False
    logger.warning("Speech synthesis not available. Install pyttsx3 package.")

class VoiceInteraction:
    """Handles voice input and output for Jarvis."""
    
    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.engine = pyttsx3.init() if SPEECH_SYNTHESIS_AVAILABLE else None
        
        if self.engine:
            # Configure speech synthesis
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        Listen for voice input.
        
        Args:
            timeout: How long to wait for the wake word
            phrase_time_limit: Maximum length of the command phrase
            
        Returns:
            The recognized text, or None if no valid input
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            logger.error("Speech recognition not available")
            return None
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Listen for wake word
                logger.info(f"Listening for wake word: {self.wake_word}")
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    # Check for wake word
                    if self.wake_word in text:
                        # Remove wake word and clean up
                        command = text.replace(self.wake_word, "").strip()
                        logger.info(f"Recognized command: {command}")
                        return command
                    
                except sr.WaitTimeoutError:
                    logger.debug("No wake word detected")
                    return None
                except sr.UnknownValueError:
                    logger.debug("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    logger.error(f"Could not request results: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Error in voice recognition: {e}")
            return None
    
    def speak(self, text: str, interrupt: bool = False) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: The text to speak
            interrupt: Whether to interrupt current speech
            
        Returns:
            True if successful, False otherwise
        """
        if not SPEECH_SYNTHESIS_AVAILABLE:
            logger.error("Speech synthesis not available")
            return False
        
        try:
            if interrupt and self.engine.isBusy():
                self.engine.stop()
            
            self.engine.say(text)
            self.engine.runAndWait()
            return True
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if voice interaction is available."""
        return SPEECH_RECOGNITION_AVAILABLE and SPEECH_SYNTHESIS_AVAILABLE 