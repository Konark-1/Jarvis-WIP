#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant

Core Jarvis class implementation.
"""

import os
import time
import threading
import dotenv
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

# Import memory components
from .memory.long_term import LongTermMemory
from .memory.medium_term import MediumTermMemory
from .memory.short_term import ShortTermMemory

# Import utility components
from .utils.speech import SpeechRecognizer, TextToSpeech
from .utils.command_parser import CommandParser
from .utils.logger import setup_logger

# Import skill components
from .skills.system_skills import SystemSkills
from .skills.web_skills import WebSkills
from .skills.file_skills import FileSkills

class Jarvis(BaseModel):
    """Main Jarvis class that integrates all components"""
    
    # Memory systems
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    medium_term_memory: MediumTermMemory = Field(default_factory=MediumTermMemory)
    short_term_memory: ShortTermMemory = Field(default_factory=ShortTermMemory)
    
    # Utility components
    speech_recognizer: SpeechRecognizer = Field(default_factory=SpeechRecognizer)
    text_to_speech: TextToSpeech = Field(default_factory=TextToSpeech)
    command_parser: CommandParser = Field(default_factory=CommandParser)
    
    # Skill components
    system_skills: SystemSkills = Field(default_factory=SystemSkills)
    web_skills: WebSkills = Field(default_factory=WebSkills)
    file_skills: FileSkills = Field(default_factory=FileSkills)
    
    # State
    is_listening: bool = Field(default=False)
    is_running: bool = Field(default=False)
    
    # Configuration
    wake_word: str = Field(default="jarvis")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("jarvis")
        self.logger.info("Jarvis initialized")
        
        # Load environment variables
        dotenv.load_dotenv()
        
        # Configure from environment
        self._configure_from_env()
        
    def _configure_from_env(self):
        """Configure Jarvis from environment variables"""
        # Wake word
        env_wake_word = os.environ.get("JARVIS_WAKE_WORD")
        if env_wake_word:
            self.wake_word = env_wake_word.lower()
            
        # Voice settings
        voice_rate = os.environ.get("JARVIS_VOICE_RATE")
        if voice_rate and voice_rate.isdigit():
            self.text_to_speech.rate = int(voice_rate)
            
        voice_volume = os.environ.get("JARVIS_VOICE_VOLUME")
        if voice_volume:
            try:
                self.text_to_speech.volume = float(voice_volume)
            except ValueError:
                pass
            
        # Speech recognition settings
        energy_threshold = os.environ.get("SPEECH_ENERGY_THRESHOLD")
        if energy_threshold and energy_threshold.isdigit():
            self.speech_recognizer.energy_threshold = int(energy_threshold)
            
        pause_threshold = os.environ.get("SPEECH_PAUSE_THRESHOLD")
        if pause_threshold:
            try:
                self.speech_recognizer.pause_threshold = float(pause_threshold)
            except ValueError:
                pass
            
        phrase_time_limit = os.environ.get("SPEECH_PHRASE_TIME_LIMIT")
        if phrase_time_limit and phrase_time_limit.isdigit():
            self.speech_recognizer.phrase_time_limit = int(phrase_time_limit)
        
    def run(self):
        """Start Jarvis"""
        self.is_running = True
        self.logger.info("Starting Jarvis")
        
        # Speak greeting
        self.text_to_speech.speak("Jarvis is now online. How may I assist you?")
        
        # Start listening thread
        threading.Thread(target=self._listen_loop, daemon=True).start()
        
        try:
            # Main loop
            while self.is_running:
                time.sleep(0.1)  # Sleep to prevent high CPU usage
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop Jarvis"""
        self.is_running = False
        self.logger.info("Stopping Jarvis")
        self.text_to_speech.speak("Jarvis shutting down.")
    
    def _listen_loop(self):
        """Background thread for continuous listening"""
        self.is_listening = True
        
        while self.is_listening and self.is_running:
            try:
                # Listen for commands
                text = self.speech_recognizer.listen()
                
                if text:
                    text = text.lower()
                    
                    # Check for wake word
                    if self.wake_word in text:
                        # Remove wake word from text
                        command = text.replace(self.wake_word, "").strip()
                        self._process_command(command)
            except Exception as e:
                self.logger.error(f"Error in listen loop: {e}")
    
    def _process_command(self, command: str):
        """Process a user command"""
        if not command:
            self.text_to_speech.speak("How can I help you?")
            return
        
        # Log command
        self.logger.info(f"Processing command: {command}")
        
        # Add to short-term memory
        self.short_term_memory.add_interaction("user", command)
        
        # Parse command
        intent, parameters = self.command_parser.parse(command)
        
        # Execute command based on intent
        response = self._execute_intent(intent, parameters)
        
        # Add response to short-term memory
        self.short_term_memory.add_interaction("jarvis", response)
        
        # Speak response
        self.text_to_speech.speak(response)
    
    def _execute_intent(self, intent: str, parameters: Dict[str, Any]) -> str:
        """Execute an intent with parameters"""
        # Intent routing
        if intent.startswith("system."):
            return self.system_skills.execute(intent, parameters)
        elif intent.startswith("web."):
            return self.web_skills.execute(intent, parameters)
        elif intent.startswith("file."):
            return self.file_skills.execute(intent, parameters)
        else:
            return "I'm not sure how to help with that."