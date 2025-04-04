"""
Centralized configuration management using Pydantic Settings.

Loads configuration from environment variables and/or a .env file.
"""

import logging
import os
from typing import Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, __version__ as PYDANTIC_VERSION
from pathlib import Path

logger = logging.getLogger(__name__)

# <<< REMOVED Debugging >>>
# print(f"[DEBUG config.py] Checking environment variables before Settings definition:")
# print(f"[DEBUG config.py] os.environ.get('GROQ_API_KEY'): {os.environ.get('GROQ_API_KEY')}")
# print(f"[DEBUG config.py] os.environ.get('TAVILY_API_KEY'): {os.environ.get('TAVILY_API_KEY')}")
# <<< END Debugging >>>

# <<< Calculate project root and .env path >>>
# Assumes config.py is in jarvis/ folder
PROJECT_ROOT = Path(__file__).parent.parent 
DOTENV_PATH = PROJECT_ROOT / '.env'
print(f"[DEBUG config.py] Expecting .env file at: {DOTENV_PATH}")

class GroqSettings(BaseSettings):
    """Groq API Configuration"""
    api_key: Optional[str] = None
    default_model: str = "llama3-8b-8192"
    model_config = SettingsConfigDict(extra='ignore')

class TavilySettings(BaseSettings):
    """Tavily API Configuration"""
    api_key: Optional[str] = None
    model_config = SettingsConfigDict(extra='ignore')

class JarvisSettings(BaseSettings):
    """Jarvis Application Configuration"""
    model_config = SettingsConfigDict(env_prefix='JARVIS_', case_sensitive=False, extra='ignore')

    wake_word: str = "jarvis"
    voice_rate: int = 150
    voice_volume: float = 1.0

class SpeechSettings(BaseSettings):
    """Speech Recognition Settings"""
    model_config = SettingsConfigDict(env_prefix='SPEECH_', case_sensitive=False, extra='ignore')

    energy_threshold: int = 300
    pause_threshold: float = 0.8
    phrase_time_limit: int = 10

class Settings(BaseSettings):
    """Main configuration class using Pydantic BaseSettings."""
    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # --- Load top-level keys directly from .env ---
    GROQ_API_KEY: Optional[str] = None 
    TAVILY_API_KEY: Optional[str] = None
    # ----------------------------------------------

    # --- Nested settings objects (initialized post-load) ---
    groq: Optional[GroqSettings] = None
    tavily: Optional[TavilySettings] = None
    # -----------------------------------------------------

    # General App Settings
    log_level: str = "INFO"
    workspace_sandbox_path: str = "./workspace_sandbox"
    # Add other settings as needed
    
    # <<< ADDED model_post_init >>>
    # Use model_post_init for Pydantic v2
    def model_post_init(self, __context: Any) -> None:
        """Initialize nested settings after main settings are loaded."""
        print("[DEBUG config.py post_init] Initializing nested settings...")
        print(f"[DEBUG config.py post_init] Loaded GROQ_API_KEY: {self.GROQ_API_KEY is not None}")
        print(f"[DEBUG config.py post_init] Loaded TAVILY_API_KEY: {self.TAVILY_API_KEY is not None}")
        
        self.groq = GroqSettings(api_key=self.GROQ_API_KEY)
        self.tavily = TavilySettings(api_key=self.TAVILY_API_KEY)
        
        print(f"[DEBUG config.py post_init] Nested groq.api_key set: {self.groq.api_key is not None}")
        print(f"[DEBUG config.py post_init] Nested tavily.api_key set: {self.tavily.api_key is not None}")

# Create a single settings instance for the application to use
settings = Settings()
print(f"[DEBUG config.py] Global settings instance created. groq set: {settings.groq is not None}, tavily set: {settings.tavily is not None}")

# Single instance to be imported by other modules
try:
    # settings = Settings() # <<< REMOVE redundant instantiation
    # Log loaded settings (excluding sensitive keys)
    sensitive_keys = {'api_key'}
    def log_settings(config_model, prefix=""):
        for key, value in config_model.model_dump().items():
            full_key = f"{prefix}{key}"
            if isinstance(value, BaseSettings):
                log_settings(value, f"{full_key}.")
            elif key.lower() not in sensitive_keys:
                logger.debug(f"Config Loaded: {full_key} = {value}")
            else:
                logger.debug(f"Config Loaded: {full_key} = ******")
    # log_settings(settings) # Uncomment for verbose config logging on startup
except Exception as e:
    logger.error(f"CRITICAL: Failed to load settings: {e}", exc_info=True)
    # Handle error appropriately, maybe exit or use defaults
    settings = Settings() # Attempt to use defaults 