"""
Centralized configuration management using Pydantic Settings.

Loads configuration from environment variables and/or a .env file.
"""

import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class GroqSettings(BaseSettings):
    """Groq API Configuration"""
    # Pydantic V2 requires explicit model_config for prefix/env_file
    model_config = SettingsConfigDict(env_prefix='GROQ_', case_sensitive=False, extra='ignore')

    api_key: Optional[str] = None
    default_model: str = "llama3-8b-8192" # Use 8b as default

class TavilySettings(BaseSettings):
    """Tavily API Configuration"""
    model_config = SettingsConfigDict(env_prefix='TAVILY_', case_sensitive=False, extra='ignore')

    api_key: Optional[str] = None

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
    """Main application settings, aggregating others."""
    # Load from .env file if it exists
    # Note: Ensure python-dotenv is installed (`pip install python-dotenv pydantic-settings`)
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # API Keys / LLM Provider Settings
    # No prefix needed here as they are nested models
    groq: GroqSettings = GroqSettings()
    tavily: TavilySettings = TavilySettings()
    # Add other providers like OpenAI, Anthropic here if re-enabled

    # Jarvis Behavior Settings
    jarvis: JarvisSettings = JarvisSettings()

    # Speech Recognition Settings
    speech: SpeechSettings = SpeechSettings()

    # Other potential top-level settings
    log_level: str = "INFO"
    debug_mode: bool = False

# Single instance to be imported by other modules
try:
    settings = Settings()
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