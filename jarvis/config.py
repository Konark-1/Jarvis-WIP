"""
Centralized configuration management using Pydantic Settings.

Loads configuration from environment variables and/or a .env file.
"""

import logging
import os
from typing import Optional, Dict, Any
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# <<< Import dotenv >>>
from dotenv import load_dotenv

# Use utils logger
from utils.logger import setup_logger

logger = setup_logger(__name__)

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

# <<< Import V2 BaseSettings >>>
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any 

# <<< Import dotenv >>>
from dotenv import load_dotenv

# --- Load .env MANUALLY first --- 
loaded_env = load_dotenv(dotenv_path=DOTENV_PATH, override=True) # Load and override existing env vars
logger.debug(f".env file loaded via dotenv: {loaded_env}")
# -----------------------------

# --- Nested Settings Models (using V2 BaseSettings) --- 
class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore') 
    api_key: Optional[str] = Field(None, description="GROQ API Key")
    default_model: str = Field("llama3-8b-8192", description="Default Groq model")
    # Add other Groq-specific settings if needed
    # e.g., default_model: str = "llama3-8b-8192"

class TavilySettings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore')
    api_key: Optional[str] = Field(None, description="Tavily Search API Key")
    # Add other Tavily-specific settings if needed

# --- Main Settings Class (using V2 BaseSettings) --- 
class Settings(BaseSettings):
    """Main application settings, loaded from .env file."""
    # --- V2 Configuration --- 
    model_config = SettingsConfigDict(
        env_file='.env',         # Specify .env file
        env_file_encoding='utf-8', # Specify encoding
        extra='ignore',           # Ignore extra fields from .env
        env_nested_delimiter='_' # <<< ADD Nested Delimiter
    )
    # ------------------------
    
    # Top-level settings (can be loaded from .env directly)
    log_level: str = Field(default="INFO", description="Logging level")
    # Example: OPENAI_API_KEY: Optional[str] = None (if you add OpenAI)

    # Nested settings models - Instances of BaseSettings
    # No need for alias in nested model if delimiter is used
    groq: GroqSettings = GroqSettings()
    tavily: TavilySettings = TavilySettings()
    # Add other nested settings sections as needed
    # e.g., openai: OpenAISettings = OpenAISettings()
    
    # Removed model_post_init as V2 BaseSettings handles .env loading directly

# --- Global Settings Instance & MANUAL Key Injection --- 
try:
    # Instantiate settings (loads only fields defined above, like log_level)
    settings = Settings() 
    
    # Manually load keys from os.environ (populated by load_dotenv)
    settings.groq.api_key = os.environ.get('GROQ_API_KEY')
    settings.tavily.api_key = os.environ.get('TAVILY_API_KEY')
    
    logger.debug("Global settings instance created; Keys injected manually from env.")
    # Validation check
    groq_key_loaded = bool(settings.groq.api_key)
    tavily_key_loaded = bool(settings.tavily.api_key)
    logger.debug(f"  GROQ Key Loaded: {groq_key_loaded}")
    logger.debug(f"  Tavily Key Loaded: {tavily_key_loaded}")
    missing_keys = []
    if not groq_key_loaded: missing_keys.append("GROQ_API_KEY")
    if not tavily_key_loaded: missing_keys.append("TAVILY_API_KEY")
    if missing_keys:
        error_msg = f"Missing required API key configurations (check .env or environment variables): {', '.join(missing_keys)}"
        logger.error(error_msg)
        raise ValueError(error_msg) 

except Exception as e:
    logger.error(f"CRITICAL ERROR: Failed to load/validate settings: {e}", exc_info=True)
    raise # Re-raise the exception

# Removed redundant instantiation inside try/except

# --- Accessing Settings --- 
# Other modules can now `from jarvis.config import settings`
# and access values like `settings.log_level`, `settings.groq.api_key`, etc.

# Create a single settings instance for the application to use
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