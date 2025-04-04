"""
LLM Integration module for Jarvis
---------------------------------
This module provides integration with Large Language Models,
with a primary focus on Groq API.
"""

from typing import Dict, List, Any, Optional, Union, Type, TYPE_CHECKING
import os
import json
import time
from pydantic import BaseModel, Field, ConfigDict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import instructor
from jarvis.config import settings

# <<< ADDED TYPE_CHECKING block >>>
if TYPE_CHECKING:
    from jarvis.planning import PlanningSystem, Task
    from jarvis.execution import ExecutionSystem
    from jarvis.memory.unified_memory import UnifiedMemorySystem
    from jarvis.skills.registry import SkillRegistry
# <<< END TYPE_CHECKING block >>>

from utils.logger import setup_logger

# Initialize logger EARLY
logger = logging.getLogger("jarvis.llm")

# --- LangChain Imports --- (Simplified for Groq)
try:
    from langchain_groq import ChatGroq
    LANGCHAIN_GROQ_AVAILABLE = True
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False
    ChatGroq = None # Define as None if not available
    logger.warning("Langchain Groq integration not available. Install with 'pip install langchain-groq'")

# Remove OpenAI/Anthropic langchain imports
# try:
#     from langchain_openai import ChatOpenAI
#     LANGCHAIN_OPENAI_AVAILABLE = True
# except ImportError:
#     LANGCHAIN_OPENAI_AVAILABLE = False
#     ChatOpenAI = None
# 
# try:
#     from langchain_anthropic import ChatAnthropic
#     LANGCHAIN_ANTHROPIC_AVAILABLE = True
# except ImportError:
#     LANGCHAIN_ANTHROPIC_AVAILABLE = False
#     ChatAnthropic = None

# Define BaseChatModel for type hinting
try:
    from langchain_core.language_models import BaseChatModel
except ImportError:
    BaseChatModel = Any # Fallback if langchain_core is not available

# Custom Exceptions
class LLMError(Exception):
    """Base exception for LLM related errors."""
    pass

class LLMConfigurationError(LLMError):
    """Error related to LLM client configuration."""
    pass

class LLMCommunicationError(LLMError):
    """Error during communication with the LLM API."""
    def __init__(self, provider: str, original_exception: Exception):
        self.provider = provider
        self.original_exception = original_exception
        super().__init__(f"Error communicating with {provider}: {original_exception}")

class LLMTokenLimitError(LLMError):
    """Error related to exceeding token limits."""
    pass

# Initialize logger
logger = logging.getLogger("jarvis.llm")

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq Python SDK not available. Install with 'pip install groq'")

# Remove OpenAI/Anthropic SDK imports
# try:
#     import openai
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False
#     class OpenAI: pass
#     logger.warning("OpenAI Python SDK not available")
# 
# try:
#     import anthropic
#     from anthropic import Anthropic
#     ANTHROPIC_AVAILABLE = True
# except ImportError:
#     ANTHROPIC_AVAILABLE = False
#     class Anthropic: pass
#     logger.warning("Anthropic Python SDK not available")

class Message(BaseModel):
    """A message in a conversation with an LLM."""
    role: str  # system, user, assistant
    content: str

class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    usage: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

class LLMClient(BaseModel):
    """Client for interacting with LLMs (Groq Only)."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    primary_provider: str = "groq" # Fixed to groq
    groq_api_key: Optional[str] = None
    # Remove OpenAI/Anthropic fields
    # openai_api_key: Optional[str] = None
    # anthropic_api_key: Optional[str] = None
    
    # Default models
    groq_model: str = "llama3-8b-8192" # Switch default to 8b
    # Remove OpenAI/Anthropic defaults
    # openai_model: str = "gpt-4o-mini"
    # anthropic_model: str = "claude-3-haiku-20240307"
    
    # Clients
    groq_client: Optional[groq.Client] = None
    # Remove OpenAI/Anthropic clients
    # openai_client: Optional[OpenAI] = None
    # anthropic_client: Optional[Anthropic] = None
    
    # Tokenizer
    tokenizer: Any = None
    
    # Tracking
    last_response: Optional[LLMResponse] = None
    available_clients: Dict[str, bool] = Field(default_factory=dict)
    logger: logging.Logger
    
    def __init__(self, **data):
        # Ensure logger exists before Pydantic validation
        if 'logger' not in data:
            from utils.logger import setup_logger
            data['logger'] = setup_logger("jarvis.llm")
            
        # <<< REMOVED: Import settings inside __init__ >>>
        # from jarvis.config import settings 
        
        # <<< ADDED: Set fields directly from settings >>>
        data['groq_api_key'] = settings.groq.api_key
        data['groq_model'] = settings.groq.default_model
        data['primary_provider'] = "groq" # Keep fixed for now

        super().__init__(**data)
        
        # Post-Pydantic initialization steps
        self._initialize_clients()
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Could not initialize tiktoken tokenizer: {e}. Token estimation will be basic.")
            self.tokenizer = None
        
        # Patch clients with instructor if available
        # self._patch_clients_with_instructor() # <<< TEMPORARILY COMMENTED OUT

        # Check if Groq client was successfully initialized
        self.available_clients = {"groq": self.groq_client is not None}
        
        if not self.available_clients["groq"]:
            # Error logged in _initialize_clients if key missing
            # Re-raise here if initialization failed for other reasons
            if not self.groq_client:
                 raise LLMConfigurationError("Groq LLM provider initialization failed (client object is None).")
        else:
            self.logger.info(f"LLMClient initialized successfully. Available providers: groq")
    
    def _patch_clients_with_instructor(self):
        """Applies instructor patch to the Groq client."""
        self.logger.info("Attempting to patch Groq client with Instructor...")
        try:
            if self.groq_client:
                # Groq might need specific handling or might work if OpenAI compatible
                try:
                    self.groq_client = instructor.patch(self.groq_client)
                    self.logger.info("Groq client patched with Instructor.")
                except Exception as e:
                    self.logger.warning(f"Could not patch Groq client with Instructor (may require specific setup): {e}")
        except Exception as e:
            self.logger.error(f"Error during instructor patching: {e}", exc_info=True)

    def _initialize_clients(self):
        """Initialize the Groq client."""
        # Only initialize Groq
        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.groq_client = groq.Client(api_key=self.groq_api_key)
                self.logger.info("Groq client initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
                # Keep self.groq_client as None
        elif not GROQ_AVAILABLE:
            self.logger.warning("Groq SDK not installed, cannot initialize Groq client.")
        elif not self.groq_api_key:
            self.logger.error("Groq API key not found (check GROQ_API_KEY env var or .env). Cannot initialize Groq client.")
            # Raise error? Yes, critical.
            raise LLMConfigurationError("Groq API key not found.")

    def _prepare_messages(self, messages: List[Union[Message, Dict]]) -> List[Dict]:
        """Ensures messages are in the dictionary format expected by APIs."""
        prepared_messages = []
        
        # <<< ADDED: Input type check >>>
        if not isinstance(messages, list):
            self.logger.error(f"_prepare_messages received non-list input: type={type(messages)}, value={messages}")
            return [] # Return empty list to prevent errors downstream
            
        for msg in messages:
            if isinstance(msg, Message):
                prepared_messages.append(msg.model_dump())
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                prepared_messages.append(msg)
            else:
                self.logger.warning(f"Skipping invalid message format: {msg}")
        return prepared_messages

    def estimate_tokens(self, text: Union[str, List[Dict]]) -> int:
        """Estimates the number of tokens for a given text or message list."""
        if not self.tokenizer:
            # Basic estimation if tokenizer is unavailable
            return len(str(text).split())

        try:
            if isinstance(text, str):
                return len(self.tokenizer.encode(text))
            elif isinstance(text, list):
                # Estimate tokens for a list of message dictionaries
                # Based on OpenAI's cookbook example
                num_tokens = 0
                for message in text:
                    num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                    for key, value in message.items():
                        if value:
                             num_tokens += len(self.tokenizer.encode(str(value)))
                        if key == "name":  # If there's a name, the role is omitted
                            num_tokens -= 1  # Role is always required and always 1 token
                num_tokens += 2  # Every reply is primed with <im_start>assistant

                return num_tokens
            else:
                return len(str(text).split()) # Fallback
        except Exception as e:
            self.logger.warning(f"Token estimation failed: {e}. Falling back to basic word count.")
            return len(str(text).split())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def process_with_llm(self, 
                         prompt: Optional[str] = None, # Added for compatibility 
                         messages: Optional[List[Union[Message, Dict]]] = None, 
                         model: Optional[str] = None, 
                         system_prompt: Optional[str] = None, # Added for compatibility
                         max_tokens: Optional[int] = 1000, # Added for compatibility
                         temperature: Optional[float] = 0.7, # Added for compatibility
                         **kwargs) -> Union[str, LLMResponse]: # Return string for compatibility
        """Invokes the Groq LLM with retry logic. (Renamed from invoke)
           Added parameters for compatibility with ExecutionSystem calls.
        """
        
        provider = "groq"
        effective_model = model or self.groq_model
        client = self.groq_client
        prepared_messages: List[Dict] = [] # Initialize prepared_messages

        if not client:
            raise LLMConfigurationError(f"Groq client is not available or configured.")

        # Determine message source
        if messages is not None:
            # Use provided messages list directly
            self.logger.debug("Using provided messages list for LLM call.")
            prepared_messages = self._prepare_messages(messages)
        elif prompt is not None:
            # Construct messages from prompt and system_prompt
            self.logger.debug("Constructing messages from prompt/system_prompt for LLM call.")
            constructed_messages = []
            if system_prompt:
                 constructed_messages.append(Message(role="system", content=system_prompt))
            constructed_messages.append(Message(role="user", content=prompt))
            prepared_messages = self._prepare_messages(constructed_messages)
        else:
             # No valid input provided
             raise ValueError("Either 'prompt' or 'messages' must be provided.")
             
        # Ensure prepared_messages is not empty before proceeding
        if not prepared_messages:
             logger.error("Message preparation resulted in an empty list. Check input prompt/messages.")
             # Returning an empty string or raising might be appropriate depending on caller expectation
             # For now, let the API call fail with Groq's error message.
             # Or raise LLMError("Prepared message list is empty.")
             pass # Let Groq API handle the empty list error for now

        try:
            start_time = time.time()
            # --- Direct synchronous call to Groq --- 
            groq_params = {
                "messages": prepared_messages,
                "model": effective_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            self.logger.debug(f"Calling Groq sync with params: {groq_params}")
            api_response = client.chat.completions.create(**groq_params)
            # ---------------------------------------
            duration = time.time() - start_time
            
            content = api_response.choices[0].message.content
            usage = api_response.usage.model_dump() if api_response.usage else {}
            metadata = {"duration_ms": int(duration * 1000)}
            
            response_obj = LLMResponse(
                content=content,
                model=effective_model,
                provider=provider,
                usage=usage,
                metadata=metadata
            )
            self.last_response = response_obj
            # Return only content string for compatibility with callers
            return response_obj.content 
            # return response_obj # Original return
        except Exception as e:
            logger.error(f"Error invoking {provider} model {effective_model}: {e}", exc_info=True)
            raise LLMCommunicationError(provider, e)

    async def _call_groq(self, client: groq.Client, messages: List[Dict], model: str, **kwargs) -> Any:
        """Makes the actual API call to Groq."""
        # Add default parameters or handle kwargs specific to Groq
        params = {
            "messages": messages,
            "model": model,
            **kwargs # Allow overriding defaults
        }
        self.logger.debug(f"Calling Groq with params: {params}")
        response = await client.chat.completions.create(**params)
        return response

    # Remove _call_openai and _call_anthropic methods

    def get_token_limit(self, model_name: Optional[str] = None) -> int:
        """Gets the approximate context window size (token limit) for a given model (Groq only)."""
        # Simplifed for Groq, using known limits. Refine as needed.
        model = model_name or self.groq_model
        if "llama3-70b" in model:
            return 8192
        elif "llama3-8b" in model:
            return 8192
        elif "mixtral-8x7b" in model:
            return 32768 # Groq's Mixtral limit
        elif "gemma-7b" in model:
            return 8192
        else:
            self.logger.warning(f"Unknown token limit for Groq model: {model}. Returning default 8192.")
            return 8192 # Default fallback 