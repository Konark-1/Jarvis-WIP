"""
LLM Integration module for Jarvis
---------------------------------
This module provides integration with Large Language Models,
with a primary focus on Groq API.
"""

from typing import Dict, List, Any, Optional, Union, Type
import os
import json
import time
from pydantic import BaseModel, Field, ConfigDict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import instructor

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

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Keep OpenAI class defined for type hinting even if not available
    class OpenAI: pass
    logger.warning("OpenAI Python SDK not available")

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Keep Anthropic class defined for type hinting even if not available
    class Anthropic: pass
    logger.warning("Anthropic Python SDK not available")

class Message(BaseModel):
    """A message in a conversation with an LLM."""
    role: str  # system, user, assistant
    content: str

class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

class LLMClient(BaseModel):
    """Client for interacting with LLMs."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    primary_provider: str = "groq"  # groq, openai, anthropic
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Default models
    groq_model: str = "llama-3.3-70b-versatile"  # Updated default to 70b
    openai_model: str = "gpt-4o-mini"  # Updated default model
    anthropic_model: str = "claude-3-haiku-20240307"
    
    # Clients
    groq_client: Optional[groq.Client] = None
    openai_client: Optional[OpenAI] = None
    anthropic_client: Optional[Anthropic] = None
    
    # Tokenizer
    tokenizer: Any = None  # To store the tiktoken tokenizer
    
    # Tracking
    last_response: Optional[LLMResponse] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_clients()
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken tokenizer: {e}. Token estimation will be basic.")
            self.tokenizer = None
        
        # Patch clients with instructor if available
        self._patch_clients_with_instructor()
    
    def _patch_clients_with_instructor(self):
        """Applies instructor patch to compatible LLM clients."""
        logger.info("Attempting to patch LLM clients with Instructor...")
        patched_any = False
        try:
            if self.openai_client:
                self.openai_client = instructor.patch(self.openai_client)
                logger.info("OpenAI client patched with Instructor.")
                patched_any = True
            if self.anthropic_client:
                 # Check instructor version compatibility or specific patching method if needed
                 try:
                     self.anthropic_client = instructor.patch(self.anthropic_client)
                     logger.info("Anthropic client patched with Instructor.")
                     patched_any = True
                 except Exception as e:
                      logger.warning(f"Could not patch Anthropic client with Instructor (may require specific setup): {e}")
            if self.groq_client:
                 # Groq might need specific handling or might work if OpenAI compatible
                 try:
                     self.groq_client = instructor.patch(self.groq_client)
                     logger.info("Groq client patched with Instructor.")
                     patched_any = True
                 except Exception as e:
                      logger.warning(f"Could not patch Groq client with Instructor (may require specific setup or be incompatible): {e}")
            
            if not patched_any:
                 logger.warning("Instructor could not patch any available LLM clients.")
                 
        except Exception as e:
             logger.error(f"Error during Instructor client patching: {e}", exc_info=True)
    
    def _initialize_clients(self):
        """Initialize the LLM clients."""
        logger.debug("Attempting to initialize LLM clients...")

        # Try loading API keys from environment if not provided
        if not self.groq_api_key:
            logger.debug("Groq API key not provided directly, attempting to load from env...")
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            logger.debug(f"GROQ_API_KEY loaded from env: {'Exists' if self.groq_api_key else 'Not Found'}")
        else:
            logger.debug("Groq API key provided directly.")

        if not self.openai_api_key:
            logger.debug("OpenAI API key not provided directly, attempting to load from env...")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            logger.debug(f"OPENAI_API_KEY loaded from env: {'Exists' if self.openai_api_key else 'Not Found'}")
        else:
            logger.debug("OpenAI API key provided directly.")

        if not self.anthropic_api_key:
            logger.debug("Anthropic API key not provided directly, attempting to load from env...")
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            logger.debug(f"ANTHROPIC_API_KEY loaded from env: {'Exists' if self.anthropic_api_key else 'Not Found'}")
        else:
            logger.debug("Anthropic API key provided directly.")

        # Initialize Groq
        logger.debug(f"Checking Groq: Key provided? {'Yes' if self.groq_api_key else 'No'}, Library available? {GROQ_AVAILABLE}")
        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                logger.debug("Attempting to initialize Groq client...")
                self.groq_client = groq.Client(api_key=self.groq_api_key)
                logger.info("Groq client initialized successfully.") # Changed log level for success
            except Exception as e:
                logger.error(f"Error initializing Groq client: {e}", exc_info=True) # Added exc_info
                self.groq_client = None # Ensure client is None on error
        else:
            logger.info("Groq client not initialized (key missing or library unavailable).")

        # Initialize OpenAI
        logger.debug(f"Checking OpenAI: Key provided? {'Yes' if self.openai_api_key else 'No'}, Library available? {OPENAI_AVAILABLE}")
        if self.openai_api_key and OPENAI_AVAILABLE:
            try:
                logger.debug("Attempting to initialize OpenAI client...")
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully.") # Changed log level for success
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}", exc_info=True) # Added exc_info
                self.openai_client = None # Ensure client is None on error
        else:
            logger.info("OpenAI client not initialized (key missing or library unavailable).")

        # Initialize Anthropic
        logger.debug(f"Checking Anthropic: Key provided? {'Yes' if self.anthropic_api_key else 'No'}, Library available? {ANTHROPIC_AVAILABLE}")
        if self.anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                logger.debug("Attempting to initialize Anthropic client...")
                self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic client initialized successfully.") # Changed log level for success
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}", exc_info=True) # Added exc_info
                self.anthropic_client = None # Ensure client is None on error
        else:
            logger.info("Anthropic client not initialized (key missing or library unavailable).")

        # Log the available providers
        available_providers = []
        if self.groq_client:
            available_providers.append("groq")
        if self.openai_client:
            available_providers.append("openai")
        if self.anthropic_client:
            available_providers.append("anthropic")

        logger.debug(f"Final check - Initialized clients: groq={bool(self.groq_client)}, openai={bool(self.openai_client)}, anthropic={bool(self.anthropic_client)}")

        if available_providers:
            logger.info(f"Successfully initialized LLM providers: {', '.join(available_providers)}")
        else:
            logger.error("Failed to initialize any LLM providers!") # Changed level to ERROR

        # Validate primary provider
        if self.primary_provider not in available_providers:
            logger.warning(f"Configured primary provider '{self.primary_provider}' is not available.")
            if available_providers:
                self.primary_provider = available_providers[0]
                logger.warning(f"Switching primary provider to the first available: '{self.primary_provider}'.")
            else:
                logger.error("CRITICAL: No LLM providers could be initialized, and no primary provider can be set.")
    
    def get_model_name(self) -> str:
        """Returns the model name configured for the primary provider."""
        if self.primary_provider == "groq":
            return self.groq_model
        elif self.primary_provider == "openai":
            return self.openai_model
        elif self.primary_provider == "anthropic":
            return self.anthropic_model
        else:
            return "unknown_model"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _call_llm(self,
                 provider: str,
                 model: str,
                 messages: List[Message],
                 max_tokens: int = 1000,
                 temperature: float = 0.7) -> LLMResponse:
        """
        Make the actual API call to the LLM provider with retry logic.

        Args:
            provider: The LLM provider to use
            model: The specific model to use
            messages: List of messages for the conversation
            max_tokens: Maximum tokens in the response
            temperature: Temperature for generation (0-1)

        Returns:
            An LLMResponse object

        Raises:
            LLMCommunicationError: If there's an issue communicating with the API.
            LLMTokenLimitError: If the request exceeds token limits.
            ValueError: If the provider is not supported or client not initialized.
            LLMConfigurationError: If the client for the provider is missing.
        """
        start_time = time.time()
        logger.debug(f"Calling LLM provider: {provider}, model: {model}")

        client = None
        if provider == "groq":
            client = self.groq_client
        elif provider == "openai":
            client = self.openai_client
        elif provider == "anthropic":
            client = self.anthropic_client

        if not client:
            logger.error(f"Client for provider {provider} is not initialized.")
            raise LLMConfigurationError(f"Client for provider {provider} is not initialized.")

        try:
            if provider == "groq":
                groq_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
                response = client.chat.completions.create(
                    model=model,
                    messages=groq_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
                finish_reason = response.choices[0].finish_reason

            elif provider == "openai":
                openai_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
                response = client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
                finish_reason = response.choices[0].finish_reason

            elif provider == "anthropic":
                system_prompt = None
                anthropic_messages = []
                for msg in messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                    else:
                        anthropic_messages.append({"role": msg.role, "content": msg.content})

                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=anthropic_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                content = response.content[0].text
                usage = {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens": getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0)
                }
                finish_reason = response.stop_reason
            else:
                # This case should theoretically not be reached due to the initial check
                raise ValueError(f"Provider {provider} not supported.")

            response_time = time.time() - start_time
            logger.debug(f"LLM call successful. Provider: {provider}, Time: {response_time:.2f}s")

            return LLMResponse(
                content=content,
                model=model,
                provider=provider,
                usage=usage,
                metadata={
                    "response_time": response_time,
                    "finish_reason": finish_reason
                }
            )

        # Specific SDK Error Handling
        except (groq.APIError if GROQ_AVAILABLE else Exception) as e:
            response_time = time.time() - start_time
            if provider == "groq":
                 logger.error(f"Groq API Error after {response_time:.2f}s: {e}")
                 # Check for specific Groq error types if needed (e.g., RateLimitError)
                 if isinstance(e, groq.RateLimitError):
                     logger.warning("Groq rate limit exceeded. Consider backoff.")
                 elif isinstance(e, groq.AuthenticationError):
                     logger.error("Groq authentication failed. Check API key.")
                 # Re-raise as our custom error for consistent handling
                 raise LLMCommunicationError(provider, e) from e
            else: # Should not happen if provider matches
                logger.error(f"Caught Groq error for non-Groq provider {provider}? Error: {e}")
                raise LLMCommunicationError(provider, e) from e

        except (openai.APIError if OPENAI_AVAILABLE else Exception) as e:
            response_time = time.time() - start_time
            if provider == "openai":
                logger.error(f"OpenAI API Error after {response_time:.2f}s: {e}")
                # Example specific OpenAI error checks
                if isinstance(e, openai.RateLimitError):
                     logger.warning("OpenAI rate limit exceeded. Consider backoff.")
                elif isinstance(e, openai.AuthenticationError):
                     logger.error("OpenAI authentication failed. Check API key.")
                elif isinstance(e, openai.BadRequestError) and 'context_length_exceeded' in str(e):
                     logger.error("OpenAI context length exceeded.")
                     raise LLMTokenLimitError("OpenAI context length exceeded.") from e
                raise LLMCommunicationError(provider, e) from e
            else:
                logger.error(f"Caught OpenAI error for non-OpenAI provider {provider}? Error: {e}")
                raise LLMCommunicationError(provider, e) from e

        except (anthropic.APIError if ANTHROPIC_AVAILABLE else Exception) as e:
            response_time = time.time() - start_time
            if provider == "anthropic":
                logger.error(f"Anthropic API Error after {response_time:.2f}s: {e}")
                if isinstance(e, anthropic.RateLimitError):
                    logger.warning("Anthropic rate limit exceeded. Consider backoff.")
                elif isinstance(e, anthropic.AuthenticationError):
                    logger.error("Anthropic authentication failed. Check API key.")
                elif isinstance(e, anthropic.BadRequestError) and 'max_tokens' in str(e):
                     logger.error("Anthropic context length potentially exceeded.")
                     # Anthropic might not have a specific token limit error type easily distinguishable
                     raise LLMTokenLimitError("Anthropic request potentially exceeded context limits.") from e
                raise LLMCommunicationError(provider, e) from e
            else:
                logger.error(f"Caught Anthropic error for non-Anthropic provider {provider}? Error: {e}")
                raise LLMCommunicationError(provider, e) from e

        # Catch-all for other unexpected errors during the API call phase
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Unexpected LLM call error after {response_time:.2f}s. Provider: {provider}, Error: {type(e).__name__}: {e}")
            # Re-raise as a generic communication error, but log specifics
            raise LLMCommunicationError(provider, e) from e
    
    def process_with_llm(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         provider: Optional[str] = None,
                         model: Optional[str] = None,
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> str:
        """
        Process a prompt with an LLM.
        
        Args:
            prompt: The user prompt to process
            system_prompt: Optional system prompt for context
            provider: The LLM provider to use (groq, openai, anthropic)
            model: The specific model to use
            max_tokens: Maximum tokens in the response
            temperature: Temperature for generation (0-1)
            
        Returns:
            The LLM response content as a string
        """
        # Use the specified provider or fall back to primary
        provider = provider or self.primary_provider
        
        # Set the appropriate model based on provider
        if model is None:
            if provider == "groq":
                model = self.groq_model
            elif provider == "openai":
                model = self.openai_model
            elif provider == "anthropic":
                model = self.anthropic_model
        
        # Format messages
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        
        response = self._call_llm(
            provider=provider, 
            model=model, 
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Store the response
        self.last_response = response
        
        # Return just the content
        return response.content
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             provider: Optional[str] = None,
             model: Optional[str] = None) -> str:
        """
        Have a chat conversation with the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            provider: The LLM provider to use (groq, openai, anthropic)
            model: The specific model to use
            
        Returns:
            The LLM response content as a string
        """
        # Convert dict messages to Message objects
        formatted_messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        # Use the last user message as the prompt
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        prompt = user_messages[-1]["content"] if user_messages else ""
        
        # Extract system prompt if present
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        system_prompt = system_messages[0]["content"] if system_messages else None
        
        # Call the LLM
        response = self._call_llm(
            provider=provider or self.primary_provider,
            model=model or getattr(self, f"{self.primary_provider}_model"),
            messages=formatted_messages
        )
        
        # Store the response
        self.last_response = response
        
        # Return just the content
        return response.content
    
    def get_token_estimate(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text using tiktoken if available.
        Falls back to a simple heuristic if tiktoken is not available.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed: {e}. Falling back to heuristic.")
                # Fallback heuristic
                return len(text) // 4
        else:
            # Fallback heuristic
            return len(text) // 4

    def get_langchain_llm(self, provider: Optional[str] = None) -> Optional[Any]:
        """Returns the initialized LangChain-compatible LLM client object for the specified provider.

        Args:
            provider: The LLM provider ("groq", "openai", "anthropic"). 
                      Defaults to self.primary_provider.

        Returns:
            The initialized client object (e.g., groq.Client, openai.OpenAI) or None if unavailable.
        """
        target_provider = provider or self.primary_provider
        client = None
        if target_provider == "groq":
            client = self.groq_client
        elif target_provider == "openai":
            client = self.openai_client
        elif target_provider == "anthropic":
            client = self.anthropic_client
        
        if not client:
            logger.warning(f"LangChain-compatible client requested for provider '{target_provider}', but it is not initialized or available.")
            return None
            
        # CrewAI often expects specific LangChain wrappers (e.g., ChatGroq, ChatOpenAI)
        # We might need to wrap the raw client here, or ensure CrewAI handles the raw SDK client.
        # For now, return the raw client, assuming CrewAI/LangChain might handle it or 
        # specific wrappers might be needed later.
        logger.debug(f"Providing LangChain LLM client instance for '{target_provider}'")
        return client

    def process_structured_output(self,
                                  prompt: str,
                                  system_prompt: Optional[str] = None,
                                  provider: Optional[str] = None,
                                  model: Optional[str] = None,
                                  max_tokens: int = 1000,
                                  temperature: float = 0.7,
                                  response_model: Optional[Type[BaseModel]] = None) -> Union[str, BaseModel]:
        """Processes a prompt with the primary LLM, optionally enforcing a Pydantic model.

        Args:
            prompt: The user prompt.
            system_prompt: An optional system prompt.
            provider: Override the primary provider.
            model: Override the default model for the provider.
            max_tokens: Max tokens for the response.
            temperature: Generation temperature.
            response_model: Optional Pydantic model to enforce on the response.

        Returns:
            The LLM response content as a string, or an instance of response_model if provided.

        Raises:
            LLMConfigurationError: If no provider is available.
            LLMCommunicationError: If API communication fails.
            LLMTokenLimitError: If token limits are hit.
            ValueError: If response_model is provided but instructor patching failed.
            Exception: For other unexpected errors.
        """
        target_provider = provider or self.primary_provider
        
        if target_provider == "groq":
            target_model = model or self.groq_model
        elif target_provider == "openai":
            target_model = model or self.openai_model
        elif target_provider == "anthropic":
            target_model = model or self.anthropic_model
        else:
            logger.error(f"Unsupported provider specified: {target_provider}")
            raise LLMConfigurationError(f"Unsupported provider: {target_provider}")

        # Construct messages
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        # --- Call LLM --- 
        # Use the internal _call_llm method, potentially adapted for instructor
        logger.info(f"Processing prompt with {target_provider}/{target_model}. Response model: {response_model.__name__ if response_model else 'None'}")

        try:
            # Determine the correct client
            client = None
            if target_provider == "groq": client = self.groq_client
            elif target_provider == "openai": client = self.openai_client
            elif target_provider == "anthropic": client = self.anthropic_client

            if not client:
                 raise LLMConfigurationError(f"Client for provider {target_provider} not initialized.")

            # Prepare API messages format
            api_messages = [{"role": m.role, "content": m.content} for m in messages]

            if response_model:
                 # Check if the client was successfully patched
                 # This check assumes instructor adds a specific attribute or method, adjust if needed
                 # A simple check might be if the object's class name contains 'Instructor'
                 client_type_str = str(type(client))
                 if "instructor" not in client_type_str.lower() and not hasattr(client, '_instructor_is_patched'): # Basic check
                      logger.error(f"Instructor response_model requested, but client '{target_provider}' appears unpatched.")
                      raise ValueError(f"Cannot use response_model with unpatched {target_provider} client.")
                 
                 # Use the patched client's create method with response_model
                 structured_response = client.chat.completions.create(
                     model=target_model,
                     messages=api_messages,
                     max_tokens=max_tokens,
                     temperature=temperature,
                     response_model=response_model # Pass the model to instructor
                 )
                 # Store minimal response info (usage might not be standard)
                 self.last_response = LLMResponse(
                      content=str(structured_response), # Store Pydantic model as string? Or handle separately?
                      model=target_model,
                      provider=target_provider,
                      usage={}, # Usage info might be different/absent
                      metadata={'response_type': 'structured', 'model_name': response_model.__name__}
                 )
                 return structured_response
                 
            else:
                 # Standard call without response_model (use existing _call_llm for consistency?)
                 # Replicating _call_llm logic here for simplicity, could refactor later
                 response = client.chat.completions.create(
                     model=target_model,
                     messages=api_messages,
                     max_tokens=max_tokens,
                     temperature=temperature
                 )
                 content = response.choices[0].message.content
                 usage = getattr(response, 'usage', None)
                 usage_dict = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                 } if usage else {}
                 finish_reason = response.choices[0].finish_reason

                 self.last_response = LLMResponse(
                      content=content,
                      model=target_model,
                      provider=target_provider,
                      usage=usage_dict,
                      metadata={'finish_reason': finish_reason}
                 )
                 return content

        # --- Error Handling --- (Simplified from _call_llm for this method)
        except (groq.APIConnectionError, openai.APIConnectionError, anthropic.APIConnectionError) as e:
            raise LLMCommunicationError(target_provider, e)
        except (groq.RateLimitError, openai.RateLimitError, anthropic.RateLimitError) as e:
            logger.warning(f"Rate limit exceeded for {target_provider}. Retrying...")
            raise LLMCommunicationError(target_provider, e) # Let retry handle
        except (groq.APIStatusError, openai.APIStatusError, anthropic.APIStatusError) as e:
             # Handle specific errors like token limits if possible
             # Example for OpenAI:
             if hasattr(e, 'code') and e.code == 'context_length_exceeded':
                  logger.error(f"Token limit likely exceeded for {target_provider}: {e}")
                  raise LLMTokenLimitError(f"Context length exceeded: {e}")
             else:
                  logger.error(f"API Error from {target_provider}: {e}")
                  raise LLMCommunicationError(target_provider, e)
        except Exception as e:
            logger.exception(f"Unexpected error during LLM call ({target_provider}): {e}")
            raise LLMError(f"Unexpected LLM error: {e}") 