"""
LLM Integration module for Jarvis
---------------------------------
This module provides integration with Large Language Models,
with a primary focus on Groq API.
"""

from typing import Dict, List, Any, Optional, Union
import os
import json
import time
from pydantic import BaseModel, Field
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

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
    primary_provider: str = "groq"  # groq, openai, anthropic
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Default models
    groq_model: str = "llama3-8b-8192"  # Default to Llama 3 8B
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
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_clients()
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken tokenizer: {e}. Token estimation will be basic.")
            self.tokenizer = None
    
    def _initialize_clients(self):
        """Initialize the LLM clients."""
        # Try loading API keys from environment if not provided
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize Groq
        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                self.groq_client = groq.Client(api_key=self.groq_api_key)
                logger.info("Groq client initialized")
            except Exception as e:
                logger.error(f"Error initializing Groq client: {e}")
        
        # Initialize OpenAI
        if self.openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
        
        # Initialize Anthropic
        if self.anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")
        
        # Log the available providers
        available_providers = []
        if self.groq_client:
            available_providers.append("groq")
        if self.openai_client:
            available_providers.append("openai")
        if self.anthropic_client:
            available_providers.append("anthropic")
        
        if available_providers:
            logger.info(f"Available LLM providers: {', '.join(available_providers)}")
        else:
            logger.warning("No LLM providers available!")
        
        # Validate primary provider
        if self.primary_provider not in available_providers:
            if available_providers:
                self.primary_provider = available_providers[0]
                logger.warning(f"Primary provider not available. Switching to {self.primary_provider}")
            else:
                logger.error("No LLM providers available!")
    
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
        """
        start_time = time.time()
        logger.debug(f"Calling LLM provider: {provider}, model: {model}")
        
        try:
            if provider == "groq" and self.groq_client:
                # Convert messages to Groq format
                groq_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
                
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=groq_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content
                
                # Prepare usage data
                usage = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                # Record response time
                response_time = time.time() - start_time
                
                logger.debug(f"LLM call successful. Provider: {provider}, Time: {response_time:.2f}s")
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider="groq",
                    usage=usage,
                    metadata={
                        "response_time": response_time,
                        "finish_reason": response.choices[0].finish_reason
                    }
                )
                
            elif provider == "openai" and self.openai_client:
                # Convert messages to OpenAI format
                openai_messages = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
                
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                content = response.choices[0].message.content
                
                # Prepare usage data
                usage = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                # Record response time
                response_time = time.time() - start_time
                
                logger.debug(f"LLM call successful. Provider: {provider}, Time: {response_time:.2f}s")
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider="openai",
                    usage=usage,
                    metadata={
                        "response_time": response_time,
                        "finish_reason": response.choices[0].finish_reason
                    }
                )
                
            elif provider == "anthropic" and self.anthropic_client:
                # Convert messages to Anthropic format
                # Anthropic requires system prompt separately and messages alternate user/assistant
                system_prompt = None
                anthropic_messages = []
                for msg in messages:
                    if msg.role == "system":
                        system_prompt = msg.content
                    else:
                        # Ensure alternating user/assistant roles if needed,
                        # though typically input is just system (optional) + user
                        anthropic_messages.append({"role": msg.role, "content": msg.content})
                
                response = self.anthropic_client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=anthropic_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                content = response.content[0].text
                
                # Record response time
                response_time = time.time() - start_time
                
                logger.debug(f"LLM call successful. Provider: {provider}, Time: {response_time:.2f}s")
                
                return LLMResponse(
                    content=content,
                    model=model,
                    provider="anthropic",
                    usage={
                        # Anthropic uses input/output tokens
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    },
                    metadata={
                        "response_time": response_time,
                        "finish_reason": response.stop_reason
                    }
                )
                
            else:
                logger.error(f"Provider {provider} not supported or client not initialized.")
                raise ValueError(f"Provider {provider} not supported or client not initialized.")
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"LLM call failed after {response_time:.2f}s. Provider: {provider}, Error: {e}")
            # Re-raise the exception to allow tenacity to handle retries
            raise ValueError(f"LLM call failed for provider {provider}: {e}") from e
    
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