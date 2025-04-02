import os
import re
import json
from typing import Tuple, Dict, Any, List, Optional

from pydantic import BaseModel, Field
from utils.logger import setup_logger

# Try to import language model libraries if available
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class CommandParser(BaseModel):
    """
    Command parser utility for Jarvis
    Uses natural language processing to understand user commands
    """
    
    # Intent patterns for rule-based matching
    intent_patterns: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Flag to use AI for parsing (if available)
    use_ai: bool = True
    
    # OpenAI client
    openai_client: Any = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("command_parser")
        
        # Initialize intent patterns
        self._initialize_intent_patterns()
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.use_ai:
            try:
                self.openai_client = OpenAI()
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")
                self.openai_client = None
    
    def _initialize_intent_patterns(self):
        """Initialize intent patterns for rule-based matching"""
        self.intent_patterns = {
            "system.open": [
                r"open (.*)",
                r"launch (.*)",
                r"start (.*)",
                r"run (.*)"
            ],
            "system.close": [
                r"close (.*)",
                r"exit (.*)",
                r"quit (.*)",
                r"stop (.*)"
            ],
            "file.search": [
                r"find (?:file|files) (.*)",
                r"search (?:for )?(?:file|files) (.*)",
                r"locate (?:file|files) (.*)"
            ],
            "file.open": [
                r"open file (.*)",
                r"show (?:file|content of) (.*)",
                r"read (?:file )?(.*)"
            ],
            "file.edit": [
                r"edit (?:file )?(.*)",
                r"modify (?:file )?(.*)",
                r"change (?:file )?(.*)"
            ],
            "web.search": [
                r"search (?:for|about)? (.*)",
                r"google (.*)",
                r"find information (?:about|on) (.*)"
            ],
            "web.open": [
                r"open (?:website|site|page|url) (.*)",
                r"go to (.*\.(com|org|net|io|gov))",
                r"visit (.*\.(com|org|net|io|gov))"
            ]
        }
    
    def _rule_based_parse(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse command using rule-based matching
        Returns a tuple of (intent, parameters)
        """
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    # Extract parameters from match groups
                    parameters = {}
                    if len(match.groups()) > 0:
                        parameters["target"] = match.group(1)
                    
                    self.logger.info(f"Matched intent: {intent} with parameters: {parameters}")
                    return intent, parameters
        
        return None, {}
    
    def _ai_parse(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse command using AI (OpenAI API)
        Returns a tuple of (intent, parameters)
        """
        if not self.openai_client:
            self.logger.warning("OpenAI client not available, falling back to rule-based parsing")
            return self._rule_based_parse(text)
        
        try:
            # Create system prompt with available intents
            system_prompt = """
            You are a command parser for an AI assistant named Jarvis. 
            Extract the user's intent and parameters from their command.
            
            Available intents:
            - system.open: Open an application
            - system.close: Close an application
            - file.search: Search for files
            - file.open: Open a file
            - file.edit: Edit a file
            - web.search: Search the web
            - web.open: Open a website
            
            Return a JSON object with 'intent' and 'parameters' fields.
            For parameters, extract relevant information like 'target', 'query', etc.
            
            Example:
            User: "open chrome"
            Response: {"intent": "system.open", "parameters": {"target": "chrome"}}
            """
            
            # Get completion from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            intent = result.get("intent", "unknown")
            parameters = result.get("parameters", {})
            
            self.logger.info(f"AI parsed intent: {intent} with parameters: {parameters}")
            return intent, parameters
        
        except Exception as e:
            self.logger.error(f"Error in AI parsing: {e}")
            return self._rule_based_parse(text)
    
    def parse(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a command into intent and parameters
        Returns a tuple of (intent, parameters)
        """
        self.logger.info(f"Parsing command: {text}")
        
        # Try AI parsing first if available
        if self.openai_client and self.use_ai:
            intent, parameters = self._ai_parse(text)
        else:
            # Fall back to rule-based parsing
            intent, parameters = self._rule_based_parse(text)
        
        # If no intent was matched, use a default
        if not intent:
            intent = "unknown"
            parameters = {"text": text}
        
        return intent, parameters 