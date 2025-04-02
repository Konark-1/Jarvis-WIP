#!/usr/bin/env python3
"""
Simple Jarvis - A direct API version of the Jarvis AI Assistant

This simplified version bypasses the voice recognition and
text-to-speech systems, providing direct access to Jarvis's
intent parsing and execution capabilities.
"""

import os
import sys
import re
import json
import dotenv
from typing import Dict, Any, Tuple, Optional, List

# Setup environment
dotenv.load_dotenv()

# Simple intent patterns
INTENT_PATTERNS = {
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

class SimpleJarvis:
    """A simplified version of Jarvis with direct API access"""
    
    def __init__(self):
        self.conversation_history = []
        print("Simple Jarvis initialized")
    
    def parse_command(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a command into intent and parameters using rule-based matching"""
        text = text.strip().lower()
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "text": text})
        
        # Check for direct commands
        if text in ["help", "?", "commands"]:
            return "help", {}
        
        # Match against intent patterns
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    # Extract parameters from match groups
                    parameters = {}
                    if len(match.groups()) > 0:
                        parameters["target"] = match.group(1)
                    
                    print(f"Matched intent: {intent} with parameters: {parameters}")
                    return intent, parameters
        
        # No match found
        return "unknown", {"text": text}
    
    def execute_intent(self, intent: str, parameters: Dict[str, Any]) -> str:
        """Execute an intent with parameters"""
        # Implementation for system intents
        if intent == "system.open":
            target = parameters.get("target", "")
            return self._open_application(target)
        elif intent == "system.close":
            target = parameters.get("target", "")
            return self._close_application(target)
            
        # Implementation for file intents
        elif intent == "file.search":
            query = parameters.get("target", "")
            return self._search_files(query)
        elif intent == "file.open":
            filepath = parameters.get("target", "")
            return self._open_file(filepath)
            
        # Implementation for web intents
        elif intent == "web.search":
            query = parameters.get("target", "")
            return self._search_web(query)
        elif intent == "web.open":
            url = parameters.get("target", "")
            return self._open_website(url)
            
        # Help intent
        elif intent == "help":
            return self._show_help()
            
        # Unknown intent
        else:
            return f"I'm not sure how to help with that. Try asking for 'help' to see available commands."
    
    def process_command(self, text: str) -> str:
        """Process a user command and return the response"""
        intent, parameters = self.parse_command(text)
        response = self.execute_intent(intent, parameters)
        
        # Add to conversation history
        self.conversation_history.append({"role": "jarvis", "text": response})
        
        return response
    
    def _open_application(self, app_name: str) -> str:
        """Simulate opening an application"""
        if not app_name:
            return "No application specified"
        return f"Opening {app_name}"
    
    def _close_application(self, app_name: str) -> str:
        """Simulate closing an application"""
        if not app_name:
            return "No application specified"
        return f"Closing {app_name}"
    
    def _search_files(self, query: str) -> str:
        """Simulate searching for files"""
        if not query:
            return "No search query specified"
        return f"Searching for files matching '{query}'"
    
    def _open_file(self, filepath: str) -> str:
        """Simulate opening a file"""
        if not filepath:
            return "No file specified"
        return f"Opening file: {filepath}"
    
    def _search_web(self, query: str) -> str:
        """Simulate searching the web"""
        if not query:
            return "No search query specified"
        return f"Searching the web for '{query}'"
    
    def _open_website(self, url: str) -> str:
        """Simulate opening a website"""
        if not url:
            return "No URL specified"
        
        # Add https:// if missing
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
            
        return f"Opening website: {url}"
    
    def _show_help(self) -> str:
        """Show help information"""
        return """
Available commands:
- Open applications: "open [app_name]", "launch [app_name]"
- Close applications: "close [app_name]"
- Search files: "find files [pattern]", "search for files [pattern]"
- Open files: "open file [path]", "read [filename]"
- Web search: "search for [query]", "google [query]"
- Open websites: "open website [url]", "go to [url]"
- Help: "help", "commands", "?"
        """

def main():
    """Main function to run Simple Jarvis"""
    jarvis = SimpleJarvis()
    
    print("Simple Jarvis is running. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Jarvis: Shutting down.")
                break
                
            response = jarvis.process_command(user_input)
            print(f"Jarvis: {response}")
            
        except KeyboardInterrupt:
            print("\nJarvis: Shutting down.")
            break
        except Exception as e:
            print(f"Jarvis: Sorry, an error occurred: {str(e)}")

if __name__ == "__main__":
    main() 