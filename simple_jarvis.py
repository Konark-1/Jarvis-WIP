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
import subprocess
import platform
import webbrowser
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
        self.system = platform.system()
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
        """Open an application by name"""
        if not app_name:
            return "No application specified"
        
        app_name = app_name.lower()
        print(f"Attempting to open application: {app_name}")
        
        try:
            # Windows-specific application handling
            if self.system == "Windows":
                common_apps = {
                    "chrome": "chrome",
                    "google chrome": "chrome",
                    "firefox": "firefox",
                    "edge": "msedge",
                    "microsoft edge": "msedge",
                    "word": "winword",
                    "excel": "excel",
                    "powerpoint": "powerpnt",
                    "notepad": "notepad",
                    "calculator": "calc",
                    "file explorer": "explorer",
                    "explorer": "explorer",
                    "cmd": "cmd",
                    "command prompt": "cmd",
                    "powershell": "powershell",
                    "visual studio": "devenv",
                    "vs code": "code",
                    "visual studio code": "code"
                }
                
                if app_name in common_apps:
                    app_name = common_apps[app_name]
                
                subprocess.Popen(f"start {app_name}", shell=True)
                return f"Opening {app_name}"
            
            # macOS-specific application handling
            elif self.system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
                return f"Opening {app_name}"
            
            # Linux-specific application handling
            elif self.system == "Linux":
                subprocess.Popen([app_name])
                return f"Opening {app_name}"
            
            else:
                return f"Unsupported operating system: {self.system}"
        
        except Exception as e:
            print(f"Error opening application {app_name}: {e}")
            return f"Failed to open {app_name}: {str(e)}"
    
    def _close_application(self, app_name: str) -> str:
        """Close an application by name"""
        if not app_name:
            return "No application specified"
        
        app_name = app_name.lower()
        print(f"Attempting to close application: {app_name}")
        
        try:
            # Windows-specific application handling
            if self.system == "Windows":
                # Map common names to process names
                common_apps = {
                    "chrome": "chrome.exe",
                    "google chrome": "chrome.exe",
                    "firefox": "firefox.exe",
                    "edge": "msedge.exe",
                    "microsoft edge": "msedge.exe",
                    "word": "winword.exe",
                    "excel": "excel.exe",
                    "powerpoint": "powerpnt.exe",
                    "notepad": "notepad.exe",
                    "calculator": "calc.exe",
                    "cmd": "cmd.exe",
                    "command prompt": "cmd.exe",
                    "powershell": "powershell.exe",
                    "visual studio": "devenv.exe",
                    "vs code": "code.exe",
                    "visual studio code": "code.exe"
                }
                
                process_name = common_apps.get(app_name, f"{app_name}.exe")
                subprocess.run(f"taskkill /f /im {process_name}", shell=True)
                return f"Closing {app_name}"
            
            # macOS-specific application handling
            elif self.system == "Darwin":
                subprocess.run(["killall", app_name])
                return f"Closing {app_name}"
            
            # Linux-specific application handling
            elif self.system == "Linux":
                subprocess.run(["pkill", app_name])
                return f"Closing {app_name}"
            
            else:
                return f"Unsupported operating system: {self.system}"
        
        except Exception as e:
            print(f"Error closing application {app_name}: {e}")
            return f"Failed to close {app_name}: {str(e)}"
    
    def _search_files(self, query: str) -> str:
        """Search for files matching a pattern"""
        if not query:
            return "No search query specified"
        
        print(f"Searching for files: {query}")
        
        try:
            # Use current directory as default
            directory = os.getcwd()
            
            # Search pattern
            if self.system == "Windows":
                cmd = f'dir /s /b "*{query}*"'
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if stderr:
                    return f"Error searching for files: {stderr.decode('utf-8', errors='ignore')}"
                
                results = stdout.decode('utf-8', errors='ignore').strip().split('\n')
                if not results or results[0] == '':
                    return f"No files found matching '{query}'"
                
                # Limit results for display
                display_results = results[:10]
                result_text = "\n".join(display_results)
                
                if len(results) > 10:
                    result_text += f"\n...and {len(results) - 10} more files"
                
                return f"Found {len(results)} files matching '{query}':\n{result_text}"
            else:
                return f"File search not fully implemented for {self.system}"
        
        except Exception as e:
            print(f"Error searching files: {e}")
            return f"Failed to search for files: {str(e)}"
    
    def _open_file(self, filepath: str) -> str:
        """Open a file with the system default application"""
        if not filepath:
            return "No file specified"
        
        print(f"Opening file: {filepath}")
        
        try:
            if self.system == "Windows":
                os.startfile(os.path.abspath(filepath))
            elif self.system == "Darwin":  # macOS
                subprocess.call(['open', filepath])
            else:  # Linux
                subprocess.call(['xdg-open', filepath])
                
            return f"Opening file: {filepath}"
        except Exception as e:
            print(f"Error opening file: {e}")
            return f"Failed to open file: {str(e)}"
    
    def _search_web(self, query: str) -> str:
        """Search the web using the default search engine"""
        if not query:
            return "No search query specified"
        
        print(f"Searching web for: {query}")
        
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return f"Searching the web for '{query}'"
        except Exception as e:
            print(f"Error searching web: {e}")
            return f"Failed to search the web: {str(e)}"
    
    def _open_website(self, url: str) -> str:
        """Open a website in the default web browser"""
        if not url:
            return "No URL specified"
        
        print(f"Opening website: {url}")
        
        try:
            # Add https:// if missing
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
                
            webbrowser.open(url)
            return f"Opening website: {url}"
        except Exception as e:
            print(f"Error opening website: {e}")
            return f"Failed to open website: {str(e)}"
    
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