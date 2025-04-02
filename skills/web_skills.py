import os
import webbrowser
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus, urlparse

from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

from utils.logger import setup_logger

class WebSkills(BaseModel):
    """Web skills for Jarvis to interact with the internet"""
    
    # Default search engine
    default_search_engine: str = "google"
    
    # Search engine URLs
    search_engines: Dict[str, str] = {
        "google": "https://www.google.com/search?q={}",
        "bing": "https://www.bing.com/search?q={}",
        "duckduckgo": "https://duckduckgo.com/?q={}",
        "youtube": "https://www.youtube.com/results?search_query={}"
    }
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("web_skills")
    
    def execute(self, intent: str, parameters: Dict[str, Any]) -> str:
        """Execute a web-related intent"""
        self.logger.info(f"Executing web intent: {intent} with parameters: {parameters}")
        
        if intent == "web.search":
            return self.search_web(parameters.get("target", ""))
        elif intent == "web.open":
            return self.open_website(parameters.get("target", ""))
        else:
            return f"Unknown web intent: {intent}"
    
    def search_web(self, query: str, search_engine: Optional[str] = None) -> str:
        """Search the web using the specified search engine"""
        if not query:
            return "No search query specified"
        
        self.logger.info(f"Searching web for: {query}")
        
        try:
            # Use specified search engine or default
            engine = search_engine or self.default_search_engine
            engine = engine.lower()
            
            # Get search URL
            if engine in self.search_engines:
                search_url = self.search_engines[engine].format(quote_plus(query))
            else:
                search_url = self.search_engines[self.default_search_engine].format(quote_plus(query))
            
            # Open in browser
            webbrowser.open(search_url)
            
            return f"Searching for '{query}' using {engine.capitalize()}"
        
        except Exception as e:
            self.logger.error(f"Error searching web for {query}: {e}")
            return f"Failed to search for '{query}': {str(e)}"
    
    def open_website(self, url: str) -> str:
        """Open a website in the default web browser"""
        if not url:
            return "No URL specified"
        
        self.logger.info(f"Opening website: {url}")
        
        try:
            # Validate and format URL
            parsed = urlparse(url)
            if not parsed.scheme:
                # Add https:// if no scheme specified
                url = "https://" + url
            
            # Open in browser
            webbrowser.open(url)
            
            return f"Opening {url}"
        
        except Exception as e:
            self.logger.error(f"Error opening website {url}: {e}")
            return f"Failed to open '{url}': {str(e)}"
    
    def get_webpage_content(self, url: str) -> str:
        """Get the text content of a webpage"""
        if not url:
            return "No URL specified"
        
        self.logger.info(f"Getting content from: {url}")
        
        try:
            # Validate and format URL
            parsed = urlparse(url)
            if not parsed.scheme:
                # Add https:// if no scheme specified
                url = "https://" + url
            
            # Send request
            headers = {
                "User-Agent": "Jarvis AI Assistant"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > 2000:
                text = text[:2000] + "...\n[Content truncated]"
            
            return text
        
        except Exception as e:
            self.logger.error(f"Error getting content from {url}: {e}")
            return f"Failed to get content from '{url}': {str(e)}" 