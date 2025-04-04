import logging
import os
from typing import Dict, Any

# Assuming usage of Tavily for web search, as often used with CrewAI/LangChain
# Ensure 'tavily-python' is installed (pip install tavily-python)
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None # Handle optional dependency

# Potential Base class import (if defined)
# from .base import Skill

logger = logging.getLogger(__name__)

# class WebSearchSkill(Skill):
class WebSearchSkill:
    """A skill to perform web searches using the Tavily API."""

    name = "web_search"
    description = "Performs a web search for a given query using the Tavily API and returns the results."

    def __init__(self):
        """Initializes the skill, checking for the Tavily API key."""
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = None
        if not self.api_key:
            logger.warning("TAVILY_API_KEY environment variable not found. WebSearchSkill will be unavailable.")
        elif TavilyClient is None:
             logger.warning("'tavily-python' library not installed. WebSearchSkill will be unavailable. Run 'pip install tavily-python'")
        else:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("TavilyClient initialized successfully for WebSearchSkill.")
            except Exception as e:
                logger.error(f"Failed to initialize TavilyClient: {e}", exc_info=True)
                self.client = None # Ensure client is None if init fails

    def is_available(self) -> bool:
        """Checks if the skill can be used (API key and library available)."""
        return self.client is not None

    def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        """Executes a web search for the given query.

        Args:
            query: The search query string.
            max_results: The maximum number of search results to return.
            **kwargs: Additional keyword arguments (ignored for now, but allows flexibility).

        Returns:
            A string containing the search results, or an error message.
        """
        if not self.is_available():
            return "Error: WebSearchSkill is unavailable. Check TAVILY_API_KEY and tavily-python installation."

        if not query:
            return "Error: No search query provided."

        try:
            logger.info(f"Performing Tavily web search for query: '{query}'")
            # Using search method which returns a dictionary
            response = self.client.search(query=query, search_depth="basic", max_results=max_results)
            # Extract results or format the whole response? Let's format relevant parts.
            results_str = f"Search results for '{query}':\n"
            if 'results' in response and response['results']:
                for i, result in enumerate(response['results']):
                    results_str += f"{i+1}. Title: {result.get('title', 'N/A')}\n   URL: {result.get('url', 'N/A')}\n   Content Snippet: {result.get('content', 'N/A')[:200]}...\n"
            else:
                results_str += "No results found."

            return results_str

        except Exception as e:
            error_msg = f"Error during Tavily web search: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg 