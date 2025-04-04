import logging
import os
from typing import Dict, Any, List, Optional

# Assuming usage of Tavily for web search, as often used with CrewAI/LangChain
# Ensure 'tavily-python' is installed (pip install tavily-python)
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None # Handle optional dependency

# Import base class and result model
from jarvis.skills.base import Skill, SkillResult
from jarvis.memory.unified_memory import MemoryEntry
from jarvis.state import JarvisState
from jarvis.config import settings # <<< MOVE import back to top level
# from jarvis.constants import MAX_RETRIEVED_KNOWLEDGE_ITEMS # <<< REMOVE this import

logger = logging.getLogger(__name__)

class WebSearchSkill(Skill):
    """A skill to perform web searches using the Tavily API."""

    # Use a class variable for the client to avoid re-initialization
    # This assumes API key doesn't change during runtime
    tavily_client: Optional[TavilyClient] = None
    api_key_status: str = "Not Checked"

    def __init__(self, tavily_client: Optional[TavilyClient] = None):
        """Initializes the Tavily client if not already done."""
        # Only initialize if not already initialized or failed permanently
        if WebSearchSkill.api_key_status not in ["Initialized", "Missing"]:
            # <<< REMOVED: Import settings inside __init__ >>>
            # from jarvis.config import settings 
            
            api_key = settings.tavily.api_key
            if not api_key:
                logger.warning("Tavily API key not found in settings. WebSearchSkill will be disabled.")
                WebSearchSkill.api_key_status = "Missing"
            else:
                # API key exists, try to initialize
                try:
                    WebSearchSkill.tavily_client = TavilyClient(api_key=api_key)
                    WebSearchSkill.api_key_status = "Initialized"
                    logger.info("Tavily client initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize Tavily client: {e}", exc_info=True)
                    WebSearchSkill.api_key_status = f"Initialization Error: {e}"
        # <<< REMOVED invalid second 'else' block >>>
        # Log current status regardless of initialization attempt in this call
        logger.debug(f"WebSearchSkill __init__ check complete. Status: {WebSearchSkill.api_key_status}")

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web using Tavily API to answer questions or find information. Use this when you need up-to-date information or knowledge beyond your internal data."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "required": True, "description": "The search query string."},
            {"name": "max_results", "type": "integer", "required": False, "default": 5, "description": "Maximum number of search results to return."},
            {"name": "search_depth", "type": "string", "required": False, "default": "basic", "description": "Search depth ('basic' or 'advanced'). Advanced costs more."}
            # Add other parameters supported by TavilyClient.search if needed
        ]

    def execute(self, **kwargs: Any) -> SkillResult:
        """Executes a web search for the given query.

        Args:
            **kwargs: Must include 'query'. Can include 'max_results' (default 5).

        Returns:
            A SkillResult object containing the search results or an error.
        """
        if WebSearchSkill.api_key_status == "Missing":
            return SkillResult(success=False, error="WebSearchSkill is unavailable. Check TAVILY_API_KEY and tavily-python installation.")

        # Use base class validation
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return SkillResult(success=False, error=f"Parameter validation failed: {validation_error}")

        query = kwargs.get('query')
        max_results = kwargs.get('max_results', 5)
        search_depth = kwargs.get('search_depth', 'basic')

        # Redundant check, but safe
        if not query:
            return SkillResult(success=False, error="No search query provided.")

        try:
            logger.info(f"Performing Tavily web search for query: '{query}', max_results={max_results}, depth={search_depth}")
            # Using search method which returns a dictionary
            response = WebSearchSkill.tavily_client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                # Add other parameters from kwargs if needed, e.g.:
                # include_answer=kwargs.get('include_answer', False),
                # include_raw_content=kwargs.get('include_raw_content', False),
                # include_images=kwargs.get('include_images', False)
            )

            # Format results
            results_list = []
            if 'results' in response and response['results']:
                for result in response['results']:
                    results_list.append({
                        "title": result.get('title', 'N/A'),
                        "url": result.get('url', 'N/A'),
                        "content": result.get('content', 'N/A')
                    })

            message = f"Found {len(results_list)} results for query '{query}'."
            return SkillResult(success=True, message=message, data={'query': query, 'results': results_list})

        except Exception as e:
            error_msg = f"Error during Tavily web search: {e}"
            logger.error(error_msg, exc_info=True)
            return SkillResult(success=False, error=error_msg) 