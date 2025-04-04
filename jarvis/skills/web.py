import logging
import requests # Requires 'pip install requests beautifulsoup4'
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Literal

from jarvis.skills.base import Skill, SkillResult

logger = logging.getLogger("jarvis.skills.web")

class WebSkill(Skill):
    """Provides web browsing and search capabilities."""

    @property
    def name(self) -> str:
        return "web"

    @property
    def description(self) -> str:
        return "Skills for interacting with the web: browsing specific URLs or performing web searches."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "action",
                "type": "string",
                "required": True,
                "description": "The action to perform: 'browse' or 'search'.",
                "enum": ["browse", "search"] # Helps validation and LLM understanding
            },
            {
                "name": "url",
                "type": "string",
                "required": False, # Required only if action is 'browse'
                "description": "The URL to browse (required for 'browse' action)."
            },
            {
                "name": "query",
                "type": "string",
                "required": False, # Required only if action is 'search'
                "description": "The search query (required for 'search' action)."
            },
            {
                "name": "num_results",
                "type": "integer",
                "required": False,
                "default": 5,
                "description": "Number of results for 'search' action (default: 5)."
            },
            {
                "name": "timeout",
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Request timeout in seconds for 'browse' action (default: 10)."
            }
        ]

    def execute(self, **kwargs: Any) -> SkillResult:
        action = kwargs.get("action")

        if action == "browse":
            url = kwargs.get("url")
            timeout = kwargs.get("timeout", 10) # Use default from parameters
            if not url:
                return SkillResult(success=False, error="Missing required parameter 'url' for action 'browse'.")
            # Call the existing static method (could also move logic here)
            return self.__class__.browse_url(url=url, timeout=timeout)

        elif action == "search":
            query = kwargs.get("query")
            num_results = kwargs.get("num_results", 5) # Use default from parameters
            if not query:
                return SkillResult(success=False, error="Missing required parameter 'query' for action 'search'.")
            # Call the existing static method
            return self.__class__.search_web(query=query, num_results=num_results)

        else:
            return SkillResult(success=False, error=f"Invalid action '{action}'. Must be 'browse' or 'search'.")

    @staticmethod
    def browse_url(url: str, timeout: int = 10) -> SkillResult:
        """
        Fetches the textual content of a given URL.

        Args:
            url: The URL to browse.
            timeout: Request timeout in seconds.

        Returns:
            SkillResult containing the extracted text or an error message.
        """
        logger.info(f"Browsing URL: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text, trying to be somewhat clean
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose() # Remove script and style elements

            text = soup.get_text(separator='\n', strip=True)

            # Limit length to avoid overwhelming results
            max_length = 5000
            if len(text) > max_length:
                text = text[:max_length] + "... [Content Truncated]"

            logger.info(f"Successfully fetched content from {url}. Length: {len(text)}")
            return SkillResult(success=True, data={"url": url, "content": text})

        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred while fetching URL: {url}")
            return SkillResult(success=False, error=f"Timeout fetching URL: {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return SkillResult(success=False, error=f"Failed to fetch URL {url}: {e}")
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {e}", exc_info=True)
            return SkillResult(success=False, error=f"Error parsing content from {url}: {e}")

    @staticmethod
    def search_web(query: str, num_results: int = 5) -> SkillResult:
        """
        Performs a web search using DuckDuckGo and returns results.
        (Note: Requires 'pip install duckduckgo_search')

        Args:
            query: The search query.
            num_results: The maximum number of results to return.

        Returns:
            SkillResult containing a list of search results or an error message.
        """
        logger.info(f"Performing web search for: '{query}'")
        try:
            # Attempt to import duckduckgo_search dynamically
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.error("duckduckgo_search library not found. Cannot perform web search. Run 'pip install duckduckgo_search'")
                return SkillResult(success=False, error="Web search dependency (duckduckgo_search) not installed.")

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            if not results:
                logger.warning(f"No search results found for query: '{query}'")
                return SkillResult(success=True, data={"query": query, "results": []}, message="No results found.")

            # Format results
            formatted_results = [
                {"title": r.get('title'), "url": r.get('href'), "snippet": r.get('body')}
                for r in results
            ]

            logger.info(f"Found {len(formatted_results)} search results for '{query}'")
            return SkillResult(success=True, data={"query": query, "results": formatted_results})

        except Exception as e:
            logger.error(f"Error during web search for '{query}': {e}", exc_info=True)
            return SkillResult(success=False, error=f"Web search failed: {e}")

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test browse (using instance now)
    web_skill = WebSkill()
    browse_result = web_skill.execute(action="browse", url="https://example.com")
    print("\n--- Browse Result ---")
    print(f"Success: {browse_result.success}")
    if browse_result.success:
        print(f"URL: {browse_result.data['url']}")
        print(f"Content Snippet:\n{browse_result.data['content'][:200]}...")
    else:
        print(f"Error: {browse_result.error}")

    # Test search (requires duckduckgo_search installed)
    search_result = web_skill.execute(action="search", query="large language models", num_results=3)
    print("\n--- Search Result ---")
    print(f"Success: {search_result.success}")
    if search_result.success:
        print(f"Query: {search_result.data['query']}")
        print("Results:")
        for res in search_result.data['results']:
            print(f"  - Title: {res['title']}")
            print(f"    URL: {res['url']}")
            print(f"    Snippet: {res['snippet'][:100]}...")
    else:
        print(f"Error: {search_result.error}") 