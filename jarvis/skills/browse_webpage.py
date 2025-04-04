import logging
from typing import Dict, Any, List, Optional
from pydantic import Field

# Attempt safe imports for libraries assumed to be in the user's environment
try:
    import requests
    from bs4 import BeautifulSoup
    _IMPORTS_AVAILABLE = True
except ImportError:
    _IMPORTS_AVAILABLE = False
    requests = None
    BeautifulSoup = None

from jarvis.skills.base import BaseSkill, SkillResult, register_skill

logger = logging.getLogger(__name__)

@register_skill
class BrowseWebpageSkill(BaseSkill):
    """Skill to fetch content from a URL and extract information.

    NOTE: Requires 'requests' and 'beautifulsoup4' to be installed in the environment.
    """
    # Dependencies (checked at runtime)
    _dependencies: List[str] = Field(default=["requests", "beautifulsoup4"], description="Libraries required for this skill.")

    @property
    def name(self) -> str:
        return "browse_webpage"

    @property
    def description(self) -> str:
        return (
            "Fetches the HTML content of a given URL and extracts the main textual content. "
            "Optionally, attempts to answer a specific question based on the extracted text using an LLM."
        )

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "url", "type": "string", "required": True, "description": "The fully qualified URL to browse (e.g., 'https://example.com')."},
            {"name": "question", "type": "string", "required": False, "description": "(Optional) A specific question to answer based on the page content."}
        ]

    def _extract_text(self, html_content: str) -> str:
        """Extracts meaningful text content from HTML using BeautifulSoup."""
        if not BeautifulSoup: return "Error: BeautifulSoup library not found."
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
                
            # Get text, trying to preserve paragraphs somewhat
            # Find main content areas if possible (common tags/ids, heuristic)
            potential_main = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(id='main') or soup.body
            if potential_main:
                text = potential_main.get_text(separator='\n', strip=True)
            else: # Fallback to body if no main content found
                 text = soup.get_text(separator='\n', strip=True)
                 
            # Basic cleaning of excessive newlines
            lines = [line for line in text.split('\n') if line.strip()]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return f"Error: Could not parse HTML content ({e})."
            
    def execute(self, llm_client: Optional['LLMClient'] = None, **kwargs: Any) -> SkillResult:
        """
        Fetches URL, parses HTML, extracts text, optionally answers question.

        Args:
            llm_client: An LLMClient instance (required if 'question' is provided).
            **kwargs: Must include 'url'. Can include 'question'.

        Returns:
            A SkillResult object containing the extracted text or answer.
        """
        # Check for dependencies in the actual runtime environment
        if not _IMPORTS_AVAILABLE:
            missing = []
            if not requests: missing.append("'requests'")
            if not BeautifulSoup: missing.append("'beautifulsoup4'")
            error_msg = f"Execution failed: Required libraries not found: {', '.join(missing)}. Please install them."
            logger.error(error_msg)
            return SkillResult(success=False, error=error_msg)

        # Validate parameters using base class method
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return SkillResult(success=False, error=f"Parameter validation failed: {validation_error}")

        url = kwargs.get('url')
        question = kwargs.get('question')

        # --- Simulate Fetching and Parsing (Placeholder for Execution Context) ---
        # In a real scenario, this block would use requests and BeautifulSoup
        logger.info(f"Attempting to browse URL: {url}")
        try:
            headers = {'User-Agent': 'JarvisAgent/1.0'} # Basic user agent
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            html_content = response.text
            logger.info(f"Successfully fetched content from {url} (Status: {response.status_code}). Length: {len(html_content)}")
            
            extracted_text = self._extract_text(html_content)
            
            if not extracted_text.startswith("Error:"):
                 logger.info(f"Extracted text preview: '{extracted_text[:200]}...'")
            else:
                 logger.warning(f"Failed to extract text properly from {url}.")
                 # Return error from extraction
                 return SkillResult(success=False, error=extracted_text, data={"url": url})

            # --- Answer question based on text (if provided) ---
            if question:
                if not llm_client:
                    return SkillResult(success=False, error="LLMClient required to answer question, but not provided.", data={"url": url})
                
                logger.info(f"Answering question '{question}' based on content from {url}")
                try:
                    # Keep the context relatively small to focus on the extracted text
                    MAX_CONTEXT_LEN = 8000 # Rough character limit for context
                    context_for_llm = extracted_text[:MAX_CONTEXT_LEN] + ('...' if len(extracted_text) > MAX_CONTEXT_LEN else '')
                    
                    prompt = f"Based *only* on the following text extracted from the webpage {url}, answer the question.\n\nExtracted Text:\n---\n{context_for_llm}\n---\n\nQuestion: {question}\n\nAnswer:"
                    system_prompt = "You are an assistant designed to answer questions based *solely* on the provided text from a webpage."
                    
                    answer = llm_client.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=300
                    ).strip()
                    
                    logger.info("Successfully generated answer using LLM.")
                    return SkillResult(success=True, message=f"Answer based on content from {url}", data={"url": url, "question": question, "answer": answer})
                    
                except Exception as llm_err:
                     logger.error(f"LLM failed to answer question based on content from {url}: {llm_err}", exc_info=True)
                     # Return the extracted text even if LLM fails? Or fail the step? Let's return text.
                     return SkillResult(success=False, message="LLM failed to answer question, returning extracted text instead.", error=str(llm_err), data={"url": url, "extracted_text": extracted_text})

            # --- Return extracted text if no question ---
            else:
                logger.info(f"Returning extracted text from {url}.")
                return SkillResult(success=True, message=f"Successfully extracted text from {url}", data={"url": url, "extracted_text": extracted_text})

        except requests.exceptions.Timeout:
            error_msg = f"Browsing failed: Request timed out for URL: {url}"
            logger.error(error_msg)
            return SkillResult(success=False, error=error_msg, data={"url": url})
        except requests.exceptions.RequestException as e:
            error_msg = f"Browsing failed: Could not fetch URL {url}. Error: {e}"
            logger.error(error_msg)
            return SkillResult(success=False, error=error_msg, data={"url": url})
        except Exception as e:
            error_msg = f"An unexpected error occurred during browsing {url}: {e}"
            logger.exception(error_msg) # Log full traceback
            return SkillResult(success=False, error=error_msg, data={"url": url})

# Example of how LLMClient might be defined (Needs actual implementation elsewhere)
# class LLMClient:
#     def process_with_llm(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
#         # Placeholder for actual LLM interaction
#         return f"LLM Answer based on: {prompt[:100]}..." 