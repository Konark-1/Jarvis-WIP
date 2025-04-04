import os
from textwrap import dedent
import logging
from typing import List, Dict, Any

from crewai import Agent, Task, Crew, Process
from langchain_core.tools import tool as lc_tool # For LangChain style tools if needed
# from crewai.tools import Tool # <-- Removed this incorrect import

# Assume LLMClient provides a LangChain compatible object
# from jarvis.llm import LLMClient # Not directly needed here, passed in
from jarvis.skills.registry import SkillRegistry # Import SkillRegistry
from jarvis.skills.skill_manager import execute_skill_wrapper # Helper for safe execution

# Configure logging for the crew file
logger = logging.getLogger(__name__)

# Tool Setup (Example: Use Serper for search)
# Ensure SERPER_API_KEY is in your .env file
# search_tool = SerperDevTool()

# --- Planning Crew Definition ---

def create_planning_crew(llm_config: object, skill_registry: SkillRegistry) -> Crew:
    """Creates the planning crew with agents, tasks, and tools."""
    logger.info(f"Creating planning crew with LLM config: {type(llm_config)}")

    # Define Tools based on available skills
    crew_tools = []
    web_search_skill = skill_registry.get_skill("web_search")

    if web_search_skill:
        # Define a wrapper function decorated as a LangChain tool
        @lc_tool
        def run_web_search(query: str) -> str:
            """Performs a web search to find up-to-date information on a given topic or query.""" # Docstring becomes tool description
            logger.info(f"Crew Agent using web_search tool with query: '{query}'")
            result = execute_skill_wrapper(web_search_skill, {"query": query})
            return str(result)

        # Add the decorated function directly to the list
        crew_tools.append(run_web_search)
        logger.info("Added 'run_web_search' (decorated with @lc_tool) to planning crew tools.")
    else:
        logger.warning("'web_search' skill not found in registry. Planning crew will lack web search tool.")

    # Add more tools here as needed, wrapping other skills...

    # Define Agents
    planner_agent = Agent(
        role="Master Planner",
        goal=(
            "Based on the user's objective and provided context, decompose the objective "
            "into a sequence of specific, actionable tasks. Each task should ideally correspond "
            "to an available skill or a logical step. Identify the necessary skill and arguments for each task." # Updated goal
        ),
        backstory="""\
            You are an expert planner specializing in breaking down complex objectives into manageable steps.
            You analyze the goal, leverage available context and tools (like web search), and produce a clear, ordered list of tasks.
            Your output *must* be a JSON list of dictionaries, where each dictionary represents a task and includes keys like 'description', 'skill' (if applicable), and 'arguments' (as a dictionary).
            Example Task Dictionary: {'description': 'Search for recent AI advancements', 'skill': 'web_search', 'arguments': {'query': 'recent AI advancements'}}
            """,
        verbose=True,
        allow_delegation=False,
        llm=llm_config,  # Pass the LangChain compatible LLM object
        tools=crew_tools # Assign the list containing the decorated function
    )

    # Define Tasks
    decompose_task = Task(
        description=(
            "1. Understand the user's main objective: {objective}.\n"
            "2. Consider the provided context: {context}.\n"
            "3. Use available tools (like WebSearch) if necessary to gather information for planning.\n" # Mention tool use
            "4. Break down the objective into a list of specific, numbered tasks.\n"
            "5. For each task, specify a brief 'description', the 'skill' required (if any, from the available tools or known system skills), and the necessary 'arguments' as a dictionary.\n"
            "6. Ensure the final output is *only* a valid JSON list of task dictionaries. Do not include any preamble, explanation, or markdown formatting around the JSON."
        ),
        expected_output=(
            "A valid JSON list of dictionaries. Each dictionary must represent a task "
            "and contain at least a 'description' key. It should also include 'skill' and 'arguments' keys where applicable. "
            "Example: `[{\"description\": \"Search for X\", \"skill\": \"web_search\", \"arguments\": {\"query\": \"X\"}}, {\"description\": \"Summarize findings\", \"skill\": null, \"arguments\": {}}]`"
        ),
        agent=planner_agent,
    )

    # Create and return the Crew
    planning_crew_instance = Crew(
        agents=[planner_agent],
        tasks=[decompose_task],
        process=Process.sequential,
        verbose=2,  # Enable verbose logging for crew execution
        # memory=True/False # Optional: Add memory if needed for complex planning across turns
    )
    logger.info("Planning crew instance created.")
    return planning_crew_instance

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from jarvis.llm import LLMClient # For testing
    from jarvis.skills.registry import SkillRegistry # For testing
    from jarvis.skills.web_search import WebSearchSkill # For testing

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Setup for testing
    test_llm_client = LLMClient(config={
        "default_provider": "groq", # Or your preferred provider
        "providers": {
            "groq": {"api_key": os.getenv("GROQ_API_KEY"), "default_model": "llama3-70b-8192"},
            # Add other providers if needed for testing
        }
    })
    test_skill_registry = SkillRegistry()
    test_skill_registry.register_skill(WebSearchSkill()) # Register the skill for testing

    test_llm_instance = test_llm_client.get_langchain_llm() # Get the LLM instance

    if test_llm_instance:
        # Create the crew using the test setup
        crew = create_planning_crew(llm_config=test_llm_instance, skill_registry=test_skill_registry)

        # Define a test objective and context
        test_objective = "Find out the current weather in London and the capital of France."
        test_context = "User is planning a trip."

        # Kick off the crew
        logger.info(f"Kicking off test crew with objective: {test_objective}")
        try:
            result = crew.kickoff(inputs={'objective': test_objective, 'context': test_context})
            logger.info("\n--- Test Crew Result ---")
            print(result)
            logger.info("--- End Test Crew Result ---")
        except Exception as e:
            logger.error(f"Error during test crew execution: {e}", exc_info=True)
    else:
        logger.error("Failed to get LLM instance for testing the crew.") 