import logging
from typing import List, Dict, Any, Optional, Callable, Type
import json
import inspect

from crewai import Agent, Task, Crew, Process # Uncommented
from langchain_core.language_models import BaseChatModel # Uncommented
# Try importing BaseTool from crewai.tools
from crewai.tools import BaseTool as CrewAIBaseTool # Uncommented
from pydantic import BaseModel, Field, create_model, ConfigDict

from jarvis.skills.registry import SkillRegistry
from jarvis.skills.base import Skill, SkillResult

logger = logging.getLogger(__name__)

# --- Adapter Tool Class (Uncommented) --- 
class SkillAdapterTool(CrewAIBaseTool):
    """Adapts a Jarvis Skill to the CrewAI BaseTool interface (attempt 4)."""
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    skill_instance: Skill

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, skill: Skill, **kwargs):
        fields = {}
        param_descriptions = []
        type_map = {'string': str, 'integer': int, 'boolean': bool, 'any': Any}
        for param in skill.parameters:
            param_name = param['name']
            param_type = type_map.get(param.get('type', 'any'), Any)
            field_args = {"description": param.get('description', '')}
            is_required = param.get('required', False)
            if not is_required:
                field_args["default"] = param.get('default', None)
                param_type = Optional[param_type]
            if is_required:
                fields[param_name] = (param_type, Field(**field_args))
            else:
                fields[param_name] = (param_type, Field(default=field_args.get("default"), description=field_args.get("description")))
            param_descriptions.append(f"- {param_name} ({param.get('type', 'any')}){' (optional)' if not is_required else ''}: {param.get('description', 'No description')}")

        full_description = f"{skill.description}\n\nParameters:\n" + "\n".join(param_descriptions)

        # Create the schema but don't assign directly to self yet
        created_args_schema: Optional[Type[BaseModel]] = None
        if fields:
            created_args_schema = create_model(f"{skill.name.capitalize()}ToolArgs", **fields)
        else:
            created_args_schema = create_model(f"{skill.name.capitalize()}ToolArgs")

        # Prepare data for super().__init__
        init_data = kwargs.copy()
        init_data['skill_instance'] = skill
        init_data['name'] = skill.name
        init_data['description'] = full_description
        init_data['args_schema'] = created_args_schema # Pass the created schema to parent

        # Call super().__init__
        super().__init__(**init_data)

    def _run(self, **kwargs: Any) -> str:
        """Executes the underlying skill and formats the result as a string."""
        logger.info(f"Crew Agent invoking skill '{self.name}' via adapter with args: {kwargs}")
        try:
            result: SkillResult = self.skill_instance.execute(**kwargs)
            if result.success:
                output = f"Skill '{self.name}' executed successfully."
                if result.message: output += f" Message: {result.message}"
                if result.data: output += f"\nData: {json.dumps(result.data, indent=2)}"
                return output
            else:
                error_output = f"Skill '{self.name}' failed."
                if result.error: error_output += f" Error: {result.error}"
                if result.message: error_output += f" Message: {result.message}"
                return error_output
        except Exception as e:
            logger.error(f"Unexpected error executing skill '{self.name}' via adapter: {e}", exc_info=True)
            return f"Error: Unexpected exception while executing skill '{self.name}': {e}"

# --- Remove the dynamic function creation helper --- 
# def _create_tool_function_from_skill(skill_instance: Skill) -> Callable:
#     pass # Remove implementation

# --- Planning Crew Definition (Uncommented) --- 
def create_planning_crew(llm: BaseChatModel, skill_registry: SkillRegistry) -> Crew:
    logger.info(f"Creating planning crew with LLM: {type(llm)}")
    # --- Add detailed logging for the LLM instance ---
    if hasattr(llm, 'model_name'):
        logger.info(f"  LLM Model Name (Original): {llm.model_name}")
        # --- Force LiteLLM provider format --- 
        if not llm.model_name.startswith("groq/"):
             original_name = llm.model_name
             llm.model_name = f"groq/{original_name}"
             logger.info(f"  LLM Model Name (Modified for LiteLLM): {llm.model_name}")
        # ------------------------------------
    if hasattr(llm, 'client'): # Check if it has a client attribute (like LangChain wrappers might)
         logger.info(f"  LLM Client Type: {type(llm.client)}")
    logger.info(f"  LLM Object Details: {llm}")
    # --- End of added logging ---

    crew_tools = []
    available_skills = skill_registry.get_all_skills()
    if not available_skills:
        logger.warning("SkillRegistry provided no skills...")
    else:
        logger.info(f"Generating tools for planning crew from skills: {list(available_skills.keys())}")
        for skill_name, skill_instance in available_skills.items():
            try:
                # Use the SkillAdapterTool class inheriting from crewai.tools.BaseTool
                adapter_tool = SkillAdapterTool(skill=skill_instance)
                crew_tools.append(adapter_tool)
                logger.info(f"Successfully created adapter tool '{skill_name}' for planning crew.")
            except Exception as e:
                logger.error(f"Failed to create adapter tool for skill '{skill_name}': {e}", exc_info=True)

    # Define Agents (Backstory remains simplified)
    planner_agent = Agent(
        role="Master Planner",
        goal=(
            "Based on the user's objective and provided context, decompose the objective "
            "into a sequence of specific, actionable tasks. Each task should ideally correspond "
            "to an available skill or a logical step. Identify the necessary skill and arguments for each task."
        ),
        backstory="You are an expert planner. Your goal is to create a JSON task list based on the objective and context.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
        tools=crew_tools # Pass list of adapter instances
    )
    
    # Define Tasks
    decompose_task = Task(
        description=(
            "1. Understand the user's main objective: {objective}.\n"
            "2. Consider the provided context: {context}.\n"
            "3. Use available tools if necessary to gather information for planning.\n"
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
        llm=llm,
        verbose=False,
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
    test_llm_client = LLMClient() # Initialize with default env vars or config
    test_skill_registry = SkillRegistry()
    # Use discovery instead of manual registration for testing consistency
    test_skill_registry.discover_skills()

    # test_llm_instance = test_llm_client.get_langchain_llm() # Get the LLM instance
    # Modify the test setup to use the new get_langchain_llm method
    test_llm_instance = test_llm_client.get_langchain_llm() # Get the LLM instance

    if test_llm_instance:
        # Create the crew using the test setup
        # Pass the LLM instance directly, not the whole config
        # crew = create_planning_crew(llm=test_llm_instance, skill_registry=test_skill_registry)

        # Define a test objective and context
        test_objective = "Find out the current weather in London and the capital of France."
        test_context = "User is planning a trip."

        # Kick off the crew
        logger.info(f"Kicking off test crew with objective: {test_objective}")
        try:
            # result = crew.kickoff(inputs={'objective': test_objective, 'context': test_context})
            logger.info("\n--- Test Crew Result ---")
            # print(result)
            logger.info("--- End Test Crew Result ---")
        except Exception as e:
            logger.error(f"Error during test crew execution: {e}", exc_info=True)
    else:
        logger.error("Failed to get LLM instance for testing the crew.") 