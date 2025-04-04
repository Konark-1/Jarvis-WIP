import logging
from typing import Any, Dict

# Assuming skills have an 'execute' method and potentially an 'is_available' check
# from .base import Skill # Import Base Skill if needed for type hinting

logger = logging.getLogger(__name__)

def execute_skill_wrapper(skill: Any, args: Dict[str, Any]) -> str:
    """Safely executes a skill's execute method and returns the result as a string.

    Args:
        skill: The skill instance to execute.
        args: A dictionary of arguments for the skill's execute method.

    Returns:
        A string representation of the skill's execution result or an error message.
    """
    if not hasattr(skill, 'execute') or not callable(skill.execute):
        error_msg = f"Error: Skill '{getattr(skill, 'name', 'UnknownSkill')}' has no callable 'execute' method."
        logger.error(error_msg)
        return error_msg

    # Optional: Check if skill is available before execution
    if hasattr(skill, 'is_available') and callable(skill.is_available):
        if not skill.is_available():
            error_msg = f"Error: Skill '{getattr(skill, 'name', 'UnknownSkill')}' is not available (check dependencies/config)."
            logger.warning(error_msg)
            return error_msg

    try:
        logger.info(f"Executing skill '{getattr(skill, 'name', 'UnknownSkill')}' with args: {args}")
        result = skill.execute(**args)
        # Ensure the result is converted to a string for CrewAI tools
        result_str = str(result)
        logger.info(f"Skill '{getattr(skill, 'name', 'UnknownSkill')}' executed successfully. Result preview: {result_str[:100]}...")
        return result_str
    except Exception as e:
        error_msg = f"Error executing skill '{getattr(skill, 'name', 'UnknownSkill')}': {e}"
        logger.error(error_msg, exc_info=True) # Log the full traceback
        return error_msg # Return the error message as a string 