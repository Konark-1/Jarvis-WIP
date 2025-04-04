# jarvis/skills/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class SkillResult(BaseModel):
    """Standard result format for skill execution."""
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Skill(ABC):
    """Abstract base class for all skills."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the skill (e.g., 'web_search')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of what the skill does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Dict[str, Any]]:
        """A list defining the parameters the skill expects.
        Example: [
            {"name": "query", "type": "string", "required": True, "description": "Search query"},
            {"name": "num_results", "type": "integer", "required": False, "default": 5}
        ]
        """
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> SkillResult:
        """Executes the skill with the given parameters.

        Args:
            **kwargs: Parameters required by the skill.

        Returns:
            A SkillResult object.
        """
        pass

    def validate_parameters(self, provided_params: Dict[str, Any]) -> Optional[str]:
        """Validates provided parameters against the skill's definition.

        Args:
            provided_params: The parameters provided for execution.

        Returns:
            An error message string if validation fails, otherwise None.
        """
        errors = []
        defined_params = {p['name']: p for p in self.parameters}

        # Check for required parameters
        for param_def in self.parameters:
            if param_def.get('required', False) and param_def['name'] not in provided_params:
                errors.append(f"Missing required parameter: '{param_def['name']}'")

        # Check for unknown parameters (optional, could allow extra args)
        # for param_name in provided_params:
        #     if param_name not in defined_params:
        #         errors.append(f"Unknown parameter provided: '{param_name}'")

        # Basic type checking (can be expanded)
        for param_name, value in provided_params.items():
            if param_name in defined_params:
                expected_type = defined_params[param_name].get('type')
                if expected_type == 'string' and not isinstance(value, str):
                    errors.append(f"Parameter '{param_name}' should be a string, got {type(value).__name__}")
                elif expected_type == 'integer' and not isinstance(value, int):
                     # Allow string conversion for integers
                    if isinstance(value, str) and value.isdigit():
                        provided_params[param_name] = int(value) # Convert in-place
                    else:
                        errors.append(f"Parameter '{param_name}' should be an integer, got {type(value).__name__}")
                elif expected_type == 'boolean' and not isinstance(value, bool):
                     # Allow string conversion for booleans
                    if isinstance(value, str) and value.lower() in ['true', 'false']:
                        provided_params[param_name] = value.lower() == 'true' # Convert in-place
                    else:
                        errors.append(f"Parameter '{param_name}' should be a boolean, got {type(value).__name__}")
                # Add more types (float, list, etc.) as needed

        return "; ".join(errors) if errors else None 