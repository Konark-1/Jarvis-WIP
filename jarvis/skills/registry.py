# jarvis/skills/registry.py

import os
import importlib
import inspect
import logging
from typing import Dict, Type, Optional, List, Any

from .base import Skill

logger = logging.getLogger(__name__)

class SkillRegistry:
    """Discovers and manages available skills."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skill_classes: Dict[str, Type[Skill]] = {}

    def discover_skills(self, skill_dir: str = "jarvis/skills"):
        """Dynamically discovers skills in the specified directory."""
        self._skills = {}
        self._skill_classes = {}
        logger.info(f"Discovering skills in '{skill_dir}'...")

        try:
            for filename in os.listdir(skill_dir):
                if filename.endswith(".py") and not filename.startswith("__") and filename != "base.py" and filename != "registry.py":
                    module_name = filename[:-3]
                    module_path = f"{skill_dir.replace('/', '.')}.{module_name}"
                    try:
                        module = importlib.import_module(module_path)
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, Skill) and obj is not Skill:
                                try:
                                    skill_instance = obj()
                                    if skill_instance.name in self._skills:
                                        logger.warning(f"Duplicate skill name '{skill_instance.name}' found in {module_path}. Overwriting.")
                                    self._skills[skill_instance.name] = skill_instance
                                    self._skill_classes[skill_instance.name] = obj
                                    logger.info(f"Discovered skill: '{skill_instance.name}' from {module_path}")
                                except Exception as e:
                                    logger.error(f"Failed to instantiate skill class '{name}' from {module_path}: {e}")
                    except ImportError as e:
                        logger.error(f"Failed to import module {module_path}: {e}")
        except FileNotFoundError:
            logger.error(f"Skill directory '{skill_dir}' not found.")
        except Exception as e:
            logger.error(f"Error discovering skills: {e}", exc_info=True)

        logger.info(f"Skill discovery complete. Found {len(self._skills)} skills: {list(self._skills.keys())}")

    def get_skill(self, name: str) -> Optional[Skill]:
        """Gets a skill instance by name."""
        return self._skills.get(name)

    def get_skill_class(self, name: str) -> Optional[Type[Skill]]:
        """Gets a skill class type by name."""
        return self._skill_classes.get(name)

    def get_all_skills(self) -> Dict[str, Skill]:
        """Returns a dictionary of all registered skill instances."""
        return self._skills

    def get_skill_definitions(self) -> List[Dict[str, Any]]:
        """Returns a list of skill definitions (name, description, parameters)."""
        definitions = []
        for name, skill in self._skills.items():
            try:
                definitions.append({
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": skill.parameters
                })
            except Exception as e:
                logger.error(f"Error getting definition for skill '{name}': {e}")
        return definitions

# Create a default instance and discover skills upon import
# skill_registry = SkillRegistry() # Removed global instance
# skill_registry.discover_skills() # Removed auto-discovery call 