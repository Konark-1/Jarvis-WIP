import logging
import subprocess
import sys
import os
from typing import Dict, Any

# Potential Base class import (if defined)
# from .base import Skill

logger = logging.getLogger(__name__)

# class ExecutePythonFileSkill(Skill):
class ExecutePythonFileSkill:
    """A skill to execute a Python file and capture its output."""

    name = "execute_python_file"
    description = (
        "Executes a given Python script file (.py) and returns its standard output and standard error. "
        "Provide the relative path to the file from the workspace root."
    )

    def __init__(self):
        """Initializes the skill."""
        # No specific initialization needed for now
        pass

    def is_available(self) -> bool:
        """Checks if the skill can be used."""
        # Available by default unless specific constraints are needed
        return True

    def execute(self, file_path: str, **kwargs: Any) -> str:
        """Executes the specified Python file.

        Args:
            file_path: The relative path to the Python file to execute.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A string containing the standard output and standard error from the script execution,
            or an error message if execution fails.
        """
        if not file_path or not file_path.endswith(".py"):
            return "Error: Invalid or non-Python file path provided."

        # Ensure the file path is treated as relative to the workspace root
        # Note: Assuming the process running Jarvis has the workspace as CWD.
        # If not, might need to pass workspace root path and use os.path.join.
        absolute_file_path = os.path.abspath(file_path)

        if not os.path.isfile(absolute_file_path):
             return f"Error: Python file not found at specified path: {absolute_file_path}"

        try:
            logger.info(f"Executing Python file: {absolute_file_path}")
            # Use subprocess to run the script and capture output
            process = subprocess.run(
                [sys.executable, absolute_file_path],
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit code
                timeout=60 # Add a timeout (e.g., 60 seconds)
            )

            output = f"--- Execution Output for {file_path} ---\n"
            if process.stdout:
                output += f"Standard Output:\n{process.stdout}\n"
            if process.stderr:
                 output += f"Standard Error:\n{process.stderr}\n"
            if not process.stdout and not process.stderr:
                output += "(No output produced)\n"

            output += f"--- Exit Code: {process.returncode} ---"

            if process.returncode != 0:
                logger.warning(f"Python script {file_path} exited with non-zero code: {process.returncode}")
            else:
                 logger.info(f"Python script {file_path} executed successfully.")

            return output

        except subprocess.TimeoutExpired:
            error_msg = f"Error: Execution of {file_path} timed out."
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing Python file {file_path}: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg 