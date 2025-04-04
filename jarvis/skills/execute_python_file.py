import logging
import subprocess
import sys
import os
from typing import Dict, Any, List

# Import base class and result model
from .base import Skill, SkillResult

logger = logging.getLogger(__name__)

# --- <<< ADDED: Configuration for Safe Execution >>> ---
# Define the directory relative to the workspace root where execution is allowed
SAFE_EXEC_DIR_NAME = "workspace_sandbox"
# Get the absolute path of the workspace root (assuming the script is run from there)
# More robust detection might be needed if run from elsewhere.
WORKSPACE_ROOT = os.getcwd()
SAFE_EXEC_DIR_ABS = os.path.abspath(os.path.join(WORKSPACE_ROOT, SAFE_EXEC_DIR_NAME))
# --- <<< END SECTION >>> ---

class ExecutePythonFileSkill(Skill):
    """A skill to execute a Python file and capture its output."""

    def __init__(self):
        """Initializes the skill."""
        # No specific initialization needed
        pass

    @property
    def name(self) -> str:
        return "execute_python_file"

    @property
    def description(self) -> str:
        return (
            "Executes a given Python script file (.py) within a designated sandbox directory and returns its standard output and standard error. "
            f"Provide the relative path to the file *within* the '{SAFE_EXEC_DIR_NAME}' directory."
        )

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "file_path", "type": "string", "required": True, "description": f"The relative path *within the '{SAFE_EXEC_DIR_NAME}' directory* to the Python file (.py) to execute."},
            {"name": "timeout", "type": "integer", "required": False, "default": 60, "description": "Timeout in seconds for the script execution."}
        ]

    def execute(self, **kwargs: Any) -> SkillResult:
        """Executes the specified Python file safely within the sandbox.

        Args:
            **kwargs: Must include 'file_path'. Can include 'timeout' (default 60).

        Returns:
            A SkillResult object containing the execution output or an error.
        """
        # Use base class validation
        validation_error = self.validate_parameters(kwargs)
        if validation_error:
            return SkillResult(success=False, error=f"Parameter validation failed: {validation_error}")

        file_path = kwargs.get('file_path')
        timeout = kwargs.get('timeout', 60)

        # Redundant checks, but safe
        if not file_path or not isinstance(file_path, str) or not file_path.endswith(".py"):
            return SkillResult(success=False, error="Invalid or non-Python file path provided.")

        # --- <<< ADDED: Security Checks >>> ---
        # 1. Prevent path traversal
        # Normalize path to handle different separators and redundant components
        normalized_input_path = os.path.normpath(file_path)
        if ".." in normalized_input_path.split(os.sep):
            error_msg = f"Security Error: Path traversal ('..') detected in file path: '{file_path}'"
            logger.error(error_msg)
            return SkillResult(success=False, error=error_msg)
        # Also check if the normalized path starts with '..' after normalization
        if normalized_input_path.startswith("..") or normalized_input_path.startswith(os.sep):
             error_msg = f"Security Error: Path appears to be absolute or attempts traversal from root: '{file_path}'"
             logger.error(error_msg)
             return SkillResult(success=False, error=error_msg)

        # 2. Ensure path is relative and resolves within the safe directory
        # First, make sure the safe directory exists
        if not os.path.exists(SAFE_EXEC_DIR_ABS):
            try:
                os.makedirs(SAFE_EXEC_DIR_ABS)
                logger.info(f"Created safe execution directory: {SAFE_EXEC_DIR_ABS}")
            except OSError as e:
                error_msg = f"Security Error: Could not create safe execution directory '{SAFE_EXEC_DIR_ABS}': {e}"
                logger.error(error_msg)
                return SkillResult(success=False, error=error_msg)
        elif not os.path.isdir(SAFE_EXEC_DIR_ABS):
             error_msg = f"Security Error: Safe execution path '{SAFE_EXEC_DIR_ABS}' exists but is not a directory."
             logger.error(error_msg)
             return SkillResult(success=False, error=error_msg)

        # Construct the absolute path by joining the SAFE directory and the (normalized) relative input path
        absolute_file_path = os.path.abspath(os.path.join(SAFE_EXEC_DIR_ABS, normalized_input_path))

        # Check if the resolved absolute path is *still* within the safe directory
        # This guards against more complex traversal tricks potentially missed by simple '..' check
        if not os.path.commonpath([SAFE_EXEC_DIR_ABS]) == os.path.commonpath([SAFE_EXEC_DIR_ABS, absolute_file_path]):
             error_msg = f"Security Error: Execution attempt outside of designated sandbox directory ('{SAFE_EXEC_DIR_NAME}'). Path: '{file_path}'"
             logger.error(error_msg)
             logger.debug(f"Resolved path: {absolute_file_path}, Safe dir: {SAFE_EXEC_DIR_ABS}")
             return SkillResult(success=False, error=error_msg)
        # --- <<< END Security Checks >>> ---

        # Check if file exists *after* security checks
        if not os.path.isfile(absolute_file_path):
             # Provide the original relative path in the error message for clarity
             return SkillResult(success=False, error=f"Python file not found within '{SAFE_EXEC_DIR_NAME}' at specified path: {file_path} (resolved to: {absolute_file_path})")

        try:
            logger.info(f"Executing Python file within sandbox: {absolute_file_path} with timeout {timeout}s")
            # Use subprocess to run the script and capture output
            process = subprocess.run(
                [sys.executable, absolute_file_path],
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit code
                timeout=timeout,
                cwd=SAFE_EXEC_DIR_ABS # <<< ADDED: Set CWD to sandbox dir
            )

            stdout = process.stdout
            stderr = process.stderr
            return_code = process.returncode

            if return_code != 0:
                logger.warning(f"Python script {file_path} exited with code {return_code}. Stderr: {stderr}")
                # Return success=False but include output
                return SkillResult(
                    success=False,
                    message=f"Script exited with non-zero code: {return_code}",
                    error=stderr or "Script exited with non-zero code but no stderr.",
                    data={"file_path": file_path, "stdout": stdout, "stderr": stderr, "return_code": return_code}
                )
            else:
                 logger.info(f"Python script {file_path} executed successfully.")
                 return SkillResult(
                     success=True,
                     message="Script executed successfully.",
                     data={"file_path": file_path, "stdout": stdout, "stderr": stderr, "return_code": return_code}
                 )

        except subprocess.TimeoutExpired:
            error_msg = f"Execution of {file_path} timed out after {timeout} seconds."
            logger.error(error_msg)
            return SkillResult(success=False, error=error_msg, data={"file_path": file_path})
        except Exception as e:
            error_msg = f"Error executing Python file {file_path}: {e}"
            logger.error(error_msg, exc_info=True)
            return SkillResult(success=False, error=error_msg, data={"file_path": file_path}) 