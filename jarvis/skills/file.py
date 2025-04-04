# jarvis/skills/file.py

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import Skill, SkillResult

logger = logging.getLogger(__name__)

class ReadFileSkill(Skill):
    """Reads the content of a specified file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Reads the content of a specified file. Can read the whole file or specific lines."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "file_path", "type": "string", "required": True, "description": "The relative or absolute path to the file."},
            {"name": "start_line", "type": "integer", "required": False, "description": "The 1-based starting line number (inclusive)."},
            {"name": "end_line", "type": "integer", "required": False, "description": "The 1-based ending line number (inclusive)."},
            {"name": "max_chars", "type": "integer", "required": False, "default": 5000, "description": "Maximum characters to return to prevent overload."},
        ]

    def execute(self, **kwargs: Any) -> SkillResult:
        file_path_str = kwargs.get('file_path')
        start_line = kwargs.get('start_line')
        end_line = kwargs.get('end_line')
        max_chars = kwargs.get('max_chars', 5000)

        if not file_path_str:
            return SkillResult(success=False, error="Missing required parameter: file_path")

        logger.info(f"Executing read file skill for path: '{file_path_str}'")

        try:
            # Attempt to resolve the path relative to the workspace or absolutely
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                # Assuming execution from workspace root for relative paths
                # TODO: Consider making workspace root configurable or discoverable
                workspace_root = Path(os.getcwd())
                file_path = workspace_root / file_path_str

            if not file_path.exists():
                return SkillResult(success=False, error=f"File not found: {file_path}")
            if not file_path.is_file():
                return SkillResult(success=False, error=f"Path is not a file: {file_path}")

            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Handle line ranges
            if start_line is not None or end_line is not None:
                start_idx = (start_line - 1) if start_line is not None else 0
                end_idx = end_line if end_line is not None else len(lines)
                
                # Clamp indices to valid range
                start_idx = max(0, start_idx)
                end_idx = min(len(lines), end_idx)
                
                if start_idx >= end_idx:
                     return SkillResult(success=True, message="Start line is after end line, no content returned.", data={"content": "", "lines_read": 0})
                
                content_lines = lines[start_idx:end_idx]
                content = "".join(content_lines)
                lines_read = len(content_lines)
                message=f"Read lines {start_idx + 1} to {end_idx} from {file_path}."
            else:
                content = "".join(lines)
                lines_read = len(lines)
                message=f"Read entire file {file_path} ({lines_read} lines)."

            # Truncate content if too long
            truncated = False
            if len(content) > max_chars:
                content = content[:max_chars]
                truncated = True
                message += f" Content truncated to {max_chars} characters."

            return SkillResult(
                success=True,
                message=message,
                data={
                    "file_path": str(file_path),
                    "content": content,
                    "lines_read": lines_read,
                    "truncated": truncated
                }
            )

        except Exception as e:
            logger.error(f"Error reading file {file_path_str}: {e}", exc_info=True)
            return SkillResult(success=False, error=f"Failed to read file: {e}")

# TODO: Add WriteFileSkill, ListDirectorySkill etc. 