import os
import glob
import shutil
from typing import Dict, Any, List, Optional, Tuple

from pydantic import BaseModel, Field

from utils.logger import setup_logger

class FileSkills(BaseModel):
    """File skills for Jarvis to interact with the file system"""
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("file_skills")
    
    def execute(self, intent: str, parameters: Dict[str, Any]) -> str:
        """Execute a file-related intent"""
        self.logger.info(f"Executing file intent: {intent} with parameters: {parameters}")
        
        if intent == "file.search":
            return self.search_files(parameters.get("target", ""))
        elif intent == "file.open":
            return self.open_file(parameters.get("target", ""))
        elif intent == "file.edit":
            content = parameters.get("content", "")
            return self.edit_file(parameters.get("target", ""), content)
        else:
            return f"Unknown file intent: {intent}"
    
    def search_files(self, pattern: str, directory: str = None) -> str:
        """Search for files matching a pattern"""
        if not pattern:
            return "No search pattern specified"
        
        self.logger.info(f"Searching for files: {pattern}")
        
        try:
            # Use current directory if none specified
            if directory is None:
                directory = os.getcwd()
            
            # Ensure directory exists
            if not os.path.exists(directory):
                return f"Directory not found: {directory}"
            
            # Search for files
            search_path = os.path.join(directory, f"**/*{pattern}*")
            files = glob.glob(search_path, recursive=True)
            
            # Filter out directories
            files = [f for f in files if os.path.isfile(f)]
            
            if not files:
                return f"No files found matching '{pattern}'"
            
            # Format results
            results = "\n".join(files[:10])
            if len(files) > 10:
                results += f"\n...and {len(files) - 10} more files"
            
            return f"Found {len(files)} files matching '{pattern}':\n{results}"
        
        except Exception as e:
            self.logger.error(f"Error searching for files with pattern {pattern}: {e}")
            return f"Failed to search for '{pattern}': {str(e)}"
    
    def open_file(self, filepath: str) -> str:
        """Open a file and read its contents"""
        if not filepath:
            return "No file specified"
        
        self.logger.info(f"Opening file: {filepath}")
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            # Check if it's a file (not a directory)
            if not os.path.isfile(filepath):
                return f"Not a file: {filepath}"
            
            # Get file extension
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            # Check if it's a text file
            text_extensions = ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.csv', '.log']
            
            if ext in text_extensions:
                # Read file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Truncate if too large
                if len(content) > 2000:
                    content = content[:2000] + "...\n[Content truncated]"
                
                return f"Content of {filepath}:\n\n{content}"
            else:
                # For non-text files, try to open with system default application
                if os.name == 'nt':  # Windows
                    os.startfile(filepath)
                elif os.name == 'posix':  # macOS and Linux
                    import subprocess
                    subprocess.call(('xdg-open', filepath))
                
                return f"Opening {filepath} with system default application"
        
        except Exception as e:
            self.logger.error(f"Error opening file {filepath}: {e}")
            return f"Failed to open '{filepath}': {str(e)}"
    
    def edit_file(self, filepath: str, content: str = None) -> str:
        """Edit a file (create if it doesn't exist)"""
        if not filepath:
            return "No file specified"
        
        self.logger.info(f"Editing file: {filepath}")
        
        try:
            # Create directories if they don't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # If content is provided, write to file
            if content is not None:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"File {filepath} has been updated"
            
            # Otherwise, open with system default editor
            else:
                if os.name == 'nt':  # Windows
                    os.system(f'notepad "{filepath}"')
                elif os.name == 'posix':  # macOS and Linux
                    editor = os.environ.get('EDITOR', 'nano')
                    os.system(f'{editor} "{filepath}"')
                
                return f"Opening {filepath} for editing"
        
        except Exception as e:
            self.logger.error(f"Error editing file {filepath}: {e}")
            return f"Failed to edit '{filepath}': {str(e)}"
    
    def copy_file(self, source: str, destination: str) -> str:
        """Copy a file from source to destination"""
        if not source or not destination:
            return "Source and destination must be specified"
        
        self.logger.info(f"Copying file from {source} to {destination}")
        
        try:
            # Check if source exists
            if not os.path.exists(source):
                return f"Source file not found: {source}"
            
            # Create destination directory if it doesn't exist
            dest_dir = os.path.dirname(destination)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Copy file
            shutil.copy2(source, destination)
            
            return f"File copied from {source} to {destination}"
        
        except Exception as e:
            self.logger.error(f"Error copying file from {source} to {destination}: {e}")
            return f"Failed to copy file: {str(e)}"
    
    def move_file(self, source: str, destination: str) -> str:
        """Move a file from source to destination"""
        if not source or not destination:
            return "Source and destination must be specified"
        
        self.logger.info(f"Moving file from {source} to {destination}")
        
        try:
            # Check if source exists
            if not os.path.exists(source):
                return f"Source file not found: {source}"
            
            # Create destination directory if it doesn't exist
            dest_dir = os.path.dirname(destination)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Move file
            shutil.move(source, destination)
            
            return f"File moved from {source} to {destination}"
        
        except Exception as e:
            self.logger.error(f"Error moving file from {source} to {destination}: {e}")
            return f"Failed to move file: {str(e)}"
    
    def delete_file(self, filepath: str) -> str:
        """Delete a file"""
        if not filepath:
            return "No file specified"
        
        self.logger.info(f"Deleting file: {filepath}")
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            # Delete file
            if os.path.isfile(filepath):
                os.remove(filepath)
                return f"File {filepath} has been deleted"
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
                return f"Directory {filepath} has been deleted"
            else:
                return f"Not a file or directory: {filepath}"
        
        except Exception as e:
            self.logger.error(f"Error deleting file {filepath}: {e}")
            return f"Failed to delete '{filepath}': {str(e)}" 