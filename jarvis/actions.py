"""
Application-specific actions and web interactions for Jarvis.
"""

import os
import platform
import subprocess
import webbrowser
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

# Initialize logger
logger = logging.getLogger("jarvis.actions")

class ApplicationActions:
    """Handles application-specific actions."""
    
    @staticmethod
    def open_application(application_name: str) -> Dict[str, Any]:
        """
        Open a specified application.
        
        Args:
            application_name: Name or path of the application to open
            
        Returns:
            Dict containing success status and any output/error
        """
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Try to find the application in common locations
                app_paths = [
                    f"C:\\Program Files\\{application_name}",
                    f"C:\\Program Files (x86)\\{application_name}",
                    os.path.expanduser(f"~\\AppData\\Local\\Programs\\{application_name}"),
                    os.path.expanduser(f"~\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\{application_name}")
                ]
                
                # Try to find the executable
                for path in app_paths:
                    if os.path.exists(path):
                        subprocess.Popen([path])
                        return {"success": True, "output": f"Opened {application_name}"}
                
                # If not found, try using the start command
                subprocess.Popen(["start", application_name])
                return {"success": True, "output": f"Opened {application_name}"}
                
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", application_name])
                return {"success": True, "output": f"Opened {application_name}"}
                
            elif system == "linux":
                subprocess.Popen([application_name])
                return {"success": True, "output": f"Opened {application_name}"}
            
            else:
                return {"success": False, "error": f"Unsupported operating system: {system}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def close_application(application_name: str) -> Dict[str, Any]:
        """
        Close a specified application.
        
        Args:
            application_name: Name of the application to close
            
        Returns:
            Dict containing success status and any output/error
        """
        try:
            system = platform.system().lower()
            
            if system == "windows":
                subprocess.run(["taskkill", "/F", "/IM", f"{application_name}.exe"])
                return {"success": True, "output": f"Closed {application_name}"}
                
            elif system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", f'tell application "{application_name}" to quit'])
                return {"success": True, "output": f"Closed {application_name}"}
                
            elif system == "linux":
                subprocess.run(["pkill", application_name])
                return {"success": True, "output": f"Closed {application_name}"}
            
            else:
                return {"success": False, "error": f"Unsupported operating system: {system}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

class FileActions:
    """Handles file system operations."""
    
    @staticmethod
    def search_files(query: str, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for files on the system.
        
        Args:
            query: Search query (filename, pattern, etc.)
            directory: Directory to search within (defaults to user's home)
            
        Returns:
            Dict containing success status and list of found files
        """
        try:
            if directory is None:
                directory = os.path.expanduser("~")
            
            # Convert query to regex pattern
            pattern = query.replace("*", ".*").replace("?", ".")
            
            # Search for files
            found_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if re.match(pattern, file, re.IGNORECASE):
                        found_files.append(os.path.join(root, file))
            
            return {
                "success": True,
                "output": found_files,
                "count": len(found_files)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def create_directory(path: str) -> Dict[str, Any]:
        """
        Create a new directory.
        
        Args:
            path: Path of the directory to create
            
        Returns:
            Dict containing success status and any output/error
        """
        try:
            os.makedirs(path, exist_ok=True)
            return {"success": True, "output": f"Created directory: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def delete_file(path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            path: Path of the file to delete
            
        Returns:
            Dict containing success status and any output/error
        """
        try:
            os.remove(path)
            return {"success": True, "output": f"Deleted file: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class WebActions:
    """Handles web-related actions."""
    
    @staticmethod
    def web_search(query: str) -> Dict[str, Any]:
        """
        Perform a web search.
        
        Args:
            query: Search query
            
        Returns:
            Dict containing success status and search URL
        """
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            return {"success": True, "output": f"Opened web search: {search_url}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def open_url(url: str) -> Dict[str, Any]:
        """
        Open a URL in the default browser.
        
        Args:
            url: URL to open
            
        Returns:
            Dict containing success status and any output/error
        """
        try:
            webbrowser.open(url)
            return {"success": True, "output": f"Opened URL: {url}"}
        except Exception as e:
            return {"success": False, "error": str(e)} 