import os
import sys
import subprocess
import platform
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field
import pyautogui

from utils.logger import setup_logger

class SystemSkills(BaseModel):
    """System skills for Jarvis to interact with the operating system"""
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("system_skills")
        self.system = platform.system()
    
    def execute(self, intent: str, parameters: Dict[str, Any]) -> str:
        """Execute a system-related intent"""
        self.logger.info(f"Executing system intent: {intent} with parameters: {parameters}")
        
        if intent == "system.open":
            return self.open_application(parameters.get("target", ""))
        elif intent == "system.close":
            return self.close_application(parameters.get("target", ""))
        else:
            return f"Unknown system intent: {intent}"
    
    def open_application(self, app_name: str) -> str:
        """Open an application by name"""
        if not app_name:
            return "No application specified"
        
        app_name = app_name.lower()
        self.logger.info(f"Attempting to open application: {app_name}")
        
        try:
            # Windows-specific application handling
            if self.system == "Windows":
                common_apps = {
                    "chrome": "chrome",
                    "google chrome": "chrome",
                    "firefox": "firefox",
                    "edge": "msedge",
                    "microsoft edge": "msedge",
                    "word": "winword",
                    "excel": "excel",
                    "powerpoint": "powerpnt",
                    "notepad": "notepad",
                    "calculator": "calc",
                    "file explorer": "explorer",
                    "explorer": "explorer",
                    "cmd": "cmd",
                    "command prompt": "cmd",
                    "powershell": "powershell",
                    "visual studio": "devenv",
                    "vs code": "code",
                    "visual studio code": "code"
                }
                
                if app_name in common_apps:
                    app_name = common_apps[app_name]
                
                subprocess.Popen(f"start {app_name}", shell=True)
                return f"Opening {app_name}"
            
            # macOS-specific application handling
            elif self.system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
                return f"Opening {app_name}"
            
            # Linux-specific application handling
            elif self.system == "Linux":
                subprocess.Popen([app_name])
                return f"Opening {app_name}"
            
            else:
                return f"Unsupported operating system: {self.system}"
        
        except Exception as e:
            self.logger.error(f"Error opening application {app_name}: {e}")
            return f"Failed to open {app_name}: {str(e)}"
    
    def close_application(self, app_name: str) -> str:
        """Close an application by name"""
        if not app_name:
            return "No application specified"
        
        app_name = app_name.lower()
        self.logger.info(f"Attempting to close application: {app_name}")
        
        try:
            # Windows-specific application handling
            if self.system == "Windows":
                # Map common names to process names
                common_apps = {
                    "chrome": "chrome.exe",
                    "google chrome": "chrome.exe",
                    "firefox": "firefox.exe",
                    "edge": "msedge.exe",
                    "microsoft edge": "msedge.exe",
                    "word": "winword.exe",
                    "excel": "excel.exe",
                    "powerpoint": "powerpnt.exe",
                    "notepad": "notepad.exe",
                    "calculator": "calc.exe",
                    "cmd": "cmd.exe",
                    "command prompt": "cmd.exe",
                    "powershell": "powershell.exe",
                    "visual studio": "devenv.exe",
                    "vs code": "code.exe",
                    "visual studio code": "code.exe"
                }
                
                process_name = common_apps.get(app_name, f"{app_name}.exe")
                subprocess.run(f"taskkill /f /im {process_name}", shell=True)
                return f"Closing {app_name}"
            
            # macOS-specific application handling
            elif self.system == "Darwin":
                subprocess.run(["killall", app_name])
                return f"Closing {app_name}"
            
            # Linux-specific application handling
            elif self.system == "Linux":
                subprocess.run(["pkill", app_name])
                return f"Closing {app_name}"
            
            else:
                return f"Unsupported operating system: {self.system}"
        
        except Exception as e:
            self.logger.error(f"Error closing application {app_name}: {e}")
            return f"Failed to close {app_name}: {str(e)}"
    
    def take_screenshot(self) -> str:
        """Take a screenshot and save it"""
        try:
            # Create screenshots directory if it doesn't exist
            screenshots_dir = "screenshots"
            os.makedirs(screenshots_dir, exist_ok=True)
            
            # Generate filename based on timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            
            return f"Screenshot saved to {filename}"
        
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return f"Failed to take screenshot: {str(e)}"
    
    def get_system_info(self) -> str:
        """Get system information"""
        try:
            info = {
                "System": platform.system(),
                "Node": platform.node(),
                "Release": platform.release(),
                "Version": platform.version(),
                "Machine": platform.machine(),
                "Processor": platform.processor(),
                "Python": sys.version
            }
            
            # Format as string
            info_str = "\n".join([f"{key}: {value}" for key, value in info.items()])
            
            return f"System Information:\n{info_str}"
        
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return f"Failed to get system information: {str(e)}" 