from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union
import subprocess
import os
import json
import time
from datetime import datetime
import platform
import re
import threading
import queue
import traceback
from pathlib import Path
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pyttsx3
    SPEECH_SYNTHESIS_AVAILABLE = True
except ImportError:
    SPEECH_SYNTHESIS_AVAILABLE = False

# --- Check for LLM API libraries ---
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# --- Memory System ---

class Memory(BaseModel):
    """Base model for Jarvis' memory system."""
    content: Any
    timestamp: float = Field(default_factory=time.time)
    
    @property
    def age(self) -> float:
        """Returns the age of this memory in seconds."""
        return time.time() - self.timestamp

class CoreMemory(Memory):
    """Long-term, permanent memory for critical information and programming."""
    type: Literal["core"] = "core"
    description: str

class ObjectiveMemory(Memory):
    """Medium-term memory for tracking active objectives and context."""
    type: Literal["objective"] = "objective"
    objective: str
    progress: float = 0.0
    completed: bool = False
    sub_tasks: List[str] = Field(default_factory=list)
    priority: int = 1  # 1-5, where 5 is highest priority

class ContextMemory(Memory):
    """Short-term memory for conversation and immediate task context."""
    type: Literal["context"] = "context"
    expires_after: float = 3600  # Default to 1 hour in seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if this context memory has expired."""
        return self.age > self.expires_after

class MemorySystem:
    """Manages Jarvis' different memory systems."""
    
    def __init__(self, memory_dir="./memory"):
        self.memory_dir = memory_dir
        self.core_memory_file = os.path.join(memory_dir, "core_memory.json")
        self.objective_memory_file = os.path.join(memory_dir, "objective_memory.json")
        self.context_memories: List[ContextMemory] = []
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize memory files if they don't exist
        self._init_memory_files()
        
        # Load existing memories
        self.core_memories = self._load_core_memories()
        self.objective_memories = self._load_objective_memories()
    
    def _init_memory_files(self):
        """Initialize memory files if they don't exist."""
        if not os.path.exists(self.core_memory_file):
            with open(self.core_memory_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.objective_memory_file):
            with open(self.objective_memory_file, 'w') as f:
                json.dump([], f)
    
    def _load_core_memories(self) -> List[CoreMemory]:
        """Load core memories from disk."""
        try:
            with open(self.core_memory_file, 'r') as f:
                data = json.load(f)
                return [CoreMemory(**item) for item in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _load_objective_memories(self) -> List[ObjectiveMemory]:
        """Load objective memories from disk."""
        try:
            with open(self.objective_memory_file, 'r') as f:
                data = json.load(f)
                return [ObjectiveMemory(**item) for item in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_core_memories(self):
        """Save core memories to disk."""
        with open(self.core_memory_file, 'w') as f:
            json.dump([memory.model_dump() for memory in self.core_memories], f, indent=2)
    
    def _save_objective_memories(self):
        """Save objective memories to disk."""
        with open(self.objective_memory_file, 'w') as f:
            json.dump([memory.model_dump() for memory in self.objective_memories], f, indent=2)
    
    def add_core_memory(self, content: Any, description: str):
        """Add a new core memory (long-term)."""
        memory = CoreMemory(content=content, description=description)
        self.core_memories.append(memory)
        self._save_core_memories()
        return memory
    
    def add_objective_memory(self, objective: str, sub_tasks: List[str] = None, priority: int = 1):
        """Add a new objective memory (medium-term)."""
        memory = ObjectiveMemory(
            content={"objective": objective, "created": datetime.now().isoformat()},
            objective=objective,
            sub_tasks=sub_tasks or [],
            priority=priority
        )
        self.objective_memories.append(memory)
        self._save_objective_memories()
        return memory
    
    def add_context_memory(self, content: Any, expires_after: float = 3600):
        """Add a new context memory (short-term)."""
        memory = ContextMemory(content=content, expires_after=expires_after)
        self.context_memories.append(memory)
        return memory
    
    def get_active_objectives(self) -> List[ObjectiveMemory]:
        """Get all active (non-completed) objectives, sorted by priority."""
        active = [mem for mem in self.objective_memories if not mem.completed]
        return sorted(active, key=lambda x: x.priority, reverse=True)
    
    def get_context_memories(self, max_age: float = None) -> List[ContextMemory]:
        """Get context memories, optionally filtering by maximum age in seconds."""
        # Clean expired memories first
        self.context_memories = [mem for mem in self.context_memories if not mem.is_expired]
        
        if max_age is not None:
            return [mem for mem in self.context_memories if mem.age <= max_age]
        return self.context_memories
    
    def search_memory(self, query: str) -> Dict[str, List[Any]]:
        """Simple search across all memory types."""
        # In a real implementation, this would use semantic search/embeddings
        results = {
            "core": [],
            "objective": [],
            "context": []
        }
        
        # Simple substring matching for now
        for memory in self.core_memories:
            if query.lower() in str(memory.content).lower() or query.lower() in memory.description.lower():
                results["core"].append(memory)
        
        for memory in self.objective_memories:
            if query.lower() in memory.objective.lower():
                results["objective"].append(memory)
        
        for memory in self.context_memories:
            if query.lower() in str(memory.content).lower():
                results["context"].append(memory)
        
        return results
    
    def complete_objective(self, objective_description: str):
        """Mark an objective as completed."""
        for memory in self.objective_memories:
            if memory.objective == objective_description:
                memory.completed = True
                memory.progress = 1.0
                self._save_objective_memories()
                return True
        return False
    
    def update_objective_progress(self, objective_description: str, progress: float):
        """Update the progress of an objective."""
        for memory in self.objective_memories:
            if memory.objective == objective_description:
                memory.progress = min(max(progress, 0.0), 1.0)
                self._save_objective_memories()
                return True
        return False

# --- Define Command Schemas ---

class BaseCommand(BaseModel):
    """Base model for all Jarvis commands."""
    command_type: str = Field(..., description="The type of command to execute.")
    thought: Optional[str] = Field(None, description="Jarvis's reasoning for choosing this command.")

class ExecuteTerminalCommand(BaseCommand):
    """Executes a shell command."""
    command_type: Literal["execute_terminal"] = "execute_terminal"
    command: str = Field(..., description="The terminal command to execute.")
    response: Optional[str] = Field(None, description="The output of the command.")

class OpenApplicationCommand(BaseCommand):
    """Opens a specified application."""
    command_type: Literal["open_application"] = "open_application"
    application_name: str = Field(..., description="The name or path of the application to open.")

class SearchFilesCommand(BaseCommand):
    """Searches for files on the system."""
    command_type: Literal["search_files"] = "search_files"
    query: str = Field(..., description="The search query (filename, pattern, etc.).")
    directory: Optional[str] = Field(None, description="The directory to search within (defaults to user's home).")
    results: Optional[List[str]] = Field(None, description="List of found file paths.")

class BrowseWebCommand(BaseCommand):
    """Browse the web or search for information."""
    command_type: Literal["browse_web"] = "browse_web"
    url: str = Field(..., description="The URL to open or search query.")
    search_engine: Optional[str] = Field("google", description="Search engine to use if the URL is a search query.")

class ModifyFileCommand(BaseCommand):
    """Modify the contents of a text file."""
    command_type: Literal["modify_file"] = "modify_file"
    file_path: str = Field(..., description="Path to the file to modify.")
    operation: Literal["append", "prepend", "replace", "delete_lines"] = Field(..., description="Type of modification.")
    content: Optional[str] = Field(None, description="Content to write (for append, prepend, replace operations).")
    line_start: Optional[int] = Field(None, description="Starting line for operations (0-indexed).")
    line_end: Optional[int] = Field(None, description="Ending line for operations (0-indexed).")

class SetObjectiveCommand(BaseCommand):
    """Set a new objective for Jarvis to track."""
    command_type: Literal["set_objective"] = "set_objective"
    objective: str = Field(..., description="Description of the objective.")
    sub_tasks: Optional[List[str]] = Field(None, description="List of sub-tasks to complete the objective.")
    priority: Optional[int] = Field(1, description="Priority level (1-5, where 5 is highest).")

class RecallMemoryCommand(BaseCommand):
    """Recall information from Jarvis's memory systems."""
    command_type: Literal["recall_memory"] = "recall_memory"
    query: str = Field(..., description="The information to recall or search for.")
    memory_type: Optional[Literal["core", "objective", "context", "all"]] = Field("all", description="Type of memory to search in.")

# Combined command type
JarvisCommand = (
    ExecuteTerminalCommand | 
    OpenApplicationCommand | 
    SearchFilesCommand | 
    BrowseWebCommand | 
    ModifyFileCommand | 
    SetObjectiveCommand | 
    RecallMemoryCommand
)

# --- Command Patterns for Simple Parsing ---
COMMAND_PATTERNS = [
    # Terminal commands
    (r"(?:run|execute|terminal)\s+(?:command)?\s*:\s*(.+)", "execute_terminal"),
    
    # Open application
    (r"(?:open|launch|start)\s+(?:app|application)?\s*:\s*(.+)", "open_application"),
    
    # Search files
    (r"(?:search|find)\s+(?:file|files)\s*:\s*(.+?)(?:\s+in\s+(.+))?$", "search_files"),
    
    # Browse web
    (r"(?:browse|search|open|go to)\s+(?:web|internet|url|website)?\s*:\s*(.+?)(?:\s+with\s+(.+))?$", "browse_web"),
    
    # Modify file
    (r"(?:modify|edit|change|append to|prepend to|replace in|delete lines from)\s+file\s*:\s*(.+?)\s+(append|prepend|replace|delete_lines)\s+(?:content\s*:\s*)?(.+?)(?:\s+from line\s+(\d+))?(?:\s+to line\s+(\d+))?$", "modify_file"),
    
    # Set objective
    (r"(?:set|create|add)\s+objective\s*:\s*(.+?)(?:\s+with priority\s+(\d+))?(?:\s+subtasks\s*:\s*(.+))?$", "set_objective"),
    
    # Recall memory
    (r"(?:recall|remember|search memory|what do you know about)\s*:\s*(.+?)(?:\s+in\s+(core|objective|context|all)\s+memory)?$", "recall_memory")
]

# --- Simple Command Parser ---
def parse_command(text: str) -> JarvisCommand:
    """Parse a natural language command into a JarvisCommand object."""
    for pattern, command_type in COMMAND_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            if command_type == "execute_terminal":
                return ExecuteTerminalCommand(
                    command=match.group(1).strip(),
                    thought=f"The user wants me to execute a terminal command: {match.group(1).strip()}"
                )
            elif command_type == "open_application":
                return OpenApplicationCommand(
                    application_name=match.group(1).strip(),
                    thought=f"The user wants me to open the application: {match.group(1).strip()}"
                )
            elif command_type == "search_files":
                return SearchFilesCommand(
                    query=match.group(1).strip(),
                    directory=match.group(2).strip() if match.group(2) else None,
                    thought=f"The user wants to search for files matching: {match.group(1).strip()}"
                )
            elif command_type == "browse_web":
                return BrowseWebCommand(
                    url=match.group(1).strip(),
                    search_engine=match.group(2).strip() if match.group(2) else "google",
                    thought=f"The user wants to browse the web at: {match.group(1).strip()}"
                )
            elif command_type == "modify_file":
                return ModifyFileCommand(
                    file_path=match.group(1).strip(),
                    operation=match.group(2).strip(),
                    content=match.group(3).strip() if match.group(3) else None,
                    line_start=int(match.group(4)) if match.group(4) else None,
                    line_end=int(match.group(5)) if match.group(5) else None,
                    thought=f"The user wants to {match.group(2).strip()} content in file: {match.group(1).strip()}"
                )
            elif command_type == "set_objective":
                return SetObjectiveCommand(
                    objective=match.group(1).strip(),
                    priority=int(match.group(2)) if match.group(2) else 1,
                    sub_tasks=match.group(3).strip().split(",") if match.group(3) else None,
                    thought=f"The user wants to set an objective: {match.group(1).strip()}"
                )
            elif command_type == "recall_memory":
                return RecallMemoryCommand(
                    query=match.group(1).strip(),
                    memory_type=match.group(2).strip() if match.group(2) else "all",
                    thought=f"The user wants to recall information about: {match.group(1).strip()}"
                )
    
    # If no pattern matches, default to executing as terminal command
    if text.strip():
        return ExecuteTerminalCommand(
            command=text.strip(),
            thought="No specific command pattern matched, interpreting as a terminal command."
        )
    
    return None

# --- LLM Integration ---
class LLMIntegration:
    """Manages the integration with Language Model APIs."""
    
    def __init__(self):
        """Initialize the LLM integration with available API keys."""
        self.available_clients = []
        self.active_client = None
        self.active_model = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Set up connections to various LLM providers based on available API keys."""
        # Check for Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and GROQ_AVAILABLE:
            try:
                groq_client = groq.Client(api_key=groq_api_key)
                self.available_clients.append(("groq", groq_client))
                if not self.active_client:
                    self.active_client = groq_client
                    self.active_model = "llama3-8b-8192"  # Default model
                    print("Groq LLM initialized as primary model.")
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
        
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                openai.api_key = openai_api_key
                self.available_clients.append(("openai", openai))
                if not self.active_client:
                    self.active_client = openai
                    self.active_model = "gpt-3.5-turbo"  # Default model
                    print("OpenAI LLM initialized as primary model.")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
        
        # Check for Anthropic API key
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                self.available_clients.append(("anthropic", anthropic_client))
                if not self.active_client:
                    self.active_client = anthropic_client
                    self.active_model = "claude-3-haiku-20240307"  # Default model
                    print("Anthropic LLM initialized as primary model.")
            except Exception as e:
                print(f"Error initializing Anthropic client: {e}")
        
        # If no clients available
        if not self.available_clients:
            print("No LLM clients available. Install an API library and set the API key.")
    
    def process_with_llm(self, user_input: str, system_prompt: str = None) -> str:
        """Process user input with the active LLM model."""
        if not self.active_client:
            return "No LLM model available. Please check your API keys and installed libraries."
            
        try:
            client_name = self.available_clients[0][0] if self.available_clients else None
            
            if client_name == "groq":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_input})
                
                response = self.active_client.chat.completions.create(
                    model=self.active_model,
                    messages=messages,
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
                
            elif client_name == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_input})
                
                response = self.active_client.chat.completions.create(
                    model=self.active_model,
                    messages=messages,
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
                
            elif client_name == "anthropic":
                content = []
                if system_prompt:
                    content.append({"type": "text", "text": system_prompt})
                content.append({"type": "text", "text": user_input})
                
                response = self.active_client.messages.create(
                    model=self.active_model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": content}]
                )
                return response.content[0].text
                
            else:
                return "No supported LLM client found."
                
        except Exception as e:
            error_msg = f"Error processing with LLM: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
    
    def generate_command_from_natural_language(self, user_input: str, available_commands: List[str]) -> str:
        """Use LLM to convert natural language to a structured command."""
        if not self.active_client:
            return None
            
        system_prompt = f"""
        You are Jarvis, an AI assistant that helps users perform tasks on their computer.
        Convert the user's natural language request into one of the following command formats:
        
        {'\n'.join(available_commands)}
        
        If the user's request doesn't match any command format, respond with the most appropriate command.
        Only respond with the command, nothing else.
        """
        
        try:
            response = self.process_with_llm(user_input, system_prompt)
            return response
        except Exception as e:
            print(f"Error generating command: {e}")
            traceback.print_exc()
            return None

# --- Voice Integration ---
class VoiceSystem:
    """Handles speech recognition and synthesis for Jarvis."""
    
    def __init__(self, voice_enabled=True, voice_rate=200):
        """Initialize the voice system if dependencies are available."""
        self.voice_enabled = voice_enabled and SPEECH_SYNTHESIS_AVAILABLE
        self.listen_enabled = voice_enabled and SPEECH_RECOGNITION_AVAILABLE
        self.voice_rate = voice_rate
        self.engine = None
        self.recognizer = None
        
        if self.voice_enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', self.voice_rate)
                
                # Set a voice that sounds like Jarvis if available
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if "david" in voice.name.lower() or "mark" in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            except Exception as e:
                print(f"Error initializing speech synthesis: {e}")
                self.voice_enabled = False
        
        if self.listen_enabled:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 4000
                self.recognizer.dynamic_energy_threshold = True
            except Exception as e:
                print(f"Error initializing speech recognition: {e}")
                self.listen_enabled = False
    
    def speak(self, text):
        """Convert text to speech if voice is enabled."""
        if self.voice_enabled and self.engine:
            try:
                print(f"Jarvis: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
                return True
            except Exception as e:
                print(f"Error with speech synthesis: {e}")
                return False
        else:
            # Just print the text if voice is not enabled
            print(f"Jarvis: {text}")
            return False
    
    def listen(self, timeout=5, phrase_time_limit=None):
        """Listen for voice commands and convert to text."""
        if not self.listen_enabled or not self.recognizer:
            return None
            
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
                print("Processing speech...")
                return self.recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        except Exception as e:
            print(f"Error with speech recognition: {e}")
            return None

# --- Jarvis Agent ---
class JarvisAgent:
    def __init__(self, voice_enabled=True):
        # Initialize memory system
        self.memory = MemorySystem()
        
        # Initialize voice system
        self.voice = VoiceSystem(voice_enabled=voice_enabled)
        
        # Initialize LLM integration
        self.llm = LLMIntegration()
        
        # Set up command queue for background processing
        self.command_queue = queue.Queue()
        self.background_thread = None
        self.is_running = False
        
        # Add basic core memories
        self._initialize_core_memories()
        
        # Add system info to context memory
        self._initialize_system_context()
        
        # Available command formats for natural language processing
        self.command_formats = [
            "execute terminal: [command]",
            "open application: [app_name]",
            "search files: [query] in [directory]",
            "browse web: [url/query] with [search_engine]",
            "modify file: [path] [operation] content: [text] from line [#] to [#]",
            "set objective: [description] with priority [1-5] subtasks: [a,b,c]",
            "recall memory: [query] in [memory_type]"
        ]
    
    def _initialize_core_memories(self):
        """Initialize basic core memories if they don't exist."""
        if not self.memory.core_memories:
            self.memory.add_core_memory(
                content={
                    "identity": "Jarvis",
                    "purpose": "I am an autonomous AI assistant designed to help with various tasks on this computer.",
                    "capabilities": [
                        "Executing terminal commands",
                        "Opening applications",
                        "Searching for files",
                        "Browsing the web",
                        "Modifying text files",
                        "Managing objectives and tasks",
                        "Voice interaction" if self.voice.voice_enabled else "Text interaction"
                    ]
                },
                description="Jarvis core identity and purpose"
            )
            
            # Add operating system info to core memory
            self.memory.add_core_memory(
                content={
                    "operating_system": platform.system(),
                    "os_version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                },
                description="Host system information"
            )
    
    def _initialize_system_context(self):
        """Add system information to context memory."""
        system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now().isoformat(),
            "voice_enabled": self.voice.voice_enabled,
            "listen_enabled": self.voice.listen_enabled,
            "llm_enabled": bool(self.llm.active_client)
        }
        self.memory.add_context_memory(
            content={"system_info": system_info},
            expires_after=86400 * 7  # Keep for a week
        )
        
        # Add user's home directory to context
        user_home = os.path.expanduser("~")
        self.memory.add_context_memory(
            content={"user_home": user_home},
            expires_after=86400 * 30  # Keep for a month
        )

    def listen(self) -> str:
        """Listens for user input via voice or text."""
        if self.voice.listen_enabled:
            # Try to listen via voice
            voice_input = self.voice.listen()
            if voice_input:
                print(f"You (voice): {voice_input}")
                return voice_input
        
        # Fall back to text input
        return input("You: ")

    def understand(self, user_input: str) -> JarvisCommand | str:
        """Parse user input into a structured command, using LLM if available."""
        try:
            # Store the user input in context memory
            self.memory.add_context_memory(
                content={"user_input": user_input, "timestamp": datetime.now().isoformat()},
                expires_after=3600  # Keep for an hour
            )
            
            # First try pattern matching
            parsed_command = parse_command(user_input)
            if parsed_command:
                return parsed_command
                
            # If pattern matching fails and LLM is available, try LLM parsing
            if self.llm.active_client:
                llm_parsed_command = self.llm.generate_command_from_natural_language(
                    user_input, self.command_formats
                )
                if llm_parsed_command:
                    # Try parsing again with LLM-generated command
                    try:
                        return parse_command(llm_parsed_command)
                    except Exception as e:
                        print(f"Error parsing LLM command: {e}")
            
            # If all else fails
            return f"I'm not sure how to process: '{user_input}'. Please try a different command format."
        except Exception as e:
            print(f"Error understanding command: {e}")
            traceback.print_exc()
            # Fallback or error handling
            return f"Sorry, I couldn't parse that command. Error: {e}"

    def execute(self, command: JarvisCommand) -> str:
        """Executes the parsed command."""
        print(f"Executing: {command.command_type}")
        if command.thought:
            print(f"Thought: {command.thought}")

        try:
            # Store command in context memory
            self.memory.add_context_memory(
                content={"executed_command": command.model_dump()},
                expires_after=3600  # Keep for an hour
            )
            
            if isinstance(command, ExecuteTerminalCommand):
                # Security Note: Be VERY careful executing arbitrary commands.
                # Consider sandboxing or whitelisting allowed commands.
                result = subprocess.run(command.command, shell=True, capture_output=True, text=True, check=True)
                response = f"""Command executed.
STDOUT:
{result.stdout}
STDERR:
{result.stderr}"""
                command.response = result.stdout or result.stderr # Store output back if needed
                
            elif isinstance(command, OpenApplicationCommand):
                # Cross-platform application opening
                if platform.system() == "Windows":
                    try:
                        subprocess.run(['start', '', command.application_name], shell=True, check=True)
                        response = f"Opened {command.application_name}."
                    except subprocess.CalledProcessError:
                        response = f"Failed to open {command.application_name}. Make sure it exists."
                elif platform.system() == "Darwin":  # macOS
                    try:
                        subprocess.run(['open', command.application_name], check=True)
                        response = f"Opened {command.application_name}."
                    except subprocess.CalledProcessError:
                        response = f"Failed to open {command.application_name}. Make sure it exists."
                else:  # Linux and others
                    try:
                        subprocess.run(['xdg-open', command.application_name], check=True)
                        response = f"Opened {command.application_name}."
                    except subprocess.CalledProcessError:
                        response = f"Failed to open {command.application_name}. Make sure it exists."
                
            elif isinstance(command, SearchFilesCommand):
                # Basic file search example
                search_dir = command.directory or os.path.expanduser("~")
                found_files = []
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        if command.query.lower() in file.lower():  # Case-insensitive matching
                            found_files.append(os.path.join(root, file))
                command.results = found_files[:20]  # Limit results
                response = f"Found {len(found_files)} files matching '{command.query}'. First {len(command.results)} listed."
                if command.results:
                    response += "\n" + "\n".join(command.results)
            
            elif isinstance(command, BrowseWebCommand):
                # Open URL or search query in default browser
                url = command.url
                if not url.startswith(('http://', 'https://')):
                    # Treat as search query
                    search_engines = {
                        "google": "https://www.google.com/search?q=",
                        "bing": "https://www.bing.com/search?q=",
                        "duckduckgo": "https://duckduckgo.com/?q="
                    }
                    base_url = search_engines.get(command.search_engine.lower(), search_engines["google"])
                    url = f"{base_url}{url.replace(' ', '+')}"
                
                if platform.system() == "Windows":
                    os.startfile(url)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(['open', url], check=True)
                else:  # Linux and others
                    subprocess.run(['xdg-open', url], check=True)
                
                response = f"Opened {url} in your default browser."
            
            elif isinstance(command, ModifyFileCommand):
                if not os.path.exists(command.file_path):
                    return f"Error: File '{command.file_path}' does not exist."
                
                try:
                    with open(command.file_path, 'r', encoding='utf-8') as file:
                        content = file.readlines()
                    
                    if command.operation == "append":
                        with open(command.file_path, 'a', encoding='utf-8') as file:
                            file.write(command.content)
                        response = f"Appended content to {command.file_path}."
                        
                    elif command.operation == "prepend":
                        with open(command.file_path, 'w', encoding='utf-8') as file:
                            file.write(command.content)
                            file.writelines(content)
                        response = f"Prepended content to {command.file_path}."
                        
                    elif command.operation == "replace":
                        line_start = command.line_start or 0
                        line_end = command.line_end or len(content)
                        
                        # Validate line numbers
                        if line_start < 0 or line_end > len(content) or line_start > line_end:
                            return f"Error: Invalid line numbers. File has {len(content)} lines."
                        
                        new_content = []
                        for i, line in enumerate(content):
                            if i < line_start or i > line_end:
                                new_content.append(line)
                            elif i == line_start:
                                new_content.append(command.content + '\n')
                        
                        with open(command.file_path, 'w', encoding='utf-8') as file:
                            file.writelines(new_content)
                        response = f"Replaced content in {command.file_path} from line {line_start} to {line_end}."
                        
                    elif command.operation == "delete_lines":
                        line_start = command.line_start or 0
                        line_end = command.line_end or len(content)
                        
                        # Validate line numbers
                        if line_start < 0 or line_end > len(content) or line_start > line_end:
                            return f"Error: Invalid line numbers. File has {len(content)} lines."
                        
                        new_content = [line for i, line in enumerate(content) if i < line_start or i > line_end]
                        
                        with open(command.file_path, 'w', encoding='utf-8') as file:
                            file.writelines(new_content)
                        response = f"Deleted lines {line_start} to {line_end} in {command.file_path}."
                    
                    else:
                        response = f"Unsupported file operation: {command.operation}"
                
                except Exception as e:
                    response = f"Error modifying file: {str(e)}"
            
            elif isinstance(command, SetObjectiveCommand):
                # Add a new objective to memory
                objective = self.memory.add_objective_memory(
                    objective=command.objective,
                    sub_tasks=command.sub_tasks or [],
                    priority=command.priority or 1
                )
                response = f"New objective set: '{command.objective}' with priority {objective.priority}."
                if objective.sub_tasks:
                    response += f"\nSub-tasks: {', '.join(objective.sub_tasks)}"
            
            elif isinstance(command, RecallMemoryCommand):
                # Search memory systems
                results = self.memory.search_memory(command.query)
                
                memory_types_to_search = []
                if command.memory_type == "all":
                    memory_types_to_search = ["core", "objective", "context"]
                else:
                    memory_types_to_search = [command.memory_type]
                
                response = f"Memory recall for '{command.query}':\n"
                for mem_type in memory_types_to_search:
                    if results[mem_type]:
                        response += f"\n{mem_type.capitalize()} memories ({len(results[mem_type])}):\n"
                        for i, memory in enumerate(results[mem_type][:5]):  # Show top 5
                            if mem_type == "core":
                                response += f"- {memory.description}: {str(memory.content)[:100]}...\n"
                            elif mem_type == "objective":
                                response += f"- {memory.objective} (Priority: {memory.priority}, Progress: {int(memory.progress * 100)}%)\n"
                            else:  # context
                                response += f"- {str(memory.content)[:100]}...\n"
                    else:
                        response += f"\nNo {mem_type} memories found for '{command.query}'.\n"
            
            else:
                response = f"Command type '{command.command_type}' not implemented yet."

            # Store the result in context memory
            self.memory.add_context_memory(
                content={"command_result": response[:500] + "..." if len(response) > 500 else response},
                expires_after=3600  # Keep for an hour
            )
            
            return response

        except subprocess.CalledProcessError as e:
            error_msg = f"Error executing command '{command.command_type}': {e}"
            if hasattr(e, 'stderr'):
                error_msg += f"\nSTDERR: {e.stderr}"
            
            # Store the error in context memory
            self.memory.add_context_memory(
                content={"command_error": error_msg},
                expires_after=3600  # Keep for an hour
            )
            
            return error_msg
        except Exception as e:
            error_msg = f"Error during execution: {e}"
            traceback.print_exc()
            
            # Store the error in context memory
            self.memory.add_context_memory(
                content={"command_error": error_msg},
                expires_after=3600  # Keep for an hour
            )
            
            return error_msg
    
    def _background_process(self):
        """Background thread to process commands from the queue."""
        while self.is_running:
            try:
                command = self.command_queue.get(timeout=0.5)
                if command == "STOP":
                    break
                    
                response = self.execute(command)
                # Speak the response if voice is enabled
                self.voice.speak(response)
                
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in background processing: {e}")
                traceback.print_exc()
    
    def start_background_processing(self):
        """Start the background processing thread."""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.is_running = True
            self.background_thread = threading.Thread(target=self._background_process)
            self.background_thread.daemon = True
            self.background_thread.start()
    
    def stop_background_processing(self):
        """Stop the background processing thread."""
        if self.background_thread and self.background_thread.is_alive():
            self.is_running = False
            self.command_queue.put("STOP")
            self.background_thread.join(timeout=2)

    def run_loop(self):
        """Main agent loop: Listen -> Understand -> Execute -> Respond."""
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                           JARVIS ACTIVATED                               ║
║                                                                          ║
║  Commands:                                                               ║
║  - execute terminal: [command]                                           ║
║  - open application: [app_name]                                          ║
║  - search files: [query] in [directory]                                  ║
║  - browse web: [url/query] with [search_engine]                          ║
║  - modify file: [path] [operation] content: [text] from line [#] to [#]  ║
║  - set objective: [description] with priority [1-5] subtasks: [a,b,c]    ║
║  - recall memory: [query] in [memory_type]                               ║
║                                                                          ║
║  Type 'exit', 'quit', or 'stop' to shut down Jarvis.                     ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
        # Start the background processing thread
        self.start_background_processing()
        
        # Welcome message
        welcome_msg = "Jarvis initialized and ready to assist."
        if self.voice.voice_enabled:
            self.voice.speak(welcome_msg)
        else:
            print(f"Jarvis: {welcome_msg}")
        
        try:
            while True:
                user_input = self.listen()
                if user_input.lower() in ["exit", "quit", "stop", "jarvis stop"]:
                    farewell = "Jarvis shutting down."
                    if self.voice.voice_enabled:
                        self.voice.speak(farewell)
                    else:
                        print(f"Jarvis: {farewell}")
                    break

                parsed_command = self.understand(user_input)

                if isinstance(parsed_command, BaseCommand):
                    if isinstance(parsed_command, RecallMemoryCommand):
                        # Execute memory-related commands immediately
                        response = self.execute(parsed_command)
                        if self.voice.voice_enabled:
                            self.voice.speak(response)
                        else:
                            print(f"Jarvis: {response}")
                    else:
                        # Queue other commands for background processing
                        self.command_queue.put(parsed_command)
                        processing_msg = "Processing your request."
                        if self.voice.voice_enabled:
                            self.voice.speak(processing_msg)
                        else:
                            print(f"Jarvis: {processing_msg}")
                else:
                    # Handle cases where understanding failed
                    if self.voice.voice_enabled:
                        self.voice.speak(parsed_command)
                    else:
                        print(f"Jarvis: {parsed_command}")
        finally:
            # Make sure to stop the background thread when exiting
            self.stop_background_processing()

# --- Main Execution ---
if __name__ == "__main__":
    jarvis = JarvisAgent(voice_enabled=True)
    jarvis.run_loop()
