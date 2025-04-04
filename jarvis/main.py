#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant

Main entry point for Jarvis that integrates all components.

Phase 0:
- [X] Replace CrewAI Planner
- [X] Fix Synthesis Execution Bug (plan_id metadata added)
- [X] Resolve Pydantic `LLMClient.model_rebuild()` Crash (Warning deferred)
- [X] Fix/Test LLM Skill Parsing Fallback (Underlying LLM call fixed)
- [X] Address Logging Inconsistency (Library levels configured)
- [ ] Test & Secure Core Skills (Security added, Testing blocked by file paths)

Phase 1: (Next)
- [ ] Define/Refine Non-Planning Agent Roles
- [ ] Defer CrewAI Planning Crew
- [ ] Setup/Refine CrewAI Project Structure (If Applicable)
- [ ] Implement Initial Agent Tools
"""

import os
import sys
import argparse
import threading
import queue
from typing import Optional, TYPE_CHECKING
from datetime import datetime
import logging # For type hinting Logger
import time
from pydantic import ValidationError
import asyncio
from dotenv import load_dotenv, find_dotenv
import json
import warnings # <<< ADDED
from jarvis.config import settings # <<< ADDED

# --- Load .env file EARLY --- <<< ADDED
# This ensures environment variables from .env are available when Pydantic settings are loaded.
load_dotenv(find_dotenv(raise_error_if_not_found=False)) 
# --- END Load .env ---

# --- Suppress specific Pydantic V1/V2 warning --- ADDED
# This warning often originates from dependencies (like langchain) using older
# Pydantic versions or constructs. Suppressing it allows development to proceed
# but should be revisited when dependencies are updated.
warnings.filterwarnings(
    "ignore",
    message="Mixing V1 models and V2 models.*is not supported.*upgrade `Settings` to V2",
    category=UserWarning,
    module="pydantic._internal._generate_schema" # Be specific about the source
)
# --- END Warning Suppression ---

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Conditional imports for type checking to avoid circular dependencies
if TYPE_CHECKING:
    from .agent import JarvisAgent
    from .memory.unified_memory import UnifiedMemorySystem
    from .planning import PlanningSystem, Task
    from .execution import ExecutionSystem, ExecutionResult
    # from utils.speech import SpeechRecognizer, TextToSpeech
    
# Import for runtime
from .agent import JarvisAgent 
from .memory.unified_memory import UnifiedMemorySystem
from .planning import PlanningSystem
from .execution import ExecutionSystem
from .llm import LLMClient # Import LLMClient
from .graph import build_graph # Import graph builder
from .state import JarvisState # Import state definition
from .skills.registry import SkillRegistry # <-- ADDED Import
from utils.logger import setup_logger
from .state import UserInput # Correct path to UserInput model
# from utils.speech import SpeechRecognizer, TextToSpeech # Commented out - file missing

# Import specific skills to register
from .skills.web_search import WebSearchSkill
from .skills.execute_python_file import ExecutePythonFileSkill

# --- Centralized Pydantic Model Rebuild ---
try:
    print("Attempting centralized Pydantic model rebuild...")
    # Order matters: Build dependencies first if they have forward refs
    # Assuming LLMClient, SkillRegistry, Memory components might be used in others
    LLMClient.model_rebuild()
    UnifiedMemorySystem.model_rebuild() # Rebuild Memory first
    # SkillRegistry.model_rebuild() # REMOVED - SkillRegistry is not a Pydantic model
    # Now rebuild systems that depend on the above
    PlanningSystem.model_rebuild() # Planning depends on Memory/LLM
    ExecutionSystem.model_rebuild() # Execution depends on Memory/LLM/Skills/Planning
    # JarvisAgent rebuild is not strictly needed here as it's not instantiated directly anymore
    # JarvisAgent.model_rebuild()
    print("Centralized Pydantic model rebuild completed.")
except NameError as e:
    print(f"Warning: NameError during model rebuild - check imports: {e}")
except AttributeError as e:
    # Catch if a class doesn't have model_rebuild (e.g., not a Pydantic model)
    print(f"Warning: AttributeError during model rebuild - non-Pydantic model?: {e}")
except Exception as e:
    print(f"Warning: An unexpected error occurred during model rebuild: {e}")
# --- End Centralized Model Rebuild ---

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Jarvis AI Assistant")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    # parser.add_argument("--voice", action="store_true", help="Enable voice interaction") # Commented out
    # parser.add_argument("--no-voice", action="store_true", help="Disable voice interaction") # Commented out
    return parser.parse_args()

def setup_environment() -> logging.Logger:
    """Set up environment variables and logging using centralized settings."""
    # Set up logging using level from settings
    log_level_str = settings.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger("main", log_level)
    
    # Load environment variables (still useful for .env loading by BaseSettings)
    # load_dotenv() # <<< REMOVED redundant call
    logger.info("Environment variables loaded by dotenv (if .env exists) before settings initialization.") # <<< UPDATED log message
    
    # Check required API keys via settings
    missing_keys = []
    if not settings.groq.api_key:
        missing_keys.append("GROQ_API_KEY")
    if not settings.tavily.api_key:
        missing_keys.append("TAVILY_API_KEY")
    # Add checks for other critical keys if needed

    if missing_keys:
        logger.error(f"Missing required API key configurations (check .env or environment variables): {', '.join(missing_keys)}")
        sys.exit(1)
    
    logger.debug(f"Logging configured at level: {log_level_str}")
    return logger

# def voice_input_loop(input_queue: queue.Queue, stop_event: threading.Event):
#     """Background thread for voice input"""
#     try:
#         recognizer = SpeechRecognizer()
#         while not stop_event.is_set():
#             try:
#                 text = recognizer.listen()
#                 if text:
#                     input_queue.put(text)
#             except Exception as e:
#                 logger.error(f"Error in voice input: {e}")
#                 continue
#     except Exception as e:
#         logger.error(f"Voice input thread error: {e}")

async def main():
    """Main asynchronous function to initialize and run Jarvis."""
    print("[DEBUG] Entering main async function...")
    # Setup environment and logger based on settings
    logger = setup_environment()
    logger.info("Initializing Jarvis components...")

    # 1. Initialize LLM Client (now uses settings internally)
    print("[DEBUG] Initializing LLMClient...")
    # <<< REMOVED llm_config dictionary >>>
    llm_client = LLMClient()
    # <<< No need to pass config dict >>>
    # LLMClient.model_rebuild() # Rebuild happens within central block now
    print("[DEBUG] LLMClient initialized.")

    # 2. Initialize Memory System
    print("[DEBUG] Initializing UnifiedMemorySystem...")
    # TODO: Add configuration for memory persistence, embeddings, vector DB
    memory_system = UnifiedMemorySystem()
    print("[DEBUG] UnifiedMemorySystem initialized.")

    # 3. Initialize Skill Registry and Register Skills
    print("[DEBUG] Initializing SkillRegistry...")
    skill_registry = SkillRegistry()
    # Register core skills - ensure dependencies (like tavily API key for web search) are available
    try:
        # Discover skills from the default directory
        skill_registry.discover_skills() # Use discovery instead of manual registration
        logger.info(f"Discovered skills: {list(skill_registry.get_all_skills().keys())}")

    except Exception as e:
        logger.error(f"Error during skill discovery: {e}", exc_info=True)
    print("[DEBUG] SkillRegistry initialized.")

    # 4. Initialize Planning System
    print("[DEBUG] Initializing PlanningSystem...")
    # TODO: Configure planner persistence
    planning_system = PlanningSystem(unified_memory=memory_system, llm_client=llm_client)
    print("[DEBUG] PlanningSystem initialized.")

    # 5. Initialize Execution System (needs Skill Registry)
    print("[DEBUG] Initializing ExecutionSystem...")
    execution_system = ExecutionSystem(
        skill_registry=skill_registry,
        unified_memory=memory_system,
        planning_system=planning_system,
        llm_client=llm_client
    )
    print("[DEBUG] ExecutionSystem initialized.")

    # 6. Build the LangGraph
    print("[DEBUG] Building LangGraph...")
    # Pass all core components to the graph builder
    app = build_graph(
        llm_client=llm_client,
        planning_system=planning_system,
        execution_system=execution_system,
        memory_system=memory_system,
        skill_registry=skill_registry # Pass the registry
    )
    logger.info("Jarvis agent graph built successfully.")
    print("[DEBUG] LangGraph built.")

    # 7. Initialize the main Agent interface (if needed for interaction loop)
    # agent = JarvisAgent(graph=app, memory=memory_system, llm=llm_client)
    # logger.info("JarvisAgent initialized.")

    # --- Interaction Loop --- (Example)
    print("[DEBUG] Entering interaction loop...")
    print("\nJarvis Initialized. Enter your objective or 'quit' to exit.")
    print("[DEBUG] About to prompt for input...")
    while True:
        raw_user_input = input("Objective: ")
        if raw_user_input.lower() == 'quit':
            logger.info("Exiting Jarvis.")
            break
        if not raw_user_input:
            continue

        # --- <<< ADDED: Input Validation >>> ---
        try:
            validated_input = UserInput(query=raw_user_input)
            user_input = validated_input.query # Use the validated/sanitized query
        except ValidationError as e:
            logger.warning(f"Invalid user input: {e}")
            print(f"Validation Error: {e}. Please try again.")
            continue # Skip processing this input
        # --- <<< END Input Validation >>> ---

        logger.info(f"Received objective: {user_input}")

        # Prepare initial state for the graph
        # initial_state: JarvisState = { # REMOVED - State is implicitly managed by LangGraph
        #     "user_input": user_input,

        # Define inputs for the graph invocation
        # Key must match the expected key in JarvisState for the entry point
        inputs = {"original_query": user_input} # Use original_query instead of user_input

        logger.info("Invoking Jarvis agent graph...")
        try:
            # --- Invoke graph and get final state --- 
            final_state = await app.ainvoke(inputs)
            
            # --- Display Final Response --- 
            final_response = final_state.get('final_response', None)
            if final_response:
                 print(f"\nJarvis Response:\n--------------------\n{final_response}\n--------------------")
            else:
                 error_msg = final_state.get('error_message', "Unknown error or no response generated.")
                 print(f"\nJarvis Error:\n--------------------\n{error_msg}\n--------------------")
            
            # --- Optional: Print final state for debugging ---
            # logger.debug(f"Final State: {json.dumps(final_state, indent=2, default=str)}") # Use default=str for non-serializable

            # --- REMOVED astream_events loop --- 
            # async for event in app.astream_events(inputs, version="v1"):
            #     kind = event["event"]
            #     # Print node start/end events for tracing
            #     if kind == "on_chain_start" or kind == "on_chain_end":
            #          if event["name"] == "LangGraph": # Skip outer graph start/end
            #              continue
            #          indent = "  " * (len(event.get("tags", [])) -1) # Basic indenting
            #          print(f"{indent}Event: {kind} | Node: {event['name']}")
            # # ... (rest of streaming loop commented out) ...
            # print("\n--- Run Finished ---") # Placeholder

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print(f"\nError: An error occurred during processing: {e}")

if __name__ == "__main__":
    # Ensure the script runs in an environment with an event loop
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # --- Load environment variables --- (Added)
    load_dotenv()
    # -----------------------------------

    # Define a top-level logger for critical errors before main() setup
    logger = logging.getLogger(__name__)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Jarvis stopped by user.") # logger is now guaranteed to exist
    except Exception as e:
        logger.critical(f"Jarvis encountered a critical error: {e}", exc_info=True) # logger is now guaranteed to exist 