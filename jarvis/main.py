#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant

Main entry point for Jarvis that integrates all components.
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
from dotenv import load_dotenv

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
# Call rebuild here after all classes are imported to resolve forward references
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

def setup_environment(args: argparse.Namespace) -> logging.Logger:
    """Set up environment variables and logging"""
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("main", log_level)
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
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
    logger.info("Initializing Jarvis components...")

    # 1. Initialize LLM Client
    # TODO: Load configuration from a dedicated config file (e.g., YAML)
    llm_config = {
        "default_provider": os.getenv("DEFAULT_LLM_PROVIDER", "groq"),
        "providers": {
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY"),
                "default_model": os.getenv("GROQ_DEFAULT_MODEL", "llama3-70b-8192")
            },
            "openai": {
                 "api_key": os.getenv("OPENAI_API_KEY"),
                 "default_model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
            },
            "anthropic": {
                 "api_key": os.getenv("ANTHROPIC_API_KEY"),
                 "default_model": os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-haiku-20240307")
            }
            # Add other providers as needed
        }
    }
    llm_client = LLMClient(config=llm_config)

    # 2. Initialize Memory System
    # TODO: Add configuration for memory persistence, embeddings, vector DB
    memory_system = UnifiedMemorySystem()

    # 3. Initialize Skill Registry and Register Skills
    skill_registry = SkillRegistry()
    # Register core skills - ensure dependencies (like tavily API key for web search) are available
    try:
        # Discover skills from the default directory
        skill_registry.discover_skills() # Use discovery instead of manual registration
        logger.info(f"Discovered skills: {list(skill_registry.get_all_skills().keys())}")

    except Exception as e:
        logger.error(f"Error during skill discovery: {e}", exc_info=True)

    # 4. Initialize Planning System
    # TODO: Configure planner persistence
    planning_system = PlanningSystem(unified_memory=memory_system, llm_client=llm_client)

    # 5. Initialize Execution System (needs Skill Registry)
    execution_system = ExecutionSystem(
        skill_registry=skill_registry,
        unified_memory=memory_system,
        planning_system=planning_system,
        llm_client=llm_client
    )

    # 6. Build the LangGraph
    # Pass all core components to the graph builder
    app = build_graph(
        llm_client=llm_client,
        planning_system=planning_system,
        execution_system=execution_system,
        memory_system=memory_system,
        skill_registry=skill_registry # Pass the registry
    )
    logger.info("Jarvis agent graph built successfully.")

    # 7. Initialize the main Agent interface (if needed for interaction loop)
    # agent = JarvisAgent(graph=app, memory=memory_system, llm=llm_client)
    # logger.info("JarvisAgent initialized.")

    # --- Interaction Loop --- (Example)
    print("\nJarvis Initialized. Enter your objective or 'quit' to exit.")
    while True:
        user_input = input("Objective: ")
        if user_input.lower() == 'quit':
            logger.info("Exiting Jarvis.")
            break
        if not user_input:
            continue

        logger.info(f"Received objective: {user_input}")

        # Prepare initial state for the graph
        initial_state: JarvisState = {
            "user_input": user_input,
            "objective": None, # Will be created by understand_query_node
            "plan": None,
            "tasks_executed": [],
            "knowledge": "",
            "conversation_history": "", # TODO: Populate history
            "final_response": None,
            "error_message": None
        }

        # Define inputs for the graph invocation
        inputs = {"user_input": user_input} # Pass only the trigger input

        logger.info("Invoking Jarvis agent graph...")
        try:
            # Stream events from the graph execution
            async for event in app.astream_events(initial_state, version="v1"):
                kind = event["event"]
                # Print node start/end events for tracing
                if kind == "on_chain_start" or kind == "on_chain_end":
                     if event["name"] == "LangGraph": # Skip outer graph start/end
                         continue
                     indent = "  " * (len(event.get("tags", [])) -1) # Basic indenting
                     print(f"{indent}Event: {kind} | Node: {event['name']}")
                # You can add more detailed event handling here if needed
                # e.g., print intermediate state updates, tool calls, etc.

            # Retrieve the final state after execution
            # Note: astream_events doesn't directly return final state easily,
            # We might need `ainvoke` or manage state updates within the loop if needed immediately.
            # For simplicity, we assume the graph runs to completion here.
            # To get final state: final_state_result = await app.ainvoke(initial_state)
            # print(f"\nFinal State: {final_state_result}")
            # print(f"\nFinal Response: {final_state_result.get('final_response', 'N/A')}")

            print("\n--- Run Finished ---") # Placeholder

        except Exception as e:
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            print(f"\nError: An error occurred during processing: {e}")

if __name__ == "__main__":
    # Ensure the script runs in an environment with an event loop
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Define a top-level logger for critical errors before main() setup
    logger = logging.getLogger(__name__)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Jarvis stopped by user.") # logger is now guaranteed to exist
    except Exception as e:
        logger.critical(f"Jarvis encountered a critical error: {e}", exc_info=True) # logger is now guaranteed to exist 