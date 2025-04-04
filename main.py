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
from typing import Optional
from datetime import datetime

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from jarvis.agent import JarvisAgent
from memory.unified_memory import UnifiedMemorySystem
from jarvis.planning import PlanningSystem
from jarvis.execution import ExecutionSystem
from utils.logger import setup_logger
# from utils.speech import SpeechRecognizer, TextToSpeech # Commented out - file missing

class Config:
    # Pydantic config to allow arbitrary types
    arbitrary_types_allowed = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Jarvis AI Assistant")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    # parser.add_argument("--voice", action="store_true", help="Enable voice interaction") # Commented out
    # parser.add_argument("--no-voice", action="store_true", help="Disable voice interaction") # Commented out
    return parser.parse_args()

def setup_environment():
    """Set up environment variables and logging"""
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("main", log_level)
    
    # Load environment variables
    from dotenv import load_dotenv
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

def main():
    """Main entry point"""
    global args, logger
    
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    logger = setup_environment()
    logger.info("Starting Jarvis...")
    
    try:
        # Initialize components
        memory_system = UnifiedMemorySystem()
        
        # Initialize Jarvis agent
        jarvis = JarvisAgent(
            memory_system=memory_system
        )
        
        # Set up voice interaction if enabled
        # use_voice = args.voice and not args.no_voice # Commented out
        use_voice = False # Disabled voice interaction
        # if use_voice:
        #     tts = TextToSpeech()
        #     input_queue = queue.Queue()
        #     stop_event = threading.Event()
        #     
        #     # Start voice input thread
        #     voice_thread = threading.Thread(
        #         target=voice_input_loop,
        #         args=(input_queue, stop_event)
        #     )
        #     voice_thread.daemon = True
        #     voice_thread.start()
        
        # Set up memory consolidation
        last_consolidation_time = datetime.now()
        consolidation_interval = 3600  # 1 hour
        
        # Track time for periodic actions
        last_periodic_action_time = datetime.now()
        periodic_action_interval = 600  # 10 minutes
        # Track time for objective review
        last_objective_review_time = datetime.now()
        objective_review_interval = 1800 # 30 minutes
        # Track time for self-assessment
        last_assessment_time = datetime.now()
        assessment_interval = 3600 # 1 hour
        # Track time for knowledge organization
        last_knowledge_org_time = datetime.now()
        knowledge_org_interval = 14400 # 4 hours
        
        # Main interaction loop
        logger.info("Jarvis is ready. Type 'exit' to quit.")
        print("\nJarvis: Hello! How can I help you today?")
        
        while True:
            try:
                # Current time for checks
                now = datetime.now()
                
                # Periodic memory consolidation
                if (now - last_consolidation_time).total_seconds() > consolidation_interval:
                    logger.info("Running periodic memory consolidation")
                    try:
                        memory_system.consolidate_memories()
                    except Exception as e:
                        logger.error(f"Error during memory consolidation: {e}")
                    last_consolidation_time = now
                
                # Periodic agent objective review
                if (now - last_objective_review_time).total_seconds() > objective_review_interval:
                    logger.info("Running periodic objective review")
                    try:
                        # Pass the LLM client instance to the review method
                        if jarvis.llm:
                            jarvis._review_and_refine_objectives()
                        else:
                            logger.warning("Cannot run objective review: LLM client not available.")
                    except Exception as e:
                         logger.error(f"Error during objective review: {e}")
                    last_objective_review_time = now
                
                # Self Assessment (New)
                if (now - last_assessment_time).total_seconds() > assessment_interval:
                    logger.info("Running periodic self-assessment")
                    try:
                        jarvis._perform_self_assessment()
                    except Exception as e:
                         logger.error(f"Error during self-assessment: {e}")
                    last_assessment_time = now
                
                # Knowledge Organization (New)
                if (now - last_knowledge_org_time).total_seconds() > knowledge_org_interval:
                    logger.info("Running periodic knowledge organization")
                    try:
                        # Example: Rebuild index and summarize
                        memory_system.organize_knowledge(summarize=True, rebuild_stm_index=True)
                    except Exception as e:
                         logger.error(f"Error during knowledge organization: {e}")
                    last_knowledge_org_time = now
                
                # Periodic agent actions (Placeholder - refine this)
                if (now - last_periodic_action_time).total_seconds() > periodic_action_interval:
                    logger.info("Running periodic agent actions (placeholder)")
                    # jarvis.periodic_actions() # Commented out if not implemented
                    last_periodic_action_time = now
                
                # Get input
                # if use_voice: # Commented out
                #     # Check for voice input
                #     try:
                #         text = input_queue.get(timeout=0.1)
                #     except queue.Empty:
                #         text = None
                #     
                #     # Fall back to text input if no voice input
                #     if not text:
                #         text = input("You: ").strip()
                # else:
                text = input("You: ").strip() # Always use text input for now
                
                # Check for exit command
                if text.lower() in ["exit", "quit", "bye"]:
                    print("\nJarvis: Goodbye!")
                    break
                
                # Process input
                response = jarvis.process_input(text)
                
                # Output response
                print(f"\nJarvis: {response}")
                
                # if use_voice:
                #     tts.speak(response)
                
            except KeyboardInterrupt:
                print("\nJarvis: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print("\nJarvis: I encountered an error. Please try again.")
        
        # Clean up
        # if use_voice:
        #     stop_event.set()
        #     voice_thread.join()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 