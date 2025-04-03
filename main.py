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

from jarvis.agent import JarvisAgent
from jarvis.memory.unified_memory import UnifiedMemorySystem
from jarvis.planning import PlanningSystem
from jarvis.execution import ExecutionSystem
from utils.logger import setup_logger
from utils.speech import SpeechRecognizer, TextToSpeech

class Config:
    # Pydantic config to allow arbitrary types
    arbitrary_types_allowed = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Jarvis AI Assistant")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--voice", action="store_true", help="Enable voice interaction")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice interaction")
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

def voice_input_loop(input_queue: queue.Queue, stop_event: threading.Event):
    """Background thread for voice input"""
    try:
        recognizer = SpeechRecognizer()
        while not stop_event.is_set():
            try:
                text = recognizer.listen()
                if text:
                    input_queue.put(text)
            except Exception as e:
                logger.error(f"Error in voice input: {e}")
                continue
    except Exception as e:
        logger.error(f"Voice input thread error: {e}")

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
        use_voice = args.voice and not args.no_voice
        if use_voice:
            tts = TextToSpeech()
            input_queue = queue.Queue()
            stop_event = threading.Event()
            
            # Start voice input thread
            voice_thread = threading.Thread(
                target=voice_input_loop,
                args=(input_queue, stop_event)
            )
            voice_thread.daemon = True
            voice_thread.start()
        
        # Set up memory consolidation
        last_consolidation_time = datetime.now()
        consolidation_interval = 3600  # 1 hour
        
        # Main interaction loop
        logger.info("Jarvis is ready. Type 'exit' to quit.")
        print("\nJarvis: Hello! How can I help you today?")
        
        # Track time for periodic actions
        last_periodic_action_time = datetime.now()
        periodic_action_interval = 600  # 10 minutes
        
        while True:
            try:
                # Current time for checks
                now = datetime.now()
                
                # Periodic memory consolidation
                if (now - last_consolidation_time).total_seconds() > consolidation_interval:
                    logger.info("Running periodic memory consolidation")
                    memory_system.consolidate_memories()
                    last_consolidation_time = now
                
                # Periodic agent actions
                if (now - last_periodic_action_time).total_seconds() > periodic_action_interval:
                    logger.info("Running periodic agent actions")
                    jarvis.periodic_actions()
                    last_periodic_action_time = now
                
                # Get input
                if use_voice:
                    # Check for voice input
                    try:
                        text = input_queue.get(timeout=0.1)
                    except queue.Empty:
                        text = None
                    
                    # Fall back to text input if no voice input
                    if not text:
                        text = input("You: ").strip()
                else:
                    text = input("You: ").strip()
                
                # Check for exit command
                if text.lower() in ["exit", "quit", "bye"]:
                    print("\nJarvis: Goodbye!")
                    break
                
                # Process input
                response = jarvis.process_input(text)
                
                # Output response
                print(f"\nJarvis: {response}")
                
                if use_voice:
                    tts.speak(response)
                
            except KeyboardInterrupt:
                print("\nJarvis: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print("\nJarvis: I encountered an error. Please try again.")
        
        # Clean up
        if use_voice:
            stop_event.set()
            voice_thread.join()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 