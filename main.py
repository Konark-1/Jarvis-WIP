#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant

Main entry point for the Jarvis agent.
"""

import os
import sys
import dotenv

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Jarvis
from jarvis.jarvis import Jarvis

def main():
    """Main function to start Jarvis"""
    print("Initializing Jarvis...")
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Initialize Jarvis
    jarvis = Jarvis()
    
    # Use text interface for testing
    print("Jarvis is now online. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Jarvis: Shutting down.")
                break
                
            # Process command
            command = user_input.lower()
            if jarvis.wake_word in command:
                command = command.replace(jarvis.wake_word, "").strip()
            
            # Parse command
            intent, parameters = jarvis.command_parser.parse(command)
            
            # Execute command
            response = jarvis._execute_intent(intent, parameters)
            
            # Display response
            print(f"Jarvis: {response}")
            
        except KeyboardInterrupt:
            print("\nJarvis: Shutting down.")
            break
        except Exception as e:
            print(f"Jarvis: Sorry, an error occurred: {str(e)}")

if __name__ == "__main__":
    main() 