#!/usr/bin/env python3
"""
Jarvis - Agentic AI Assistant

Main entry point for the Jarvis agent.
"""

import os
import sys
import logging
import dotenv
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("jarvis.log")
    ]
)

logger = logging.getLogger("jarvis.main")

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Jarvis agent
from jarvis.agent import JarvisAgent
from jarvis.llm import LLMClient

def initialize_agent(groq_api_key: Optional[str] = None) -> JarvisAgent:
    """Initialize the Jarvis agent."""
    # Set API keys if provided
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Create LLM client with Groq as primary
    llm_client = LLMClient(primary_provider="groq")
    
    # Create and return agent
    agent = JarvisAgent(llm=llm_client)
    
    return agent

def main():
    """Main function to start Jarvis."""
    logger.info("Initializing Jarvis...")
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Initialize Jarvis agent
    jarvis = initialize_agent(groq_api_key)
    
    # Print welcome message
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                          JARVIS AI ASSISTANT                             ║
║                                                                          ║
║  Enhanced with agentic planning, memory, and reasoning capabilities.     ║
║  Type your objectives or questions, and Jarvis will autonomously plan    ║
║  and execute the steps needed to accomplish them.                        ║
║                                                                          ║
║  Type 'exit', 'quit', or 'shutdown' to exit.                            ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Use text interface
    print("Jarvis is now online. Type your request or 'exit' to quit.")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'shutdown']:
                print("Jarvis: Shutting down.")
                break
            
            # Process input through agent
            response = jarvis.process_input(user_input)
            
            # Display response
            print(f"Jarvis: {response}")
            
        except KeyboardInterrupt:
            print("\nJarvis: Shutting down.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Jarvis: I encountered an error: {str(e)}")

if __name__ == "__main__":
    main() 