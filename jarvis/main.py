#!/usr/bin/env python3
"""
Jarvis - Personal AI Assistant

Main entry point for the Jarvis agent.
"""

import os
import sys
import dotenv

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Jarvis
from jarvis import Jarvis

def main():
    """Main function to start Jarvis"""
    print("Initializing Jarvis...")
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Initialize Jarvis
    jarvis = Jarvis()
    
    # Start Jarvis
    jarvis.run()

if __name__ == "__main__":
    main() 