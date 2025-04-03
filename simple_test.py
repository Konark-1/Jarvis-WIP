#!/usr/bin/env python3
"""
Simplified Jarvis test
"""

import os
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple Jarvis Test")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice interaction")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Import the simple Jarvis
    try:
        from simple_jarvis import SimpleJarvis
        print("Successfully imported SimpleJarvis")
        
        # Create instance
        jarvis = SimpleJarvis()
        print("Successfully created SimpleJarvis instance")
        
        # Test basic functionality
        response = jarvis.process_command("help")
        print(f"Response to 'help':\n{response}")
        
        print("Simple test completed successfully")
    except Exception as e:
        print(f"Error in simple test: {e}")

if __name__ == "__main__":
    main() 