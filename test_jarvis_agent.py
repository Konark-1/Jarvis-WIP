#!/usr/bin/env python3
"""
Test the Jarvis agent to validate full system functionality
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test imports
from jarvis.agent import JarvisAgent
from jarvis.memory.unified_memory import UnifiedMemorySystem

def test_jarvis_agent():
    """Test the Jarvis agent"""
    print("Testing JarvisAgent...")
    
    # Create memory system
    print("Initializing memory system...")
    memory_system = UnifiedMemorySystem()
    
    # Create Jarvis agent
    print("Creating Jarvis agent...")
    jarvis = JarvisAgent(memory_system=memory_system)
    
    # Test basic functionality
    print("\nTesting basic commands...")
    
    # Test reflection command
    print("\n1. Testing reflection command:")
    response = jarvis.process_input("Reflect on your capabilities")
    print(f"Response: {response}")
    
    # Test objective command
    print("\n2. Testing objective command:")
    response = jarvis.process_input("Create an objective to optimize my code")
    print(f"Response: {response}")
    
    # Test plan command
    print("\n3. Testing plan command:")
    response = jarvis.process_input("Show me the plan for this objective")
    print(f"Response: {response}")
    
    # Test status command
    print("\n4. Testing status command:")
    response = jarvis.process_input("Show me your current status")
    print(f"Response: {response}")
    
    print("\nJarvis agent tests completed successfully.")

if __name__ == "__main__":
    print("Testing Jarvis Agent System")
    print("=" * 50)
    
    # Create memory directories if they don't exist
    os.makedirs("memory/db", exist_ok=True)
    os.makedirs("jarvis/memory/db", exist_ok=True)
    
    # Run tests
    test_jarvis_agent()
    
    print("\nJarvis system tests completed.") 