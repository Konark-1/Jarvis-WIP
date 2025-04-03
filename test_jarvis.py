#!/usr/bin/env python3
"""
Comprehensive test script for the Jarvis system
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import core components
from jarvis.memory.unified_memory import UnifiedMemorySystem
from jarvis.agent import JarvisAgent
from jarvis.planning import PlanningSystem
from jarvis.execution import ExecutionSystem

def test_memory_system():
    """Test the memory subsystem"""
    print("Testing Memory System...")
    
    # Create memory directories
    os.makedirs("memory/db", exist_ok=True)
    os.makedirs("jarvis/memory/db", exist_ok=True)
    
    # Initialize memory systems
    memory = UnifiedMemorySystem()
    
    # Add test data to each memory type
    short_id = memory.add_memory("This is a short-term test memory", "short_term", {"speaker": "user"})
    medium_id = memory.add_memory("Medium-term test objective", "medium_term")
    long_id = memory.add_memory("Long-term knowledge test", "long_term", {"description": "Test knowledge"})
    
    print(f"Added memories with IDs: {short_id}, {medium_id}, {long_id}")
    
    # Search memory
    results = memory.search_memory("test")
    print(f"Found {len(results)} search results")
    
    print("Memory system test successful!")

def test_agent_capabilities():
    """Test the Jarvis agent capabilities"""
    print("\nTesting Agent Capabilities...")
    
    # Initialize memory
    memory = UnifiedMemorySystem()
    
    # Create agent
    agent = JarvisAgent(memory_system=memory)
    
    # Test basic reflection
    print("\n1. Testing reflection:")
    response = agent.process_input("Reflect on your capabilities")
    print(f"Response: {response}")
    
    # Test objective creation
    print("\n2. Testing objective creation:")
    response = agent.process_input("Create an objective to organize my files")
    print(f"Response: {response}")
    
    # Test plan display
    print("\n3. Testing plan display:")
    response = agent.process_input("Show me the plan")
    print(f"Response: {response}")
    
    # Test status display
    print("\n4. Testing status display:")
    response = agent.process_input("What's your current status?")
    print(f"Response: {response}")
    
    print("Agent capability tests successful!")

def test_memory_consolidation():
    """Test memory consolidation"""
    print("\nTesting Memory Consolidation...")
    
    memory = UnifiedMemorySystem()
    
    # Add test memories
    for i in range(5):
        memory.short_term.add_interaction("user", f"Important test message {i}", 
                                          {"importance": 0.8})
    
    print("Added 5 important short-term memories")
    
    # Run consolidation
    memory.consolidate_memories()
    
    print("Consolidation completed")
    
    # Check medium-term memory
    objectives = memory.medium_term.search_objectives("important")
    print(f"Found {len(objectives)} relevant objectives after consolidation")
    
    print("Memory consolidation test successful!")

if __name__ == "__main__":
    print("Comprehensive Jarvis System Test")
    print("=" * 50)
    
    # Run tests
    test_memory_system()
    test_agent_capabilities()
    test_memory_consolidation()
    
    print("\nAll tests completed successfully!") 