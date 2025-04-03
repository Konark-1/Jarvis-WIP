#!/usr/bin/env python3
"""
Test the Jarvis memory system to validate fixes
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test imports
from jarvis.memory.unified_memory import UnifiedMemorySystem
from jarvis.memory.short_term import ShortTermMemory
from jarvis.memory.medium_term import MediumTermMemory
from jarvis.memory.long_term import LongTermMemory

def test_memory_systems():
    """Test all memory systems"""
    print("Testing individual memory systems...")
    
    # Test short-term memory
    print("\nTesting ShortTermMemory...")
    try:
        short_term = ShortTermMemory()
        memory_id = short_term.add_interaction("user", "This is a test message")
        print(f"Added interaction with ID: {memory_id}")
        print("ShortTermMemory initialized successfully.")
    except Exception as e:
        print(f"Error initializing ShortTermMemory: {e}")
        
    # Test medium-term memory
    print("\nTesting MediumTermMemory...")
    try:
        medium_term = MediumTermMemory()
        objective_id = medium_term.create_objective("Test objective")
        print(f"Created objective with ID: {objective_id}")
        print("MediumTermMemory initialized successfully.")
    except Exception as e:
        print(f"Error initializing MediumTermMemory: {e}")
        
    # Test long-term memory
    print("\nTesting LongTermMemory...")
    try:
        long_term = LongTermMemory()
        long_term.add_knowledge("test_knowledge", "This is test knowledge")
        print("Added knowledge to LongTermMemory")
        print("LongTermMemory initialized successfully.")
    except Exception as e:
        print(f"Error initializing LongTermMemory: {e}")

def test_unified_memory():
    """Test the unified memory system"""
    print("\nTesting UnifiedMemorySystem...")
    try:
        memory_system = UnifiedMemorySystem()
        
        # Add memories to each system
        short_id = memory_system.add_memory(
            "Test short-term memory", 
            "short_term", 
            {"speaker": "user"}
        )
        
        medium_id = memory_system.add_memory(
            "Test medium-term objective", 
            "medium_term", 
            {"priority": 3}
        )
        
        long_id = memory_system.add_memory(
            "Test long-term knowledge", 
            "long_term", 
            {"description": "Important knowledge"}
        )
        
        print(f"Added memories with IDs: {short_id}, {medium_id}, {long_id}")
        
        # Test search functionality
        results = memory_system.search_memory("test", ["short_term", "medium_term", "long_term"])
        print(f"Found {len(results)} search results")
        
        print("UnifiedMemorySystem initialized and tested successfully.")
    except Exception as e:
        print(f"Error with UnifiedMemorySystem: {e}")

if __name__ == "__main__":
    print("Testing Jarvis Memory Systems")
    print("=" * 50)
    
    # Create memory directories if they don't exist
    os.makedirs("memory/db", exist_ok=True)
    os.makedirs("jarvis/memory/db", exist_ok=True)
    
    # Run tests
    test_memory_systems()
    test_unified_memory()
    
    print("\nMemory system tests completed.") 