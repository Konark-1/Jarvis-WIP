#!/usr/bin/env python3
"""
Fix the Jarvis directory structure.
This script copies memory files from the root memory directory into the jarvis/memory directory.
"""

import os
import shutil

# Create jarvis/memory directory if it doesn't exist
os.makedirs("jarvis/memory", exist_ok=True)
os.makedirs("jarvis/memory/db", exist_ok=True)

# Files to copy from memory -> jarvis/memory
memory_files = [
    "short_term.py",
    "medium_term.py",
    "long_term.py",
    "__init__.py",
    "core_memory.json",
    "objective_memory.json"
]

for file in memory_files:
    source = f"memory/{file}"
    dest = f"jarvis/memory/{file}"
    if os.path.exists(source) and not os.path.exists(dest):
        print(f"Copying {source} -> {dest}")
        shutil.copy2(source, dest)

print("Directory structure fixed!") 