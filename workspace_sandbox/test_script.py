#!/usr/bin/env python3
import sys

print("Hello from test_script.py!")
print("Arguments received:", sys.argv)
# Example of writing to stderr
print("This is an error message simulation", file=sys.stderr) 