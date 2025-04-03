import glob
import os

def check_file_for_null_bytes(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        has_null = b'\x00' in data
    return has_null

# Check all Python files
results = {}
for file in glob.glob('**/*.py', recursive=True):
    try:
        results[file] = check_file_for_null_bytes(file)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Print files with null bytes
print("Files with null bytes:")
for file, has_null in results.items():
    if has_null:
        print(f"- {file}")

if not any(results.values()):
    print("No files with null bytes found") 