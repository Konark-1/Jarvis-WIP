import subprocess
import sys
import os

log_file = "run_log.txt"
# Construct the command using sys.executable to ensure the correct Python interpreter is used
command = [sys.executable, "-m", "jarvis.main", "--verbose"]

print(f"Executing: {' '.join(command)}")
print(f"Redirecting output to {log_file}")

try:
    # Run the command, capture output, handle potential encoding issues
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace', # Replace characters that can't be decoded
        check=False # Don't raise exception for non-zero exit codes
    )

    # Write the captured output to the log file
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("--- Captured STDOUT ---\n")
        f.write(result.stdout)
        f.write("\n\n--- Captured STDERR ---\n")
        f.write(result.stderr)
        f.write(f"\n\n--- Process Exit Code: {result.returncode} ---\n")

    print(f"Finished running. Exit code: {result.returncode}. Check {log_file} for details.")

except FileNotFoundError:
    print(f"Error: Could not find Python executable '{sys.executable}' or module 'jarvis.main'. Ensure Python is in PATH and Jarvis is installed correctly.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while trying to run jarvis.main: {e}", file=sys.stderr)
    # Attempt to write the runner error to the log file as well
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"\n--- Error in run_and_log.py ---\n{str(e)}\n")
    except Exception:
        pass # Ignore errors during error logging
    sys.exit(1) 