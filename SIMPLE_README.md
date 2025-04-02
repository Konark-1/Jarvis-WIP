# Simple Jarvis

A lightweight implementation of the Jarvis AI Assistant that uses text-based commands instead of voice recognition. This version provides the core functionality of Jarvis without the complexity of speech recognition and text-to-speech.

## Features

- **Application Control**: Open and close applications on Windows, macOS, and Linux
- **File Management**: Search for files, open files with default applications
- **Web Browsing**: Search the web, open websites in your default browser
- **Simple Text Interface**: No voice recognition required

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required packages:
```
pip install python-dotenv webbrowser
```

## Usage

1. Run Simple Jarvis:
```
python simple_jarvis.py
```

2. Type commands at the prompt:
```
You: open chrome
Jarvis: Opening chrome

You: search for python tutorials
Jarvis: Searching the web for 'python tutorials'

You: help
Jarvis: [Displays list of available commands]
```

## Available Commands

- **Opening Applications**: `open chrome`, `launch notepad`, `start calculator`
- **Closing Applications**: `close chrome`, `exit notepad`
- **Searching Files**: `find files report`, `search for files budget`
- **Opening Files**: `open file document.txt`, `read README.md`
- **Web Searching**: `search for weather`, `google python tutorials`
- **Opening Websites**: `open website github.com`, `go to example.com`
- **Getting Help**: `help`, `commands`, `?`

## Customization

You can easily extend Simple Jarvis by:

1. Adding new intent patterns in the `INTENT_PATTERNS` dictionary
2. Implementing new methods in the `SimpleJarvis` class
3. Adding new command handlers in the `execute_intent` method

For example, to add a weather checking capability:

```python
# Add pattern
INTENT_PATTERNS["weather.check"] = [
    r"what(?:'s| is) the weather(?: in| for) (.*)",
    r"weather(?: in| for) (.*)"
]

# Add implementation
def _check_weather(self, location: str) -> str:
    # Implementation to check weather for location
    return f"The weather in {location} is sunny and 72°F"

# Add to execute_intent
elif intent == "weather.check":
    location = parameters.get("target", "")
    return self._check_weather(location)
```

## Limitations

- No speech recognition or text-to-speech capability
- Limited natural language understanding compared to full Jarvis
- Basic functionality without AI-powered reasoning