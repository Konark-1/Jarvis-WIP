# Jarvis - Personal AI Assistant

An autonomous AI agent inspired by Jarvis from Iron Man, capable of understanding voice commands and performing various tasks on your computer.

## Features

- **Voice Recognition**: Listens to your commands and responds with speech
- **Application Control**: Opens and closes applications
- **File Management**: Searches, opens, edits files
- **Web Browsing**: Searches the web, opens websites
- **Multi-layered Memory**:
  - **Long-term Memory**: Stores core programming knowledge
  - **Medium-term Memory**: Tracks objectives and tasks
  - **Short-term Memory**: Maintains conversation context

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd jarvis
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Run Jarvis:
```
python main.py
```

2. Wake Jarvis with the wake word "jarvis" followed by your command:
```
"Jarvis, open Chrome"
"Jarvis, search for pictures of cats"
"Jarvis, what's the weather today?"
```

## Project Structure

```
jarvis/
├── main.py              # Main entry point
├── jarvis.py            # Core Jarvis class
├── memory/              # Memory components
│   ├── long_term.py     # Long-term memory (core knowledge)
│   ├── medium_term.py   # Medium-term memory (objectives)
│   ├── short_term.py    # Short-term memory (context)
│   └── __init__.py
├── utils/               # Utility components
│   ├── logger.py        # Logging utility
│   ├── speech.py        # Speech recognition and synthesis
│   ├── command_parser.py # Command parsing
│   └── __init__.py
├── skills/              # Skill components
│   ├── system_skills.py # System interactions
│   ├── web_skills.py    # Web interactions
│   ├── file_skills.py   # File interactions
│   └── __init__.py
└── requirements.txt     # Dependencies
```

## Memory Architecture

### Long-term Memory (Core Programming)
- Persistent knowledge base stored in vector database
- Retains information across restarts
- Used for core functionality and learned procedures

### Medium-term Memory (Objective-centric)
- Tracks user objectives and progress
- Persists across sessions
- Helps Jarvis maintain focus on goals

### Short-term Memory (Context-relevant)
- Maintains recent conversation history
- Stores context variables
- Reset when Jarvis is restarted

## Extending Jarvis

### Adding New Skills

1. Create a new skill module in the `skills/` directory
2. Implement the skill class using Pydantic's BaseModel
3. Add the skill to the Jarvis class in `jarvis.py`
4. Register intents in the `CommandParser` class

### Customizing Voice

You can modify the voice settings in `utils/speech.py`:

```python
# Example: Change speech rate and voice
text_to_speech = TextToSpeech(rate=150, volume=1.0)

# List available voices
voices = text_to_speech.get_available_voices()
print(voices)

# Set a specific voice
text_to_speech.set_voice(voices[1])
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for language model capabilities
- Pydantic for data validation
- ChromaDB for vector storage
- SpeechRecognition and pyttsx3 for speech processing 