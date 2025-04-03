# Jarvis - Agentic AI Assistant

Jarvis is an autonomous AI assistant that can execute tasks, manage objectives, and help users with various computer-related tasks. It features memory systems with different time horizons, agentic planning, and execution capabilities.

## Features

- **Agentic Planning**: Autonomously creates plans for accomplishing objectives
- **Memory System**: Short-term, medium-term, and long-term memory for context awareness
- **Self-reflection**: Can evaluate its own performance and improve
- **Objective Management**: Creates and tracks objectives over time
- **Error Recovery**: Detects failures and attempts recovery strategies
- **Voice Interaction**: Optional voice input and output capabilities
- **Terminal Command Execution**: Can run commands on your behalf

## Quick Start

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r jarvis/requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GROQ_API_KEY=your_groq_key
   ```
4. Run Jarvis:
   ```
   python main.py
   ```

## Usage

Once running, you can interact with Jarvis using text commands:

```
You: Set a new objective to organize my documents folder
Jarvis: I've set a new objective to organize your documents folder. I'll help you accomplish this by breaking it down into manageable tasks.

You: What are my current objectives?
Jarvis: You currently have 1 active objective:
1. Organize documents folder (Priority: 2, Progress: 0%)

You: Help me with the documents folder organization
Jarvis: I'll help you organize your documents folder. First, let's create a plan...
```

## Command Line Options

- `--verbose` or `-v`: Enable detailed logging
- `--no-voice`: Disable voice interactions (text only)

## Architecture

Jarvis consists of several key components:

1. **Agent System**: Core decision-making and execution loop
2. **Memory System**: Manages information across different time horizons
3. **Planning System**: Creates and manages plans to achieve objectives
4. **LLM Integration**: Connects to language models for reasoning and generation
5. **Action Registry**: Collection of capabilities Jarvis can execute

## How It Works

1. User provides input
2. Jarvis analyzes the input to understand intent
3. If needed, Jarvis creates a plan with multiple steps
4. Each step is executed, with monitoring for success/failure
5. Results are stored in memory for future reference
6. Jarvis provides a summarized response to the user

## Development

This is a work in progress. The implementation in the `jarvis/` directory contains the latest agentic version of Jarvis.

### Project Structure

- `jarvis/agent.py`: Core agentic implementation
- `jarvis/memory/`: Memory system implementation
- `jarvis/planning.py`: Planning and task management
- `jarvis/llm.py`: LLM client integrations
- `main.py`: Entry point for running Jarvis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 