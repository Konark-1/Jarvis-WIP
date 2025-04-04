"""
Benchmark script for testing LLM capabilities within the Jarvis agent framework.

Focuses on:
- Task Decomposition (Planning)
- Skill Parsing (Execution)
- Error Diagnosis (Execution)
- Context Assembly (Memory)
- Reflection (Memory)
- Objective Review (Agent)
- Self-Assessment (Agent)
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Core Jarvis Components ---
# Attempt to import core components, handle potential errors
try:
    from dotenv import load_dotenv
    load_dotenv() # Load .env file for API keys
except ImportError:
    print("python-dotenv not found. Please install it: pip install python-dotenv")
    sys.exit(1)

try:
    from jarvis.llm import LLMClient, Message
    from jarvis.memory.unified_memory import UnifiedMemorySystem
    from jarvis.memory.long_term import LongTermMemory
    from jarvis.memory.medium_term import MediumTermMemory, Objective
    from jarvis.memory.short_term import ShortTermMemory
    from jarvis.planning import PlanningSystem, Task, Plan
    from jarvis.execution import ExecutionSystem, ExecutionResult
    from jarvis.agent import JarvisAgent, AgentState
    from jarvis.skills.registry import SkillRegistry
    from jarvis.skills.base import Skill
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Error importing Jarvis components: {e}")
    print("Ensure you are running this script from the project root or the path is set correctly.")
    print("Also check if all dependencies in requirements.txt are installed.")
    sys.exit(1)

# --- Mock Components ---
# Simple mock classes for isolated testing

class MockSkillRegistry(SkillRegistry):
    def __init__(self):
        # Use Dict[str, Dict] for _skills for simplicity in mock
        # Or potentially create simple Skill mock objects
        self._skills: Dict[str, Dict[str, Any]] = {
            "web_search": {"name": "web_search", "description": "Search the web for information.", "parameters": ["query"]},
            "read_file": {"name": "read_file", "description": "Read the contents of a file.", "parameters": ["filepath"]},
            "send_email": {"name": "send_email", "description": "Send an email.", "parameters": ["recipient", "subject", "body"]},
            "no_params_skill": {"name": "no_params_skill", "description": "A skill with no parameters.", "parameters": []},
        }

    def get_skill(self, name: str) -> Optional[Dict[str, Any]]: # Return Dict for mock
        return self._skills.get(name)

    def list_skills(self) -> List[Dict[str, Any]]: # Return List[Dict]
        return list(self._skills.values())

    def get_skill_definitions(self, skill_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        # Return definitions in the format expected by execution system's LLM prompt
        defs = {}
        skills_to_list = self._skills.keys() if skill_names is None else skill_names
        for name in skills_to_list:
            if name in self._skills:
                skill = self._skills[name]
                defs[skill["name"]] = {
                    "description": skill["description"],
                    "parameters": {p: "string" for p in skill["parameters"]} # Assume string for simplicity
                }
        return defs

class MockUnifiedMemory(UnifiedMemorySystem):
     # Override methods that interact heavily with storage if needed for benchmarks
     def __init__(self, logger: logging.Logger, llm_client: Optional[LLMClient] = None, **data):
         # Properly initialize the parent BaseModel
         # Pass logger and llm_client, let Pydantic handle other fields via defaults
         init_data = data.copy()
         init_data['logger'] = logger
         init_data['llm_client'] = llm_client
         super().__init__(**init_data)

         # We can now access self.short_term, self.medium_term etc. as they were initialized by Pydantic

         # Add some dummy data AFTER initialization
         try:
            # Make sure the memory components were initialized
            if hasattr(self, 'short_term') and self.short_term:
                 self.short_term.add_interaction("user", "What's the weather today?")
                 self.short_term.add_interaction("assistant", "Fetching weather...")
            else:
                 self.logger.warning("Mock short_term memory not initialized, skipping dummy data.")

            if hasattr(self, 'medium_term') and self.medium_term:
                 self.medium_term.create_objective("Plan Project Phoenix presentation", {"priority": 5})
            else:
                self.logger.warning("Mock medium_term memory not initialized, skipping dummy data.")

            if hasattr(self, 'long_term') and self.long_term:
                 self.long_term.add_knowledge("project_phoenix_summary", "Project Phoenix aims to deliver X by Y.", {"keywords": "phoenix, project, summary"})
            else:
                 self.logger.warning("Mock long_term memory not initialized, skipping dummy data.")

         except Exception as e:
             self.logger.warning(f"Could not add dummy data to mock memory: {e}")


# --- Logger Setup ---
log_level = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logger("benchmark_llm", log_level=log_level)

# --- Benchmark Functions ---

def run_benchmark(func, name: str, llm_client: LLMClient, **kwargs) -> Dict[str, Any]:
    """Helper to run a benchmark function and capture timing/results."""
    logger.info(f"--- Running Benchmark: {name} --- Provider: {llm_client.primary_provider}")
    start_time = time.time()
    results = []
    errors = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    try:
        # Assuming the benchmark function returns a list of individual results
        benchmark_outputs = func(llm_client=llm_client, **kwargs)
        for output in benchmark_outputs:
            results.append(output.get("output", "N/A"))
            if "error" in output and output["error"]:
                errors.append(output["error"])
            if "usage" in output and output["usage"]:
                total_prompt_tokens += output["usage"].get("prompt_tokens", 0)
                total_completion_tokens += output["usage"].get("completion_tokens", 0)
                total_tokens += output["usage"].get("total_tokens", 0)

    except Exception as e:
        logger.error(f"Error during benchmark {name}: {e}", exc_info=True)
        errors.append(str(e))
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- Benchmark {name} Finished --- Duration: {duration:.2f}s")

    return {
        "name": name,
        "provider": llm_client.primary_provider,
        "model": llm_client.get_model_name(),
        "duration_seconds": duration,
        "num_runs": len(results) + len(errors),
        "num_success": len(results),
        "num_errors": len(errors),
        "results": results, # Raw outputs for quality scoring
        "errors": errors,
        "usage": {
             "prompt_tokens": total_prompt_tokens,
             "completion_tokens": total_completion_tokens,
             "total_tokens": total_tokens
        }
    }

def benchmark_decompose_objective(llm_client: LLMClient, planning_system: PlanningSystem, objectives: List[str]) -> List[Dict[str, Any]]:
    """Benchmarks the _decompose_objective method."""
    results = []
    for objective_desc in objectives:
        start_time = time.time()
        output = None
        error = None
        usage = None
        try:
            # We need to call the internal method directly or replicate its logic
            # Replicating logic is safer to avoid side effects of create_plan
            
            system_prompt = """
            You are Jarvis, an AI assistant that excels at breaking down objectives into practical, actionable tasks.
            For the given objective, create a comprehensive and logical sequence of tasks needed to accomplish it.
            Consider dependencies between tasks and different phases like planning, research, execution, and validation.
            
            Return the tasks as a JSON array of objects. Each object MUST have a "description" (string) and "dependencies" (list of integers, representing the 0-based index of tasks that must precede this one).
            Optionally include a "phase" (string: planning|research|execution|validation).
            Example structure:
            [
                {
                    "description": "Task 1 description",
                    "phase": "planning",
                    "dependencies": []
                },
                {
                    "description": "Task 2 description",
                    "phase": "execution",
                    "dependencies": [0]
                }
            ]
            Ensure the output is ONLY the JSON array, without any introductory text or markdown formatting.
            """
            prompt = f"""
            Objective: {objective_desc}
            Create a logical sequence of tasks to accomplish this objective following the specified JSON format.
            Output ONLY the JSON array.
            """
            
            logger.debug(f"Sending decomposition prompt to LLM for objective: {objective_desc}")
            response_content = llm_client.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=2000
            )
            output = response_content # Store raw JSON string for quality check
            usage = llm_client.last_response.usage if llm_client.last_response else {}

            # Basic validation (can be expanded)
            try:
                # Clean potential markdown
                if output.strip().startswith("```json"):
                    output = output.strip()[7:-3].strip()
                elif output.strip().startswith("```"):
                    output = output.strip()[3:-3].strip()
                json.loads(output)
            except json.JSONDecodeError as json_err:
                 error = f"JSON Decode Error: {json_err} - Output: {output[:100]}..."
                 logger.warning(error)
                 # Keep the potentially invalid JSON in 'output' for review

        except Exception as e:
            logger.error(f"Error decomposing objective '{objective_desc}': {e}", exc_info=True)
            error = str(e)
            if llm_client.last_response:
                 usage = llm_client.last_response.usage

        duration = time.time() - start_time
        results.append({"input": objective_desc, "output": output, "error": error, "duration": duration, "usage": usage})
    return results


def benchmark_parse_task_for_skill(llm_client: LLMClient, execution_system: ExecutionSystem, tasks: List[str]) -> List[Dict[str, Any]]:
    """Benchmarks the _parse_task_for_skill method."""
    results = []
    skill_defs = execution_system.skill_registry.get_skill_definitions() # Use mock registry here
    skill_json = json.dumps(skill_defs, indent=2)

    for task_desc in tasks:
        start_time = time.time()
        output = None
        error = None
        usage = None
        try:
             # Replicate logic from _parse_task_for_skill
            system_prompt = f"""
            You are an AI assistant responsible for parsing a natural language task description and mapping it to an available skill call.
            Analyze the task description and select the most appropriate skill from the provided list.
            Extract the necessary parameters for the selected skill based *only* on the task description.
            If a parameter value is not present in the task description, DO NOT invent one. Use an empty string "" or null/None if the format allows.

            Available Skills (JSON format):
            {skill_json}

            Respond with a JSON object containing "skill_name" and "parameters" (a dictionary of parameter names and their extracted values).
            Example: {{"skill_name": "web_search", "parameters": {{"query": "latest AI news"}}}}
            Example: {{"skill_name": "read_file", "parameters": {{"filepath": "/path/to/notes.txt"}}}}
            Example: {{"skill_name": "no_params_skill", "parameters": {{}}}}

            Output ONLY the JSON object.
            """
            prompt = f"Task Description: {task_desc}\n\nSelect the skill and extract parameters. Output ONLY JSON."

            logger.debug(f"Sending skill parsing prompt for task: {task_desc}")
            response_content = llm_client.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2, # Lower temperature for structured output
                max_tokens=500
            )
            output = response_content
            usage = llm_client.last_response.usage if llm_client.last_response else {}

            # Basic validation
            try:
                # Clean potential markdown
                if output.strip().startswith("```json"):
                    output = output.strip()[7:-3].strip()
                elif output.strip().startswith("```"):
                    output = output.strip()[3:-3].strip()

                parsed = json.loads(output)
                if not isinstance(parsed, dict) or "skill_name" not in parsed or "parameters" not in parsed:
                    raise ValueError("Output JSON missing required keys.")
                if not isinstance(parsed["parameters"], dict):
                     raise ValueError("'parameters' key is not a dictionary.")

            except (json.JSONDecodeError, ValueError) as json_err:
                error = f"JSON Validation Error: {json_err} - Output: {output[:100]}..."
                logger.warning(error)
                # Keep the potentially invalid JSON in 'output' for review

        except Exception as e:
            logger.error(f"Error parsing task '{task_desc}': {e}", exc_info=True)
            error = str(e)
            if llm_client.last_response:
                 usage = llm_client.last_response.usage

        duration = time.time() - start_time
        results.append({"input": task_desc, "output": output, "error": error, "duration": duration, "usage": usage})
    return results


def benchmark_diagnose_execution_error(llm_client: LLMClient, execution_system: ExecutionSystem, errors: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Benchmarks the _diagnose_execution_error method."""
    results = []
    for task_desc, error_msg in errors:
        start_time = time.time()
        output = None
        error = None
        usage = None
        try:
            # Replicate logic from _diagnose_execution_error
            context = f"Task Description: {task_desc}\n"
            # context += f"Task Metadata: {{}}\n" # Mock metadata if needed
            context += f"Error Encountered: {error_msg}\n"

            system_prompt = """
            You are an error analysis assistant for the Jarvis agent's execution system.
            A task failed during execution.
            Analyze the task details (description, metadata) and the error message.
            Provide a brief diagnosis of the likely cause (e.g., bad parameters, skill bug, external API issue, LLM hallucination).
            Suggest a recovery strategy if possible (e.g., retry with different parameters, modify plan, use fallback skill, report bug).
            Format as: "Diagnosis: [Your diagnosis]. Suggestion: [Your suggestion]"
            """
            prompt = f"""
            {context}
            Analyze the execution failure and provide a diagnosis and suggestion.
            """
            
            logger.debug(f"Sending error diagnosis prompt for task: {task_desc}, error: {error_msg}")
            response_content = llm_client.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=200
            )
            output = response_content.strip()
            usage = llm_client.last_response.usage if llm_client.last_response else {}

        except Exception as e:
            logger.error(f"Error diagnosing error for task '{task_desc}': {e}", exc_info=True)
            error = str(e)
            if llm_client.last_response:
                 usage = llm_client.last_response.usage

        duration = time.time() - start_time
        results.append({"input": (task_desc, error_msg), "output": output, "error": error, "duration": duration, "usage": usage})
    return results


def benchmark_assemble_context(llm_client: LLMClient, memory_system: UnifiedMemorySystem, queries: List[str]) -> List[Dict[str, Any]]:
    """Benchmarks the assemble_context method (indirectly testing retrieval + formatting)."""
    results = []
    # Ensure we have an LLM client with a tokenizer for token counting
    if not memory_system.llm_client or not memory_system.llm_client.tokenizer:
         logger.warning("Skipping assemble_context benchmark: LLMClient with tokenizer required in UnifiedMemorySystem.")
         return []

    for query in queries:
        start_time = time.time()
        output = None
        error = None
        num_tokens = 0
        try:
            output = memory_system.assemble_context(query=query, max_tokens=4096) # Use a reasonable token limit
            # Estimate tokens using the tokenizer
            num_tokens = len(memory_system.llm_client.tokenizer.encode(output))

        except Exception as e:
            logger.error(f"Error assembling context for query '{query}': {e}", exc_info=True)
            error = str(e)

        duration = time.time() - start_time
        # Outputting the first 200 chars for brevity in results, and token count
        results.append({
            "input": query,
            "output": f"{output[:200]}... (Tokens: {num_tokens})" if output else None,
            "error": error,
            "duration": duration,
            "usage": {"total_tokens": num_tokens} # Store token count here
        })
    return results


# --- Placeholder Benchmarks for Agentic Capabilities ---

def benchmark_reflection(llm_client: LLMClient, memory_system: UnifiedMemorySystem, reflection_queries: List[str]) -> List[Dict[str, Any]]:
    """Benchmarks the memory reflection capability."""
    logger.info("Starting memory reflection benchmark...")
    results = []
    if not llm_client:
        logger.error("LLM client not available for memory reflection benchmark.")
        return [{"input": reflection_queries, "output": None, "error": "LLM Client not configured", "duration": 0, "usage": {}}]

    # Helper to count tokens safely
    def _count_tokens_helper(text: str) -> int:
         if hasattr(llm_client, 'tokenizer') and llm_client.tokenizer:
             try: return len(llm_client.tokenizer.encode(text))
             except Exception: return len(text) // 3 # Fallback estimation
         else: return len(text) // 3 # Fallback estimation

    system_prompt = """
    You are an AI assistant specialized in analyzing and synthesizing information from memory logs.
    Review the provided memories carefully. Your goal is to extract meaningful insights, identify recurring patterns or themes, summarize the key information, establish connections between memories, and suggest potential next steps or areas for further investigation based ONLY on the provided memories.

    Focus on:
    1.  **Insights:** What significant conclusions or understandings can be drawn?
    2.  **Patterns:** Are there recurring topics, behaviors, or outcomes?
    3.  **Knowledge:** What factual information or key takeaways should be retained?
    4.  **Connections:** How do different memories relate to each other?
    5.  **Suggestions:** Based on the analysis, what actions, questions, or objectives seem relevant?
    6.  **Summary:** Provide a concise overall summary of the analyzed memories.

    Present your analysis in a structured JSON format. Ensure the JSON is valid.
    Example structure:
    {
        "summary": "A brief summary of the key themes in the memories.",
        "insights": ["Insight 1", "Insight 2"],
        "patterns": ["Pattern A identified", "Pattern B recurring"],
        "knowledge": ["Key fact 1", "Important detail 2"],
        "connections": ["Memory 1 relates to Memory 5 because...", "Theme X connects memories 2, 4, 7"],
        "suggested_actions": ["Follow up on topic Y", "Verify fact Z"]
    }
    Output ONLY the JSON object.
    """

    for query in reflection_queries:
        start_time = time.time()
        output = None
        error = None
        usage = None
        formatted_memories = "No relevant memories found or formatted."
        memory_count = 0
        try:
            # 1. Retrieve memories (using mock memory search)
            # Note: search_memory returns MemoryEntry objects now
            relevant_memories = memory_system.search_memory(query=query, k_per_type=10) # k=10 per type, max ~30
            memory_count = len(relevant_memories)
            logger.debug(f"Retrieved {memory_count} memories for reflection query: '{query}'")

            if relevant_memories:
                # 2. Format memories for prompt
                memory_text = ""
                current_tokens = 0
                max_context_tokens = 3500 # Reserve some tokens for the rest of the prompt

                # Sort by timestamp (most recent first)
                relevant_memories.sort(key=lambda x: x.timestamp, reverse=True)

                for memory in relevant_memories[:20]: # Limit total memories in prompt
                    memory_source = f"[{memory.memory_type.replace('_', ' ').upper()}]"
                    memory_time = memory.timestamp.strftime("%Y-%m-%d %H:%M")
                    content_str = str(memory.content)
                    if isinstance(memory.content, dict): content_str = json.dumps(memory.content)

                    part = f"{memory_source} ({memory_time}):\\n{content_str}\\n---\\n"
                    part_tokens = _count_tokens_helper(part)

                    if current_tokens + part_tokens < max_context_tokens:
                        memory_text += part
                        current_tokens += part_tokens
                    else:
                        logger.warning("Reached token limit while formatting memories for reflection.")
                        break
                
                formatted_memories = memory_text.strip() if memory_text else "No memories formatted."


            # 3. Construct prompt
            reflection_prompt = f"""
            Analyze the following memories related to the query: "{query}"
            Memory Log:
            ---
            {formatted_memories}
            ---
            Provide your analysis in the specified JSON format. Output ONLY JSON.
            """

            # 4. Call LLM
            response_content = llm_client.process_with_llm(
                prompt=reflection_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=1500
            )
            output = response_content
            usage = llm_client.last_response.usage if llm_client.last_response else {}

            # 5. Basic Validation
            try:
                if output.strip().startswith("```json"):
                    output = output.strip()[7:-3].strip()
                elif output.strip().startswith("```"):
                    output = output.strip()[3:-3].strip()
                
                parsed = json.loads(output)
                required_keys = {"summary", "insights", "patterns", "knowledge", "connections", "suggested_actions"}
                if not required_keys.issubset(parsed.keys()):
                     logger.warning(f"Reflection response missing keys for query '{query}'. Got: {parsed.keys()}")
                     # Don't raise error, just log. Keep output for review.

            except (json.JSONDecodeError, ValueError) as json_err:
                error = f"JSON Validation Error: {json_err} - Output: {output[:100]}..."
                logger.warning(error)

        except Exception as e:
            logger.error(f"Error during reflection benchmark for query '{query}': {e}", exc_info=True)
            error = str(e)
            if llm_client.last_response:
                 usage = llm_client.last_response.usage

        duration = time.time() - start_time
        results.append({
            "input": query,
            "context_preview": f"({memory_count} memories) {formatted_memories[:100]}...",
            "output": output,
            "error": error,
            "duration": duration,
            "usage": usage
        })
    logger.info("Finished memory reflection benchmark.")
    return results


def benchmark_objective_review(llm_client: LLMClient, agent: JarvisAgent, review_triggers: List[str]) -> List[Dict[str, Any]]:
    """Benchmarks the objective review and refinement capability."""
    logger.info("Starting objective review benchmark...")
    results = []
    if not llm_client:
        logger.error("LLM client not available for objective review benchmark.")
        return [{"input": review_triggers, "output": None, "error": "LLM Client not configured", "duration": 0, "usage": {}}]

    system_prompt = """
    You are an objective review assistant for the Jarvis agent.
    Analyze the list of active objectives in the context of the agent's recent activity and memory.
    Suggest refinements, completion flags, or mark objectives as potentially irrelevant.
    Output suggestions as a JSON list of objects, each with 'objective_id' and 'suggestion' (e.g., 'refine: [new description]', 'mark_complete', 'mark_irrelevant', 'keep').
    Example: `[{\"objective_id\": \"obj_123\", \"suggestion\": \"refine: Focus search on Python libraries\"}, {\"objective_id\": \"obj_456\", \"suggestion\": \"mark_complete\"}]`
    Output ONLY the JSON list.
    """
    
    # Run the review process multiple times if needed (using review_triggers as symbolic inputs)
    for trigger_desc in review_triggers:
        start_time = time.time()
        output = None
        error = None
        usage = None
        context_preview = "No context assembled."
        objective_list_str = "No active objectives found."
        try:
            # 1. Get active objectives (from mock memory)
            active_objectives = [
                obj for obj in agent.memory_system.medium_term.search_objectives("", n_results=10)
                # Assuming search_objectives returns list of dicts matching Objective structure
                if obj["metadata"].get("status") == "active"
            ]

            if not active_objectives:
                logger.info("No active objectives found in mock memory for review.")
                error = "No active objectives found in mock memory."
                objective_list_str = "No active objectives found."
            else:
                # Format for prompt
                objective_list = []
                for o in active_objectives:
                    # Handle potential missing keys gracefully
                    obj_id = o.get('objective_id', o.get('id', 'unknown'))
                    desc = o.get('description', 'No description')
                    objective_list.append(f"- ID: {obj_id}, Desc: {desc}")
                objective_list_str = "\n".join(objective_list)

                # 2. Assemble context
                context = agent.memory_system.assemble_context(
                    query=f"Review active objectives triggered by: {trigger_desc}",
                    max_tokens=2500
                )
                context_preview = f"{context[:150]}..."

                # 3. Construct prompt
                prompt = f"""
                Recent Context & Memory:
                {context}

                Active Objectives:
                {objective_list_str}

                Review the objectives based on the context and suggest actions in the specified JSON format. Output ONLY JSON.
                """

                # 4. Call LLM
                response_content = llm_client.process_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.4,
                    max_tokens=500
                )
                output = response_content
                usage = llm_client.last_response.usage if llm_client.last_response else {}

                # 5. Basic Validation
                try:
                    if output.strip().startswith("```json"):
                        output = output.strip()[7:-3].strip()
                    elif output.strip().startswith("```"):
                        output = output.strip()[3:-3].strip()

                    suggestions = json.loads(output)
                    if not isinstance(suggestions, list):
                         raise ValueError("Output is not a list.")
                    # Further check structure if needed
                    # for item in suggestions: ...

                except (json.JSONDecodeError, ValueError) as json_err:
                    error = f"JSON Validation Error: {json_err} - Output: {output[:100]}..."
                    logger.warning(error)

        except Exception as e:
            logger.error(f"Error during objective review benchmark for trigger '{trigger_desc}': {e}", exc_info=True)
            error = str(e)
            if llm_client.last_response:
                 usage = llm_client.last_response.usage

        duration = time.time() - start_time
        results.append({
            "input": trigger_desc,
            "context_preview": context_preview,
            "objectives_reviewed": objective_list_str,
            "output": output,
            "error": error,
            "duration": duration,
            "usage": usage
        })
    logger.info("Finished objective review benchmark.")
    return results

def benchmark_self_assessment(llm_client: LLMClient, agent: JarvisAgent, assessment_triggers: List[str]) -> List[Dict[str, Any]]:
    """Benchmarks the agent self-assessment capability."""
    logger.info("Starting self-assessment benchmark...")
    results = []
    if not llm_client:
        logger.error("LLM client not available for self-assessment benchmark.")
        return [{"input": assessment_triggers, "output": None, "error": "LLM Client not configured", "duration": 0, "usage": {}}]

    system_prompt = """
    You are a self-assessment module for the Jarvis AI agent.
    Analyze the provided context (recent interactions, errors, feedback, memory snippets).
    Identify key strengths, weaknesses, and areas for improvement in the agent's performance.
    Suggest 1-3 concrete improvement goals or learning objectives.
    Output the assessment as a JSON object with keys: "strengths" (list[str]), "weaknesses" (list[str]), "improvement_goals" (list[str]).
    Example: `{\"strengths\": [\"Good at web searches\"], \"weaknesses\": [\"Struggles with complex planning\"], \"improvement_goals\": [\"Improve plan decomposition for multi-step tasks\"]}`
    Output ONLY the JSON object.
    """

    # Run assessment multiple times if needed (using triggers as symbolic inputs)
    for trigger_desc in assessment_triggers:
        start_time = time.time()
        output = None
        error = None
        usage = None
        context_preview = "No context assembled."
        try:
            # 1. Assemble context
            context = agent.memory_system.assemble_context(
                query=f"Self-assessment trigger: {trigger_desc}",
                max_tokens=3000
            )
            context_preview = f"{context[:150]}..."

            # 2. Construct prompt
            prompt = f"""
            Context for Self-Assessment:
            {context}

            Perform self-assessment and output the result as JSON. Output ONLY JSON.
            """

            # 3. Call LLM
            response_content = llm_client.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=600
            )
            output = response_content
            usage = llm_client.last_response.usage if llm_client.last_response else {}

            # 4. Basic Validation
            try:
                if output.strip().startswith("```json"):
                    output = output.strip()[7:-3].strip()
                elif output.strip().startswith("```"):
                    output = output.strip()[3:-3].strip()
                
                assessment_data = json.loads(output)
                required_keys = {"strengths", "weaknesses", "improvement_goals"}
                if not isinstance(assessment_data, dict) or not required_keys.issubset(assessment_data.keys()):
                     raise ValueError("Assessment JSON missing required keys.")
                # Further validation if needed

            except (json.JSONDecodeError, ValueError) as json_err:
                error = f"JSON Validation Error: {json_err} - Output: {output[:100]}..."
                logger.warning(error)

        except Exception as e:
            logger.error(f"Error during self-assessment benchmark for trigger '{trigger_desc}': {e}", exc_info=True)
            error = str(e)
            if llm_client.last_response:
                 usage = llm_client.last_response.usage

        duration = time.time() - start_time
        results.append({
            "input": trigger_desc,
            "context_preview": context_preview,
            "output": output,
            "error": error,
            "duration": duration,
            "usage": usage
        })
    logger.info("Finished self-assessment benchmark.")
    return results


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM capabilities for Jarvis.")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "openai", "anthropic"], help="LLM provider to use.")
    parser.add_argument("--model", type=str, default=None, help="Specific model name to use (optional, defaults to provider's default). Provider defaults: groq=llama3-8b-8192, openai=gpt-4o-mini, anthropic=claude-3-haiku-20240307")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="File to save benchmark results.")
    args = parser.parse_args()

    logger.info(f"Starting benchmarks for provider: {args.provider}, Model: {args.model or 'Default'}")

    # --- Initialize Components ---
    try:
        llm = LLMClient(primary_provider=args.provider)
        if args.model:
             # Set the specific model for the chosen provider
             if args.provider == 'groq':
                 llm.groq_model = args.model
             elif args.provider == 'openai':
                 llm.openai_model = args.model
             elif args.provider == 'anthropic':
                 llm.anthropic_model = args.model
             logger.info(f"Using specified model: {llm.get_model_name()}")

        # Use mock memory and skill registry for more isolated testing initially
        mock_skill_registry = MockSkillRegistry()
        mock_memory = MockUnifiedMemory(logger=setup_logger("mock_memory"), llm_client=llm) # Give memory the LLM client too

        # Systems needed for benchmarks (using mocks where appropriate)
        planning = PlanningSystem(unified_memory=mock_memory, llm_client=llm, logger=setup_logger("planning"))
        execution = ExecutionSystem(unified_memory=mock_memory, llm_client=llm, skill_registry=mock_skill_registry, logger=setup_logger("execution"))
        
        # Agent needed for some benchmarks (using mocks)
        agent = JarvisAgent(memory_system=mock_memory, llm=llm, logger=setup_logger("agent"))
        agent.planning_system = planning # Ensure agent uses the same instance
        agent.execution_system = execution # Ensure agent uses the same instance
        agent.skill_registry = mock_skill_registry # Ensure agent uses the same instance

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)


    # --- Define Benchmark Data ---
    decomposition_objectives = [
        "Plan my upcoming week, scheduling focused work blocks and personal appointments.",
        "Research the latest advancements in AI agent memory systems, focusing on long-term consolidation techniques.",
        "Draft a concise email to the Project Phoenix team summarizing the key decisions from today's sync meeting and outlining next steps.",
        "Organize my digital notes on project management methodologies.",
        "Find and summarize three recent articles about using Groq for low-latency LLM applications.",
    ]

    skill_parsing_tasks = [
        "Search the web for tutorials on Python Pydantic V2 model config.",
        "Can you read the main points from the file named 'meeting_minutes_apr4.md'?",
        "Send an email to manager@example.com with the subject 'Weekly Update' and body containing my progress report.",
        "Perform a web search about the weather forecast for tomorrow.",
        "Read the contents of /home/user/documents/important_notes.txt",
        "Email Bob (bob@example.com) subject: Quick question - body: Did you push the latest code?",
        "Just run the skill that requires no parameters.",
        "Look up the capital of France online.",
        "What does the file 'config.yaml' contain?",
        "Tell alice@example.com via email that the report is ready. Subject: Report Finalized",
    ]

    error_diagnosis_scenarios = [
        ("Search the web for Groq API docs", "APIError: Connection timed out after 30 seconds."),
        ("Read the file '/data/project_alpha.log'", "FileNotFoundError: [Errno 2] No such file or directory: '/data/project_alpha.log'"),
        ("Send email to deployment@notify.service", "SMTPServerDisconnected: Connection unexpectedly closed"),
        ("Summarize the report at https://example.com/long_report.pdf", "SkillError: Failed to download or parse PDF content."),
        ("Generate a complex graph based on user data", "LLMError: Response generation failed due to content filter violation."),
    ]

    context_assembly_queries = [
        "What were the key points from the last Project Phoenix meeting?",
        "Summarize my recent interactions about AI memory systems.",
        "Give me context for planning my week.",
        "What's the current status of the 'Draft email' objective?",
    ]

    # --- Run Benchmarks ---
    all_results = []

    all_results.append(run_benchmark(
        benchmark_decompose_objective,
        "Task Decomposition",
        llm,
        planning_system=planning,
        objectives=decomposition_objectives
    ))

    all_results.append(run_benchmark(
        benchmark_parse_task_for_skill,
        "Skill Parsing",
        llm,
        execution_system=execution,
        tasks=skill_parsing_tasks
    ))

    all_results.append(run_benchmark(
        benchmark_diagnose_execution_error,
        "Error Diagnosis",
        llm,
        execution_system=execution,
        errors=error_diagnosis_scenarios
    ))

    all_results.append(run_benchmark(
        benchmark_assemble_context,
        "Context Assembly",
        llm,
        memory_system=mock_memory,
        queries=context_assembly_queries
    ))

    # Placeholders for Agentic Capabilities
    all_results.append(run_benchmark(
        benchmark_reflection,
        "Memory Reflection",
        llm,
        memory_system=mock_memory,
        reflection_queries=context_assembly_queries
    ))

    all_results.append(run_benchmark(
        benchmark_objective_review,
        "Objective Review",
        llm,
        agent=agent,
        review_triggers=context_assembly_queries
    ))

    all_results.append(run_benchmark(
        benchmark_self_assessment,
        "Self Assessment",
        llm,
        agent=agent,
        assessment_triggers=context_assembly_queries
    ))


    # --- Save Results ---
    try:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save benchmark results: {e}")

if __name__ == "__main__":
    main() 