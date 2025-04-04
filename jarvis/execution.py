"""
Execution system that handles task execution and error recovery.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import logging

from utils.logger import setup_logger
# Import skill components
from jarvis.skills import skill_registry, SkillResult

# Assume Task structure is defined elsewhere (e.g., planning.py)
# For type hinting here:
class Task(BaseModel):
    task_id: str
    description: str
    status: str = "pending"
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of executing a task"""
    task_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExecutionStrategy(BaseModel):
    """Strategy for executing a task"""
    name: str
    description: str
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay in seconds
    timeout: float = 30.0  # Maximum execution time
    fallback_strategies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExecutionSystem(BaseModel):
    """System for executing tasks with error recovery and skill dispatch."""

    logger: Any = None
    llm: Any = None # LLMClient expected here
    skill_registry: Any = None # SkillRegistry expected here

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("execution_system")

        # Initialize LLM client if not provided
        if not self.llm:
            try:
                from jarvis.llm import LLMClient
                self.llm = LLMClient()
            except Exception as e:
                self.logger.warning(f"Could not initialize LLM client: {e}")
        
        # Use the globally discovered skill registry
        from jarvis.skills import skill_registry as default_skill_registry
        self.skill_registry = data.get('skill_registry', default_skill_registry)

        # Initialize strategies
        self._strategies = {}
        self._load_execution_strategies()
    
    def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a task with error recovery, attempting skill dispatch first."""
        start_time = time.time()
        self.logger.info(f"Executing task {task.task_id}: '{task.description}'")
        try:
            strategy = self._get_execution_strategy(task)
            execution_result = None

            for attempt in range(strategy.max_retries + 1): # +1 for initial try
                self.logger.info(f"Attempt {attempt + 1}/{strategy.max_retries + 1} for task {task.task_id}")
                try:
                    execution_result = self._execute_with_strategy(task, strategy)
                    if execution_result.success:
                        self.logger.info(f"Task {task.task_id} completed successfully in attempt {attempt + 1}.")
                        break # Exit retry loop on success

                    self.logger.warning(f"Attempt {attempt + 1} failed for task {task.task_id}: {execution_result.error}")
                    # Increment task's internal error count (if Task model allows)
                    # task.error_count += 1

                    if attempt < strategy.max_retries:
                        delay = strategy.retry_delay * (2 ** attempt) # Exponential backoff
                        self.logger.info(f"Retrying task {task.task_id} in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        self.logger.error(f"Task {task.task_id} failed after maximum retries ({strategy.max_retries}).")
                        break # Exit loop after max retries

                except Exception as e:
                    # Catch errors during the execution attempt itself
                    self.logger.error(f"Exception during attempt {attempt + 1} for task {task.task_id}: {e}", exc_info=True)
                    execution_result = ExecutionResult(
                        task_id=task.task_id,
                        success=False,
                        output=None,
                        error=f"Exception during execution attempt: {str(e)}",
                        execution_time=time.time() - start_time # Partial time
                    )
                    if attempt >= strategy.max_retries:
                         self.logger.error(f"Task {task.task_id} failed due to exception after maximum retries.")
                         break # Exit loop after max retries
                    # Wait before retrying after an exception
                    delay = strategy.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying task {task.task_id} in {delay:.2f} seconds after exception...")
                    time.sleep(delay)

            # Fallback strategies (if primary + retries failed)
            if not execution_result or not execution_result.success:
                self.logger.info(f"Primary execution failed for task {task.task_id}. Trying fallback strategies...")
                for fallback_name in strategy.fallback_strategies:
                    fallback_strategy = self._strategies.get(fallback_name)
                    if fallback_strategy:
                        self.logger.info(f"Attempting fallback strategy '{fallback_name}' for task {task.task_id}")
                        try:
                            fallback_result = self._execute_with_strategy(task, fallback_strategy)
                            if fallback_result.success:
                                self.logger.info(f"Task {task.task_id} completed successfully with fallback strategy '{fallback_name}'.")
                                execution_result = fallback_result
                                break # Exit fallback loop on success
                            else:
                                self.logger.warning(f"Fallback strategy '{fallback_name}' failed for task {task.task_id}: {fallback_result.error}")
                        except Exception as e:
                            self.logger.error(f"Exception during fallback strategy '{fallback_name}' for task {task.task_id}: {e}", exc_info=True)
                            # Store the error from the fallback attempt
                            execution_result = ExecutionResult(
                                task_id=task.task_id,
                                success=False,
                                output=None,
                                error=f"Exception during fallback '{fallback_name}': {str(e)}",
                                execution_time=time.time() - start_time
                            )

            # Final result creation
            final_execution_time = time.time() - start_time
            if not execution_result:
                # Should not happen if loop logic is correct, but as a safeguard
                error_msg = "All execution strategies failed without producing a result."
                self.logger.error(f"Task {task.task_id}: {error_msg}")
                execution_result = ExecutionResult(
                    task_id=task.task_id, success=False, output=None,
                    error=error_msg, execution_time=final_execution_time
                )
            else:
                # Update execution time to final value
                execution_result.execution_time = final_execution_time

            # Log final status
            self.logger.info(f"Finished execution for task {task.task_id}. Success: {execution_result.success}, Time: {final_execution_time:.2f}s")
            
            # If failed, attempt diagnosis
            if not execution_result.success and execution_result.error:
                 diagnosis = self._diagnose_execution_error(task, execution_result.error)
                 execution_result.metadata['diagnosis'] = diagnosis # Add diagnosis to result metadata
                 self.logger.info(f"Task {task.task_id} failure diagnosis: {diagnosis}")
            
            return execution_result

        except Exception as e:
            # Catch errors in the main execute_task orchestration
            error_msg = f"Fatal error executing task {task.task_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            final_execution_time = time.time() - start_time
            # Attempt diagnosis even for fatal errors
            diagnosis = self._diagnose_execution_error(task, error_msg)
            return ExecutionResult(
                task_id=task.task_id, success=False, output=None, error=error_msg,
                execution_time=final_execution_time, metadata={'diagnosis': diagnosis}
            )

    def _diagnose_execution_error(self, task: Task, error_message: str) -> str:
        """Use LLM to diagnose task execution errors."""
        diagnosis = "Diagnosis unavailable."
        if self.llm:
            try:
                # Note: assemble_context might not be available here directly
                # Construct basic context for diagnosis
                context = f"Task Description: {task.description}\n"
                context += f"Task Metadata: {json.dumps(task.metadata)}\n"
                context += f"Error Encountered: {error_message}\n"
                
                # Include skill definitions if parsing failed within execution?
                # skill_defs = self.skill_registry.get_skill_definitions()
                # context += f"Available Skills: {json.dumps(skill_defs)}\n"
                
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
                
                diagnosis = self.llm.process_with_llm(prompt, system_prompt, temperature=0.4, max_tokens=150).strip()
            except Exception as diag_err:
                self.logger.error(f"LLM diagnosis for task {task.task_id} failed: {diag_err}")
        return diagnosis

    def _load_execution_strategies(self):
        """Load execution strategies from memory"""
        try:
            # Default strategies
            self._strategies = {
                "default": ExecutionStrategy(
                    name="default",
                    description="Default execution strategy",
                    max_retries=3,
                    retry_delay=1.0,
                    timeout=30.0,
                    fallback_strategies=["simple", "robust"]
                ),
                "simple": ExecutionStrategy(
                    name="simple",
                    description="Simple execution without retries",
                    max_retries=0,
                    timeout=10.0
                ),
                "robust": ExecutionStrategy(
                    name="robust",
                    description="Robust execution with many retries",
                    max_retries=5,
                    retry_delay=2.0,
                    timeout=60.0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error loading execution strategies: {e}")
            # Ensure at least default strategy exists
            self._strategies = {
                "default": ExecutionStrategy(
                    name="default",
                    description="Default execution strategy",
                    max_retries=3,
                    retry_delay=1.0,
                    timeout=30.0
                )
            }
    
    def _get_execution_strategy(self, task: Task) -> ExecutionStrategy:
        """Get the appropriate execution strategy for a task"""
        try:
            # Check task metadata for strategy
            strategy_name = task.metadata.get("execution_strategy")
            if strategy_name and strategy_name in self._strategies:
                return self._strategies[strategy_name]
            
            # Use default strategy
            return self._strategies["default"]
            
        except Exception as e:
            self.logger.error(f"Error getting execution strategy: {e}")
            return self._strategies["default"]
    
    def _execute_with_strategy(self, task: Task, strategy: ExecutionStrategy) -> ExecutionResult:
        """Execute a task using a specific strategy, attempting skill dispatch first."""
        start_time = time.time()
        try:
            # --- Skill Dispatch Attempt ---
            skill_name, params = self._parse_task_for_skill(task)

            if skill_name and self.skill_registry:
                skill = self.skill_registry.get_skill(skill_name)
                if skill:
                    self.logger.info(f"Attempting to execute task '{task.task_id}' using skill: '{skill_name}'")
                    validation_error = skill.validate_parameters(params) # Note: validate_parameters might modify params (e.g., type conversion)
                    if validation_error:
                        error_msg = f"Parameter validation failed for skill '{skill_name}': {validation_error}"
                        self.logger.error(error_msg)
                        # Don't immediately fail execution, let LLM fallback handle it maybe?
                        # Or return failure directly:
                        return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)

                    try:
                        skill_result: SkillResult = skill.execute(**params)
                        exec_time = time.time() - start_time
                        return ExecutionResult(
                            task_id=task.task_id,
                            success=skill_result.success,
                            output=skill_result.data or skill_result.message,
                            error=skill_result.error,
                            execution_time=exec_time,
                            metadata={'executed_by': 'skill', 'skill_name': skill_name}
                        )
                    except Exception as skill_exec_err:
                        error_msg = f"Skill '{skill_name}' execution failed: {skill_exec_err}"
                        self.logger.error(error_msg, exc_info=True)
                        return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)
                else:
                    self.logger.warning(f"Parsed skill '{skill_name}' not found in registry.")
            # --- End Skill Dispatch Attempt ---

            # --- LLM Execution Fallback ---
            self.logger.info(f"No skill dispatched for task {task.task_id}. Attempting LLM execution.")
            if self.llm:
                try:
                    # Existing LLM execution logic (can be refined)
                    system_prompt = """
                    You are Jarvis, an AI assistant executing a planned task.
                    Analyze the task description and metadata.
                    Describe the exact steps you would take to accomplish this task.
                    Summarize the outcome or result of performing these steps.
                    Be concise and focus on the execution and result.
                    """

                    prompt = f"""
                    Task to execute: {task.description}
                    Task Metadata: {json.dumps(task.metadata, indent=2)}

                    Execute this task and provide a summary of the steps and results.
                    """

                    llm_output = self.llm.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.5, # Lower temp for execution description
                        max_tokens=500
                    )
                    exec_time = time.time() - start_time
                    return ExecutionResult(
                        task_id=task.task_id,
                        success=True, # Assume success if LLM provides output
                        output=llm_output,
                        execution_time=exec_time,
                        metadata={'executed_by': 'llm'}
                    )
                except Exception as llm_err:
                    error_msg = f"Error executing task {task.task_id} with LLM: {llm_err}"
                    self.logger.error(error_msg)
                    return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)

            # --- Final Fallback (If no skill and no LLM) ---
            self.logger.warning(f"No skill or LLM available for task {task.task_id}. Using basic fallback.")
            output = f"Placeholder execution for task: {task.description}"
            success = True # Basic fallback assumes success
            exec_time = time.time() - start_time
            return ExecutionResult(
                task_id=task.task_id,
                success=success,
                output=output,
                execution_time=exec_time,
                metadata={'executed_by': 'fallback'}
            )

        except Exception as e:
            # Catch errors within the _execute_with_strategy scope
            error_msg = f"Unexpected error in strategy execution for task {task.task_id}: {e}"
            self.logger.error(error_msg, exc_info=True)
            return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)

    def _parse_task_for_skill(self, task: Task) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Attempts to parse the task to identify a skill and its parameters."""
        # Strategy 1: Check metadata first
        if "skill_name" in task.metadata:
            skill_name = task.metadata["skill_name"]
            params = task.metadata.get("parameters", {})
            if isinstance(params, dict):
                self.logger.info(f"Found skill '{skill_name}' in task metadata.")
                return skill_name, params
            else:
                 self.logger.warning(f"Task metadata has skill_name '{skill_name}' but parameters are not a dict.")

        # Strategy 2: Use LLM to parse description (if LLM available)
        if self.llm and self.skill_registry:
            try:
                # Get available skill definitions for the LLM
                skill_defs = self.skill_registry.get_skill_definitions()
                if not skill_defs:
                    return None, None # No skills to match against

                skill_defs_json = json.dumps(skill_defs, indent=2)

                system_prompt = """
                You are an expert system analyzing task descriptions to identify the appropriate skill and extract its parameters.
                Given a task description and a list of available skills (with their descriptions and parameters),
                determine which single skill is the best match for the task.
                Extract the necessary parameters for that skill from the task description.

                Available Skills:
                {skill_defs_json}

                Respond ONLY with a JSON object with this exact structure:
                {
                    "skill_name": "<name_of_the_best_matching_skill_or_null>",
                    "parameters": { <key_value_pairs_of_extracted_parameters_or_empty_dict> }
                }
                If no single skill is a clear match for the task description, return skill_name as null.
                Ensure parameter keys match the definition exactly.
                Infer parameter values from the task description.
                """
                prompt = f"Task Description: {task.description}"

                response = self.llm.process_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.1, # Low temp for precise parsing
                    max_tokens=300
                )

                parsed_data = json.loads(response)
                skill_name = parsed_data.get("skill_name")
                params = parsed_data.get("parameters", {})

                if skill_name and isinstance(params, dict):
                    self.logger.info(f"LLM parsed task description to skill: '{skill_name}', params: {params}")
                    return skill_name, params
                else:
                    self.logger.info("LLM could not identify a suitable skill for the task description.")
                    return None, None

            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse LLM response for skill parsing. Response: {response[:200]}...")
                return None, None
            except Exception as e:
                self.logger.error(f"Error using LLM for skill parsing: {e}")
                return None, None

        # Strategy 3: Simple keyword matching (optional basic fallback)
        # Add simple logic here if needed, e.g., if 'search web' in task.description.lower() -> return "web_search", {}

        self.logger.info(f"Could not parse skill from task: {task.task_id}")
        return None, None

    # _store_execution_result and _reflect_on_execution (optional, can be added later)
    # ... 