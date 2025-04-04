"""
Execution system that handles task execution and error recovery.
"""

from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import time
import json
import logging

from utils.logger import setup_logger
# Import skill components
# from jarvis.skills import skill_registry, SkillResult # Removed unused import of old global
from jarvis.skills import SkillResult # Keep SkillResult
from jarvis.skills.registry import SkillRegistry # Import the class if needed for type hints elsewhere, or rely on TYPE_CHECKING

# Conditional Imports for Type Checking
if TYPE_CHECKING:
    from .memory.unified_memory import UnifiedMemorySystem
    from .llm import LLMClient
    from .skills import SkillRegistry
    from jarvis.planning import Task # Assuming Task is defined in planning.py
    
# Import runtime dependencies
from .llm import LLMError # <-- ADDED Import

# Local Task definition for cases where full import isn't desired/possible at runtime
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

# --- Pydantic Model for Structured LLM Output (Skill Parsing) ---
class ParsedSkillCall(BaseModel):
    """Represents the structured identification of a skill and its parameters from text."""
    skill_name: Optional[str] = Field(..., description="The exact name of the skill to call, or null if no suitable skill is found.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of parameter names and values extracted for the skill.")

# ------------------------------------------------------------------

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

    logger: logging.Logger
    llm: Optional['LLMClient']
    unified_memory: 'UnifiedMemorySystem'
    skill_registry: 'SkillRegistry'
    planning_system: Optional['PlanningSystem'] = None
    strategies: Dict[str, ExecutionStrategy] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, 
                 unified_memory: 'UnifiedMemorySystem', 
                 llm_client: Optional['LLMClient'], 
                 skill_registry: 'SkillRegistry', 
                 planning_system: Optional['PlanningSystem'],
                 **data):
        # Initialize logger first if not passed in data
        # --- Logger Initialization ---
        if 'logger' not in data:
            # data['logger'] = logging.getLogger(__name__) # Use module name
            # # Basic check: assume configured if it has handlers or level is not NOTSET
            # if not data['logger'].hasHandlers() and data['logger'].level == logging.NOTSET:
            #      data['logger'].setLevel(logging.INFO) # Default level if needed
            #      print(f"WARNING: Logger '{__name__}' was not configured. Applying default level INFO.")
            data['logger'] = setup_logger(__name__) # Reverted to setup_logger
        # --- End Logger Initialization ---

        # Check required dependencies before passing them to super
        if not unified_memory:
            raise ValueError("UnifiedMemorySystem instance is required for ExecutionSystem")
        if not skill_registry:
            raise ValueError("SkillRegistry instance is required for ExecutionSystem")

        # Add dependencies to data dict for Pydantic initialization
        data['unified_memory'] = unified_memory
        data['skill_registry'] = skill_registry
        data['llm'] = llm_client
        data['planning_system'] = planning_system

        # Now call super().__init__ with all necessary data from the dict
        super().__init__(**data)

        # Post-initialization checks or actions can go here
        # The logger is now available via self.logger as Pydantic initialized it
        if not self.llm:
            self.logger.warning("LLMClient instance not provided to ExecutionSystem. Skill parsing and error diagnosis may be limited.")

        # Initialize strategies after core components are initialized by Pydantic
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
                    fallback_strategy = self.strategies.get(fallback_name)
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

    def _diagnose_execution_error(self, task: 'Task', error_message: str) -> str:
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

    def _load_execution_strategies(self) -> None:
        """Load execution strategies from memory"""
        try:
            # Default strategies
            self.strategies = {
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
            
            # TODO: Load custom strategies from memory/config if needed
            self.logger.info(f"Loaded {len(self.strategies)} execution strategies.")
            
        except Exception as e:
            self.logger.error(f"Error loading execution strategies: {e}")
    
    def _get_execution_strategy(self, task: 'Task') -> ExecutionStrategy:
        """Get the execution strategy for a task, falling back to default."""
        # Check task metadata for a specific strategy
        strategy_name = task.metadata.get("execution_strategy", "default")
        default_strategy = self.strategies["default"] # Get default first
        return self.strategies.get(strategy_name, default_strategy)
    
    def _execute_with_strategy(self, task: 'Task', strategy: ExecutionStrategy) -> ExecutionResult:
        """Execute a task using a specific strategy (internal attempt logic)."""
        start_time = time.time()
        skill_name = None
        try:
            # --- PRIORITY CHECK: Handle Synthesis Tasks FIRST ---
            task_desc_lower = task.description.lower()
            synthesis_keywords = ["synthesize", "summarize", "report", "compile", "present", "finalize", "combine", "consolidate", "review results"]
            is_synthesis_task = any(keyword in task_desc_lower for keyword in synthesis_keywords)

            if is_synthesis_task and self.planning_system:
                try:
                    self.logger.info(f"Task '{task.task_id}' identified as synthesis task based on description. Executing internal synthesis logic...")
                    synthesis_result: ExecutionResult = self._synthesize_results(task)
                    return synthesis_result # Return directly if synthesis is handled
                except Exception as synth_err:
                    error_msg = f"Internal synthesis execution failed for task {task.task_id}: {synth_err}"
                    self.logger.error(error_msg, exc_info=True)
                    # Return failure if synthesis itself errored
                    return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)
            
            # --- If NOT a synthesis task, proceed with Skill Dispatch Attempt ---
            parsed_result = self._parse_task_for_skill(task)

            # Check if parsing returned a valid skill and params
            if parsed_result:
                skill_name, params = parsed_result # Unpack only if not None
            else:
                skill_name = None # Explicitly set to None
                params = None

            if skill_name and params is not None and self.skill_registry:
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
                    # Fall through if skill not in registry

            # --- Handle cases where no skill was parsed by the LLM ---
            elif not skill_name:
                task_desc_lower = task.description.lower()
                # More comprehensive keywords for synthesis/final reporting
                synthesis_keywords = ["synthesize", "summarize", "report", "compile", "present", "finalize", "combine", "consolidate", "review results"]
                # Check if it looks like a synthesis task AND planning system is available
                is_synthesis_task = any(keyword in task_desc_lower for keyword in synthesis_keywords)

                if is_synthesis_task and self.planning_system:
                    try:
                        self.logger.info(f"Task '{task.task_id}' identified as synthesis task based on description. Executing internal synthesis logic...")
                        # Directly call the internal synthesis method
                        synthesis_result: ExecutionResult = self._synthesize_results(task)
                        # Return the result obtained from the synthesis method
                        return synthesis_result
                    except Exception as synth_err:
                         # Handle errors during the synthesis process itself
                         error_msg = f"Internal synthesis execution failed for task {task.task_id}: {synth_err}"
                         self.logger.error(error_msg, exc_info=True)
                         return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)
                else:
                    # If not a synthesis task or planning system unavailable, treat as internal step
                    self.logger.info(f"No suitable skill identified by LLM for task '{task.task_id}' ('{task.description}') and not identified as a synthesis task. Marking as completed internal step.")
                    exec_time = time.time() - start_time
                    return ExecutionResult(
                        task_id=task.task_id,
                        success=True, # Mark as success as it's an expected internal step
                        output="Internal step completed (no direct action/skill identified).",
                        error=None,
                        execution_time=exec_time,
                        metadata={'executed_by': 'internal'} # Mark execution type
                    )

            # --- Fallback Error Handling ---
            # If code reaches here, it means something went wrong earlier
            # e.g., skill parsed but not found, or parameter validation failed and wasn't caught cleanly above.
            error_msg = f"Execution failed for task {task.task_id}. Could not execute skill '{skill_name}' or handle as internal/synthesis step."
            self.logger.error(error_msg)
            return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)

        except Exception as strategy_exec_err:
             # Catch unexpected errors within the _execute_with_strategy scope
             error_msg = f"Unexpected error during execution strategy for task {task.task_id}: {strategy_exec_err}"
             self.logger.error(error_msg, exc_info=True)
             return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time)

    def _parse_task_for_skill(self, task: 'Task') -> Optional[Tuple[str, Dict[str, Any]]]:
        """Use LLM to parse task description into a structured skill call, with fallback."""
        if not self.llm:
            self.logger.warning("LLM not available for skill parsing.")
            return None
            
        # --- Determine if we can use Instructor --- 
        use_instructor = False
        if hasattr(self.llm, 'process_structured_output'):
            # Check if the *currently selected* provider's client seems patched
            client = None
            provider = self.llm.primary_provider
            if provider == "groq": client = self.llm.groq_client
            elif provider == "openai": client = self.llm.openai_client
            elif provider == "anthropic": client = self.llm.anthropic_client
            
            if client and (hasattr(client, '_instructor_is_patched') or "instructor" in str(type(client)).lower()):
                 use_instructor = True
                 self.logger.debug("Instructor appears available for the current LLM provider.")
            else:
                 self.logger.warning(f"Instructor response_model requested, but client '{provider}' appears unpatched. Falling back to manual JSON parsing.")
        else:
            self.logger.error("LLMClient does not have 'process_structured_output' method.")
            # Fallback to manual parsing if method doesn't exist

        try:
            skill_defs = self.skill_registry.get_skill_definitions()
            if not skill_defs:
                self.logger.warning("Skill registry is empty. Cannot use LLM for skill parsing.")
                return None
            skill_defs_json = json.dumps(skill_defs, indent=2)
            prompt = f"Identify the skill and parameters for the task: {task.description}"

            if use_instructor:
                # --- Instructor Path --- 
                self.logger.debug(f"Sending skill parsing prompt to LLM for task: {task.task_id} (using Instructor)")
                system_prompt = f"""... [System prompt explaining Instructor/Pydantic output] ...""" # Truncated for brevity, use original prompt
                system_prompt = f"""
                You are an expert function calling agent. Your task is to identify the most appropriate skill (function) from the provided list to fulfill the user's request (task description) and extract the necessary parameters accurately.

                Available Skills (Functions):
                ```json
                {skill_defs_json}
                ```

                Analyze the following task description:
                `{task.description}`

                Determine the single best matching skill and its parameters based *only* on the task description and the available skills.
                If no single skill is a clear and appropriate match, return `skill_name` as `null`.
                Extract parameter values directly from the task description where possible.
                Ensure parameter keys and value types match the skill definition.
                Do not invent parameters or values not present in the description.
                You MUST output a JSON object conforming to the specified Pydantic schema.
                """
                parsed_response = self.llm.process_structured_output(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response_model=ParsedSkillCall,
                    temperature=0.0,
                    max_tokens=500
                )
                if isinstance(parsed_response, ParsedSkillCall):
                    skill_name = parsed_response.skill_name
                    params = parsed_response.parameters
                    # ... (rest of validation logic as before) ...
                    if not skill_name: return None # No suitable skill
                    if self.skill_registry.get_skill(skill_name): return skill_name, params
                    else: self.logger.warning(f"Instructor identified invalid skill '{skill_name}'"); return None
                else:
                    self.logger.error(f"Instructor response was not ParsedSkillCall: {type(parsed_response)}")
                    return None
            else:
                # --- Fallback Path (Manual JSON Parsing) --- 
                self.logger.debug(f"Sending skill parsing prompt to LLM for task: {task.task_id} (manual JSON parsing)")
                system_prompt = f"""... [System prompt explaining JSON output] ...""" # Truncated for brevity, use original prompt
                system_prompt = f"""
                You are an expert function calling agent. Your task is to identify the most appropriate skill (function) from the provided list to fulfill the user's request (task description) and extract the necessary parameters accurately.

                Available Skills (Functions):
                ```json
                {skill_defs_json}
                ```

                Analyze the following task description:
                `{task.description}`

                Determine the single best matching skill and its parameters based *only* on the task description and the available skills.

                Respond ONLY with a JSON object matching this exact structure:
                {{ "skill_name": "<name_of_best_matching_skill_or_null>", "parameters": {{ <extracted_parameters_object_or_empty_object> }} }}

                Rules:
                - If a skill perfectly matches the task, provide its name and parameters.
                - If no single skill is a clear and appropriate match, return `skill_name` as `null`.
                - Extract parameter values directly from the task description where possible.
                - Ensure parameter keys and value types match the skill definition.
                - Do not invent parameters or values not present in the description.
                - Output *only* the JSON object, nothing else.
                """
                response_content = self.llm.process_with_llm(
                    prompt=prompt, 
                    system_prompt=system_prompt,
                    temperature=0.0, 
                    max_tokens=500 
                )
                try:
                    # Clean potential markdown
                    if response_content.strip().startswith("```json"):
                        response_content = response_content.strip()[7:-3].strip()
                    elif response_content.strip().startswith("```"):
                         response_content = response_content.strip()[3:-3].strip()
                    parsed_data = json.loads(response_content)
                    if not isinstance(parsed_data, dict) or "skill_name" not in parsed_data or "parameters" not in parsed_data:
                         raise ValueError("Manual JSON response does not match required structure.")
                    skill_name = parsed_data.get("skill_name")
                    params = parsed_data.get("parameters")
                    if not skill_name or not isinstance(params, dict): return None # No suitable skill
                    if self.skill_registry.get_skill(skill_name): return skill_name, params
                    else: self.logger.warning(f"Manual JSON parsing identified invalid skill '{skill_name}'"); return None
                except (json.JSONDecodeError, ValueError) as json_err:
                    self.logger.error(f"Failed to parse or validate manual JSON response: {json_err}. Raw: {response_content}")
                    return None

        except (LLMError, ValueError) as e: # Catch LLM or Value errors from either path
             self.logger.error(f"Error during skill parsing for task {task.task_id}: {e}", exc_info=True)
             return None
        except Exception as e:
            self.logger.exception(f"Unexpected error during skill parsing on task {task.task_id}: {e}")
            return None

    def _synthesize_results(self, task: 'Task') -> ExecutionResult:
        """Synthesizes results from completed tasks in the plan associated with the given task."""
        start_time = time.time()
        plan_id = task.metadata.get("plan_id")

        if not self.planning_system:
            error_msg = f"PlanningSystem not available for synthesis task {task.task_id}"
            self.logger.error(error_msg)
            return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time, metadata={'executed_by': 'synthesis'})

        if not plan_id:
            error_msg = f"Missing plan_id in metadata for synthesis task {task.task_id}"
            self.logger.error(error_msg)
            return ExecutionResult(task_id=task.task_id, success=False, output=None, error=error_msg, execution_time=time.time()-start_time, metadata={'executed_by': 'synthesis'})

        try:
            # Retrieve the plan using the PlanningSystem's method
            # Note: Using internal _get_plan might be necessary if no public getter exists
            # Adjust based on PlanningSystem's actual interface
            plan = self.planning_system._get_plan(plan_id) # Assuming this internal method exists
            
            if not plan:
                raise ValueError(f"Could not retrieve plan {plan_id} for synthesis.")

            # Gather outputs from previous completed tasks in this plan
            previous_outputs = []
            for prev_task in plan.tasks:
                if prev_task.task_id == task.task_id: # Don't include the synthesis task itself
                    continue
                # Check for completion and presence of output in metadata
                if prev_task.status == "completed" and prev_task.metadata.get("last_output"):
                    previous_outputs.append(f"Output from task '{prev_task.description}' (ID: {prev_task.task_id}):\n{prev_task.metadata['last_output']}\n---")
            
            if not previous_outputs:
                self.logger.warning(f"No previous outputs found in plan {plan_id} for synthesis task {task.task_id}.")
                synthesis_context = "No previous task outputs were available to synthesize."
            else:
                synthesis_context = "\n".join(previous_outputs)

            # Use LLM to synthesize
            if not self.llm:
                raise ValueError("LLM not available for synthesis.")
            
            synthesis_system_prompt = """
            You are Jarvis, synthesizing the results of a completed plan.
            Review the provided outputs from the completed tasks in the plan.
            Generate a final, comprehensive response or report that directly addresses the original objective based *only* on these results.
            Present the information clearly and concisely to the user.
            Focus on answering the user's original goal using the gathered information.
            Do not add commentary about the process, just provide the synthesized result.
            """
            synthesis_prompt = f"""
            Original Objective: {plan.objective_description if hasattr(plan, 'objective_description') else '(Objective description not found)'} 
            Task to Perform: {task.description}
            
            Results from Previous Tasks:
            ---
            {synthesis_context}
            ---
            
            Generate the final synthesized response for the user based *only* on the provided results:
            """

            final_response = self.llm.process_with_llm(
                prompt=synthesis_prompt,
                system_prompt=synthesis_system_prompt,
                temperature=0.6, 
                max_tokens=2000 # Allow longer summary
            )

            return ExecutionResult(
                task_id=task.task_id,
                success=True,
                output=final_response,
                execution_time=time.time() - start_time,
                metadata={'executed_by': 'synthesis'}
            )

        except Exception as e:
            error_msg = f"Error during synthesis process for task {task.task_id}: {e}"
            self.logger.error(error_msg, exc_info=True)
            return ExecutionResult(
                task_id=task.task_id,
                success=False,
                output=None,
                error=error_msg,
                execution_time=time.time() - start_time,
                metadata={'executed_by': 'synthesis'} # Still mark as synthesis attempt
            )

    # _store_execution_result and _reflect_on_execution (optional, can be added later)
    # ... 

# <<< End of ExecutionSystem class definition >>>

# Ensure dependent classes are fully defined before rebuilding models that reference them
# try:
#     from .memory.unified_memory import UnifiedMemorySystem
#     from .skills.registry import SkillRegistry # Use registry path
#     from .llm import LLMClient
#     
#     # Perform model rebuild for ExecutionSystem
#     ExecutionSystem.model_rebuild()
#     print("Rebuilt Pydantic model: ExecutionSystem") # Optional
# except ImportError as e:
#     print(f"Warning: Could not import dependencies to rebuild ExecutionSystem: {e}")
# except NameError as e:
#     print(f"Warning: Could not rebuild ExecutionSystem, definition might be missing: {e}")
# except Exception as e:
#     print(f"Warning: An unexpected error occurred during ExecutionSystem model rebuild: {e}") 