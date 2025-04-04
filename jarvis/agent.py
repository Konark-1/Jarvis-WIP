"""
Core agentic implementation of Jarvis that integrates planning, execution, and memory systems.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import logging # For Logger type hint

# Conditional imports for type checking
if TYPE_CHECKING:
    from .planning import PlanningSystem, Task
    from .execution import ExecutionSystem
    from .memory.unified_memory import UnifiedMemorySystem
    from .skills import SkillRegistry
    from .llm import LLMClient
    
# Runtime imports
from .planning import PlanningSystem, Task
from .execution import ExecutionSystem
from .memory.unified_memory import UnifiedMemorySystem
from .skills.registry import SkillRegistry
from utils.logger import setup_logger
from jarvis.llm import LLMClient

# --- Consolidated Model Rebuild --- REMOVED
# try:
#     # Dependencies first (if they have forward refs themselves, though unlikely here)
#     UnifiedMemorySystem.model_rebuild()
#     LLMClient.model_rebuild() # Assuming LLMClient might have forward refs
# 
#     # Systems depending on Memory, LLM, Skills
#     PlanningSystem.model_rebuild()
#     ExecutionSystem.model_rebuild()
# 
#     print("Rebuilt core system Pydantic models (UnifiedMemory, LLMClient, Planning, Execution).")
# except NameError as e:
#     print(f"Warning: A NameError occurred during core model rebuild - likely a missing import or definition: {e}")
# except AttributeError as e:
#     # Catch if a class doesn't have model_rebuild (e.g., not a Pydantic model)
#     print(f"Warning: An AttributeError occurred during core model rebuild - potentially tried to rebuild a non-Pydantic model: {e}")
# except Exception as e:
#     print(f"Warning: An unexpected error occurred during core model rebuild: {e}")
# --- End Consolidated Model Rebuild ---

# Rebuild models to resolve forward references and dependencies
# PlanningSystem.model_rebuild() # Commented out - moved above
# ExecutionSystem.model_rebuild() # Commented out - moved above

class AgentState(BaseModel):
    """Current state of the Jarvis agent"""
    is_active: bool = True
    current_objective_id: Optional[str] = None
    current_plan_id: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    last_maintenance: Optional[datetime] = Field(default_factory=datetime.now)

class JarvisAgent(BaseModel):
    """Main agentic implementation of Jarvis"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    memory_system: 'UnifiedMemorySystem'
    logger: logging.Logger
    planning_system: 'PlanningSystem'
    execution_system: 'ExecutionSystem'
    state: AgentState = Field(default_factory=AgentState) # Initialize with default factory
    llm: Optional['LLMClient'] = None # Optional, as it can be initialized later
    skill_registry: 'SkillRegistry' # Now just a required field
    
    def __init__(self, memory_system: Optional['UnifiedMemorySystem'] = None, llm: Optional['LLMClient'] = None, **data):
        # Prepare data dictionary for Pydantic initialization
        init_data = data.copy()

        # 1. Initialize Logger (required)
        # --- Logger Initialization ---
        if 'logger' not in init_data:
            # init_data['logger'] = logging.getLogger(__name__) # Use module name
            # # Basic check: assume configured if it has handlers or level is not NOTSET
            # if not init_data['logger'].hasHandlers() and init_data['logger'].level == logging.NOTSET:
            #      init_data['logger'].setLevel(logging.INFO) # Default level if needed
            #      print(f"WARNING: Logger '{__name__}' was not configured. Applying default level INFO.")
            init_data['logger'] = setup_logger(__name__) # Reverted to setup_logger
        # --- End Logger Initialization ---

        # 2. Initialize Memory System (required)
        # Use provided instance or create default
        if memory_system:
            init_data['memory_system'] = memory_system
        elif 'memory_system' not in init_data: # Only create default if not passed via data either
            from .memory.unified_memory import UnifiedMemorySystem
            init_data['memory_system'] = UnifiedMemorySystem()
        
        # 3. Initialize LLM Client (optional field, but needed for other components)
        # Use provided instance or create default
        llm_instance = llm # Use passed llm argument first
        if not llm_instance and 'llm' not in init_data:
            from .llm import LLMClient
            llm_instance = LLMClient() # Create default if not provided
        elif 'llm' in init_data:
             llm_instance = init_data['llm'] # Use instance from data if provided
        init_data['llm'] = llm_instance # Add to init_data for Pydantic field

        # 4. Initialize Skill Registry (required)
        # Create default instance if not provided, then discover skills
        if 'skill_registry' in init_data:
            skill_registry_instance = init_data['skill_registry']
            # Assume already discovered if passed explicitly, or discover here?
            # Let's assume explicit pass means it's ready.
        else:
            # Create a new instance if not provided
            skill_registry_instance = SkillRegistry()
            # Discover skills on the new instance
            skill_registry_instance.discover_skills() 
            init_data['skill_registry'] = skill_registry_instance # Add the created instance to init_data

        # Ensure components needed for Planning/Execution are available
        current_memory_system = init_data['memory_system']
        # Use the instance we just ensured exists
        current_skill_registry = init_data['skill_registry'] 

        # 5. Initialize Planning System (required)
        if 'planning_system' not in init_data:
            init_data['planning_system'] = PlanningSystem(
                unified_memory=current_memory_system, 
                llm_client=llm_instance
            )

        # 6. Initialize Execution System (required)
        if 'execution_system' not in init_data:
            # Make sure the correct skill_registry instance is passed
            init_data['execution_system'] = ExecutionSystem(
                unified_memory=current_memory_system,
                llm_client=llm_instance,
                skill_registry=current_skill_registry, # Pass the managed instance
                planning_system=init_data['planning_system']
            )

        # 7. Initialize State (required, uses default factory)
        if 'state' not in init_data:
            init_data['state'] = AgentState()

        # Now call super().__init__ with all required fields prepared in init_data
        super().__init__(**init_data)
        
        # Post-initialization actions (fields are now available via self)
        self.logger.info("JarvisAgent core components initialized by Pydantic.")

        # Load previous state if available
        self._load_state()
        self.logger.info("JarvisAgent initialization complete.")
    
    def _load_state(self) -> None:
        """Load agent state from memory"""
        try:
            # Original loading logic (now re-enabled):
            self.logger.info("Loading agent state from long-term memory...")
            # Query LTM for the state
            knowledge_id = f"agent_state"
            state_data = self.memory_system.long_term.retrieve_knowledge(
                knowledge_id,
                n_results=1
            )
            
            if state_data and state_data[0]["metadata"]:
                self.logger.debug(f"Found previous state data: {state_data[0]['metadata']}")
                # Convert string values back to appropriate types
                metadata = state_data[0]["metadata"]
                
                # Parse string-encoded values
                if "context" in metadata and isinstance(metadata["context"], str):
                    if metadata["context"] == "{}":
                        metadata["context"] = {}
                    else:
                        try:
                            metadata["context"] = json.loads(metadata["context"])
                        except Exception as json_err:
                             self.logger.warning(f"Could not parse context JSON: {json_err}. Resetting context.")
                             metadata["context"] = {}
                
                # Convert empty strings back to None (only for specific known optional fields if needed)
                # Example: if self.state.current_plan_id can be None
                if "current_plan_id" in metadata and metadata["current_plan_id"] == "":
                     metadata["current_plan_id"] = None
                if "current_objective_id" in metadata and metadata["current_objective_id"] == "":
                     metadata["current_objective_id"] = None
                # Add others as necessary
                
                # Convert bools stored as strings
                if "is_active" in metadata and isinstance(metadata["is_active"], str):
                    metadata["is_active"] = metadata["is_active"].lower() == 'true'
                
                # Convert ints stored as strings
                if "error_count" in metadata and isinstance(metadata["error_count"], str):
                     try:
                         metadata["error_count"] = int(metadata["error_count"])
                     except ValueError:
                         self.logger.warning("Could not parse error_count from state. Resetting to 0.")
                         metadata["error_count"] = 0

                self.state = AgentState(**metadata)
            else:
                 self.logger.info("No previous state found in long-term memory. Using default state.")
                 self.state = AgentState()

        except Exception as e:
            self.logger.error(f"Error during state loading: {e}", exc_info=True)
            # Ensure default state in case of any errors during loading
            self.logger.warning("Using default state due to loading error.")
            self.state = AgentState()
    
    def _save_state(self) -> None:
        """Save current agent state to memory"""
        # Convert state dict to ensure all values are scalar
        state_dict = self.state.model_dump()
        
        # Make sure all values are serializable primitives for ChromaDB
        # None values are not allowed
        for key, value in list(state_dict.items()):
            if value is None:
                state_dict[key] = ""
            elif not isinstance(value, (str, int, float, bool)):
                state_dict[key] = str(value)
        
        self.memory_system.long_term.add_knowledge(
            "agent_state",
            "Current state of Jarvis agent",
            metadata=state_dict
        )
    
    def process_input(self, text: str) -> str:
        """Process user input and return response"""
        try:
            # Add to short-term memory
            self.memory_system.short_term.add_interaction("user", text)
            
            # Check if we need to reset context due to errors
            if self.state.error_count > 3:
                self.reset_context()
                self.state.error_count = 0
                self._save_state()
            
            # Check for self-reflection commands
            if self._is_reflection_command(text):
                return self._handle_reflection_command(text)
            
            # Check for objective-related commands
            if self._is_objective_command(text):
                return self._handle_objective_command(text)
            
            # Check for plan-related commands
            if self._is_plan_command(text):
                return self._handle_plan_command(text)
            
            # Check for status commands
            if self._is_status_command(text):
                return self._handle_status_command(text)
            
            # Check for feedback command
            if text.lower().startswith("feedback"):
                 return self._handle_feedback_command(text)
            
            # Default to task execution
            return self._execute_task(text)
        
        except Exception as e:
            error_time = datetime.now()
            error_type_name = type(e).__name__
            error_message = str(e)
            self.logger.error(f"Error processing input '{text}': {error_type_name}: {error_message}", exc_info=True)
            self.state.error_count += 1
            last_error_str = f"[{error_time.isoformat()}] {error_type_name}: {error_message}"
            self.state.last_error = last_error_str
            
            # Attempt LLM-based diagnosis
            diagnosis = "Diagnosis unavailable."
            suggestion = "Please try again or rephrase your request."
            if self.llm:
                try:
                    # Use assemble_context for better context gathering
                    context = self.memory_system.assemble_context(query=f"Error analysis for input: {text}, Error: {last_error_str}")
                    
                    system_prompt = """
                    You are an error analysis assistant for the Jarvis agent.
                    An error occurred while processing user input or executing a task.
                    Analyze the error type, message, user input, and provided context.
                    Provide a brief diagnosis of the likely cause (e.g., planning failure, skill error, memory issue, external API problem, LLM hallucination, invalid input).
                    Suggest a concrete, actionable recovery strategy for the *agent* or the *user*. Examples:
                    - Agent Actions: retry, retry_with_different_params, use_fallback_skill <skill_name>, replan_objective, reset_context
                    - User Actions: rephrase_input, check_api_key, provide_more_info, wait_and_retry
                    - Other: report_bug, none
                    Format as: "Diagnosis: [Your diagnosis]. Suggestion: [recovery_action_keyword] | [Explanation/Details for user/agent]"
                    Example: "Diagnosis: Skill parsing failed due to ambiguous input. Suggestion: rephrase_input | Could you please rephrase the request more clearly?"
                    Example: "Diagnosis: Web browse timed out. Suggestion: retry | Trying to access the URL again."
                    Example: "Diagnosis: Planning failed for complex objective. Suggestion: replan_objective | Let me try breaking down that objective again."
                    """
                    prompt = f"""
                    Error Type: {error_type_name}
                    Error Message: {error_message}
                    User Input: {text}
                    Context:
                    {context}
                    
                    Analyze the error and provide diagnosis and suggestion in the specified format.
                    """
                    raw_diagnosis = self.llm.process_with_llm(prompt, system_prompt, temperature=0.3, max_tokens=200).strip()
                    self.logger.info(f"LLM Error Diagnosis: {raw_diagnosis}")
                    
                    # Parse diagnosis and suggestion
                    diagnosis_part = raw_diagnosis.split("Suggestion:")[0].replace("Diagnosis:", "").strip()
                    suggestion_part = raw_diagnosis.split("Suggestion:")[-1].strip()
                    suggestion_keyword = suggestion_part.split("|")[0].strip().lower()
                    suggestion_details = suggestion_part.split("|")[-1].strip() if "|" in suggestion_part else "Please try again."
                    
                    diagnosis = diagnosis_part # Store the parsed diagnosis
                    suggestion = suggestion_details # Store the user-facing message or agent action explanation
                    
                    # Store detailed log
                    self.memory_system.add_memory(
                        content=f"Error processing input '{text}': {last_error_str}",
                        memory_type="long_term",
                        metadata={
                            "type": "error_log",
                            "timestamp": error_time.isoformat(),
                            "error_type": error_type_name,
                            "diagnosis": diagnosis,
                            "suggestion_keyword": suggestion_keyword,
                            "suggestion_details": suggestion_details
                        },
                        importance=0.7 # Errors are fairly important
                    )

                    # --- Attempt Automatic Recovery based on keyword ---
                    if suggestion_keyword == "retry":
                        self.logger.info("Attempting automatic retry based on LLM suggestion.")
                        self.state.error_count = max(0, self.state.error_count - 1) # Decrement error count for retry attempt
                        self._save_state()
                        # Re-call process_input. Be careful of infinite loops!
                        # Add a retry counter or mechanism if needed.
                        # return self.process_input(text) # Potential loop risk!
                        return f"{suggestion} (Retrying...)" # Safer: Inform user and agent can retry internally next cycle
                    elif suggestion_keyword == "reset_context":
                         self.logger.info("Attempting context reset based on LLM suggestion.")
                         self.reset_context() # Calls consolidate and clears STM
                         self.state.error_count = 0 # Reset error count after context reset
                         self._save_state()
                         return f"{suggestion} I've reset my short-term context. Please try your request again."
                    elif suggestion_keyword == "replan_objective" and self.state.current_objective_id:
                         self.logger.info(f"Attempting replan for objective {self.state.current_objective_id} based on LLM suggestion.")
                         try:
                              # Invalidate current plan if one exists
                              if self.state.current_plan_id:
                                   self.planning_system.update_plan_status(self.state.current_plan_id, "failed", reason="Replanning triggered by error")
                              # Trigger replanning
                              plan = self.planning_system.create_plan(self.state.current_objective_id)
                              self.state.current_plan_id = plan.plan_id
                              self.state.error_count = 0 # Reset error count after replan
                              self._save_state()
                              return f"{suggestion} I've created a new plan for the objective."
                         except Exception as replan_err:
                              self.logger.error(f"Replanning failed: {replan_err}")
                              suggestion = f"I tried to replan, but encountered another error: {replan_err}. Please review the objective or try something else."
                    # Add other recovery actions here (e.g., use_fallback_skill)

                except Exception as diag_err:
                    self.logger.error(f"LLM diagnosis or recovery attempt failed: {diag_err}")
                    # Fall back to default suggestion if diagnosis fails
                    suggestion = "Please try again or rephrase your request."
            
            # Save state after potential modifications
            self._save_state()
            # Return the (potentially updated) user-facing suggestion
            return f"I encountered an error ({error_type_name}). {suggestion}"
    
    def _is_reflection_command(self, text: str) -> bool:
        """Check if the input is a reflection-related command"""
        reflection_keywords = [
            "reflect", "think", "analyze", "review", "assess",
            "self", "evaluation", "improve", "learning", "progress"
        ]
        return any(keyword in text.lower() for keyword in reflection_keywords)
    
    def _is_objective_command(self, text: str) -> bool:
        """Check if the input is an objective-related command"""
        objective_keywords = [
            "objective", "goal", "task", "mission", "purpose",
            "set", "create", "new", "start", "begin"
        ]
        return any(keyword in text.lower() for keyword in objective_keywords)
    
    def _is_plan_command(self, text: str) -> bool:
        """Check if the input is a plan-related command"""
        plan_keywords = [
            "plan", "strategy", "steps", "process", "method",
            "show", "display", "list", "view", "status"
        ]
        return any(keyword in text.lower() for keyword in plan_keywords)
    
    def _is_status_command(self, text: str) -> bool:
        """Check if the input is a status-related command"""
        status_keywords = [
            "status", "progress", "state", "condition", "situation",
            "how", "what", "where", "when", "why"
        ]
        return any(keyword in text.lower() for keyword in status_keywords)
    
    def _handle_reflection_command(self, text: str) -> str:
        """Handle reflection-related commands"""
        # Perform self-reflection on progress and abilities
        recent_interactions = self.memory_system.short_term.get_recent_interactions(10)
        objectives = self.memory_system.medium_term.search_objectives("")
        
        reflection_text = "Based on my recent interactions and progress:\n"
        
        # 1. Analyze recent interactions for patterns
        user_queries = [i.text for i in recent_interactions if i.speaker == "user"]
        if user_queries:
            reflection_text += "\n- Recent user queries seem focused on: "
            if any("code" in q.lower() for q in user_queries):
                reflection_text += "coding tasks, "
            if any("organize" in q.lower() for q in user_queries):
                reflection_text += "organization tasks, "
            if any("find" in q.lower() or "search" in q.lower() for q in user_queries):
                reflection_text += "information retrieval, "
            reflection_text = reflection_text.rstrip(", ") + "\n"
        
        # 2. Review objectives and progress
        active_objectives = [o for o in objectives if o["metadata"]["status"] == "active"]
        if active_objectives:
            reflection_text += f"\n- I have {len(active_objectives)} active objectives\n"
            for obj in active_objectives[:3]:  # Show top 3
                reflection_text += f"  - {obj['description']}\n"
        
        # 3. Consider creating a new objective based on patterns
        if "create" in text.lower() or "new" in text.lower():
            new_objective = self._create_autonomous_objective(user_queries)
            if new_objective:
                reflection_text += f"\n- I've created a new objective: {new_objective}\n"
        
        return reflection_text
    
    def _create_autonomous_objective(self, recent_queries: List[str]) -> Optional[str]:
        """Create an autonomous objective based on patterns, memory, and agent state."""
        try:
            # 1. Gather Context
            self.logger.info("Attempting to create autonomous objective...")
            
            # Memory Reflection (using the existing reflection method)
            reflection_query = "overall agent performance and user needs"
            if recent_queries:
                reflection_query = f"user needs based on recent queries like: {recent_queries[0]}"

            # <<< ADDED CALL TO assemble_context >>>
            assembled_context = self.memory_system.assemble_context(query=reflection_query)
            # <<< Use assembled_context in the prompt below, replacing the manually built 'context' variable >>>

            # 2. Define LLM Prompt for Objective Generation
            system_prompt = """
            You are Jarvis, an advanced agentic assistant. Your task is to analyze the provided context 
            and propose a specific, actionable, and relevant **new** objective for the agent to pursue autonomously.

            Context includes recent interactions and relevant memories/insights.
            Consider:
            - Implicit or explicit user needs.
            - The agent's current state, errors, and active objective (if any) within the context.
            - Insights and patterns derived from memory within the context.
            - The overall goal of being a helpful, proactive assistant.

            Requirements for the objective:
            - Must be a *new* objective, not a restatement of the current one if provided in context.
            - Must be actionable and achievable by an AI assistant.
            - Should ideally address a gap, recurring theme, or opportunity identified in the context.
            - Be concise and clearly worded.

            Analyze the context and generate **one single best objective** based on your analysis.
            Return ONLY the objective text, without any explanation, preamble, or markdown formatting.
            If no suitable new objective can be identified from the context, return the exact string "NULL".
            """

            prompt = f"""
            **Context Analysis for New Objective Generation**

            {assembled_context}
            
            **Task:**
            Based *only* on the context provided above, propose one single, new, actionable, and relevant objective for the agent.
            Remember to return ONLY the objective text or the string "NULL".
            """

            # 3. Get Objective from LLM
            objective_text = None
            if self.llm:
                try:
                    raw_response = self.llm.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.6, # Slightly lower temperature for more focused objective
                        max_tokens=150 # Objectives should be relatively concise
                    ).strip()

                    if raw_response and raw_response != "NULL":
                        objective_text = raw_response
                        self.logger.info(f"LLM proposed new objective: {objective_text}")
                    else:
                        self.logger.info("LLM indicated no suitable new objective (NULL response).")

                except Exception as e:
                    self.logger.error(f"Error generating objective with LLM: {e}")
            else:
                 self.logger.warning("LLM client not available for objective generation.")

            # 4. Create Objective (No fallback needed if LLM reliably returns NULL)
            if objective_text:
                 # Check if it's substantially different from the current one
                is_new = True
                if self.state.current_objective_id:
                    current_obj_details = self.memory_system.medium_term.get_objective(self.state.current_objective_id)
                    if current_obj_details and objective_text.lower() == current_obj_details.description.lower():
                        is_new = False
                        self.logger.info("Proposed objective is the same as the current one. Skipping creation.")

                if is_new:
                    self.logger.info(f"Creating and planning for new objective: {objective_text}")
                    objective_id = self.memory_system.medium_term.create_objective(objective_text)
                    self.state.current_objective_id = objective_id
                    self.state.current_plan_id = None # Reset plan ID for the new objective
                    self._save_state()

                    # Create a plan for this objective
                    try:
                        plan = self.planning_system.create_plan(objective_id)
                        self.state.current_plan_id = plan.plan_id
                        self._save_state()
                        self.logger.info(f"Successfully created plan {plan.plan_id} for objective {objective_id}")
                        return objective_text
                    except Exception as plan_err:
                        self.logger.error(f"Error creating plan for new objective {objective_id}: {plan_err}")
                        # Objective created, but planning failed. Agent state reflects this.
                        return f"Objective '{objective_text}' created, but planning failed: {plan_err}" # Return text with error

            # Return None if no new objective was created
            return None

        except Exception as e:
            self.logger.error(f"Error creating autonomous objective: {e}", exc_info=True)
            return None
    
    def _handle_objective_command(self, text: str) -> str:
        """Handle objective-related commands"""
        try:
            if "set" in text.lower() or "create" in text.lower():
                # Create new objective
                objective_id = self.memory_system.medium_term.create_objective(text)
                self.state.current_objective_id = objective_id
                self._save_state()
                
                # Create plan for objective
                try:
                    plan = self.planning_system.create_plan(objective_id)
                    self.state.current_plan_id = plan.plan_id
                    self._save_state()
                    
                    return f"I've created a new objective and plan. Let's start working on it."
                except Exception as e:
                    self.logger.error(f"Error creating plan for objective: {e}")
                    return f"I've created a new objective, but couldn't create a plan: {str(e)}"
            
            elif "list" in text.lower() or "show" in text.lower():
                # List objectives
                objectives = self.memory_system.medium_term.search_objectives("")
                if not objectives:
                    return "You don't have any active objectives."
                
                response = "Current objectives:\n"
                for obj in objectives:
                    response += f"- {obj['description']} (Status: {obj['metadata']['status']})\n"
                return response
            
            else:
                return "I'm not sure what you want to do with objectives. Try 'set objective' or 'list objectives'."
        except Exception as e:
            self.logger.error(f"Error handling objective command: {e}")
            return f"I encountered an error with the objective command: {str(e)}"
    
    def _handle_plan_command(self, text: str) -> str:
        """Handle plan-related commands"""
        if not self.state.current_plan_id:
            return "No active plan. Set an objective first."
        
        if "show" in text.lower() or "display" in text.lower():
            # Show current plan status
            status = self.planning_system.get_plan_status(self.state.current_plan_id)
            return self._format_plan_status(status)
        
        elif "next" in text.lower():
            # Get next task
            task = self.planning_system.get_next_task(self.state.current_plan_id)
            if not task:
                return "No more tasks in the current plan."
            
            # Execute task
            result = self.execution_system.execute_task(task)
            self.planning_system.update_task_status(
                self.state.current_plan_id,
                task.task_id,
                "completed" if result.success else "failed",
                result.error
            )
            
            return f"Executed task: {task.description}\nResult: {result.output}"
        
        else:
            return "I'm not sure what you want to do with the plan. Try 'show plan' or 'next task'."
    
    def _handle_status_command(self, text: str) -> str:
        """Handle status-related commands"""
        if "status" in text.lower() or "state" in text.lower():
            # Get overall status
            status = {
                "active": self.state.is_active,
                "current_objective": self.state.current_objective_id,
                "current_plan": self.state.current_plan_id,
                "error_count": self.state.error_count,
                "last_error": self.state.last_error
            }
            
            # Get plan status if active
            if self.state.current_plan_id:
                plan_status = self.planning_system.get_plan_status(self.state.current_plan_id)
                status["plan"] = plan_status
            
            return self._format_status(status)
        
        else:
            return "I'm not sure what status information you want. Try 'show status'."
    
    def _execute_task(self, text: str) -> str:
        """
        Handles executing the next task of the current plan OR processing general user input.
        Prioritizes executing the current plan if one is active.
        """
        try:
            # --- General Input Processing (No active plan or plan finished) --- 
            self.logger.info(f"No active plan or current plan step completed. Processing input: '{text}'")
            
            # <<< Attempt skill execution FIRST >>>
            try:
                # Create a temporary task from the user input
                # Use a simple ID scheme for these one-off tasks
                temp_task_id = f"user_cmd_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                temp_task = Task(task_id=temp_task_id, description=text)
                
                self.logger.info(f"Attempting to execute direct command as task: {temp_task.description}")
                result = self.execution_system.execute_task(temp_task)
                
                if result.success:
                    self.logger.info(f"Direct command execution successful. Task: '{text}'")
                    # Add interaction to memory (using skill output)
                    response = str(result.output) if result.output else result.message # Prefer output, fallback to message
                    self.memory_system.add_memory(content=response, memory_type="short_term", metadata={"speaker": "assistant"})
                    return response
                else:
                    # Execution failed or no skill matched and LLM fallback in execution system didn't succeed
                    self.logger.warning(f"Direct command execution failed or no skill matched. Falling back to conversational LLM. Error (if any): {result.error}")
                    # Fall through to the conversational LLM below
            
            except Exception as exec_err:
                 self.logger.error(f"Error attempting direct command execution for '{text}': {exec_err}", exc_info=True)
                 # Fall through to conversational LLM on error

            # <<< Fallback to conversational LLM if skill execution didn't succeed >>>
            self.logger.info(f"Falling back to general conversational response for input: '{text}'")
            if self.llm:
                # Use the more robust assemble_context method
                assembled_context = self.memory_system.assemble_context(
                    query=f"User request: {text}", 
                    max_tokens=3000 # Adjust token limit as needed
                )

                # Use LLM to generate a response
                try:
                    system_prompt = """
                    You are Jarvis, an agentic AI assistant.
                    Generate a helpful, accurate, and concise response to the user's input based on the provided context.
                    Use the conversation history and relevant memories to maintain continuity and provide informed answers.
                    Avoid referring to unrelated past events unless directly relevant to the current input.
                    If the user input seems like a new task or objective, suggest creating one explicitly.
                    If the input is a simple command that can be executed (e.g., based on available skills), confirm understanding or ask for clarification if needed, rather than just chatting.
                    """

                    prompt = f"""
                    {assembled_context}

                    User Input: {text}

                    Jarvis Response:
                    """

                    response = self.llm.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.7
                    )

                    # Add the interaction to memory
                    self.memory_system.add_memory(content=response, memory_type="short_term", metadata={"speaker": "assistant"})

                    return response
                except Exception as e:
                    self.logger.error(f"Error generating response with LLM: {e}")
                    # Fallback response if LLM fails
                    return f"I had trouble processing that request with my language model ({type(e).__name__}). Could you please rephrase or try again?"

            # Fallback if LLM is not available
            return f"I received your input: '{text}'. My language model is currently unavailable. Please set an objective or check status."

        except Exception as e:
            # Catch errors during the task execution/response generation flow
            error_time = datetime.now()
            self.logger.error(f"Error during task execution or response generation for input '{text}': {e}", exc_info=True)
            self.state.error_count += 1
            last_error_str = f"[{error_time.isoformat()}] {type(e).__name__}: {str(e)}"
            self.state.last_error = last_error_str
            self._save_state()
            # Store error in memory
            self.memory_system.add_memory(
                content=f"Agent error processing input '{text}': {last_error_str}",
                memory_type="long_term",
                metadata={"type": "error_log", "timestamp": error_time.isoformat(), "error_type": type(e).__name__},
                importance=0.7
            )
            return f"I encountered an internal error ({type(e).__name__}) while processing your request. I've logged the issue. Please try again."
    
    def _format_plan_status(self, status: Dict[str, Any]) -> str:
        """Format plan status for display"""
        response = f"Plan Status:\n"
        response += f"- ID: {status['plan_id']}\n"
        response += f"- Objective: {status['objective_id']}\n"
        response += f"- Status: {status['status']}\n"
        response += f"- Created: {status['created_at']}\n"
        if status['completed_at']:
            response += f"- Completed: {status['completed_at']}\n"
        
        response += "\nTasks:\n"
        for status, count in status['tasks'].items():
            response += f"- {status}: {count}\n"
        
        return response
    
    def _format_status(self, status: Dict[str, Any]) -> str:
        """Format overall status for display"""
        response = "Jarvis Status:\n"
        response += f"- Active: {'Yes' if status['active'] else 'No'}\n"
        response += f"- Current Objective: {status['current_objective'] or 'None'}\n"
        response += f"- Current Plan: {status['current_plan'] or 'None'}\n"
        response += f"- Error Count: {status['error_count']}\n"
        if status['last_error']:
            response += f"- Last Error: {status['last_error']}\n"
        
        if 'plan' in status:
            response += "\nCurrent Plan Status:\n"
            response += self._format_plan_status(status['plan'])
        
        return response
    
    def reset_context(self) -> None:
        """Reset the context window, summarizing the state before clearing."""
        self.logger.warning(f"Resetting context due to error threshold ({self.state.error_count} errors). Last error: {self.state.last_error}")

        summary = "Context reset due to repeated errors."
        if self.llm:
            try:
                # 1. Gather state before reset
                state_before_reset = "State before context reset:\n"
                state_before_reset += f"- Objective ID: {self.state.current_objective_id}\n"
                state_before_reset += f"- Plan ID: {self.state.current_plan_id}\n"
                state_before_reset += f"- Error Count: {self.state.error_count}\n"
                state_before_reset += f"- Last Error: {self.state.last_error}\n"

                # Get last few interactions
                interactions = self.memory_system.short_term.get_recent_interactions(5)
                interaction_log = "\n".join([f"- [{i.timestamp.strftime('%H:%M')}] {i.speaker}: {i.text[:100]}..." for i in interactions])
                state_before_reset += f"\nRecent Interactions:\n{interaction_log}\n"

                # 2. Ask LLM to summarize
                system_prompt = """
                You are an assistant analyzing the agent's state just before a context reset caused by repeated errors.
                Summarize the situation concisely, focusing on the active objective/plan and the nature of the recent errors.
                This summary will be placed in the short-term memory after the reset.
                Keep the summary brief (1-2 sentences).
                """
                prompt = f"""
                {state_before_reset}
                
                Provide a brief summary of the situation before the context reset.
                """

                summary = self.llm.process_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5,
                    max_tokens=100
                ).strip()
                self.logger.info(f"Generated context reset summary: {summary}")

            except Exception as e:
                self.logger.error(f"Failed to generate context reset summary with LLM: {e}")
                summary = f"Context reset due to {self.state.error_count} errors. Last error: {self.state.last_error}" # Fallback summary

        # 3. Consolidate important memories (optional, depending on strategy)
        # self.memory_system.consolidate_memories() 

        # 4. Reset short-term memory
        self.memory_system.short_term.reset()

        # 5. Add reset note and summary to the now-empty short-term memory
        self.memory_system.short_term.add_interaction(
            "system",
            f"CONTEXT RESET. {summary}"
        )

        # Also reset agent error count state
        self.state.error_count = 0
        self.state.last_error = None
        self._save_state() 

        return f"Context has been reset. {summary}"
    
    def periodic_actions(self):
        """Perform periodic maintenance actions"""
        try:
            # Check if we should create a new autonomous objective
            if not self.state.current_objective_id or self._should_create_new_objective():
                self._create_periodic_objective()
            
            # Consolidate memories
            self.memory_system.consolidate_memories()
            
            # Reflect on recent interactions to improve future responses
            if self.llm:
                recent_interactions = self.memory_system.short_term.get_recent_interactions(10)
                if recent_interactions:
                    query = "What patterns and preferences can be identified from recent user interactions?"
                    try:
                        reflection = self.memory_system.reflect_on_memories(query, self.llm)
                        if reflection and "error" not in reflection:
                            # Store insights in medium-term memory
                            insights = reflection.get("insights", [])
                            if insights:
                                insight_text = "\n".join(insights)
                                self.memory_system.medium_term.add_progress(
                                    f"Interaction insights: {insight_text}",
                                    metadata={"type": "reflection", "source": "periodic_action"}
                                )
                    except Exception as e:
                        self.logger.error(f"Error during periodic reflection: {e}")
            
            # Update state
            self.state.last_maintenance = datetime.now()
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Error in periodic actions: {e}")
    
    def _should_create_new_objective(self) -> bool:
        """Determine if we should create a new autonomous objective"""
        # Check if current objective is completed
        if self.state.current_objective_id:
            objective = self.memory_system.medium_term.get_objective(self.state.current_objective_id)
            if objective and objective["metadata"]["status"] == "completed":
                return True
        
        # Check if we haven't created an objective in a while
        last_objective_time = self.state.context.get("last_objective_time")
        if last_objective_time:
            try:
                last_time = datetime.fromisoformat(last_objective_time)
                # Create a new objective if it's been more than 24 hours
                if (datetime.now() - last_time).total_seconds() > 86400:
                    return True
            except:
                return True
        else:
            return True
        
        return False
    
    def _create_periodic_objective(self):
        """Create a new autonomous objective periodically"""
        # Get recent interactions
        recent_interactions = self.memory_system.short_term.get_recent_interactions(20)
        if not recent_interactions:
            return
        
        # Get user queries from interactions
        user_queries = [i.text for i in recent_interactions if i.speaker == "user"]
        
        # Create a new objective
        new_objective = self._create_autonomous_objective(user_queries)
        
        if new_objective:
            # Update last objective time
            self.state.context["last_objective_time"] = datetime.now().isoformat()
            self._save_state()
            
            self.logger.info(f"Created new autonomous objective: {new_objective}")
            
            # Add to memory
            self.memory_system.medium_term.add_progress(
                f"Created autonomous objective: {new_objective}",
                metadata={"type": "autonomous_objective"}
            )

    # --- Periodic Review Method ---
    def _review_and_refine_objectives(self) -> None:
        """Periodically review objectives using LLM for relevance and refinement."""
        self.logger.info("Performing periodic objective review and refinement...")
        try:
            # Get active objectives
            active_objectives = [
                obj for obj in self.memory_system.medium_term.search_objectives("", n_results=10)
                if obj["metadata"].get("status") == "active"
            ]
            if not active_objectives:
                self.logger.info("No active objectives to review.")
                return

            objective_list = "\n".join([f"- ID: {o['id']}, Desc: {o['description']}" for o in active_objectives])

            # Get recent context
            context = self.memory_system.assemble_context(
                query="Review active objectives based on recent activity",
                max_tokens=2000
            )

            if not self.llm:
                self.logger.warning("LLM not available for objective review.")
                return

            system_prompt = """
            You are an objective review assistant for the Jarvis agent.
            Analyze the list of active objectives in the context of the agent's recent activity and memory.
            Suggest refinements, completion flags, or mark objectives as potentially irrelevant.
            Output suggestions as a JSON list of objects, each with 'objective_id' and 'suggestion' (e.g., 'refine: [new description]', 'mark_complete', 'mark_irrelevant', 'keep').
            Example: `[{"objective_id": "obj_123", "suggestion": "refine: Focus search on Python libraries"}, {"objective_id": "obj_456", "suggestion": "mark_complete"}]`
            Output ONLY the JSON list.
            """
            prompt = f"""
            Recent Context & Memory:
            {context}

            Active Objectives:
            {objective_list}

            Review the objectives based on the context and suggest actions in the specified JSON format.
            """

            raw_response = self.llm.process_with_llm(prompt, system_prompt, temperature=0.4, max_tokens=500)

            # Process suggestions
            try:
                # Clean potential markdown
                if raw_response.strip().startswith("```json"):
                    raw_response = raw_response.strip()[7:-3].strip()
                elif raw_response.strip().startswith("```"):
                    raw_response = raw_response.strip()[3:-3].strip()

                suggestions = json.loads(raw_response)
                if not isinstance(suggestions, list):
                    raise ValueError("LLM response is not a list.")

                for sugg in suggestions:
                    obj_id = sugg.get("objective_id")
                    action = sugg.get("suggestion")
                    if not obj_id or not action:
                        continue

                    # Find the actual objective object to potentially update
                    objective_to_update = self.memory_system.medium_term.get_objective(obj_id)
                    if not objective_to_update:
                        self.logger.warning(f"Objective {obj_id} mentioned in review suggestion not found.")
                        continue

                    self.logger.info(f"Objective Review Suggestion for {obj_id}: {action}")
                    if action == "mark_complete":
                        self.memory_system.medium_term.update_objective(obj_id, status="completed")
                        # If it was the current objective, clear agent state
                        if self.state.current_objective_id == obj_id:
                            self.state.current_objective_id = None
                            self.state.current_plan_id = None
                            self._save_state()
                    elif action == "mark_irrelevant":
                        # Consider a different status like 'archived' or 'irrelevant'?
                        self.memory_system.medium_term.update_objective(obj_id, status="failed", reason="Marked irrelevant by review")
                        if self.state.current_objective_id == obj_id:
                            self.state.current_objective_id = None
                            self.state.current_plan_id = None
                            self._save_state()
                    elif action.startswith("refine:"):
                        new_description = action.split("refine:", 1)[1].strip()
                        if new_description:
                            self.memory_system.medium_term.update_objective(obj_id, description=new_description)
                            # If it is the current objective, consider replanning?
                            if self.state.current_objective_id == obj_id:
                                self.logger.info(f"Objective {obj_id} refined. Triggering replan.")
                                self.state.current_plan_id = None # Invalidate old plan
                                try:
                                    new_plan = self.planning_system.create_plan(obj_id)
                                    self.state.current_plan_id = new_plan.plan_id
                                except Exception as replan_err:
                                    self.logger.error(f"Failed to replan after refining objective {obj_id}: {replan_err}")
                                self._save_state()
            except (json.JSONDecodeError, ValueError) as json_err:
                self.logger.error(f"Failed to parse objective review suggestions: {json_err}. Raw: {raw_response}")

        except Exception as e:
            self.logger.error(f"Error during periodic objective review: {e}", exc_info=True)
    
    # --- Self-Assessment Method ---
    def _perform_self_assessment(self) -> None:
        """Periodically perform self-assessment based on recent performance and memory."""
        self.logger.info("Performing periodic self-assessment...")
        try:
            # Gather context for assessment
            context = self.memory_system.assemble_context(
                query="Self-assessment based on recent performance, errors, and feedback",
                max_tokens=3000
            )
            if not self.llm:
                self.logger.warning("LLM not available for self-assessment.")
                return

            system_prompt = """
            You are a self-assessment module for the Jarvis AI agent.
            Analyze the provided context (recent interactions, errors, feedback, memory snippets).
            Identify key strengths, weaknesses, and areas for improvement in the agent's performance.
            Suggest 1-3 concrete improvement goals or learning objectives.
            Output the assessment as a JSON object with keys: "strengths" (list[str]), "weaknesses" (list[str]), "improvement_goals" (list[str]).
            Example: `{"strengths": ["Good at web searches"], "weaknesses": ["Struggles with complex planning"], "improvement_goals": ["Improve plan decomposition for multi-step tasks"]}`
            Output ONLY the JSON object.
            """
            prompt = f"""
            Context for Self-Assessment:
            {context}

            Perform self-assessment and output the result as JSON.
            """
            raw_response = self.llm.process_with_llm(prompt, system_prompt, temperature=0.5, max_tokens=600)

            # Process and store assessment
            try:
                # Clean potential markdown
                if raw_response.strip().startswith("```json"):
                    raw_response = raw_response.strip()[7:-3].strip()
                elif raw_response.strip().startswith("```"):
                    raw_response = raw_response.strip()[3:-3].strip()

                assessment_data = json.loads(raw_response)
                if not isinstance(assessment_data, dict) or not all(k in assessment_data for k in ["strengths", "weaknesses", "improvement_goals"]):
                     raise ValueError("Assessment JSON missing required keys.")

                self.logger.info(f"Self-Assessment Results: Strengths: {assessment_data['strengths']}, Weaknesses: {assessment_data['weaknesses']}, Goals: {assessment_data['improvement_goals']}")
                # Store the assessment in long-term memory
                self.memory_system.add_memory(
                    content="Periodic self-assessment performed.",
                    memory_type="long_term",
                    metadata={
                        "type": "self_assessment",
                        "timestamp": datetime.now().isoformat(),
                        "assessment": assessment_data # Store the full JSON
                    },
                    importance=0.6
                )
            except (json.JSONDecodeError, ValueError) as json_err:
                self.logger.error(f"Failed to parse self-assessment response: {json_err}. Raw: {raw_response}")

        except Exception as e:
            self.logger.error(f"Error during self-assessment: {e}", exc_info=True)
    
    # --- Feedback Handling ---
    def _handle_feedback_command(self, text: str) -> str:
        """Handle feedback provided by the user."""
        try:
            feedback_content = text.lower().replace("feedback", "", 1).strip()
            if not feedback_content:
                return "Please provide some feedback after the word 'feedback'."
            
            self.logger.info(f"Received user feedback: {feedback_content}")
            # Store feedback in memory
            self.memory_system.add_memory(
                content=f"User feedback: {feedback_content}",
                memory_type="long_term",
                metadata={
                    "type": "user_feedback",
                    "timestamp": datetime.now().isoformat()
                },
                importance=0.8 # Feedback is important
            )
            return "Thank you for your feedback! I've recorded it."
        except Exception as e:
            self.logger.error(f"Error handling feedback: {e}", exc_info=True)
            return "Sorry, I encountered an error trying to process your feedback."

# <<< End of JarvisAgent class definition >>>

# Ensure dependent classes are fully defined before rebuilding models that reference them
try:
    # This rebuild is specifically for JarvisAgent itself, which has forward refs
    # to PlanningSystem, ExecutionSystem, etc., which should now be defined and rebuilt.
    JarvisAgent.model_rebuild()
    print("Rebuilt Pydantic model: JarvisAgent")
except NameError as e:
     print(f"Warning: Could not rebuild JarvisAgent, definition might be missing or dependencies failed rebuild: {e}")
except Exception as e:
    print(f"Warning: An unexpected error occurred during JarvisAgent model rebuild: {e}") 