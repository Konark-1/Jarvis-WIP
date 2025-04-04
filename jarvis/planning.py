"""
Planning system for Jarvis that integrates with memory and provides agentic planning capabilities.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import logging
from enum import Enum # <-- Import Enum

# Conditional Imports for Type Checking
if TYPE_CHECKING:
    from .memory.unified_memory import UnifiedMemorySystem
    from .memory.medium_term import MediumTermMemory, Objective
    from .memory.long_term import LongTermMemory
    from .memory.short_term import ShortTermMemory
    from .llm import LLMClient
    
# Runtime Imports
from .memory.medium_term import Objective, MediumTermMemory # Keep for runtime use if needed by Pydantic
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from utils.logger import setup_logger
from .llm import LLMClient, LLMError, LLMCommunicationError, LLMTokenLimitError # Keep runtime imports for exceptions

# <<< ADDED TaskStatus Enum >>>
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    """A single task in a plan"""
    task_id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    dependencies: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    """A plan to achieve an objective"""
    plan_id: str
    objective_id: str
    objective_description: str
    tasks: List[Task] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # active, completed, failed
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PlanningSystem(BaseModel):
    """Planning system that creates and manages plans to achieve objectives"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    logger: logging.Logger
    unified_memory: 'UnifiedMemorySystem'
    llm: Optional['LLMClient'] # LLM can be optional
    
    # <<< ADDED: Cache for active plans >>>
    active_plans: Dict[str, Plan] = Field(default_factory=dict)
    
    def __init__(self, unified_memory: 'UnifiedMemorySystem', llm_client: Optional['LLMClient'], **data):
        # Prepare data for Pydantic initialization
        init_data = data.copy()

        # --- Logger Initialization ---
        if 'logger' not in init_data:
            init_data['logger'] = setup_logger(__name__) # Reverted to setup_logger
        # --- End Logger Initialization ---

        # Validate required unified_memory
        if not unified_memory:
            raise ValueError("UnifiedMemorySystem instance is required for PlanningSystem")
        
        # Add unified memory and LLM to init_data for Pydantic
        init_data['unified_memory'] = unified_memory
        init_data['llm'] = llm_client # Add llm_client (Optional)

        # Call Pydantic's __init__ with all necessary data
        super().__init__(**init_data)
        
        # Post-initialization checks (accessing fields via self)
        if not self.llm:
             self.logger.warning("LLM client not provided to PlanningSystem. Planning capabilities will be limited.")
        
        # Access sub-memory via self.unified_memory when needed
    
    def create_plan(self, objective_id: str, 
                      context_knowledge: Optional[List[Dict[str, Any]]] = None, 
                      context_history: Optional[List[Dict[str, str]]] = None) -> Plan:
        """Create a plan to achieve an objective, using provided context."""
        # Get objective details
        objective: Optional['Objective'] = self.unified_memory.medium_term.get_objective(objective_id)
        if not objective:
            raise ValueError(f"Objective {objective_id} not found")
        
        # Create plan object
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        plan = Plan(
            plan_id=plan_id,
            objective_id=objective_id,
            objective_description=objective.description
        )
        
        # Break down objective into tasks, passing context
        tasks = self._decompose_objective(
            objective.description, 
            plan_id,
            context_knowledge=context_knowledge,
            context_history=context_history
        )
        plan.tasks = tasks
        
        # Store plan in memory (persistence and active cache)
        self._store_plan(plan)
        self.active_plans[plan_id] = plan # Add to active cache (still needed for get_next_task if not fully refactored)
        self.logger.info(f"Created and cached active plan: {plan_id}")
        
        return plan
    
    def is_plan_complete(self, plan_id: str) -> bool:
        """Check if all tasks in a plan are marked as completed."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            self.logger.warning(f"Attempted to check completion status for non-existent or inactive plan: {plan_id}")
            # If the plan isn't in the active cache, it's effectively complete (or failed/removed).
            # Returning True might be misleading, but returning False could stall the main loop if the plan *was* removed because it finished.
            # Let's consider it 'not actively completable' and return False, allowing the main loop to potentially handle the state based on other factors.
            return False

        for task in plan.tasks:
            if task.status != "completed":
                return False # Found a task that is not completed
        
        # If loop finishes without returning False, all tasks are completed
        self.logger.debug(f"Plan {plan_id} confirmed as complete (all tasks are 'completed').")
        return True
    
    def _decompose_objective(self, objective_description: str, plan_id: str, 
                             context_knowledge: Optional[List[Dict[str, Any]]] = None, 
                             context_history: Optional[List[Dict[str, str]]] = None) -> List[Task]:
        """Break down an objective into tasks using LLM-based planning and provided context."""
        # REMOVED internal LTM search for similar objectives
        # similar_objectives: List[Dict[str, Any]] = self.unified_memory.long_term.retrieve_knowledge(
        #     f"objective decomposition for: {objective_description}"
        # )
        
        # Create a task ID prefix
        task_id_prefix = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Use LLM to decompose the objective if available
        if self.llm:
            try:
                # --- Format Context for Prompt --- 
                context_str = ""
                if context_history:
                    context_str += "\nRelevant Conversation History:\n"
                    for msg in context_history[-3:]: # Use last 3 messages
                         context_str += f"- {msg.get('role', 'unknown')}: {msg.get('content', '')}\n"
                    context_str += "---"
                
                if context_knowledge:
                    context_str += "\nRelevant Retrieved Knowledge:\n"
                    for i, item in enumerate(context_knowledge[:3]): # Use top 3 knowledge items
                         # Safely access content and metadata
                         content = item.get('content', 'N/A')
                         metadata_desc = item.get('metadata', {}).get('description', '')
                         # Shorten potentially long content
                         content_preview = (str(content)[:150] + '...') if len(str(content)) > 150 else str(content)
                         context_str += f"- Item {i+1} ({metadata_desc}): {content_preview}\n"
                    context_str += "---"
                # --- End Context Formatting --- 

                system_prompt = """
                You are Jarvis, an AI assistant that excels at breaking down objectives into practical, actionable tasks.
                For the given objective, create a comprehensive and logical sequence of tasks needed to accomplish it.
                Consider dependencies between tasks and different phases like planning, research, execution, and validation.
                Use the provided context (conversation history, retrieved knowledge) to inform the task breakdown if relevant.
                
                Crucially, ensure the **final task** is always responsible for synthesizing the results or outcomes from the previous tasks and preparing a final report or response for the user.
                
                Return the tasks as a JSON array of objects. Each object MUST have a "description" (string) and "dependencies" (list of integers, representing the 0-based index of tasks that must precede this one).
                Optionally include a "phase" (string: planning|research|execution|validation|reporting).
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
                
                # REMOVED logic for formatting internal LTM search results
                
                prompt = f"""
                Objective: {objective_description}
                
                {context_str} # Inject formatted context here
                
                Create a logical sequence of tasks to accomplish this objective following the specified JSON format.
                Output ONLY the JSON array.
                """
                
                self.logger.debug(f"Sending decomposition prompt to LLM for objective: {objective_description}")
                response_content = self.llm.process_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5, # Slightly lower temp for structured output
                    max_tokens=2000 # Allow for longer plans
                )
                
                # Try to parse the JSON response
                try:
                    # Clean potential markdown code blocks
                    if response_content.strip().startswith("```json"):
                        response_content = response_content.strip()[7:-3].strip()
                    elif response_content.strip().startswith("```"):
                         response_content = response_content.strip()[3:-3].strip()

                    tasks_data = json.loads(response_content)

                    # Validate structure
                    if not isinstance(tasks_data, list):
                        raise ValueError(f"LLM response is not a list. Got: {type(tasks_data)}")

                    if len(tasks_data) == 0:
                        self.logger.warning("LLM returned an empty list of tasks.")
                        # Fall through to rule-based
                    else:
                        # Convert to Task objects with validation
                        dynamic_tasks = []
                        for i, task_data in enumerate(tasks_data):
                            if not isinstance(task_data, dict):
                                self.logger.warning(f"Skipping invalid task data (not a dict) at index {i}: {task_data}")
                                continue
                            if "description" not in task_data or not isinstance(task_data["description"], str):
                                self.logger.warning(f"Skipping task data missing valid 'description' at index {i}: {task_data}")
                                continue

                            # Convert dependencies from indices to task IDs
                            dependencies = []
                            dep_indices = task_data.get("dependencies", [])
                            if isinstance(dep_indices, list):
                                for dep_idx in dep_indices:
                                    if isinstance(dep_idx, int) and 0 <= dep_idx < i:
                                        # Map 0-based index to 1-based task ID suffix
                                        dependencies.append(f"{task_id_prefix}_{dep_idx + 1}")
                                    else:
                                         self.logger.warning(f"Invalid dependency index '{dep_idx}' for task {i}. Skipping dependency.")
                            else:
                                self.logger.warning(f"Invalid 'dependencies' format for task {i}. Expected list, got {type(dep_indices)}. Ignoring dependencies.")

                            dynamic_tasks.append(Task(
                                task_id=f"{task_id_prefix}_{i + 1}",
                                description=task_data["description"],
                                dependencies=dependencies,
                                metadata={
                                    "phase": task_data.get("phase", "execution"),
                                    "plan_id": plan_id
                                }
                            ))

                        if dynamic_tasks:
                            self.logger.info(f"Successfully decomposed objective into {len(dynamic_tasks)} tasks using LLM.")
                            # Store this decomposition for future reference
                            self._store_decomposition(objective_description, dynamic_tasks)
                            return dynamic_tasks
                        else:
                            self.logger.warning("LLM response parsed, but no valid tasks were extracted. Falling back to rule-based.")

                except json.JSONDecodeError as json_err:
                    self.logger.error(f"Failed to parse LLM task decomposition response as JSON: {json_err}. Raw response:\n{response_content}")
                    # Fall through to rule-based decomposition
                except ValueError as val_err:
                     self.logger.error(f"Error validating LLM task decomposition structure: {val_err}. Raw response:\n{response_content}")
                     # Fall through to rule-based decomposition
                except Exception as e:
                     self.logger.error(f"Unexpected error processing LLM task decomposition: {type(e).__name__}: {e}")
                     # Fall through to rule-based decomposition

            # Catch specific LLM errors
            except LLMTokenLimitError as token_err:
                self.logger.error(f"LLM task decomposition failed due to token limit: {token_err}")
                # Fall through to rule-based decomposition
            except LLMCommunicationError as comm_err:
                 self.logger.error(f"LLM task decomposition failed due to communication error: {comm_err}")
                 # Fall through to rule-based decomposition
            except Exception as e:
                 self.logger.exception(f"Unexpected error during LLM task decomposition attempt: {e}")
                 # Fall through to rule-based decomposition

        # Fallback to basic task if LLM fails or is unavailable
        self.logger.warning("LLM decomposition failed or unavailable. Creating a single task for the objective.")
        # Return a list containing a single task representing the whole objective
        return [Task(task_id=f"{task_id_prefix}_1", 
                    description=objective_description, 
                    metadata={"plan_id": plan_id})
               ]
    
    def _store_plan(self, plan: Plan) -> None:
        """Store a plan in memory"""
        # Store in medium-term memory
        self.unified_memory.medium_term.add_progress(
            f"Created plan {plan.plan_id} with {len(plan.tasks)} tasks",
            metadata={"plan_id": plan.plan_id}
        )
        
        # <<< Serialize tasks for storage >>>
        # Exclude datetimes for simpler JSON, keep other fields
        tasks_serializable = [task.model_dump(exclude={'created_at', 'completed_at'}) for task in plan.tasks]
        tasks_json = json.dumps(tasks_serializable)

        # <<< Prepare complete metadata for LTM >>>
        plan_metadata = {
            "type": "plan",
            "plan_id": plan.plan_id, # Include plan_id
            "objective_id": plan.objective_id,
            "status": plan.status,
            "created_at": plan.created_at.isoformat(), # Store as ISO string
            # Store serialized tasks
            "tasks_json": tasks_json,
            # Add other relevant plan metadata if needed
            # "other_metadata": json.dumps(plan.metadata) if plan.metadata else None # Original line with potential None
        }
        # <<< Only add other_metadata if it exists >>>
        if plan.metadata:
            try:
                plan_metadata["other_metadata_json"] = json.dumps(plan.metadata)
            except TypeError as e:
                 self.logger.warning(f"Could not serialize plan.metadata for plan {plan.plan_id}: {e}")
                 # Optionally store a placeholder or skip

        # Store in long-term memory for future reference
        self.unified_memory.long_term.add_knowledge(
            f"plan_{plan.plan_id}", # Use a consistent ID format
            f"Stored plan for objective {plan.objective_id}", # Simpler content description
            metadata=plan_metadata # Store the complete metadata
        )
    
    def get_next_task(self, plan_id: str) -> Optional[Task]:
        """Get the next task to execute from a plan"""
        # Get plan from memory
        plan = self._get_plan(plan_id)
        if not plan:
            return None
        
        # Find first pending task with no pending dependencies
        for task in plan.tasks:
            if task.status == "pending":
                # Check dependencies
                dependencies_met = True
                for dep_id in task.dependencies:
                    dep_task = next((t for t in plan.tasks if t.task_id == dep_id), None)
                    if dep_task and dep_task.status != "completed":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    return task
        
        return None
    
    def _get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan from the active cache ONLY."""
        # <<< MODIFIED: Check active cache first >>>
        if plan_id in self.active_plans:
            # self.logger.debug(f"Retrieving plan {plan_id} from active cache.") # Reduce log noise
            return self.active_plans[plan_id]
        
        # <<< REMOVED: Logic to load from LTM automatically >>>
        # self.logger.debug(f"Plan {plan_id} not in active cache. Not loading automatically.")
        return None # Return None if not in active cache
    
    def load_plan_from_ltm(self, plan_id: str) -> Optional[Plan]:
        """Explicitly loads a plan from LTM and adds it to the active cache."""
        # Check if already active to avoid redundant load
        if plan_id in self.active_plans:
            self.logger.info(f"Plan {plan_id} is already in active cache.")
            return self.active_plans[plan_id]

        self.logger.info(f"Explicitly loading plan {plan_id} from LTM.")
        knowledge_id = f"plan_{plan_id}"
        plan_knowledge = self.unified_memory.long_term.retrieve_knowledge(knowledge_id)
        
        if plan_knowledge and plan_knowledge[0].get("metadata"):
            metadata = plan_knowledge[0]["metadata"]
            try:
                # Deserialize tasks
                tasks_list = []
                if metadata.get("tasks_json"): 
                    tasks_data = json.loads(metadata["tasks_json"])
                    tasks_list = [Task(**task_data) for task_data in tasks_data]
                
                other_metadata = {} 
                if metadata.get("other_metadata_json"): 
                    other_metadata = json.loads(metadata["other_metadata_json"])
                
                plan_obj = Plan(
                    plan_id=metadata.get("plan_id", plan_id),
                    objective_id=metadata.get("objective_id", "unknown"),
                    objective_description=metadata.get("objective_description", "unknown"),
                    tasks=tasks_list,
                    created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else datetime.now(),
                    status=metadata.get("status", "active"),
                    metadata=other_metadata,
                )
                self.active_plans[plan_id] = plan_obj
                self.logger.info(f"Loaded plan {plan_id} from LTM and added to active cache.")
                return plan_obj
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
                self.logger.error(f"Error reconstructing plan {plan_id} from metadata: {e}. Metadata: {metadata}", exc_info=True)
                return None
        else:
            self.logger.warning(f"Plan {plan_id} not found in LTM.")
            return None

    def update_task_status(self, plan_id: str, task_id: str, status: str, error: Optional[str] = None, output: Optional[Any] = None) -> None:
        """Update the status of a task and store its output."""
        plan = self._get_plan(plan_id)
        if not plan:
            return
        
        # Find and update task
        task = next((t for t in plan.tasks if t.task_id == task_id), None)
        if task:
            # <<< Log status change >>>
            self.logger.info(f"Updating task {task_id} in plan {plan_id} to status: {status}")
            
            task.status = status
            if status == "completed":
                task.completed_at = datetime.now()
                # Store the output in the task metadata if successful
                if output is not None:
                    # Ensure output is serializable if storing plan back to LTM eventually
                    try:
                        # Simple check for basic types, attempt json.dumps for others
                        if isinstance(output, (str, int, float, bool, list, dict)):
                             task.metadata["last_output"] = output
                        else:
                             task.metadata["last_output"] = json.dumps(output) # Or str(output)
                    except Exception as serial_err:
                         self.logger.warning(f"Could not serialize output for task {task_id}: {serial_err}. Storing as string.")
                         task.metadata["last_output"] = str(output)
            elif status == "failed":
                task.error_count += 1
                if error:
                    task.metadata["last_error"] = error
            
            # Update plan status
            plan_finished = False # <<< Flag to check if plan ended >>>
            if all(t.status == "completed" for t in plan.tasks):
                plan.status = "completed"
                plan.completed_at = datetime.now()
                plan_finished = True
                self.logger.info(f"Plan {plan_id} marked as completed.")
            elif any(t.status == "failed" for t in plan.tasks):
                plan.status = "failed"
                # Consider adding a completed_at timestamp for failed plans too?
                plan_finished = True
                self.logger.info(f"Plan {plan_id} marked as failed.")
            
            # Store updated plan (persists changes)
            self._store_plan(plan)
            
            # <<< ADDED: Remove finished plan from active cache >>>
            if plan_finished and plan_id in self.active_plans:
                del self.active_plans[plan_id]
                self.logger.info(f"Removed finished plan {plan_id} from active cache.")
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get the current status of a plan"""
        plan = self._get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        
        # Count tasks by status
        task_counts = {
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0
        }
        
        for task in plan.tasks:
            task_counts[task.status] += 1
        
        return {
            "plan_id": plan_id,
            "objective_id": plan.objective_id,
            "status": plan.status,
            "created_at": plan.created_at,
            "completed_at": plan.completed_at,
            "tasks": task_counts
        }
    
    def _store_decomposition(self, objective_description: str, tasks: List[Task]) -> None:
        """Store a task decomposition in long-term memory for future reference"""
        # Convert tasks to a serializable format (e.g., list of dicts)
        tasks_serializable = [task.model_dump(exclude={'created_at', 'completed_at'}) for task in tasks] # Exclude datetimes for simpler JSON
        # Serialize task list to JSON string
        tasks_json = json.dumps(tasks_serializable)

        # Create metadata dictionary
        decomposition_metadata = {
            "type": "task_decomposition",
            "objective": objective_description,
            "task_count": len(tasks_serializable),
            "timestamp": datetime.now().isoformat(),
            # Store the JSON string
            "tasks_json": tasks_json
        }

        # Add to long-term memory
        self.unified_memory.long_term.add_knowledge(
            f"objective_decomposition_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            f"Decomposition for: {objective_description}",
            metadata=decomposition_metadata
        )

# <<< End of PlanningSystem class definition >>>

# Ensure dependent classes are fully defined before rebuilding models that reference them
# try:
#     from .memory.unified_memory import UnifiedMemorySystem
#     from .memory.medium_term import MediumTermMemory # Add imports for all forward refs
#     from .memory.long_term import LongTermMemory
#     from .memory.short_term import ShortTermMemory
#     from .llm import LLMClient
#     
#     # Perform model rebuild for PlanningSystem
#     PlanningSystem.model_rebuild()
#     print("Rebuilt Pydantic model: PlanningSystem") # Optional
# except ImportError as e:
#     print(f"Warning: Could not import dependencies to rebuild PlanningSystem: {e}")
# except NameError as e:
#     print(f"Warning: Could not rebuild PlanningSystem, definition might be missing: {e}")
# except Exception as e:
#     print(f"Warning: An unexpected error occurred during PlanningSystem model rebuild: {e}") 