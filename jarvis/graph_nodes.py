import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from jarvis.state import JarvisState # Import the TypedDict state
from jarvis.llm import LLMClient
from jarvis.planning import PlanningSystem, Plan, Task, TaskStatus
from jarvis.execution import ExecutionSystem, ExecutionResult
from jarvis.memory.unified_memory import UnifiedMemorySystem
from jarvis.crew import create_planning_crew # Import crew creation function
from jarvis.skills.registry import SkillRegistry # Import SkillRegistry

logger = logging.getLogger(__name__)

# --- Node Definitions ---

def understand_query_node(state: JarvisState, llm: LLMClient, memory: UnifiedMemorySystem) -> Dict[str, Any]:
    """Analyzes the initial query, creates an objective in memory, and updates state."""
    query = state.get('user_input')
    logger.info(f"NODE: understand_query_node (Query: '{query}')")
    
    if not query:
        logger.error("User input query is missing in state.")
        return {"error_message": "Cannot process: User input query missing.", "timestamp": datetime.now()}

    # Simple approach: Use the original query directly as the objective description.
    # TODO: Enhance with LLM interaction to refine, clarify, or extract parameters.
    objective_desc = query
    
    # Create the objective in Medium Term Memory
    objective_id = None
    new_objective = None
    try:
        # Metadata could include query source, user info, etc. later
        objective_id = memory.medium_term.create_objective(
            description=objective_desc, 
            metadata={"source": "langgraph_query"}
        )
        if not objective_id:
             raise ValueError("Failed to create objective in MTM (returned None ID)")
        
        # Retrieve the newly created objective object to put in the state
        new_objective = memory.medium_term.get_objective(objective_id)
        if not new_objective:
             logger.error(f"Failed to retrieve newly created objective {objective_id} from MTM.")
             return {"error_message": f"Failed to retrieve objective {objective_id} after creation.", "timestamp": datetime.now()}
        else:
            logger.info(f"-> Objective created & retrieved from MTM: ID={objective_id}, Description='{objective_desc}'")

    except Exception as e:
        logger.exception(f"Failed to create/retrieve objective in Medium Term Memory: {e}")
        return {"error_message": f"Failed to store objective in memory: {e}", "timestamp": datetime.now()}

    # Return the Objective object itself, not just the ID/description
    return {
        "objective": new_objective,
        "timestamp": datetime.now()
    }

def retrieve_context_node(state: JarvisState, memory: UnifiedMemorySystem) -> Dict[str, Any]:
    """Retrieves relevant context from memory based on the objective."""
    logger.info("NODE: retrieve_context_node")
    current_objective = state.get('objective')
    
    if not current_objective or not hasattr(current_objective, 'description'):
        logger.warning("No objective object or description found in state for context retrieval.")
        return {}
        
    objective_desc = current_objective.description

    try:
        # Retrieve relevant knowledge from LTM
        # Use a query combining objective and maybe recent history if available
        query = objective_desc 
        logger.debug(f"Searching memory with query: '{query}'")
        # Search LTM and STM (adjust types and k as needed)
        search_results = memory.search_memory(
            query=query, 
            memory_types=["long_term", "short_term"], 
            k_per_type=3 
        )
        
        retrieved_knowledge_list = []
        conversation_history_list = []
        
        for entry in search_results:
            entry_dict = entry.model_dump() # Convert MemoryEntry to dict
            # Basic serialization check for content/metadata
            for key in ['content', 'metadata']:
                if key in entry_dict and entry_dict[key] is not None:
                    try: json.dumps(entry_dict[key])
                    except TypeError:
                         entry_dict[key] = str(entry_dict[key])
                         
            if entry.memory_type == "long_term":
                retrieved_knowledge_list.append(entry_dict)
            elif entry.memory_type == "short_term":
                 # Format STM entry for state
                 conv_entry = {
                     "role": entry_dict.get('metadata', {}).get('speaker', 'unknown'),
                     "content": entry_dict.get('content')
                 }
                 conversation_history_list.append(conv_entry)
        
        # Sort conversation history chronologically (search might not guarantee order)
        conversation_history_list.sort(key=lambda x: x.get('timestamp', datetime.min()))

        logger.info(f"-> Retrieved {len(retrieved_knowledge_list)} knowledge items and {len(conversation_history_list)} conversation items.")
        
        # Return updates for state
        return {
            "retrieved_knowledge": retrieved_knowledge_list,
            "conversation_history": conversation_history_list,
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.exception(f"Error during context retrieval: {e}")
        # Return empty lists or an error message?
        return {"retrieved_knowledge": [], "conversation_history": [], "error_message": f"Context retrieval failed: {e}"}

# --- Helper function to format context for crew --- 
# (Similar to the one previously added to _decompose_objective)
def format_context_for_prompt(
    context_knowledge: Optional[List[Dict[str, Any]]] = None, 
    context_history: Optional[List[Dict[str, str]]] = None) -> str:
    context_str = ""
    if context_history:
        context_str += "Relevant Conversation History:\n"
        for msg in context_history[-3:]:
             context_str += f"- {msg.get('role', 'unknown')}: {msg.get('content', '')}\n"
        context_str += "---"
    if context_knowledge:
        context_str += "\nRelevant Retrieved Knowledge:\n"
        for i, item in enumerate(context_knowledge[:3]):
             content = item.get('content', 'N/A')
             metadata_desc = item.get('metadata', {}).get('description', '')
             content_preview = (str(content)[:150] + '...') if len(str(content)) > 150 else str(content)
             context_str += f"- Item {i+1} ({metadata_desc}): {content_preview}\n"
        context_str += "---"
    return context_str

# --- Crew Node Definition ---
def run_planning_crew_node(
    objective_description: str, context_str: str, llm_client: LLMClient, skills: SkillRegistry
) -> list[dict]:
    """Runs the planning crew to decompose the objective into tasks."""
    logger.info(f"Running planning crew for objective: '{objective_description[:50]}...'")
    tasks_list = []

    try:
        # Get the LangChain compatible LLM instance
        crew_llm = llm_client.get_langchain_llm()
        if not crew_llm:
            logger.error("Could not get a valid LLM instance for the planning crew.")
            # Fallback or raise error? For now, return empty list.
            # Consider adding a basic LLM call fallback here if crew fails.
            return []

        # Create the planning crew, passing the LLM and SkillRegistry
        # Note: create_planning_crew needs to be updated to accept/use skills
        planning_crew = create_planning_crew(llm_config=crew_llm, skill_registry=skills)

        # Prepare inputs for the crew's task
        inputs = {"objective": objective_description, "context": context_str}
        logger.debug(f"Planning crew inputs: {inputs}")

        # Kick off the crew's task
        result = planning_crew.kickoff(inputs=inputs)
        logger.info(f"Planning crew finished. Raw result: {result}")

        # Attempt to parse the result as JSON (assuming crew outputs JSON string)
        # TODO: Add more robust parsing and validation (e.g., using Pydantic)
        try:
            # Assuming the result is a string containing a JSON list of tasks
            parsed_output = json.loads(result)
            if isinstance(parsed_output, list):
                # Validate structure? For now, assume it's list of dicts
                tasks_list = parsed_output
                logger.info(f"Successfully parsed {len(tasks_list)} tasks from crew output.")
            else:
                logger.warning(f"Crew output was not a JSON list: {type(parsed_output)}")
                # Attempt extraction or fallback?
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from planning crew result: {result}")
            # Fallback: maybe try a regex or simple string split if format is known?
        except Exception as e:
            logger.error(f"Error processing planning crew result: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error running planning crew: {e}", exc_info=True)
        # Fallback mechanism if crew fails entirely?

    # Ensure return type is correct even on failure
    return tasks_list if isinstance(tasks_list, list) else []

# --- Modified plan_tasks_node to use the crew node --- 
# Placeholder for other nodes
def plan_tasks_node(state: JarvisState, planner: PlanningSystem, llm: LLMClient, skills: SkillRegistry) -> dict:
    """Uses the planning crew to generate a task list based on the user query and context."""
    logger.info("Entering plan_tasks_node...")
    current_objective = state.get("objective")
    if not current_objective:
        logger.error("Objective is missing from state in plan_tasks_node.")
        # How to handle this? Maybe update state with an error or return empty plan?
        # Returning current state might lead to loop. Let's create an empty plan with error status.
        # Provide dummy values for required fields to satisfy Pydantic validation
        error_plan = Plan(
            plan_id="error_plan_missing_objective",
            objective_id="error_obj_missing",
            objective_description="Error: Objective missing in state",
            tasks=[]
        )
        return {"plan": error_plan, "status_message": "Error: Objective missing."}

    # Ensure current_objective and its description exist before proceeding
    if not hasattr(current_objective, 'description') or not current_objective.description:
        logger.error(f"Objective object exists but description is missing or empty: {current_objective}")
        # Create another error plan
        error_plan = Plan(
            plan_id="error_plan_missing_desc",
            objective_id=getattr(current_objective, 'objective_id', 'error_obj_unknown'),
            objective_description="Error: Objective description missing",
            tasks=[]
        )
        return {"plan": error_plan, "status_message": "Error: Objective description missing."}

    # Now it's safe to access description
    objective_desc = current_objective.description
    objective_id = current_objective.objective_id # Also get the ID

    logger.info(f"Starting planning for objective: {objective_desc}")

    # Format context
    knowledge_str = state.get("knowledge", "")
    history_str = state.get("conversation_history", "") # Assuming history is stored as a formatted string
    context_str = format_context_for_prompt(knowledge_str, history_str)

    try:
        # Run the planning crew via the dedicated node function
        task_dicts = run_planning_crew_node(objective_desc, context_str, llm, skills)

        if not task_dicts:
             logger.warning("Planning crew returned no tasks. Creating an empty plan.")
             # Maybe add a task indicating planning failed?
             plan = planner.create_plan(objective_desc, [])
             return {"plan": plan, "status_message": "Planning crew failed or returned no tasks."}

        # Convert dicts to Task objects
        tasks = []
        for i, task_dict in enumerate(task_dicts):
             # Basic validation: Check for 'description' key
             if "description" not in task_dict:
                 logger.warning(f"Task dictionary {i} missing 'description': {task_dict}. Skipping.")
                 continue
             # Assuming other fields might be optional or have defaults
             tasks.append(
                 Task(
                     id=i + 1,  # Assign sequential IDs
                     description=task_dict["description"],
                     status=TaskStatus.PENDING, # Default status
                     result=task_dict.get("result", ""), # Optional field
                     skill=task_dict.get("skill", ""), # Optional field
                     arguments=task_dict.get("arguments", {}) # Optional field
                 )
             )

        # Create the plan object
        plan = planner.create_plan(objective_desc, tasks)
        logger.info(f"Plan created successfully with {len(plan.tasks)} tasks.")
        return {"plan": plan, "status_message": "Plan created successfully."}

    except Exception as e:
        logger.error(f"Error during task planning: {e}", exc_info=True)
        # Fallback: Create an empty plan or a plan with an error task
        error_plan = Plan(objective=objective_desc, tasks=[Task(id=1, description=f"Planning Failed: {e}", status=TaskStatus.FAILED)])
        return {"plan": error_plan, "status_message": f"Error during planning: {e}"}

def get_next_task_node(state: JarvisState, planner: PlanningSystem) -> Dict[str, Any]:
    """Gets the next pending task from the current plan in the state."""
    logger.info("NODE: get_next_task_node")
    
    current_plan = state.get('current_plan')
    if not current_plan:
        logger.warning("No current plan found in state to get next task from.")
        return {"current_task": None, "timestamp": datetime.now()}
        
    # Ensure we have a Plan object (it might be just a dict if not handled carefully)
    # Re-instantiate from dict if necessary, assuming state dict holds serializable data
    if isinstance(current_plan, dict):
        try:
            current_plan = Plan(**current_plan) # Recreate Plan object
        except Exception as e:
            logger.error(f"Failed to recreate Plan object from state: {e}. Data: {current_plan}")
            return {"current_task": None, "error_message": "Invalid plan data in state.", "timestamp": datetime.now()}
    elif not isinstance(current_plan, Plan):
        logger.error(f"current_plan in state is not a Plan object or dict, type: {type(current_plan)}")
        return {"current_task": None, "error_message": "Invalid plan object type in state.", "timestamp": datetime.now()}

    plan_id = current_plan.plan_id
    logger.info(f"Checking plan {plan_id} in state for next task.")

    # Re-implement PlanningSystem.get_next_task logic using the state's plan object
    next_task_obj: Optional[Task] = None
    for task_data in current_plan.tasks: # Iterate task dicts/objects within Plan
        # Ensure task is a Task object
        if isinstance(task_data, dict):
             try:
                 task = Task(**task_data)
             except Exception:
                 logger.warning(f"Skipping invalid task data in plan {plan_id}: {task_data}")
                 continue 
        elif isinstance(task_data, Task):
            task = task_data
        else:
            logger.warning(f"Skipping invalid task data type {type(task_data)} in plan {plan_id}")
            continue
            
        if task.status == "pending":
            dependencies_met = True
            for dep_id in task.dependencies:
                # Find dependency task within the current_plan.tasks list
                dep_task_found = None
                for t_inner_data in current_plan.tasks:
                    task_id_to_check = None
                    if isinstance(t_inner_data, dict):
                        task_id_to_check = t_inner_data.get('task_id')
                    elif isinstance(t_inner_data, Task):
                        task_id_to_check = t_inner_data.task_id
                    
                    if task_id_to_check == dep_id:
                        # Recreate Task obj to check status if needed
                        if isinstance(t_inner_data, dict):
                            try: dep_task_found = Task(**t_inner_data)
                            except Exception: pass # Ignore if invalid
                        else:
                            dep_task_found = t_inner_data 
                        break # Found the dependency task data/obj
                        
                if dep_task_found and getattr(dep_task_found, 'status', 'unknown') != "completed":
                    dependencies_met = False
                    break # Dependency not met
            
            if dependencies_met:
                next_task_obj = task # Assign the Task object
                break # Found the next task

    if next_task_obj:
        logger.info(f"-> Next task found: {next_task_obj.task_id} ('{next_task_obj.description}')")
        # Return the task object itself, LangGraph handles dict merging
        # Convert Task object to dict for state update
        return {"current_task": next_task_obj.model_dump(), "timestamp": datetime.now()}
    else:
        logger.info(f"-> No runnable task found in plan {plan_id}.")
        return {"current_task": None, "timestamp": datetime.now()}

def execute_tool_node(state: JarvisState, executor: ExecutionSystem) -> Dict[str, Any]: # Renamed for clarity
    """Executes the current task using the ExecutionSystem."""
    logger.info("NODE: execute_tool_node")
    
    current_task = state.get('current_task')
    if not current_task:
        logger.warning("No current task found in state to execute.")
        # Return None for last_execution_result to signify no execution happened
        return {"last_execution_result": None, "timestamp": datetime.now()}
        
    # Ensure current_task is a Task object
    if isinstance(current_task, dict):
        try:
            current_task = Task(**current_task)
        except Exception as e:
             logger.error(f"Failed to recreate Task object from state: {e}. Data: {current_task}")
             # Return an error state
             return {"last_execution_result": None, "error_message": "Invalid task data in state.", "timestamp": datetime.now()}
    elif not isinstance(current_task, Task):
         logger.error(f"current_task in state is not a Task object or dict, type: {type(current_task)}")
         # Return an error state
         return {"last_execution_result": None, "error_message": "Invalid task object type in state.", "timestamp": datetime.now()}

    logger.info(f"Executing task: {current_task.task_id} ('{current_task.description}')")

    try:
        result: ExecutionResult = executor.execute_task(current_task)
        logger.info(f"-> Task {current_task.task_id} execution result: Success={result.success}")

        # ExecutionResult is Pydantic, convert to dict for state
        last_exec_result_dict = result.model_dump()
        
        # Basic serialization check for potentially complex fields (output/metadata)
        for key in ['output', 'metadata']:
             if key in last_exec_result_dict and last_exec_result_dict[key] is not None:
                 try:
                     json.dumps(last_exec_result_dict[key])
                 except TypeError:
                     logger.warning(f"{key.capitalize()} for task {result.task_id} not JSON serializable, converting to string.")
                     last_exec_result_dict[key] = str(last_exec_result_dict[key])

        history = state.get("execution_history", []) or []
        history.append(last_exec_result_dict)

        return {
            "last_execution_result": last_exec_result_dict,
            "execution_history": history,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.exception(f"Error executing task {current_task.task_id}: {e}")
        failed_exec_result = {
             "task_id": current_task.task_id,
             "success": False,
             "output": None,
             "error": f"Node-level execution error: {e}",
             "execution_time": 0.0,
             "metadata": {"executed_by": "graph_node_error"}
        }
        history = state.get("execution_history", []) or []
        history.append(failed_exec_result)
        return {
            "error_message": f"Failed to execute task {current_task.task_id}: {e}", 
            "last_execution_result": failed_exec_result,
            "execution_history": history,
            "timestamp": datetime.now()
        }

def update_plan_node(state: JarvisState, planner: PlanningSystem) -> Dict[str, Any]:
    """Updates the plan status in the state based on the last execution result."""
    logger.info("NODE: update_plan_node")

    last_result = state.get('last_execution_result')
    current_plan = state.get('current_plan')

    if not last_result:
        logger.warning("No last execution result found in state to update plan.")
        # If no result, implies no execution happened, so no update needed
        return {"timestamp": datetime.now()}

    if not current_plan:
        logger.error("No current plan found in state to update.")
        return {"error_message": "Cannot update plan: No plan found in state.", "timestamp": datetime.now()}
        
    # Ensure current_plan is a Plan object
    if isinstance(current_plan, dict):
        try:
            # Make a deep copy before modification if it's just a dict?
            # Or assume state management handles immutability if needed.
            # For now, just recreate the object.
            current_plan = Plan(**current_plan)
        except Exception as e:
            logger.error(f"Failed to recreate Plan object from state: {e}. Data: {current_plan}")
            return {"error_message": "Invalid plan data in state.", "timestamp": datetime.now()}
    elif not isinstance(current_plan, Plan):
        logger.error(f"current_plan in state is not a Plan object or dict, type: {type(current_plan)}")
        return {"error_message": "Invalid plan object type in state.", "timestamp": datetime.now()}

    plan_id = current_plan.plan_id
    task_id = last_result.get('task_id')
    success = last_result.get('success')
    error = last_result.get('error')
    output = last_result.get('output')

    if task_id is None or success is None:
         logger.error(f"Invalid execution result format: {last_result}")
         return {"error_message": "Invalid execution result format.", "timestamp": datetime.now()}

    logger.info(f"Updating plan {plan_id} in state for task {task_id} status (Success={success})")

    try:
        task_updated = False
        plan_finished = False
        new_plan_status = current_plan.status
        updated_tasks = []

        # Find and update task within the state's plan object
        for task_data in current_plan.tasks:
            task_obj = None
            if isinstance(task_data, dict):
                 try: task_obj = Task(**task_data)
                 except Exception: pass # Skip invalid
            elif isinstance(task_data, Task):
                 task_obj = task_data # Already an object
            
            if not task_obj: continue # Skip if creation failed or invalid type

            if task_obj.task_id == task_id:
                task_obj.status = "completed" if success else "failed"
                if task_obj.status == "completed":
                    task_obj.completed_at = datetime.now()
                    # Ensure output is stored appropriately
                    task_obj.metadata["last_output"] = output 
                elif task_obj.status == "failed":
                    task_obj.error_count += 1
                    task_obj.metadata["last_error"] = error
                task_updated = True
            updated_tasks.append(task_obj) # Add the (potentially updated) task object
        
        current_plan.tasks = updated_tasks # Assign the list of task objects back
        
        if not task_updated:
            logger.error(f"Task {task_id} not found in plan {plan_id} within state.")
            # This is an error, but maybe recoverable? Return current state?
            return {"error_message": f"Task {task_id} not found in state's plan object.", "timestamp": datetime.now()}

        # Update overall plan status based on tasks in the state's plan object
        if all(t.status == "completed" for t in current_plan.tasks):
            new_plan_status = "completed"
            current_plan.completed_at = datetime.now()
            plan_finished = True
            logger.info(f"Plan {plan_id} marked as completed in state.")
        elif any(t.status == "failed" for t in current_plan.tasks):
            new_plan_status = "failed"
            # Assign completed_at for failed plans? Optional, but good for tracking.
            current_plan.completed_at = datetime.now()
            plan_finished = True
            logger.info(f"Plan {plan_id} marked as failed in state.")

        current_plan.status = new_plan_status
        
        # Persist the updated plan using PlanningSystem method
        try:
             planner._store_plan(current_plan) # Pass the updated Plan object
             logger.info(f"Persisted updated plan {plan_id} to memory.")
        except Exception as store_err:
             logger.exception(f"Failed to persist updated plan {plan_id}: {store_err}")
             # Continue state update even if persistence fails?
             # Maybe add a flag to state indicating persistence failure.

        # If finished, clean up PlanningSystem's internal cache
        if plan_finished and hasattr(planner, 'active_plans') and plan_id in planner.active_plans:
             try:
                 del planner.active_plans[plan_id]
                 logger.info(f"Removed finished plan {plan_id} from PlanningSystem cache.")
             except Exception as cache_err:
                  logger.error(f"Failed to remove plan {plan_id} from PlanningSystem cache: {cache_err}")

        # Return the updated plan and status
        # Convert Plan object to dict for state update
        return {
            "current_plan": current_plan.model_dump(), 
            "plan_status": new_plan_status,
            "current_task": None, # Clear current task after updating plan
            "last_execution_result": None, # Clear last result after updating plan
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.exception(f"Unexpected error updating plan {plan_id} in state: {e}")
        return {"error_message": f"Failed to update plan state: {e}", "timestamp": datetime.now()}

# --- Conditional Edge Functions ---

def should_continue_condition(state: JarvisState) -> str:
    """Determines if the plan execution loop should continue."""
    logger.info("CONDITION: should_continue_condition")
    plan_status = state.get("plan_status")
    error_message = state.get("error_message")

    if error_message:
        logger.warning(f"-> Error detected ('{error_message}'), routing to handle_error.")
        return "handle_error"
        
    if plan_status == "completed" or plan_status == "failed":
        logger.info(f"-> Plan status is '{plan_status}', routing to handle_completion.")
        return "handle_completion"
    elif plan_status == "active":
        logger.info("-> Plan status is active, routing to get_next_task.")
        return "continue"
    else:
        # If plan status is None or unexpected, likely an error or start state
        logger.warning(f"-> Plan status is unexpected ('{plan_status}'), routing to handle_error.")
        return "handle_error" 

def task_dispatch_condition(state: JarvisState) -> str:
    """Determines if a task was found to be executed."""
    logger.info("CONDITION: task_dispatch_condition")
    current_task = state.get("current_task")
    error_message = state.get("error_message") # Check for errors from get_next_task

    if error_message:
        logger.warning(f"-> Error detected ('{error_message}'), routing to handle_error.")
        return "handle_error"
        
    if current_task:
        logger.info("-> Task found, routing to execute_tool.")
        return "execute"
    else:
        # If no task found, the plan might be implicitly complete or stalled
        logger.info("-> No task found, routing to handle_completion.")
        return "handle_completion"

# --- Placeholder Nodes for End States ---

def handle_error_node(state: JarvisState) -> Dict[str, Any]:
    """Handles errors encountered during graph execution."""
    logger.error(f"NODE: handle_error_node (Error: '{state.get('error_message')}')")
    # Potentially generate a final error response
    final_response = f"An error occurred: {state.get('error_message', 'Unknown error')}"
    return {"final_response": final_response, "timestamp": datetime.now()}

def handle_plan_completion_node(state: JarvisState) -> Dict[str, Any]:
    """Handles the completion (successful or failed) of a plan."""
    logger.info(f"NODE: handle_plan_completion_node (Status: '{state.get('plan_status')}')")
    plan_status = state.get("plan_status", "Unknown")
    final_response = f"Plan execution finished with status: {plan_status}."

    # TODO: Add final synthesis logic here if needed, based on execution_history
    # For now, just report status.
    
    return {"final_response": final_response, "timestamp": datetime.now()}


# --- Add a simple synthesis node placeholder ---
# This might be called from handle_plan_completion_node or be a separate final step
def synthesize_final_response_node(state: JarvisState, llm: LLMClient) -> Dict[str, Any]:
    """Synthesizes the final response based on execution history."""
    logger.info("NODE: synthesize_final_response_node")
    
    objective = state.get("objective_description", "Unknown objective")
    history = state.get("execution_history", [])
    
    if not history:
        return {"final_response": "No execution history found to synthesize."} 

    try:
        context = f"Original Objective: {objective}\n\nExecution Results:\n"
        for i, result in enumerate(history[-5:]): # Use last 5 results for brevity
            task_desc = result.get('task_description', f'Task {result.get("task_id")}') # Need to ensure task desc is in result metadata
            status = "Success" if result.get('success') else "Failure"
            output = result.get('output') or result.get('error') or "No output/error recorded."
            context += f"\n--- Task {i+1}: {task_desc} ({status}) ---\n{output}\n"
            
        system_prompt = "You are Jarvis, summarizing the results of a completed plan. Review the provided objective and execution results. Generate a final, comprehensive response for the user based ONLY on these results. Present the information clearly and concisely. Focus on answering the user's original goal." 
        
        final_response = llm.process_with_llm(
             prompt=f"{context}\n\nSynthesize the final response for the user:",
             system_prompt=system_prompt,
             temperature=0.6,
             max_tokens=1500 
         )
        return {"final_response": final_response}
    
    except Exception as e:
        logger.exception(f"Error during final synthesis: {e}")
        return {"final_response": f"Error synthesizing final report: {e}"}

# ... other nodes ... 