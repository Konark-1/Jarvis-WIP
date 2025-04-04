import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from jarvis.state import JarvisState, UserInput, ChatMessage, KnowledgeSnippet # Import state definitions
from jarvis.llm import LLMClient
from jarvis.planning import PlanningSystem, Plan, Task, TaskStatus
from jarvis.execution import ExecutionSystem, ExecutionResult
from jarvis.memory.unified_memory import UnifiedMemorySystem, MemoryEntry # Import MemoryEntry
from jarvis.skills.registry import SkillRegistry # Import SkillRegistry
from jarvis.crew import create_planning_crew # Import crew creation function

logger = logging.getLogger(__name__)

# --- Node Definitions ---

def understand_query_node(state: JarvisState, llm: LLMClient, memory: UnifiedMemorySystem) -> Dict[str, Any]:
    """Analyzes the initial query, creates an objective in memory, and updates state."""
    query = state.get('original_query')
    logger.info(f"NODE: understand_query_node (Query: '{query}')")
    
    if not query:
        logger.error("Original query is missing in state.")
        return {"error_message": "Cannot process: Original query missing.", "timestamp": datetime.now()}

    # Simple approach: Use the original query directly as the objective description.
    # TODO: Enhance with LLM interaction to refine, clarify, or extract parameters.
    objective_desc = query
    
    # Create the objective in Medium Term Memory
    objective_id = None
    try:
        # Metadata could include query source, user info, etc. later
        objective_id = memory.medium_term.create_objective(
            description=objective_desc, 
            metadata={"source": "langgraph_query"}
        )
        if not objective_id:
             raise ValueError("Failed to create objective in MTM (returned None ID)")
        
        logger.info(f"-> Objective created in MTM: ID={objective_id}, Description='{objective_desc}'")

    except Exception as e:
        logger.exception(f"Failed to create objective in Medium Term Memory: {e}")
        return {"error_message": f"Failed to store objective in memory: {e}", "timestamp": datetime.now()}

    # Return updates for keys defined in JarvisState
    return {
        "objective_id": objective_id,
        "objective_description": objective_desc,
        "timestamp": datetime.now()
    }

def retrieve_context_node(state: JarvisState, memory: UnifiedMemorySystem) -> Dict[str, Any]:
    """Retrieves relevant context from memory based on the objective."""
    logger.info("NODE: retrieve_context_node")
    # Corrected: Access objective_description directly from state
    objective_desc = state.get('objective_description') 
    
    if not objective_desc:
        # Check if objective_id exists, maybe description failed upstream?
        obj_id = state.get('objective_id')
        logger.warning(f"Objective description not found in state for context retrieval (Objective ID: {obj_id}).")
        # Return empty context or error?
        return {"retrieved_knowledge": [], "conversation_history": []}
        
    # objective_desc = current_objective.description # Removed old access

    try:
        # Retrieve relevant knowledge from LTM
        # Use a query combining objective and maybe recent history if available
        query = objective_desc 
        logger.debug(f"Searching memory with query: '{query}'")
        # Search LTM and STM (adjust types and k as needed)
        search_results: List[MemoryEntry] = memory.search_memory(
            query=query, 
            memory_types=["long_term", "short_term"], 
            k_per_type=3 # Consider making k configurable
        )
        
        retrieved_knowledge_list: List[KnowledgeSnippet] = []
        conversation_history_list: List[ChatMessage] = []
        
        for entry in search_results:
            # entry is a MemoryEntry Pydantic model
            if entry.memory_type == "long_term":
                try:
                    snippet = KnowledgeSnippet(
                        content=str(entry.content), # Ensure content is string
                        source=f"memory_{entry.memory_type}",
                        score=entry.metadata.get("score"), # Assuming score might be in metadata from search
                        metadata=entry.metadata
                    )
                    retrieved_knowledge_list.append(snippet)
                except ValidationError as e:
                    logger.warning(f"Failed to validate LTM entry as KnowledgeSnippet: {e}. Entry: {entry}")
            elif entry.memory_type == "short_term":
                try:
                     # Extract role, default if missing
                    role = entry.metadata.get('speaker', 'unknown') 
                    # Ensure content is string
                    content = str(entry.content) 
                    message = ChatMessage(role=role, content=content)
                    # Add timestamp if needed for sorting later?
                    # message.timestamp = entry.timestamp 
                    conversation_history_list.append(message)
                except ValidationError as e:
                     logger.warning(f"Failed to validate STM entry as ChatMessage: {e}. Entry: {entry}")
       
        # TODO: Sort conversation history chronologically if MemorySystem doesn't guarantee order?
        # Requires timestamp on ChatMessage model and MemoryEntry providing it reliably.
        # conversation_history_list.sort(key=lambda x: getattr(x, 'timestamp', datetime.min()))

        logger.info(f"-> Retrieved {len(retrieved_knowledge_list)} knowledge items and {len(conversation_history_list)} conversation items.")
        
        # Return updates for state
        return {
            "retrieved_knowledge": retrieved_knowledge_list, # List of KnowledgeSnippet objects
            "conversation_history": conversation_history_list, # List of ChatMessage objects
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.exception(f"Error during context retrieval: {e}")
        # Return empty lists or an error message?
        return {"retrieved_knowledge": [], "conversation_history": [], "error_message": f"Context retrieval failed: {e}"}

# --- Helper function to format context for crew --- 
# (Similar to the one previously added to _decompose_objective)
def format_context_for_prompt(retrieved_knowledge: List[KnowledgeSnippet], conversation_history: List[ChatMessage]) -> str:
    """Formats retrieved knowledge and conversation history for the planning prompt.
       MODIFIED: Accepts Pydantic models, includes more context (last 3 msgs, top 3 snippets).
    """
    context_str = ""
    if conversation_history:
        context_str += "Recent Conversation History:\n"
        # Take last 3 messages
        for msg in conversation_history[-3:]:
            context_str += f"- {msg.role}: {msg.content}\n"
        context_str += "---"
        
    if retrieved_knowledge:
        context_str += "\nRelevant Retrieved Knowledge:\n"
        # Take top 3 items
        for i, item in enumerate(retrieved_knowledge[:3]):
            content = item.content
            metadata_desc = item.metadata.get('description', item.source) 
            # Longer preview, ensure it's string
            content_preview = (str(content)[:300] + '...') if len(str(content)) > 300 else str(content) 
            context_str += f"- Item {i+1} ({metadata_desc}): {content_preview}\n"
        context_str += "---"
        
    # Add logic here later to estimate token count and truncate if necessary
    # logger.debug(f"Formatted context token estimate: {estimate_tokens(context_str)}")
    # MAX_CONTEXT_TOKENS = 2000 
    # while estimate_tokens(context_str) > MAX_CONTEXT_TOKENS:
    #    # Truncation logic (remove oldest message, shortest snippet, etc.)
    #    pass 
        
    return context_str

# --- Modified plan_tasks_node to use the crew node --- 
# Placeholder for other nodes
def plan_tasks_node(state: JarvisState, planner: PlanningSystem, llm: LLMClient, skills: SkillRegistry) -> dict:
    """Uses a direct LLM call to generate a task list based on the user query and context."""
    logger.info("NODE: plan_tasks_node (Direct LLM Planning)")
    objective_id = state.get("objective_id")
    objective_description = state.get("objective_description")
    
    if not objective_id or not objective_description:
        logger.error(f"Objective ID ('{objective_id}') or Description ('{objective_description}') missing from state in plan_tasks_node.")
        return {"current_plan": None, "plan_status": "failed", "error_message": "Objective details missing for planning."}

    logger.info(f"Objective for planning: '{objective_description[:60]}...' (ID: {objective_id})")

    # Format context for the planning prompt
    retrieved_knowledge = state.get("retrieved_knowledge", [])
    conversation_history = state.get("conversation_history", [])
    context_str = format_context_for_prompt(retrieved_knowledge, conversation_history)
    logger.debug(f"Context string for LLM planning:\n{context_str}")

    # Get available skill definitions
    available_skills = skills.get_skill_definitions()
    try:
        # Add our pseudo-skill explanation to the list
        pseudo_skill = {
            "name": "reasoning_skill",
            "description": "Use this INTERNAL skill when the task requires reasoning, summarization, synthesis, or analysis based on information gathered in previous steps, and no other specific skill (like web_search or read_file) is appropriate. This skill uses the LLM's general capabilities.",
            "parameters": []
        }
        # Ensure available_skills is mutable if it comes from registry directly
        skills_list = list(available_skills) 
        skills_list.append(pseudo_skill)
        skills_json_str = json.dumps(skills_list, indent=2)
    except Exception as e:
        logger.error(f"Failed to serialize skill definitions: {e}")
        skills_json_str = "[]" # Fallback to empty list

    # --- New LLM Planning Logic ---
    SYSTEM_PROMPT = f"""
You are an expert planning AI assistant for Jarvis. Your goal is to break down the user's objective into a sequence of actionable tasks using the provided tools (skills).

Objective: {objective_description}

Context:
{context_str}

Available Skills:
{skills_json_str}

Based on the objective, context, and available skills, generate a JSON list representing the plan. Each object in the list should represent a task and have the following keys:
- "description": A clear description of the task.
- "skill": The name of the *most appropriate* skill from the list above to accomplish this task.
    - **Use 'reasoning_skill' if the task involves analyzing results from previous steps, requires general knowledge application not covered by other skills, or needs synthesis.**
    - **Use 'execute_python_file' ONLY if the user specifically asks to run an existing file or if you are certain a relevant file exists based on prior context.** Provide ONLY the relative path *within* the sandbox (e.g., 'my_script.py', NOT 'workspace_sandbox/my_script.py'). Do NOT invent filenames.
    - Choose specific skills like 'web_search' or 'read_file' when they directly match the action needed.
- "arguments": (Optional) A dictionary of arguments required by the selected skill, inferred from the task description and context. Only include if "skill" is specified. Ensure argument names match the skill definition exactly.

Return *only* the JSON list (starting with '[' and ending with ']'), without any introductory text, comments, or explanations.
"""
    USER_PROMPT = "Generate the task plan JSON."
    
    generated_tasks = [] # Initialize as empty list
    try:
        logger.info("Calling LLM for task planning...")
        llm_response_str = llm.process_with_llm(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            # Consider using a smaller model for planning if needed
            # model="llama3-8b-8192", 
            temperature=0.1, # Lower temperature for more deterministic planning
            # Add response_format={'type': 'json_object'} if supported and reliable
        )
        logger.debug(f"LLM Raw Planning Response:\n{llm_response_str}")

        # Attempt to parse the response string as JSON
        # Clean potential markdown code fences
        cleaned_response = llm_response_str.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
             cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
             cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        if cleaned_response:
            try:
                parsed_output = json.loads(cleaned_response)
                if isinstance(parsed_output, list):
                    # Basic validation: check if items are dicts
                    if all(isinstance(item, dict) for item in parsed_output):
                        generated_tasks = parsed_output
                        logger.info(f"Successfully parsed {len(generated_tasks)} tasks from LLM output.")
                    else:
                         logger.warning("LLM output was a list, but contained non-dictionary items.")
                elif isinstance(parsed_output, dict) and 'tasks' in parsed_output and isinstance(parsed_output['tasks'], list):
                    # Handle cases where LLM wraps the list in a dict
                    generated_tasks = parsed_output['tasks']
                    logger.info(f"Successfully parsed {len(generated_tasks)} tasks from LLM output (extracted from 'tasks' key).")
                else:
                    logger.warning(f"LLM output was valid JSON, but not the expected list format: {type(parsed_output)}")
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON from LLM planning response: {json_err}. Response: {cleaned_response}")
            except Exception as parse_err:
                 logger.error(f"Error processing LLM planning response after JSON parsing: {parse_err}", exc_info=True)
        else:
            logger.warning("LLM returned an empty response for planning.")

    except Exception as llm_err:
        logger.error(f"Error during LLM call for planning: {llm_err}", exc_info=True)
        # Return failure state immediately if LLM call fails
        return {"current_plan": None, "plan_status": "failed", "error_message": f"LLM planning call failed: {llm_err}"}
    # --- End New LLM Planning Logic ---

    if not generated_tasks:
        logger.warning("LLM planning returned no valid tasks.")
        return {"current_plan": None, "plan_status": "failed", "error_message": "LLM planner failed to generate valid tasks."}

    # Convert the list of task dictionaries from the LLM into Plan and Task objects
    try:
        plan_id = f"plan_{objective_id}"
        tasks_list_for_plan = []
        for i, task_dict in enumerate(generated_tasks):
            # Basic validation of task_dict structure
            if not isinstance(task_dict, dict) or "description" not in task_dict:
                logger.warning(f"Skipping invalid task structure from LLM: {task_dict}")
                continue
            tasks_list_for_plan.append(
                Task(
                    task_id=f"{plan_id}_task_{i}",
                    description=task_dict.get("description", "No description provided."),
                    skill=task_dict.get("skill"), # Skill name (string) or None
                    arguments=task_dict.get("arguments", {}), # Arguments dict or empty
                    status=TaskStatus.PENDING,
                    metadata={"plan_id": plan_id}
                    # TODO: Add dependency handling if LLM provides it (complex)
                )
            )

        # Ensure at least one valid task was created
        if not tasks_list_for_plan:
             logger.error("LLM returned tasks, but none were valid after parsing.")
             return {"current_plan": None, "plan_status": "failed", "error_message": "Failed to parse valid tasks from LLM output."}

        new_plan = Plan(
            plan_id=plan_id,
            objective_id=objective_id,
            objective_description=objective_description,
            tasks=tasks_list_for_plan
        )
        logger.info(f"-> Created Plan object with {len(new_plan.tasks)} tasks from Direct LLM Planning.")
        return {"current_plan": new_plan, "plan_status": "active", "timestamp": datetime.now()}

    except Exception as e:
        logger.error(f"Failed to convert LLM output to Plan/Task objects: {e}", exc_info=True)
        return {"current_plan": None, "plan_status": "failed", "error_message": f"Failed to process generated tasks: {e}"}

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

def execute_tool_node(state: JarvisState, executor: ExecutionSystem) -> Dict[str, Any]:
    """Executes the next tool call."""
    logger.info("NODE: execute_tool_node")
    task_data = state.get('current_task') # Get task data (likely a dict)
    if not task_data:
        logger.error("execute_tool_node called with no current_task in state.")
        return {"error_message": "Cannot execute tool: No current task found."}

    # --- Ensure task is a Pydantic Task object --- 
    if isinstance(task_data, dict):
        try:
            # Import Task model locally if needed to avoid top-level circular deps
            from jarvis.planning import Task 
            task = Task(**task_data)
            logger.debug(f"Converted task data dict to Task object: {task.task_id}")
        except Exception as e:
            logger.error(f"Failed to convert task data to Task object: {e}. Data: {task_data}", exc_info=True)
            return {"error_message": f"Invalid task data format: {e}"}
    elif hasattr(task_data, 'task_id'): # Check if it might already be a Task-like object
        task = task_data # Assume it's already the correct type or compatible
    else:
        logger.error(f"current_task data is neither a dict nor a Task object. Type: {type(task_data)}")
        return {"error_message": "Invalid task data type in state."}
    # --- End Task Conversion ---

    # Execute the task using the execution system
    execution_result = executor.execute_task(task, state=state)

    # <<< Convert result to dict EARLY >>>
    result_dict = execution_result.model_dump(mode='json')

    # Update the execution history in the state
    # <<< REMOVE History update from this node >>>
    # # <<< Add the DICT to history >>>
    # current_history = state.get('execution_history', [])
    # if not isinstance(current_history, list):
    #     logger.warning(f"execution_history was not a list ({type(current_history)}), resetting.")
    #     current_history = [] 
    # updated_history = current_history + [result_dict] 

    logger.info(f"execute_tool_node returning result - Success: {result_dict.get('success')}, Task ID: {result_dict.get('task_id')}")
    
    # <<< Return ONLY the last result dict >>>
    return {
        "last_execution_result": result_dict, 
        # "execution_history": updated_history, 
        "timestamp": datetime.now()
    }

def update_plan_node(state: JarvisState, planner: PlanningSystem) -> Dict[str, Any]:
    """Updates the plan status in the state based on the last execution result."""
    logger.info("NODE: update_plan_node (Entered)")
    # <<< Expect a DICT now >>>
    last_result_dict = state.get('last_execution_result') 

    # Check if it's a dict and has needed keys
    if not isinstance(last_result_dict, dict) or 'task_id' not in last_result_dict or 'success' not in last_result_dict:
        logger.warning(f"Invalid or missing last execution result dict found in state: {last_result_dict}")
        return {"timestamp": datetime.now()} # No update if result is bad

    current_plan = state.get('current_plan')
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
    # <<< Extract info from DICT >>>
    task_id = last_result_dict.get('task_id')
    success = last_result_dict.get('success')
    error = last_result_dict.get('error')
    output = last_result_dict.get('output') 

    # Check again after extraction, though initial check should cover basic keys
    if task_id is None or success is None:
         logger.error(f"Invalid execution result dict format: {last_result_dict}")
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
    """Determines the next step based on the plan status and last execution result."""
    logger.info("CONDITION: should_continue_condition (Entered)")
    plan_status = state.get("plan_status")
    error_message = state.get("error_message")
    # <<< Expect a DICT now >>>
    last_result_dict = state.get("last_execution_result")

    # <<< Extract success status from DICT >>>
    last_success = last_result_dict.get('success') if isinstance(last_result_dict, dict) else 'N/A'
    
    logger.info(f"Evaluating condition: plan_status='{plan_status}', last_result.success='{last_success}', error_message='{error_message}'")

    # 1. Handle critical graph errors first
    if error_message:
        logger.warning(f"-> Critical error detected ('{error_message}'), routing to handle_error.")
        return "handle_error"
        
    # 2. Handle task execution failure: Re-plan
    # <<< Check dict success value >>>
    if isinstance(last_result_dict, dict) and last_success is False:
        # Extract task_id for logging
        failed_task_id = last_result_dict.get('task_id', 'Unknown')
        logger.warning(f"-> Task {failed_task_id} failed. Routing back to plan_tasks for re-planning.")
        return "plan_tasks" # ROUTE TO RE-PLAN

    # 3. Handle plan completion (successful or failed overall)
    if plan_status == "completed" or plan_status == "failed":
        logger.info(f"-> Plan status is '{plan_status}', routing to handle_completion.")
        return "handle_completion"
        
    # 4. Continue with the active plan if last task succeeded
    if plan_status == "active":
        logger.info("-> Plan active, last task OK. Routing to get_next_task.")
        return "continue"
        
    # 5. Fallback/Default: Unexpected state, route to error
    logger.warning(f"-> Plan status is unexpected ('{plan_status}') or state unclear, routing to handle_error.")
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

def handle_error_node(state: JarvisState, llm: LLMClient) -> Dict[str, Any]:
    """Handles errors encountered during graph execution by synthesizing an error response."""
    error_message = state.get('error_message', "Unknown error")
    logger.info(f"NODE: handle_error_node: {error_message}")
    
    # Leverage the synthesis logic to create a user-facing error message
    logger.info("Synthesizing final error response...")
    synthesis_result = synthesize_final_response_node(state, llm)
    final_response = synthesis_result.get("final_response", f"An error occurred: {error_message}")

    # Ensure the error message itself is also preserved in the state if needed
    return {
        "error_message": error_message, # Keep original error
        "final_response": final_response, # Provide synthesized response
        "timestamp": datetime.now()
        }

def synthesize_final_response_node(state: JarvisState, llm: LLMClient) -> Dict[str, Any]:
    """Synthesizes the final response based on objective, results, and errors."""
    logger.info("NODE: synthesize_final_response_node")
    
    objective = state.get('objective_description', 'No objective specified.')
    task_results = state.get('execution_history', [])
    error_message = state.get('error_message')

    # Format results
    results_summary = "\nExecution Results:\n"
    if task_results:
        for i, result in enumerate(task_results[-5:]):
            # --- Add Check: Ensure result is a dictionary --- 
            if not isinstance(result, dict):
                logger.warning(f"Skipping non-dict item in execution_history (index {i}): {type(result)}")
                results_summary += f"- Task {i+1}: Invalid format in history\n"
                continue # Skip to the next item
            # -----------------------------------------------

            task_id = result.get('task_id', f'Task_{i+1}')
            # <<< Safely access nested task description >>>
            task_info = result.get('task')
            task_desc = task_info.get('description', task_id) if isinstance(task_info, dict) else task_id
            # -------------------------------------------
            success = result.get('success')
            error = result.get('error')
            output = result.get('output')

            if success is True:
                status = "Success"
                results_summary += f"- {task_desc}: {status}\n"
                # <<< ADD Output to summary if successful >>>
                if output:
                    output_preview = str(output)[:500] + ('...' if len(str(output)) > 500 else '') # Longer preview for results
                    results_summary += f"  Output: {output_preview}\n"
                else:
                    results_summary += "  Output: (No output data provided)\n"
                # <<< END Add Output >>>
            elif success is False:
                status = f"Failure ({error or 'No error details'})"
                results_summary += f"- {task_desc}: {status}\n"
            else:
                # Handle cases where success key might be missing or None
                status = "Unknown Status"
                results_summary += f"- {task_desc}: {status}\n"
    else:
        results_summary += "- No tasks executed.\n"

    if error_message:
        results_summary += "\n---"
        results_summary += f"Error Encountered: {error_message}"

    results_summary += "\n---"
    results_summary += "Instruction: Based on the objective and the execution summary (including any errors), generate a concise final response for the user."
    
    final_prompt = results_summary
    logger.debug(f"Final synthesis prompt:\n{final_prompt}")

    # --- Call LLM --- 
    try:
        final_response = llm.process_with_llm(
            prompt=final_prompt,
            # provider=..., # Optional: Specify provider if needed
            # model=...,    # Optional: Specify model if needed
            max_tokens=500, # Adjust as needed
            temperature=0.5 # Adjust as needed
        )
        logger.info(f"-> Synthesized final response: '{final_response[:100]}...'")
        
    except Exception as e:
        logger.error(f"Failed to synthesize final response using LLM: {e}", exc_info=True)
        final_response = f"Error: Could not generate the final summary. Details: {e}"
        # Update error message in state as well?
        # return {"final_response": final_response, "error_message": final_response, "timestamp": datetime.now()} 

    # --- Update State --- 
    return {"final_response": final_response, "timestamp": datetime.now()}

# ... other nodes ... 