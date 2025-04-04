import logging
from functools import partial
from langgraph.graph import StateGraph, END

# State
from jarvis.state import JarvisState

# Nodes
from jarvis.graph_nodes import (
    understand_query_node,
    plan_tasks_node,
    get_next_task_node,
    execute_tool_node,
    update_plan_node,
    retrieve_context_node,
    handle_error_node,
    handle_plan_completion_node,
    synthesize_final_response_node,
    should_continue_condition,
    task_dispatch_condition
)

# Tool/System dependencies (for binding)
from jarvis.llm import LLMClient
from jarvis.planning import PlanningSystem
from jarvis.execution import ExecutionSystem
from jarvis.memory.unified_memory import UnifiedMemorySystem
from jarvis.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

def build_graph(llm_client: LLMClient, planning_system: PlanningSystem, execution_system: ExecutionSystem, memory_system: UnifiedMemorySystem, skill_registry: SkillRegistry):
    """Builds the LangGraph StateGraph for Jarvis."""
    logger.info("Building Jarvis agent graph...")

    # Initialize the graph with the JarvisState
    workflow = StateGraph(JarvisState)

    # --- Bind system components to node functions --- 
    # Use functools.partial to pass the instantiated systems to the node functions
    bound_understand_query = partial(understand_query_node, llm=llm_client, memory=memory_system)
    bound_retrieve_context = partial(retrieve_context_node, memory=memory_system)
    bound_plan_tasks = partial(plan_tasks_node, planner=planning_system, llm=llm_client, skills=skill_registry)
    bound_get_next_task = partial(get_next_task_node, planner=planning_system)
    bound_execute_tool = partial(execute_tool_node, executor=execution_system)
    bound_update_plan = partial(update_plan_node, planner=planning_system)
    bound_synthesize_final = partial(synthesize_final_response_node, llm=llm_client)
    bound_handle_error = partial(handle_error_node)

    # --- Add nodes to the graph ---
    logger.debug("Adding nodes...")
    workflow.add_node("understand_query", bound_understand_query)
    workflow.add_node("retrieve_context", bound_retrieve_context)
    workflow.add_node("plan_tasks", bound_plan_tasks)
    workflow.add_node("get_next_task", bound_get_next_task)
    workflow.add_node("execute_tool", bound_execute_tool)
    workflow.add_node("update_plan", bound_update_plan)
    workflow.add_node("handle_error", bound_handle_error)
    workflow.add_node("handle_completion", handle_plan_completion_node) # Basic completion handler
    # workflow.add_node("synthesize_final", bound_synthesize_final) # Optional: Add synthesis as separate node

    # --- Define edges --- 
    logger.debug("Adding edges...")
    # 1. Entry point
    workflow.set_entry_point("understand_query")

    # 2. Understand -> Retrieve Context
    workflow.add_edge("understand_query", "retrieve_context")

    # 3. Retrieve Context -> Plan Tasks
    workflow.add_edge("retrieve_context", "plan_tasks")

    # 4. Plan -> Conditional Check
    workflow.add_conditional_edges(
        "plan_tasks",
        should_continue_condition, # Function to decide next step
        {
            "continue": "get_next_task",    # If plan is active
            "handle_completion": "handle_completion", # If plan completed/failed immediately (unlikely)
            "handle_error": "handle_error"     # If error during planning
        }
    )

    # 5. Get Task -> Conditional Dispatch
    workflow.add_conditional_edges(
        "get_next_task",
        task_dispatch_condition, # Function to check if task exists
        {
            "execute": "execute_tool",         # Task found
            "handle_completion": "handle_completion", # No task found (plan complete/stalled)
            "handle_error": "handle_error"      # Error finding task
        }
    )

    # 6. Execute -> Update Plan
    workflow.add_edge("execute_tool", "update_plan")

    # 7. Update Plan -> Conditional Check (Loop back)
    workflow.add_conditional_edges(
        "update_plan",
        should_continue_condition, # Re-check plan status after update
        {
            "continue": "get_next_task",    # Plan still active, get next task
            "handle_completion": "handle_completion", # Plan now completed or failed
            "handle_error": "handle_error"     # Error during update
        }
    )
    
    # --- Define End Points ---
    # Nodes that lead to the end of the graph execution
    workflow.add_edge("handle_error", END)
    workflow.add_edge("handle_completion", END) # Simple end for now
    # If using separate synthesis:
    # workflow.add_edge("handle_completion", "synthesize_final")
    # workflow.add_edge("synthesize_final", END)

    # --- Compile the graph --- 
    logger.info("Compiling graph...")
    app = workflow.compile()
    logger.info("Graph compiled successfully.")
    
    return app 