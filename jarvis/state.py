from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError

from jarvis.planning import Plan, Task # Assuming Plan and Task are defined here

# --- Input Validation Model ---
class UserInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000) # Basic length validation

    @validator('query')
    def sanitize_query(cls, v):
        # Basic sanitization example (can be expanded)
        # Remove leading/trailing whitespace is already done by strip()
        # Potentially remove harmful characters or patterns if needed
        # For now, just ensure it's not excessively long
        if len(v) > 2000:
            raise ValueError("Query exceeds maximum length of 2000 characters.")
        # Add more complex sanitization/validation if required
        return v

# --- State TypedDict ---
class JarvisState(TypedDict):
    """Represents the overall state of the Jarvis workflow.

    Designed for use with LangGraph, holding information passed between nodes.
    """
    # Input/Objective
    original_query: str
    objective_id: Optional[str]
    objective_description: Optional[str]

    # Planning
    current_plan: Optional[Plan]
    plan_status: Optional[str] # e.g., "active", "completed", "failed"
    available_skills: Optional[List[Dict[str, Any]]] # Skill definitions for planning/execution

    # Execution
    current_task: Optional[Task]
    task_queue: Optional[List[Task]] # Or manage queue within Plan object?
    last_execution_result: Optional[Dict[str, Any]] # Simplified ExecutionResult
    execution_history: Optional[List[Dict[str, Any]]] # History of task executions

    # Synthesis/Output
    final_response: Optional[str]
    intermediate_results: Optional[List[Any]]

    # Context & Memory
    conversation_history: Optional[List[Dict[str, str]]] # Simplified STM
    retrieved_knowledge: Optional[List[Dict[str, Any]]] # Relevant LTM results
    # Add specific context items as needed

    # Agent/Workflow State
    error_message: Optional[str]
    error_count: int
    current_node: Optional[str] # Track position in the graph
    timestamp: datetime 