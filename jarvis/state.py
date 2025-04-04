from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError

from jarvis.planning import Plan, Task # Assuming Plan and Task are defined here
from jarvis.execution import ExecutionResult # <<< ADDED Import

# --- Pydantic Models for State Components ---
class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""
    role: str # e.g., "user", "assistant"
    content: str

class KnowledgeSnippet(BaseModel):
    """Represents a piece of knowledge retrieved from memory."""
    content: str
    source: str # e.g., "memory_ltm", "memory_stm", "web_search"
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
    # task_queue: Optional[List[Task]] # <<< REMOVED
    last_execution_result: Optional[ExecutionResult] # <<< Changed to Pydantic model
    execution_history: Optional[List[ExecutionResult]] # <<< Changed to List of Pydantic models

    # Synthesis/Output
    final_response: Optional[str]
    # intermediate_results: Optional[List[Any]] # <<< REMOVED

    # Context & Memory
    conversation_history: Optional[List[ChatMessage]] # <<< UPDATED
    retrieved_knowledge: Optional[List[KnowledgeSnippet]] # <<< UPDATED
    # Add specific context items as needed

    # Agent/Workflow State
    error_message: Optional[str]
    error_count: int
    current_node: Optional[str] # Track position in the graph
    timestamp: datetime 