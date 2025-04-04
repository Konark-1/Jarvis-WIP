"""
Planning system for Jarvis that integrates with memory and provides agentic planning capabilities.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from .memory.medium_term import Objective, MediumTermMemory
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from utils.logger import setup_logger
from .llm import LLMClient, LLMError, LLMCommunicationError, LLMTokenLimitError # Import LLM exceptions

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
    tasks: List[Task] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # active, completed, failed
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PlanningSystem(BaseModel):
    """Planning system that creates and manages plans to achieve objectives"""
    
    logger: Any = None
    long_term_memory: Any = None
    medium_term_memory: Any = None
    short_term_memory: Any = None
    llm: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("planning_system")
        self.long_term_memory = LongTermMemory()
        self.medium_term_memory = MediumTermMemory()
        self.short_term_memory = ShortTermMemory()
        
        # Initialize LLM client if not provided
        if not self.llm:
            try:
                from jarvis.llm import LLMClient
                self.llm = LLMClient()
            except Exception as e:
                self.logger.warning(f"Could not initialize LLM client: {e}")
    
    def create_plan(self, objective_id: str) -> Plan:
        """Create a plan to achieve an objective"""
        # Get objective details
        objective = self.medium_term_memory.get_objective(objective_id)
        if not objective:
            raise ValueError(f"Objective {objective_id} not found")
        
        # Create plan
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        plan = Plan(
            plan_id=plan_id,
            objective_id=objective_id
        )
        
        # Break down objective into tasks
        tasks = self._decompose_objective(objective.description)
        plan.tasks = tasks
        
        # Store plan in memory
        self._store_plan(plan)
        
        return plan
    
    def _decompose_objective(self, objective_description: str) -> List[Task]:
        """Break down an objective into tasks using LLM-based planning"""
        # Use long-term memory to find similar objectives and their successful task breakdowns
        similar_objectives = self.long_term_memory.retrieve_knowledge(
            f"objective decomposition for: {objective_description}"
        )
        
        # Create a task ID prefix
        task_id_prefix = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Use LLM to decompose the objective if available
        if self.llm:
            try:
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
                
                # Add context from similar objectives if available
                similar_context = ""
                if similar_objectives:
                    similar_context = "Here are examples of similar objectives and their task breakdowns:\n"
                    for i, obj in enumerate(similar_objectives[:3]):
                        similar_context += f"Example {i+1}: {obj['content']}\n"
                        if "tasks" in obj["metadata"]:
                            similar_context += "Tasks:\n"
                            # Ensure metadata tasks are dictionaries before accessing
                            if isinstance(obj["metadata"]["tasks"], list):
                                for task_meta in obj["metadata"]["tasks"]:
                                    if isinstance(task_meta, dict) and "description" in task_meta:
                                        similar_context += f"- {task_meta['description']}\n"
                            elif isinstance(obj["metadata"]["tasks"], str):
                                # Handle case where tasks might be stored as a string
                                similar_context += f" {obj['metadata']['tasks']}\n"
                        similar_context += "\n"
                
                prompt = f"""
                Objective: {objective_description}
                
                {similar_context}
                
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
                                    "phase": task_data.get("phase", "execution")
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

        # Fall back to rule-based decomposition if LLM fails, returns invalid data, or is not available
        self.logger.info("Falling back to rule-based task decomposition.")
        dynamic_tasks = []
        task_index = 1
        
        # Analyze the objective to create appropriate tasks
        keywords = objective_description.lower().split()
        
        # Default tasks for planning phase
        dynamic_tasks.append(Task(
            task_id=f"{task_id_prefix}_{task_index}",
            description="Analyze requirements and constraints",
            metadata={"type": "planning", "phase": "preparation"}
        ))
        task_index += 1
        
        # Information gathering phase
        if any(kw in keywords for kw in ['find', 'search', 'locate', 'research', 'information']):
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Gather information and research",
                metadata={"type": "research", "phase": "information_gathering"}
            ))
            task_index += 1
        
        # Development phase
        if any(kw in keywords for kw in ['create', 'develop', 'build', 'implement', 'code']):
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Design solution architecture",
                metadata={"type": "planning", "phase": "design"}
            ))
            task_index += 1
            
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Implement core functionality",
                metadata={"type": "execution", "phase": "implementation"}
            ))
            task_index += 1
        
        # Organization phase
        if any(kw in keywords for kw in ['organize', 'sort', 'categorize', 'arrange']):
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Scan and analyze target items",
                metadata={"type": "execution", "phase": "analysis"}
            ))
            task_index += 1
            
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Create organization scheme",
                metadata={"type": "planning", "phase": "organization"}
            ))
            task_index += 1
            
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Apply organization changes",
                metadata={"type": "execution", "phase": "implementation"}
            ))
            task_index += 1
        
        # Problem-solving phase
        if any(kw in keywords for kw in ['solve', 'fix', 'repair', 'debug', 'troubleshoot']):
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Diagnose problem root cause",
                metadata={"type": "analysis", "phase": "diagnosis"}
            ))
            task_index += 1
            
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Develop solution approach",
                metadata={"type": "planning", "phase": "solution_design"}
            ))
            task_index += 1
            
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Implement and test solution",
                metadata={"type": "execution", "phase": "implementation"}
            ))
            task_index += 1
        
        # If no specific tasks could be created, add generic tasks
        if len(dynamic_tasks) <= 1:
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Break down objective into specific tasks",
                metadata={"type": "planning", "phase": "decomposition"}
            ))
            task_index += 1
            
            dynamic_tasks.append(Task(
                task_id=f"{task_id_prefix}_{task_index}",
                description="Execute primary task",
                metadata={"type": "execution", "phase": "implementation"}
            ))
            task_index += 1
        
        # Add validation tasks
        dynamic_tasks.append(Task(
            task_id=f"{task_id_prefix}_{task_index}",
            description="Verify completion and quality",
            metadata={"type": "validation", "phase": "verification"}
        ))
        task_index += 1
        
        # Store this decomposition in long-term memory for future reference
        self._store_decomposition(objective_description, dynamic_tasks)
        
        return dynamic_tasks
    
    def _store_plan(self, plan: Plan):
        """Store a plan in memory"""
        # Store in medium-term memory
        self.medium_term_memory.add_progress(
            f"Created plan {plan.plan_id} with {len(plan.tasks)} tasks",
            metadata={"plan_id": plan.plan_id}
        )
        
        # Store in long-term memory for future reference
        self.long_term_memory.add_knowledge(
            f"plan_{plan.plan_id}",
            f"Plan for objective {plan.objective_id}: {plan.tasks}",
            metadata={"type": "plan", "objective_id": plan.objective_id}
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
        """Get a plan from memory"""
        # Try to get from long-term memory
        plan_data = self.long_term_memory.retrieve_knowledge(f"plan_{plan_id}")
        if plan_data:
            return Plan(**plan_data[0]["metadata"])
        return None
    
    def update_task_status(self, plan_id: str, task_id: str, status: str, error: Optional[str] = None):
        """Update the status of a task"""
        plan = self._get_plan(plan_id)
        if not plan:
            return
        
        # Find and update task
        task = next((t for t in plan.tasks if t.task_id == task_id), None)
        if task:
            task.status = status
            if status == "completed":
                task.completed_at = datetime.now()
            elif status == "failed":
                task.error_count += 1
                if error:
                    task.metadata["last_error"] = error
            
            # Update plan status
            if all(t.status == "completed" for t in plan.tasks):
                plan.status = "completed"
                plan.completed_at = datetime.now()
            elif any(t.status == "failed" for t in plan.tasks):
                plan.status = "failed"
            
            # Store updated plan
            self._store_plan(plan)
    
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
    
    def _store_decomposition(self, objective_description: str, tasks: List[Task]):
        """Store a task decomposition in long-term memory for future reference"""
        self.long_term_memory.add_knowledge(
            f"objective_decomposition_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            f"Decomposition for: {objective_description}",
            metadata={
                "objective": objective_description,
                "tasks": [task.dict() for task in tasks],
                "timestamp": datetime.now().isoformat()
            }
        ) 