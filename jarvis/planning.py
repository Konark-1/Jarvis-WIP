"""
Planning system for Jarvis that integrates with memory and provides agentic planning capabilities.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from jarvis.memory.medium_term import Objective, MediumTermMemory
from jarvis.memory.long_term import LongTermMemory
from jarvis.memory.short_term import ShortTermMemory
from utils.logger import setup_logger

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
                
                Return the tasks as a JSON array with this structure:
                [
                    {
                        "description": "Task description",
                        "phase": "planning|research|execution|validation",
                        "dependencies": []  # List of task indices that must be completed first
                    }
                ]
                """
                
                # Add context from similar objectives if available
                similar_context = ""
                if similar_objectives:
                    similar_context = "Here are examples of similar objectives and their task breakdowns:\n"
                    for i, obj in enumerate(similar_objectives[:3]):
                        similar_context += f"Example {i+1}: {obj['content']}\n"
                        if "tasks" in obj["metadata"]:
                            similar_context += "Tasks:\n"
                            for j, task in enumerate(obj["metadata"]["tasks"]):
                                similar_context += f"- {task['description']}\n"
                            similar_context += "\n"
                
                prompt = f"""
                Objective: {objective_description}
                
                {similar_context}
                
                Create a logical sequence of tasks to accomplish this objective.
                Consider what steps would be needed, their dependencies, and different phases of work.
                """
                
                response = self.llm.process_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.7
                )
                
                # Try to parse the JSON response
                try:
                    tasks_data = json.loads(response)
                    if isinstance(tasks_data, list) and len(tasks_data) > 0:
                        # Convert to Task objects
                        dynamic_tasks = []
                        for i, task_data in enumerate(tasks_data):
                            # Convert dependencies from indices to task IDs
                            dependencies = []
                            for dep_idx in task_data.get("dependencies", []):
                                if 0 <= dep_idx < i:
                                    dependencies.append(f"{task_id_prefix}_{dep_idx + 1}")
                            
                            dynamic_tasks.append(Task(
                                task_id=f"{task_id_prefix}_{i + 1}",
                                description=task_data["description"],
                                dependencies=dependencies,
                                metadata={
                                    "type": task_data.get("type", "task"),
                                    "phase": task_data.get("phase", "execution")
                                }
                            ))
                        
                        # Store this decomposition for future reference
                        self._store_decomposition(objective_description, dynamic_tasks)
                        return dynamic_tasks
                except Exception as e:
                    self.logger.error(f"Error parsing LLM task decomposition: {e}")
                    # Fall back to rule-based decomposition
        
        # Fall back to rule-based decomposition if LLM fails or is not available
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