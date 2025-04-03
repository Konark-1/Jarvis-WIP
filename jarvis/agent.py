"""
Core agentic implementation of Jarvis that integrates planning, execution, and memory systems.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from .planning import PlanningSystem
from .execution import ExecutionSystem
from utils.logger import setup_logger

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
    
    memory_system: Any = None  # UnifiedMemorySystem
    logger: Any = None
    planning_system: Any = None
    execution_system: Any = None
    state: AgentState = None
    llm: Any = None  # LLMClient
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("jarvis_agent")
        
        # Initialize components if not provided
        if not self.memory_system:
            from jarvis.memory.unified_memory import UnifiedMemorySystem
            self.memory_system = UnifiedMemorySystem()
            
        self.planning_system = PlanningSystem()
        self.execution_system = ExecutionSystem()
        
        # Initialize LLM if not provided
        if not self.llm:
            from jarvis.llm import LLMClient
            self.llm = LLMClient()
            
        # Initialize state
        self.state = AgentState()
        
        # Load previous state if available
        self._load_state()
    
    def _load_state(self):
        """Load previous agent state from memory"""
        try:
            state_data = self.memory_system.long_term.retrieve_knowledge("agent_state")
            if state_data and state_data[0]["metadata"]:
                # Convert string values back to appropriate types
                metadata = state_data[0]["metadata"]
                
                # Parse string-encoded values
                if "context" in metadata and isinstance(metadata["context"], str):
                    if metadata["context"] == "{}":
                        metadata["context"] = {}
                    else:
                        try:
                            metadata["context"] = json.loads(metadata["context"])
                        except:
                            metadata["context"] = {}
                
                # Convert empty strings back to None
                for key, value in metadata.items():
                    if value == "":
                        metadata[key] = None
                
                self.state = AgentState(**metadata)
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            # Keep default state in case of errors
    
    def _save_state(self):
        """Save current agent state to memory"""
        # Convert state dict to ensure all values are scalar
        state_dict = self.state.dict()
        
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
            
            # Default to task execution
            return self._execute_task(text)
        
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            self.state.error_count += 1
            self.state.last_error = str(e)
            self._save_state()
            return f"I encountered an error: {str(e)}"
    
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
        """Create an autonomous objective based on patterns in user interactions"""
        try:
            if not recent_queries:
                return None
                
            # Use LLM to analyze patterns and create an objective
            system_prompt = """
            You are Jarvis, an agentic assistant analyzing user queries to identify patterns and create objectives.
            Based on the user's recent queries, identify a key pattern or need and formulate a specific, actionable objective.
            The objective should be clear, concise, and directly address an underlying user need.
            Return ONLY the objective text without any explanation or additional text.
            """
            
            # Format recent queries for analysis
            query_history = "\n".join([f"- {q}" for q in recent_queries])
            prompt = f"""
            Here are the user's recent queries:
            
            {query_history}
            
            Based on these queries, create a single clear objective that addresses a pattern or need.
            """
            
            # Get objective from LLM
            objective_text = None
            if self.llm:
                try:
                    objective_text = self.llm.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.7
                    ).strip()
                except Exception as e:
                    self.logger.error(f"Error generating objective with LLM: {e}")
            
            # Fall back to simple pattern matching if LLM fails
            if not objective_text:
                # Derive an objective based on common themes in recent queries
                common_themes = []
                
                # Look for common coding-related requests
                if any("code" in q.lower() for q in recent_queries):
                    common_themes.append("coding")
                
                # Look for organization-related requests
                if any("organize" in q.lower() for q in recent_queries):
                    common_themes.append("organization")
                
                # Look for research-related requests
                if any("find" in q.lower() or "search" in q.lower() for q in recent_queries):
                    common_themes.append("research")
                
                # Create an objective based on identified themes
                if common_themes:
                    theme = common_themes[0]  # Take the most common theme
                    if theme == "coding":
                        objective_text = "Improve code organization and implement best practices"
                    elif theme == "organization":
                        objective_text = "Create an efficient organization system for user files"
                    elif theme == "research":
                        objective_text = "Develop better information retrieval capabilities"
            
            # Create the objective if we've determined a valid one
            if objective_text:
                objective_id = self.memory_system.medium_term.create_objective(objective_text)
                self.state.current_objective_id = objective_id
                self._save_state()
                
                # Create a plan for this objective
                self.planning_system.create_plan(objective_id)
                
                return objective_text
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating autonomous objective: {e}")
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
        """Execute a task based on user input"""
        try:
            # If we have a current objective and plan, check if this input relates to them
            if self.state.current_objective_id and self.state.current_plan_id:
                # Get the current objective
                objective = self.memory_system.medium_term.get_objective(self.state.current_objective_id)
                
                if objective:
                    # Get the next task in the plan
                    next_task = self.planning_system.get_next_task(self.state.current_plan_id)
                    
                    if next_task:
                        # Update task status
                        self.planning_system.update_task_status(
                            self.state.current_plan_id,
                            next_task.task_id,
                            "in_progress"
                        )
                        
                        # Use LLM to determine if the user input is related to the current task
                        is_related = False
                        if self.llm:
                            try:
                                # Formulate a prompt to check relevance
                                prompt = f"""
                                Current task: {next_task.description}
                                
                                User input: "{text}"
                                
                                Is the user input related to the current task? Answer with 'yes' or 'no'.
                                """
                                
                                response = self.llm.process_with_llm(
                                    prompt=prompt,
                                    temperature=0.2
                                ).strip().lower()
                                
                                is_related = "yes" in response
                            except Exception as e:
                                self.logger.error(f"Error checking task relevance with LLM: {e}")
                        
                        if is_related:
                            # Execute the task using the execution system
                            result = self.execution_system.execute_task(next_task)
                            
                            # Update task status based on result
                            if result.success:
                                self.planning_system.update_task_status(
                                    self.state.current_plan_id,
                                    next_task.task_id,
                                    "completed"
                                )
                                
                                # Save to memory
                                self.memory_system.medium_term.add_progress(
                                    f"Completed task: {next_task.description}",
                                    metadata={
                                        "task_id": next_task.task_id,
                                        "plan_id": self.state.current_plan_id,
                                        "objective_id": self.state.current_objective_id
                                    }
                                )
                                
                                return f"Task completed: {result.output}"
                            else:
                                self.planning_system.update_task_status(
                                    self.state.current_plan_id,
                                    next_task.task_id,
                                    "failed",
                                    result.error
                                )
                                
                                # If we've failed too many times, move to the next task
                                if next_task.error_count >= next_task.max_retries:
                                    self.memory_system.medium_term.add_progress(
                                        f"Failed to complete task after {next_task.error_count} attempts: {next_task.description}",
                                        metadata={
                                            "task_id": next_task.task_id,
                                            "plan_id": self.state.current_plan_id,
                                            "objective_id": self.state.current_objective_id,
                                            "error": result.error
                                        }
                                    )
                                
                                return f"Task failed: {result.error}"
            
            # If we reach here, either we don't have a current plan or the input isn't related to it
            # Use LLM to process the input and generate a response
            if self.llm:
                # Get relevant context from memory
                recent_interactions = self.memory_system.short_term.get_recent_interactions(5)
                context = "\n".join([f"{i.speaker}: {i.text}" for i in recent_interactions])
                
                # Use LLM to generate a response
                try:
                    system_prompt = """
                    You are Jarvis, an agentic AI assistant. 
                    Generate a helpful, accurate, and concise response to the user's input.
                    Use the conversation context to maintain continuity.
                    """
                    
                    prompt = f"""
                    Conversation context:
                    {context}
                    
                    User: {text}
                    
                    Jarvis:
                    """
                    
                    response = self.llm.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.7
                    )
                    
                    # Add the interaction to memory
                    self.memory_system.short_term.add_interaction("assistant", response)
                    
                    return response
                except Exception as e:
                    self.logger.error(f"Error generating response with LLM: {e}")
            
            # Fallback if LLM is not available or fails
            return f"I'm processing your request: '{text}'. Let me work on that for you."
            
        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
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
    
    def reset_context(self):
        """Reset the context window to handle error accumulation"""
        self.logger.info("Resetting context due to error threshold")
        
        # Save current objective and plan
        current_objective = self.state.current_objective_id
        current_plan = self.state.current_plan_id
        
        # Reset short-term memory
        self.memory_system.short_term.reset()
        
        # Add a note about the reset
        self.memory_system.short_term.add_interaction(
            "system",
            "Context window has been reset due to error accumulation."
        )
        
        # Remember the current objective and plan
        if current_objective:
            objective = self.memory_system.medium_term.get_objective(current_objective)
            if objective:
                self.memory_system.short_term.add_interaction(
                    "system",
                    f"Current objective: {objective.description}"
                )
                
        if current_plan:
            plan_status = self.planning_system.get_plan_status(current_plan)
            if 'error' not in plan_status:
                self.memory_system.short_term.add_interaction(
                    "system",
                    f"Current plan status: {plan_status['status']}"
                )
        
        return "Context has been reset due to error accumulation."
    
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