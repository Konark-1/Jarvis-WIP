"""
Core agentic implementation of Jarvis that integrates planning, execution, and memory systems.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from jarvis.planning import PlanningSystem
from jarvis.execution import ExecutionSystem
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
            from memory.unified_memory import UnifiedMemorySystem
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
            
            # Check for feedback command
            if text.lower().startswith("feedback"):
                 return self._handle_feedback_command(text)
            
            # Default to task execution
            return self._execute_task(text)
        
        except Exception as e:
            error_time = datetime.now()
            self.logger.error(f"Error processing input '{text}': {e}", exc_info=True)
            self.state.error_count += 1
            self.state.last_error = f"[{error_time.isoformat()}] {type(e).__name__}: {str(e)}"
            
            # Attempt LLM-based diagnosis
            diagnosis = "Diagnosis unavailable."
            if self.llm:
                try:
                    context = self.memory_system.assemble_context(query=f"Error analysis: {text}")
                    system_prompt = """
                    You are an error analysis assistant for the Jarvis agent.
                    An error occurred while processing user input.
                    Analyze the error message, the user input, and the provided context.
                    Provide a brief diagnosis of the likely cause (e.g., planning failure, skill error, memory issue, external API problem).
                    Suggest a potential recovery action if possible (e.g., retry, rephrase input, check API key, report bug).
                    Format as: "Diagnosis: [Your diagnosis]. Suggestion: [Your suggestion]"
                    """
                    prompt = f"""
                    Error Type: {type(e).__name__}
                    Error Message: {str(e)}
                    User Input: {text}
                    Context:
                    {context}
                    
                    Analyze the error and provide a diagnosis and suggestion.
                    """
                    diagnosis = self.llm.process_with_llm(prompt, system_prompt, temperature=0.4, max_tokens=150).strip()
                    self.logger.info(f"LLM Error Diagnosis: {diagnosis}")
                    # Store diagnosis in long-term memory?
                    self.memory_system.add_memory(
                        content=f"Error processing input '{text}': {type(e).__name__}: {str(e)}",
                        memory_type="long_term",
                        metadata={
                            "type": "error_log",
                            "timestamp": error_time.isoformat(),
                            "error_type": type(e).__name__,
                            "diagnosis": diagnosis
                        },
                        importance=0.6
                    )
                except Exception as diag_err:
                    self.logger.error(f"LLM diagnosis failed: {diag_err}")
            
            self._save_state()
            # Return a user-friendly message, potentially including the suggestion
            user_message = f"I encountered an error ({type(e).__name__}). "
            if "Suggestion:" in diagnosis:
                user_message += diagnosis.split("Suggestion:")[-1].strip()
            else:
                 user_message += "Please try again or rephrase your request."
            return user_message
    
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
            # Use LLM to process the input and generate a response, using assembled context
            if self.llm:
                # <<< ADDED CALL TO assemble_context >>>
                assembled_context = self.memory_system.assemble_context(query=text)
                # <<< Use assembled_context in the prompt below, replacing the manually built 'context' variable >>>

                # --- Keep old context building for now, replace usage below --- 
                # context = "\n".join([f"{i.speaker}: {i.text}" for i in recent_interactions]) # Original context

                # Use LLM to generate a response
                try:
                    system_prompt = """
                    You are Jarvis, an agentic AI assistant. 
                    Generate a helpful, accurate, and concise response to the user's input.
                    Use the provided conversation context and relevant memories to maintain continuity and provide informed answers.
                    """
                    
                    prompt = f"""
                    Context:
                    {assembled_context} # Use assembled context here
                    
                    User Input: {text}
                    
                    Jarvis Response:
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
    def _review_and_refine_objectives(self):
        """Periodically review active objectives for relevance and progress."""
        self.logger.info("Starting periodic objective review...")
        try:
            active_objectives = self.memory_system.medium_term.search_objectives(status="active")
            if not active_objectives:
                self.logger.info("No active objectives to review.")
                return

            self.logger.info(f"Reviewing {len(active_objectives)} active objectives.")

            # <<< ADDED CALL TO assemble_context >>>
            assembled_context = self.memory_system.assemble_context(query="Review active objectives for relevance and progress")
            # <<< Use assembled_context in the prompt below, replacing the manually built 'context' variable >>>

            # 1. Gather Context (Similar to autonomous creation, but focused on review)
            context = "Objective Review Context:\n" # Original context building starts here
            # ... (original context building logic remains) ...

            # 2. Prepare Objectives for LLM Review (remains the same)
            objectives_summary = "\nActive Objectives for Review:\n"
            for i, obj_data in enumerate(active_objectives):
                obj = Objective(**obj_data) # Recreate Objective object for easier access
                objectives_summary += f"Objective {i+1} (ID: {obj.objective_id}):\n"
                objectives_summary += f"  Description: {obj.description}\n"
                objectives_summary += f"  Status: {obj.status}\n"
                objectives_summary += f"  Created: {obj.created_at}\n"
                objectives_summary += f"  Priority: {obj.priority}\n"
                # Add plan status if available and relevant
                # if obj.objective_id == self.state.current_objective_id and self.state.current_plan_id:
                #     plan_status = self.planning_system.get_plan_status(self.state.current_plan_id)
                #     objectives_summary += f"  Plan Progress: ... \n" # Simplified for now
                objectives_summary += "\n"

            # 3. Define LLM Prompt for Review
            system_prompt = """
            You are Jarvis, an AI assistant reviewing its active objectives for relevance, progress, and potential refinement.
            Analyze the provided context (agent state, recent interactions, memory reflection) and the list of active objectives.

            For EACH objective listed, provide a concise assessment and recommendation. Consider:
            - Is the objective still relevant based on recent interactions and insights?
            - Is progress being made (implicitly, based on context)?
            - Should the objective be marked completed, failed, paused, or continued?
            - Should the objective description or priority be refined?
            - Is there a need for a *new* objective based on the review?

            Format your response as a JSON object containing a list called "objective_reviews".
            Each item in the list should correspond to an objective reviewed and have the following structure:
            {
                "objective_id": "<original_objective_id>",
                "assessment": "<your concise assessment>",
                "recommendation": "<continue|complete|fail|pause|refine|create_new>",
                "new_description": "<updated description if recommendation is refine, else null>",
                "new_priority": <new priority 1-5 if recommendation is refine, else null>,
                "new_objective_proposal": "<text for a new objective if recommendation is create_new, else null>"
            }
            Return ONLY the JSON object.
            """

            prompt = f"""
            **Objective Review Task**

            Context:
            {assembled_context} # Use assembled context here

            Active Objectives:
            {objectives_summary}
            
            **Instructions:**
            Review each objective based on the context and provide your assessment and recommendation in the specified JSON format.
            """

            # 4. Get Review from LLM (remains the same)
            # 4. Get Review from LLM
            if not self.llm:
                self.logger.warning("LLM client not available for objective review.")
                return

            try:
                response = self.llm.process_with_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.5, # Lower temp for more deterministic review
                    max_tokens=1000 # Allow for multiple reviews
                )
                review_data = json.loads(response)

                if "objective_reviews" not in review_data or not isinstance(review_data["objective_reviews"], list):
                    raise ValueError("LLM response missing 'objective_reviews' list.")

                self.logger.info(f"Received {len(review_data['objective_reviews'])} objective reviews from LLM.")

                # 5. Process Recommendations
                for review in review_data["objective_reviews"]:
                    obj_id = review.get("objective_id")
                    recommendation = review.get("recommendation")
                    assessment = review.get("assessment")
                    self.logger.info(f"Review for {obj_id}: Assessment='{assessment}', Recommendation='{recommendation}'")

                    if not obj_id or not recommendation:
                        self.logger.warning(f"Skipping invalid review item: {review}")
                        continue

                    # Find the original objective data
                    original_obj_data = next((o for o in active_objectives if o['objective_id'] == obj_id), None)
                    if not original_obj_data:
                        self.logger.warning(f"Objective {obj_id} from review not found in active objectives.")
                        continue

                    metadata_update = original_obj_data.get('metadata', {})
                    description_update = original_obj_data.get('description')

                    needs_update = False
                    if recommendation == "complete":
                        metadata_update['status'] = "completed"
                        metadata_update['completed_at'] = datetime.now().isoformat()
                        needs_update = True
                    elif recommendation == "fail":
                        metadata_update['status'] = "failed"
                        metadata_update['reason'] = assessment # Store assessment as reason
                        needs_update = True
                    elif recommendation == "pause":
                        metadata_update['status'] = "paused"
                        needs_update = True
                    elif recommendation == "refine":
                        if review.get("new_description"):
                            description_update = review["new_description"]
                            needs_update = True
                        if review.get("new_priority") is not None:
                            metadata_update['priority'] = review["new_priority"]
                            needs_update = True
                        metadata_update['status'] = "active" # Ensure it remains active after refining
                        metadata_update['last_refined_at'] = datetime.now().isoformat()

                    elif recommendation == "create_new":
                        new_objective_text = review.get("new_objective_proposal")
                        if new_objective_text:
                            self.logger.info(f"LLM proposed creating new objective during review: {new_objective_text}")
                            # Call creation logic (could potentially create duplicates if not careful)
                            # Consider adding a check or deferring creation
                            self.memory_system.medium_term.create_objective(new_objective_text)
                        else:
                            self.logger.warning("LLM recommended create_new but provided no objective text.")

                    if needs_update:
                        self.logger.info(f"Updating objective {obj_id} based on review recommendation '{recommendation}'.")
                        self.memory_system.medium_term.update_objective(
                            objective_id=obj_id,
                            description=description_update,
                            metadata=metadata_update
                        )
                        # If the currently active objective's status changed, update agent state
                        if obj_id == self.state.current_objective_id and metadata_update.get('status') != 'active':
                            self.logger.info(f"Current objective {obj_id} is no longer active. Clearing state.")
                            self.state.current_objective_id = None
                            self.state.current_plan_id = None
                            self._save_state()

            except json.JSONDecodeError as json_err:
                self.logger.error(f"Failed to parse JSON response from LLM during objective review: {json_err}\nResponse: {response[:500]}...")
            except Exception as review_err:
                self.logger.error(f"Error processing objective review: {review_err}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error during periodic objective review: {e}", exc_info=True) 

    # --- Self-Assessment Method ---
    def _perform_self_assessment(self):
        """Periodically assess agent performance and identify improvement areas."""
        self.logger.info("Performing self-assessment...")
        if not self.llm:
            self.logger.warning("LLM client not available for self-assessment.")
            return

        try:
            # 1. Gather Context for Assessment
            context = "Agent Self-Assessment Context:\n"
            # Agent State
            context += f"- Current Status: {'Active' if self.state.is_active else 'Inactive'}\n"
            context += f"- Current Objective ID: {self.state.current_objective_id}\n"
            context += f"- Current Plan ID: {self.state.current_plan_id}\n"
            context += f"- Recent Error Count: {self.state.error_count}\n"
            if self.state.last_error:
                context += f"- Last Error Message: {self.state.last_error}\n"
            context += f"- Last Maintenance: {self.state.last_maintenance}\n"

            # Memory Stats
            try:
                mem_stats = self.memory_system.get_memory_stats()
                context += "\nMemory Statistics:\n"
                for key, value in mem_stats.items():
                    context += f"- {key.replace('_', ' ').title()}: {value}\n"
            except Exception as mem_stat_err:
                self.logger.warning(f"Could not retrieve memory stats: {mem_stat_err}")

            # Recent Task Performance (Needs Planning/Execution systems to track this)
            # Placeholder: Assuming we could get recent task results
            # recent_tasks = self.planning_system.get_recent_task_results(10) 
            # if recent_tasks:
            #     context += "\nRecent Task Performance:\n"
            #     success_count = sum(1 for task in recent_tasks if task.success)
            #     context += f"- Success Rate (last 10): {success_count * 10}%\n"

            # Recent Objective Progress (Needs MediumTermMemory to track status changes)
            # completed_objectives = self.memory_system.medium_term.search_objectives(status="completed", n_results=5)
            # failed_objectives = self.memory_system.medium_term.search_objectives(status="failed", n_results=5)
            # context += f"- Recently Completed Objectives: {len(completed_objectives)}\n"
            # context += f"- Recently Failed Objectives: {len(failed_objectives)}\n"

            # Memory Reflection
            reflection = self.memory_system.reflect_on_memories("overall agent performance and health", self.llm)
            if reflection and "error" not in reflection:
                context += "\nMemory Reflection Summary:\n"
                context += f"- Insights: {reflection.get('insights', [])}\n"
                context += f"- Patterns: {reflection.get('patterns', [])}\n"

            # 2. Define LLM Prompt
            system_prompt = """
            You are Jarvis, performing a self-assessment of your own performance and operational health.
            Analyze the provided context, which includes your current state, memory statistics, 
            (potentially) recent task/objective performance, and a reflection on your memory contents.

            Identify:
            1. Key strengths or areas where you are performing well.
            2. Key weaknesses, recurring issues, or areas needing improvement.
            3. Potential causes for any identified issues (e.g., planning failures, skill gaps, memory issues).
            4. Concrete suggestions for improvement or adjustments (e.g., refine planning, learn new skill, optimize memory).

            Format your response as a JSON object:
            {
                "assessment_summary": "<A brief overall summary of your current state>",
                "strengths": ["<Identified strength 1>", "<Identified strength 2>", ...],
                "weaknesses": ["<Identified weakness 1>", "<Identified weakness 2>", ...],
                "potential_causes": ["<Potential cause 1>", "<Potential cause 2>", ...],
                "improvement_suggestions": ["<Suggestion 1>", "<Suggestion 2>", ...]
            }
            Return ONLY the JSON object.
            """

            prompt = f"""
            **Agent Self-Assessment Data**

            {context}
            
            **Task:**
            Perform a self-assessment based on the data above and return the results in the specified JSON format.
            """

            # 3. Get Assessment from LLM
            response = self.llm.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4, # Moderate temp for balanced analysis
                max_tokens=800 
            )
            assessment_data = json.loads(response)

            # 4. Store Assessment in Long-Term Memory
            assessment_summary = assessment_data.get("assessment_summary", "Assessment completed.")
            self.memory_system.add_memory(
                content=assessment_summary,
                memory_type="long_term",
                metadata={
                    "type": "self_assessment",
                    "timestamp": datetime.now().isoformat(),
                    "assessment_details": assessment_data # Store full details in metadata
                },
                importance=0.8 # Self-assessments are quite important
            )
            self.logger.info(f"Self-assessment complete. Summary: {assessment_summary}")
            # Optional: Trigger actions based on suggestions?
            # self._act_on_assessment(assessment_data)

        except json.JSONDecodeError as json_err:
            self.logger.error(f"Failed to parse JSON response from LLM during self-assessment: {json_err}\nResponse: {response[:500]}...")
        except Exception as e:
            self.logger.error(f"Error during self-assessment: {e}", exc_info=True) 

    # --- Feedback Handling ---
    def _handle_feedback_command(self, text: str) -> str:
        """Handles explicit user feedback on the last action/result."""
        feedback_text = text.lower().replace("feedback", "").strip()
        sentiment = "neutral"
        if feedback_text in ["good", "success", "great", "correct", "yes"]:
            sentiment = "positive"
        elif feedback_text in ["bad", "failure", "wrong", "incorrect", "no"]:
            sentiment = "negative"
        
        self.logger.info(f"Received feedback: {sentiment} ('{feedback_text}')")
        
        # Try to associate feedback with the last interaction or task
        last_interaction = self.memory_system.short_term.get_recent_interactions(1)
        if last_interaction:
            last_action_id = last_interaction[0].interaction_id
            # Could also try to get last completed task ID from state or planning system
        else:
            last_action_id = "unknown"

        # Store feedback in long-term memory
        self.memory_system.add_memory(
            content=f"User feedback: {sentiment}",
            memory_type="long_term",
            metadata={
                "type": "user_feedback",
                "timestamp": datetime.now().isoformat(),
                "sentiment": sentiment,
                "raw_text": feedback_text,
                "associated_action_id": last_action_id 
            },
            importance=0.9 # User feedback is highly important
        )

        return f"Thank you for the feedback! I've recorded it as {sentiment}." 