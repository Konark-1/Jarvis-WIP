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
            error_type_name = type(e).__name__
            error_message = str(e)
            self.logger.error(f"Error processing input '{text}': {error_type_name}: {error_message}", exc_info=True)
            self.state.error_count += 1
            last_error_str = f"[{error_time.isoformat()}] {error_type_name}: {error_message}"
            self.state.last_error = last_error_str
            
            # Attempt LLM-based diagnosis
            diagnosis = "Diagnosis unavailable."
            suggestion = "Please try again or rephrase your request."
            if self.llm:
                try:
                    # Use assemble_context for better context gathering
                    context = self.memory_system.assemble_context(query=f"Error analysis for input: {text}, Error: {last_error_str}")
                    
                    system_prompt = """
                    You are an error analysis assistant for the Jarvis agent.
                    An error occurred while processing user input or executing a task.
                    Analyze the error type, message, user input, and provided context.
                    Provide a brief diagnosis of the likely cause (e.g., planning failure, skill error, memory issue, external API problem, LLM hallucination, invalid input).
                    Suggest a concrete, actionable recovery strategy for the *agent* or the *user*. Examples:
                    - Agent Actions: retry, retry_with_different_params, use_fallback_skill <skill_name>, replan_objective, reset_context
                    - User Actions: rephrase_input, check_api_key, provide_more_info, wait_and_retry
                    - Other: report_bug, none
                    Format as: "Diagnosis: [Your diagnosis]. Suggestion: [recovery_action_keyword] | [Explanation/Details for user/agent]"
                    Example: "Diagnosis: Skill parsing failed due to ambiguous input. Suggestion: rephrase_input | Could you please rephrase the request more clearly?"
                    Example: "Diagnosis: Web browse timed out. Suggestion: retry | Trying to access the URL again."
                    Example: "Diagnosis: Planning failed for complex objective. Suggestion: replan_objective | Let me try breaking down that objective again."
                    """
                    prompt = f"""
                    Error Type: {error_type_name}
                    Error Message: {error_message}
                    User Input: {text}
                    Context:
                    {context}
                    
                    Analyze the error and provide diagnosis and suggestion in the specified format.
                    """
                    raw_diagnosis = self.llm.process_with_llm(prompt, system_prompt, temperature=0.3, max_tokens=200).strip()
                    self.logger.info(f"LLM Error Diagnosis: {raw_diagnosis}")
                    
                    # Parse diagnosis and suggestion
                    diagnosis_part = raw_diagnosis.split("Suggestion:")[0].replace("Diagnosis:", "").strip()
                    suggestion_part = raw_diagnosis.split("Suggestion:")[-1].strip()
                    suggestion_keyword = suggestion_part.split("|")[0].strip().lower()
                    suggestion_details = suggestion_part.split("|")[-1].strip() if "|" in suggestion_part else "Please try again."
                    
                    diagnosis = diagnosis_part # Store the parsed diagnosis
                    suggestion = suggestion_details # Store the user-facing message or agent action explanation
                    
                    # Store detailed log
                    self.memory_system.add_memory(
                        content=f"Error processing input '{text}': {last_error_str}",
                        memory_type="long_term",
                        metadata={
                            "type": "error_log",
                            "timestamp": error_time.isoformat(),
                            "error_type": error_type_name,
                            "diagnosis": diagnosis,
                            "suggestion_keyword": suggestion_keyword,
                            "suggestion_details": suggestion_details
                        },
                        importance=0.7 # Errors are fairly important
                    )

                    # --- Attempt Automatic Recovery based on keyword ---
                    if suggestion_keyword == "retry":
                        self.logger.info("Attempting automatic retry based on LLM suggestion.")
                        self.state.error_count = max(0, self.state.error_count - 1) # Decrement error count for retry attempt
                        self._save_state()
                        # Re-call process_input. Be careful of infinite loops!
                        # Add a retry counter or mechanism if needed.
                        # return self.process_input(text) # Potential loop risk!
                        return f"{suggestion} (Retrying...)" # Safer: Inform user and agent can retry internally next cycle
                    elif suggestion_keyword == "reset_context":
                         self.logger.info("Attempting context reset based on LLM suggestion.")
                         self.reset_context() # Calls consolidate and clears STM
                         self.state.error_count = 0 # Reset error count after context reset
                         self._save_state()
                         return f"{suggestion} I've reset my short-term context. Please try your request again."
                    elif suggestion_keyword == "replan_objective" and self.state.current_objective_id:
                         self.logger.info(f"Attempting replan for objective {self.state.current_objective_id} based on LLM suggestion.")
                         try:
                              # Invalidate current plan if one exists
                              if self.state.current_plan_id:
                                   self.planning_system.update_plan_status(self.state.current_plan_id, "failed", reason="Replanning triggered by error")
                              # Trigger replanning
                              plan = self.planning_system.create_plan(self.state.current_objective_id)
                              self.state.current_plan_id = plan.plan_id
                              self.state.error_count = 0 # Reset error count after replan
                              self._save_state()
                              return f"{suggestion} I've created a new plan for the objective."
                         except Exception as replan_err:
                              self.logger.error(f"Replanning failed: {replan_err}")
                              suggestion = f"I tried to replan, but encountered another error: {replan_err}. Please review the objective or try something else."
                    # Add other recovery actions here (e.g., use_fallback_skill)

                except Exception as diag_err:
                    self.logger.error(f"LLM diagnosis or recovery attempt failed: {diag_err}")
                    # Fall back to default suggestion if diagnosis fails
                    suggestion = "Please try again or rephrase your request."
            
            # Save state after potential modifications
            self._save_state()
            # Return the (potentially updated) user-facing suggestion
            return f"I encountered an error ({error_type_name}). {suggestion}"
    
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
        """
        Handles executing the next task of the current plan OR processing general user input.
        Prioritizes executing the current plan if one is active.
        """
        try:
            # --- Proactive Plan Execution --- 
            if self.state.current_objective_id and self.state.current_plan_id:
                self.logger.info(f"Active plan {self.state.current_plan_id} for objective {self.state.current_objective_id}. Checking for next task.")
                next_task = self.planning_system.get_next_task(self.state.current_plan_id)

                if next_task:
                    self.logger.info(f"Proceeding with task {next_task.task_id}: '{next_task.description}'")
                    # Update task status to in_progress
                    self.planning_system.update_task_status(
                        self.state.current_plan_id,
                        next_task.task_id,
                        "in_progress"
                    )

                    # Execute the task
                    # Optional: Provide recent context/user input to the execution system if relevant?
                    # execution_context = {"user_input": text, "recent_interactions": ...}
                    result = self.execution_system.execute_task(next_task)

                    # Update task status based on result
                    final_status = "completed" if result.success else "failed"
                    self.planning_system.update_task_status(
                        self.state.current_plan_id,
                        next_task.task_id,
                        final_status,
                        result.error
                    )
                    self.memory_system.medium_term.add_progress(
                        f"{final_status.capitalize()} task: {next_task.description}",
                        metadata={
                            "task_id": next_task.task_id,
                            "plan_id": self.state.current_plan_id,
                            "objective_id": self.state.current_objective_id,
                            "success": result.success,
                            "error": result.error,
                            "output_summary": str(result.output)[:200] # Store a summary
                        }
                    )

                    if result.success:
                        self.logger.info(f"Task {next_task.task_id} completed successfully.")
                        # Check if the plan is now complete
                        plan_status = self.planning_system.get_plan_status(self.state.current_plan_id)
                        if plan_status["status"] == "completed":
                            self.logger.info(f"Plan {self.state.current_plan_id} completed for objective {self.state.current_objective_id}.")
                            # Mark objective as completed in memory
                            self.memory_system.medium_term.update_objective(
                                self.state.current_objective_id,
                                status="completed"
                            )
                            # Clear agent state for objective/plan
                            completed_objective_id = self.state.current_objective_id
                            self.state.current_objective_id = None
                            self.state.current_plan_id = None
                            self._save_state()
                            return f"Objective '{completed_objective_id}' completed successfully! What should I work on next?"
                        else:
                            # Plan not finished, provide task output and indicate readiness for next
                             return f"Task '{next_task.description}' completed. Output: {str(result.output)[:500]}... Ready for the next step."
                    else:
                        # Task failed
                        self.logger.error(f"Task {next_task.task_id} failed: {result.error}")
                        # Check if plan should be marked as failed
                        plan_status = self.planning_system.get_plan_status(self.state.current_plan_id)
                        if plan_status["status"] == "failed":
                             self.logger.error(f"Plan {self.state.current_plan_id} failed for objective {self.state.current_objective_id}.")
                             # Mark objective as failed
                             self.memory_system.medium_term.update_objective(
                                 self.state.current_objective_id,
                                 status="failed"
                             )
                             failed_objective_id = self.state.current_objective_id
                             self.state.current_objective_id = None
                             self.state.current_plan_id = None
                             self._save_state()
                             return f"Objective '{failed_objective_id}' failed due to task error: {result.error}. Please review or set a new objective."
                        else:
                             # Plan not failed yet, return task error
                             return f"Task '{next_task.description}' failed: {result.error}. Trying to proceed or awaiting feedback."
                else:
                    # No more tasks in the plan, but it wasn't marked completed? Should not happen ideally.
                    self.logger.warning(f"Plan {self.state.current_plan_id} has no next task but isn't marked completed. Marking objective complete.")
                    self.memory_system.medium_term.update_objective(self.state.current_objective_id, status="completed")
                    self.state.current_objective_id = None
                    self.state.current_plan_id = None
                    self._save_state()
                    return "Current objective completed as no further tasks were found in the plan. Ready for new instructions."

            # --- General Input Processing (No active plan or plan finished) --- 
            self.logger.info(f"No active plan or current plan step completed. Processing input: '{text}'")
            # Use LLM to process the input and generate a response, using assembled context
            if self.llm:
                # Search memory for context related to the user input
                memory_results = self.memory_system.search_memory(query=text, memory_types=["short_term", "medium_term", "long_term"])
                # Format context for LLM
                assembled_context = "Relevant context for response generation:\n"
                for i, mem in enumerate(memory_results[:5]): # Limit context size
                    # Simple formatting, adjust as needed
                    content_str = str(mem.content)
                    if len(content_str) > 150:
                        content_str = content_str[:150] + "..."
                    assembled_context += f"- [{mem.memory_type} @ {mem.timestamp.isoformat()}] {content_str}\n"

                # Use LLM to generate a response
                try:
                    system_prompt = """
                    You are Jarvis, an agentic AI assistant.
                    Generate a helpful, accurate, and concise response to the user's input.
                    Use the provided conversation history and relevant memories to maintain continuity and provide informed answers.
                    If the user input seems like a new task or objective, suggest creating one explicitly.
                    """

                    prompt = f"""
                    Relevant Context:
                    {assembled_context.strip()}

                    User Input: {text}

                    Jarvis Response:
                    """

                    response = self.llm.process_with_llm(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.7
                    )

                    # Add the interaction to memory
                    self.memory_system.add_memory(content=response, memory_type="short_term", metadata={"speaker": "assistant"})

                    return response
                except Exception as e:
                    self.logger.error(f"Error generating response with LLM: {e}")
                    # Fallback response if LLM fails
                    return f"I had trouble processing that request with my language model ({type(e).__name__}). Could you please rephrase or try again?"

            # Fallback if LLM is not available
            return f"I received your input: '{text}'. My language model is currently unavailable. Please set an objective or check status."

        except Exception as e:
            # Catch errors during the task execution/response generation flow
            error_time = datetime.now()
            self.logger.error(f"Error during task execution or response generation for input '{text}': {e}", exc_info=True)
            self.state.error_count += 1
            last_error_str = f"[{error_time.isoformat()}] {type(e).__name__}: {str(e)}"
            self.state.last_error = last_error_str
            self._save_state()
            # Store error in memory
            self.memory_system.add_memory(
                content=f"Agent error processing input '{text}': {last_error_str}",
                memory_type="long_term",
                metadata={"type": "error_log", "timestamp": error_time.isoformat(), "error_type": type(e).__name__},
                importance=0.7
            )
            return f"I encountered an internal error ({type(e).__name__}) while processing your request. I've logged the issue. Please try again."
    
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
        """Periodically review active objectives using LLM for relevance, progress, and potential refinement."""
        self.logger.info("Starting periodic objective review...")
        if not self.llm:
            self.logger.warning("LLM client not available. Skipping objective review.")
            return

        try:
            active_objectives = self.memory_system.medium_term.search_objectives(status="active")
            if not active_objectives:
                self.logger.info("No active objectives to review.")
                return

            self.logger.info(f"Reviewing {len(active_objectives)} active objectives.")

            # 1. Assemble Context for Review
            # Combine recent interactions, memory reflection, and current agent state
            context_query = "Current agent status and recent activity for objective review"
            assembled_context = self.memory_system.assemble_context(query=context_query, max_tokens=3000) # Limit context size

            # Include basic agent state in the context
            state_summary = f"Agent State: Errors={self.state.error_count}, Last Error={self.state.last_error or 'None'}, Current Objective ID={self.state.current_objective_id or 'None'}"
            full_context = f"{state_summary}\n\nRelevant Memory & Interactions:\n{assembled_context}"

            # 2. Prepare Objectives Summary for LLM
            objectives_summary = "\nActive Objectives Under Review:\n"
            objective_map = {} # Store details for easier lookup after LLM response
            for i, obj_data in enumerate(active_objectives):
                # Ensure obj_data is a dict with expected keys
                if not isinstance(obj_data, dict) or 'objective_id' not in obj_data:
                    self.logger.warning(f"Skipping invalid objective data during review: {obj_data}")
                    continue
                obj_id = obj_data['objective_id']
                objective_map[obj_id] = obj_data
                objectives_summary += f"Objective {i+1} (ID: {obj_id}):\n"
                objectives_summary += f"  Description: {obj_data.get('description', 'N/A')}\n"
                objectives_summary += f"  Status: {obj_data.get('metadata', {}).get('status', 'unknown')}\n"
                objectives_summary += f"  Created: {obj_data.get('metadata', {}).get('created_at', 'N/A')}\n"
                objectives_summary += f"  Priority: {obj_data.get('metadata', {}).get('priority', 'N/A')}\n"
                # Optionally add plan progress summary here if feasible
                objectives_summary += "\n"

            # 3. Define LLM Prompt for Review
            system_prompt = """
            You are an AI assistant's strategic review component. Your task is to analyze the agent's current context (state, recent activity, memory) and its list of active objectives.

            For EACH objective listed, provide a concise assessment and a specific recommendation based *only* on the provided context and objective details.
            Consider:
            - Relevance: Is the objective still aligned with recent activity or stated goals in the context?
            - Progress: Does the context suggest progress is being made, stalled, or blocked? (Infer if possible)
            - Viability: Does the objective still seem achievable?
            - Need for Change: Should the objective be continued, completed, paused, marked as failed, or refined (description/priority)?

            **Output Format:** Respond ONLY with a JSON object containing a single key "objective_reviews", which is a list. Each item in the list MUST correspond to an objective reviewed and MUST have the following structure:
            {
                "objective_id": "<original_objective_id_string>",
                "assessment": "<your_concise_assessment_string>",
                "recommendation": "<continue|complete|fail|pause|refine>",
                "refined_description": "<updated_description_string_if_recommendation_is_refine_else_null>",
                "refined_priority": <integer_priority_1_to_5_if_recommendation_is_refine_else_null>
            }
            Ensure the `objective_id` matches exactly one from the input list. Make a recommendation for every objective provided.
            Do not propose creating new objectives here; focus solely on reviewing the existing active ones.
            Return ONLY the valid JSON object.
            """

            prompt = f"""
            **Objective Review Task**

            **Agent Context & Recent Activity:**
            {full_context}

            **Active Objectives for Review:**
            {objectives_summary}

            **Instructions:**
            Review each active objective based on the context. Provide your assessment and recommendation for *every* objective in the specified JSON format.
            """

            # 4. Get Review from LLM
            self.logger.debug("Sending objective review prompt to LLM...")
            response_content = self.llm.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3, # Low temp for focused analysis
                max_tokens=1500 # Allow ample space for reviews
            )

            # 5. Parse and Process Recommendations
            try:
                # Clean potential markdown
                if response_content.strip().startswith("```json"):
                    response_content = response_content.strip()[7:-3].strip()
                elif response_content.strip().startswith("```"):
                     response_content = response_content.strip()[3:-3].strip()

                review_data = json.loads(response_content)

                if "objective_reviews" not in review_data or not isinstance(review_data["objective_reviews"], list):
                    raise ValueError("LLM response missing 'objective_reviews' list or is not a list.")

                self.logger.info(f"Received {len(review_data['objective_reviews'])} objective reviews from LLM.")

                processed_ids = set()
                for review in review_data["objective_reviews"]:
                    if not isinstance(review, dict):
                        self.logger.warning(f"Skipping invalid review item (not a dict): {review}")
                        continue

                    obj_id = review.get("objective_id")
                    recommendation = review.get("recommendation")
                    assessment = review.get("assessment", "No assessment provided.")
                    processed_ids.add(obj_id)

                    self.logger.info(f"LLM Review for {obj_id}: Assessment='{assessment}', Recommendation='{recommendation}'")

                    if not obj_id or not recommendation:
                        self.logger.warning(f"Skipping review item with missing id or recommendation: {review}")
                        continue
                    if obj_id not in objective_map:
                        self.logger.warning(f"Objective {obj_id} from LLM review not found in the initial active list. Skipping.")
                        continue

                    original_obj_data = objective_map[obj_id]
                    metadata_update = original_obj_data.get('metadata', {}).copy() # Make a copy to modify
                    description_update = original_obj_data.get('description')
                    needs_update = False
                    new_status = metadata_update.get('status', 'active') # Default to keeping status

                    if recommendation == "complete":
                        new_status = "completed"
                        metadata_update['completed_at'] = datetime.now().isoformat()
                        metadata_update['status_reason'] = f"Marked complete by periodic review: {assessment}"
                        needs_update = True
                    elif recommendation == "fail":
                        new_status = "failed"
                        metadata_update['status_reason'] = f"Marked failed by periodic review: {assessment}"
                        needs_update = True
                    elif recommendation == "pause":
                        new_status = "paused"
                        metadata_update['status_reason'] = f"Paused by periodic review: {assessment}"
                        needs_update = True
                    elif recommendation == "refine":
                        new_desc = review.get("refined_description")
                        new_priority = review.get("refined_priority")
                        if new_desc and isinstance(new_desc, str):
                            self.logger.info(f"Refining description for {obj_id}")
                            description_update = new_desc
                            needs_update = True
                        if new_priority is not None and isinstance(new_priority, int) and 1 <= new_priority <= 5:
                            self.logger.info(f"Refining priority for {obj_id} to {new_priority}")
                            metadata_update['priority'] = new_priority
                            needs_update = True
                        if needs_update:
                             metadata_update['last_refined_at'] = datetime.now().isoformat()
                             metadata_update['status_reason'] = f"Refined by periodic review: {assessment}"
                        # Ensure status remains active or becomes active if refined from paused/etc.
                        new_status = "active"
                    elif recommendation == "continue":
                        # No change needed, but log the assessment
                        self.logger.info(f"Objective {obj_id} recommended to continue.")
                        # Optionally update a 'last_reviewed' timestamp
                        metadata_update['last_reviewed_at'] = datetime.now().isoformat()
                        metadata_update['status_reason'] = f"Reviewed and continued: {assessment}"
                        # needs_update = True # Only set if metadata actually changed
                    else:
                         self.logger.warning(f"Unknown recommendation '{recommendation}' for objective {obj_id}. Ignoring.")

                    metadata_update['status'] = new_status # Apply the decided status

                    if needs_update:
                        self.logger.info(f"Applying update to objective {obj_id}: Status={new_status}")
                        self.memory_system.medium_term.update_objective(
                            objective_id=obj_id,
                            description=description_update,
                            metadata=metadata_update,
                            status=new_status # Pass status explicitly if update_objective supports it
                        )
                        # If the *currently active* objective's status changed from active, update agent state
                        if obj_id == self.state.current_objective_id and new_status != 'active':
                            self.logger.info(f"Agent's current objective {obj_id} status changed to '{new_status}'. Clearing agent state.")
                            self.state.current_objective_id = None
                            self.state.current_plan_id = None
                            self._save_state()
                
                # Check if any objectives were *not* reviewed by the LLM
                missing_review_ids = set(objective_map.keys()) - processed_ids
                if missing_review_ids:
                     self.logger.warning(f"LLM review did not include assessments for objectives: {missing_review_ids}")

            except json.JSONDecodeError as json_err:
                self.logger.error(f"Failed to parse JSON response from LLM during objective review: {json_err}\nResponse: {response_content[:500]}...")
            except ValueError as val_err:
                self.logger.error(f"LLM objective review response validation failed: {val_err}\nResponse: {response_content[:500]}...")
            except Exception as review_err:
                self.logger.error(f"Error processing LLM objective review: {review_err}", exc_info=True)

        except Exception as e:
            self.logger.error(f"General error during periodic objective review: {e}", exc_info=True)
    
    # --- Self-Assessment Method ---
    def _perform_self_assessment(self):
        """Periodically assess agent performance using LLM and store insights."""
        self.logger.info("Performing self-assessment...")
        if not self.llm:
            self.logger.warning("LLM client not available. Skipping self-assessment.")
            return
        if not hasattr(self.memory_system, 'reflect_on_memories') or not hasattr(self.memory_system, 'get_memory_stats'):
             self.logger.warning("Memory system does not support required methods (reflect_on_memories, get_memory_stats) for self-assessment.")
             return

        try:
            # 1. Gather Context for Assessment
            self.logger.debug("Gathering context for self-assessment.")
            assessment_context = "Agent Self-Assessment Context:\n"
            
            # Basic Agent State
            assessment_context += f"- Agent State: Active={self.state.is_active}, Errors={self.state.error_count}, ObjectiveID={self.state.current_objective_id or 'None'}, PlanID={self.state.current_plan_id or 'None'}\n"
            if self.state.last_error:
                assessment_context += f"- Last Error Recorded: {self.state.last_error}\n"

            # Memory Statistics
            try:
                mem_stats = self.memory_system.get_memory_stats()
                assessment_context += "\nMemory Statistics:\n"
                for key, value in mem_stats.items():
                    assessment_context += f"- {key.replace('_', ' ').title()}: {value}\n"
            except Exception as mem_stat_err:
                self.logger.warning(f"Could not retrieve memory stats during assessment: {mem_stat_err}")
                assessment_context += "- Memory Statistics: Unavailable\n"

            # Recent Performance Indicators (from Memory Logs/Progress)
            # Search for recent errors, completed tasks/objectives, and feedback
            performance_query = "recent errors OR recent task completions OR recent objective status changes OR user feedback"
            performance_memories = self.memory_system.search_memory(query=performance_query, memory_types=["long_term", "medium_term"])
            
            assessment_context += "\nRecent Performance Indicators (from Memory):\n"
            if performance_memories:
                 for i, mem in enumerate(performance_memories[:10]): # Limit number of examples
                    content_str = str(mem.content)
                    if len(content_str) > 100:
                         content_str = content_str[:100] + "..."
                    mem_type = mem.metadata.get('type', mem.memory_type)
                    assessment_context += f"- [{mem_type} @ {mem.timestamp.isoformat()}] {content_str}\n"
            else:
                 assessment_context += "- No specific performance indicators found in recent memory.\n"

            # Memory Reflection (Use the dedicated method)
            # Ensure the llm_client is passed if reflect_on_memories requires it
            reflection_query = "overall agent performance patterns, successes, and failures"
            reflection_result = self.memory_system.reflect_on_memories(reflection_query)
            if reflection_result and "error" not in reflection_result:
                assessment_context += "\nMemory Reflection Summary:\n"
                assessment_context += f"- Summary: {reflection_result.get('summary', 'N/A')}\n"
                assessment_context += f"- Insights: {reflection_result.get('insights', [])}\n"
                assessment_context += f"- Patterns: {reflection_result.get('patterns', [])}\n"
            else:
                self.logger.warning(f"Memory reflection failed or returned error during assessment: {reflection_result.get('error', 'Unknown error')}")
                assessment_context += "- Memory Reflection: Unavailable or failed\n"

            # 2. Define LLM Prompt for Assessment
            system_prompt = """
            You are Jarvis, performing a critical self-assessment of your own performance, reliability, and efficiency.
            Analyze the provided context, which includes your current state, memory statistics, recent performance indicators (errors, completions, feedback), and a summary reflection on your memory contents.

            Your goal is to identify concrete strengths, weaknesses, potential root causes for issues, and actionable suggestions for improvement.

            Specifically, analyze and report on:
            1.  **Strengths:** Identify areas or specific instances of successful operation or positive outcomes.
            2.  **Weaknesses:** Identify recurring errors, failed objectives, negative feedback patterns, or areas of inefficiency.
            3.  **Potential Causes:** For identified weaknesses, hypothesize the likely root causes (e.g., flawed planning logic, specific skill malfunction, inadequate context retrieval, API issues, unclear objectives).
            4.  **Improvement Suggestions:** Propose specific, actionable steps the agent system could take to address weaknesses or enhance strengths (e.g., "Refine planning prompt to request dependency checks", "Improve error handling in WebSkill.browse_url", "Increase context retrieved for task execution", "Request user clarification more often for ambiguous goals").

            **Output Format:** Respond ONLY with a JSON object with the following keys:
            {
                "assessment_summary": "<A brief (1-2 sentence) overall summary of the current assessment.>",
                "strengths": ["<List of identified strengths.>"],
                "weaknesses": ["<List of identified weaknesses or recurring issues.>"],
                "potential_causes": ["<List of hypothesized causes for weaknesses.>"],
                "improvement_suggestions": ["<List of concrete, actionable improvement suggestions.>"],
                "confidence_score": <A float between 0.0 and 1.0 indicating confidence in this assessment, based on data quality/quantity>
            }
            Ensure all lists contain string elements. Return ONLY the valid JSON object.
            """

            prompt = f"""
            **Agent Self-Assessment Data**
            Timestamp: {datetime.now().isoformat()}

            {assessment_context}

            **Task:**
            Perform a self-assessment based *only* on the data provided above. Adhere strictly to the required JSON output format.
            """

            # 3. Get Assessment from LLM
            self.logger.debug("Sending self-assessment prompt to LLM.")
            response_content = self.llm.process_with_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4, # Balanced temperature for analysis
                max_tokens=1000 # Allow space for detailed analysis
            )

            # 4. Parse and Store Assessment
            try:
                # Clean potential markdown
                if response_content.strip().startswith("```json"):
                    response_content = response_content.strip()[7:-3].strip()
                elif response_content.strip().startswith("```"):
                     response_content = response_content.strip()[3:-3].strip()

                assessment_data = json.loads(response_content)

                # Basic validation of structure
                required_keys = {"assessment_summary", "strengths", "weaknesses", "potential_causes", "improvement_suggestions", "confidence_score"}
                if not required_keys.issubset(assessment_data.keys()):
                    raise ValueError(f"LLM self-assessment response missing expected keys. Got: {assessment_data.keys()}")
                
                assessment_summary = assessment_data.get("assessment_summary", "Assessment completed.")
                self.logger.info(f"Self-assessment complete. Summary: {assessment_summary}")
                self.logger.debug(f"Full assessment details: {assessment_data}")
                
                # Store the full assessment in Long-Term Memory
                memory_id = self.memory_system.add_memory(
                    content=assessment_summary, # Use summary as main content
                    memory_type="long_term",
                    metadata={
                        "type": "self_assessment",
                        "timestamp": datetime.now().isoformat(),
                        "assessment_details": json.dumps(assessment_data) # Store full JSON details as string
                    },
                    importance=0.8 # Self-assessments are important
                )
                self.logger.info(f"Stored self-assessment in long-term memory (ID: {memory_id})")

                # Optional: Automatically create tasks based on high-confidence suggestions?
                # This requires careful implementation to avoid loops or bad tasks.
                # Example placeholder:
                # if assessment_data.get("confidence_score", 0.0) > 0.7:
                #     for suggestion in assessment_data.get("improvement_suggestions", [])[:1]: # Limit auto-task creation
                #         if "Refine" in suggestion or "Improve" in suggestion:
                #              self.memory_system.medium_term.create_objective(f"Implement improvement suggestion: {suggestion}", priority=2, metadata={"source": "self_assessment"})

            except json.JSONDecodeError as json_err:
                self.logger.error(f"Failed to parse JSON response from LLM during self-assessment: {json_err}. Raw response:\n{response_content}")
            except ValueError as val_err:
                self.logger.error(f"LLM self-assessment response validation failed: {val_err}. Raw response:\n{response_content}")
            except Exception as parse_err:
                 self.logger.error(f"Error processing self-assessment response: {parse_err}", exc_info=True)

        except Exception as e:
            self.logger.error(f"General error during self-assessment: {e}", exc_info=True)
    
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