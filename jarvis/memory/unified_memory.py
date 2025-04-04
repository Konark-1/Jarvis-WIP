"""
Unified Memory System for Jarvis
-------------------
This module provides a unified memory system that integrates short-term, 
medium-term, and long-term memory with embedding-based semantic search.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import os
import json
import time
from pydantic import BaseModel, Field

from jarvis.memory.short_term import ShortTermMemory
from jarvis.memory.medium_term import MediumTermMemory
from jarvis.memory.long_term import LongTermMemory
from utils.logger import setup_logger
from jarvis.llm import LLMClient, LLMError, LLMCommunicationError, LLMTokenLimitError

class MemoryEntry(BaseModel):
    """Base class for memory entries"""
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0 to 1.0
    memory_type: str  # "long_term", "medium_term", or "short_term"

class MemorySearchResult(BaseModel):
    """A search result from memory"""
    memory_type: str  # short_term, medium_term, long_term
    content: Any
    source_id: str
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryConsolidationRule(BaseModel):
    """Rule for when to consolidate memories from one type to another"""
    source_type: str  # short_term, medium_term
    target_type: str  # medium_term, long_term
    age_threshold: float  # seconds
    importance_threshold: float  # 0-1
    repetition_threshold: int  # number of times seen

class UnifiedMemorySystem(BaseModel):
    """
    Unified memory system that integrates short-term, medium-term, and long-term memory
    with embedding-based semantic search and memory consolidation.
    """
    short_term: ShortTermMemory = Field(default_factory=ShortTermMemory)
    medium_term: MediumTermMemory = Field(default_factory=MediumTermMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)
    logger: Any = None
    llm_client: Optional[LLMClient] = None  # Add LLM client to the system
    
    consolidation_rules: List[MemoryConsolidationRule] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = setup_logger("unified_memory")
        
        # Set up default consolidation rules if none provided
        if not self.consolidation_rules:
            self._setup_default_consolidation_rules()
        
        # Initialize memory cache
        self._cache: Dict[str, MemoryEntry] = {}
        self._cache_size = 1000
    
    def _setup_default_consolidation_rules(self):
        """Set up default memory consolidation rules"""
        # Short-term to medium-term: 1 hour old, 0.7 importance, seen 3 times
        self.consolidation_rules.append(
            MemoryConsolidationRule(
                source_type="short_term",
                target_type="medium_term",
                age_threshold=3600,  # 1 hour
                importance_threshold=0.7,
                repetition_threshold=3
            )
        )
        
        # Medium-term to long-term: 1 week old, 0.8 importance, seen 5 times
        self.consolidation_rules.append(
            MemoryConsolidationRule(
                source_type="medium_term",
                target_type="long_term",
                age_threshold=604800,  # 1 week
                importance_threshold=0.8,
                repetition_threshold=5
            )
        )
    
    def add_memory(self, content: Any, memory_type: str, 
                  metadata: Optional[Dict[str, Any]] = None,
                  importance: float = 0.5) -> str:
        """Add a memory entry to the appropriate memory system"""
        try:
            # Create memory entry
            entry = MemoryEntry(
                content=content,
                metadata=metadata or {},
                importance=importance,
                memory_type=memory_type
            )
            
            # Add to appropriate memory system
            if memory_type == "long_term":
                memory_id = self.long_term.add_knowledge(
                    content,
                    metadata.get("description", ""),
                    metadata
                )
            elif memory_type == "medium_term":
                memory_id = self.medium_term.create_objective(
                    content,
                    metadata
                )
            else:  # short_term
                memory_id = self.short_term.add_interaction(
                    metadata.get("speaker", "system"),
                    content,
                    metadata
                )
            
            # Add to cache
            self._add_to_cache(memory_id, entry)
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            raise
    
    def retrieve_memory(self, memory_id: str, memory_type: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID and type"""
        try:
            # Check cache first
            cache_key = f"{memory_type}:{memory_id}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # Retrieve from appropriate memory system
            if memory_type == "long_term":
                result = self.long_term.retrieve_knowledge(memory_id)
                if result:
                    entry = MemoryEntry(
                        content=result[0]["content"],
                        metadata=result[0]["metadata"],
                        timestamp=datetime.fromisoformat(result[0]["metadata"].get("timestamp", datetime.now().isoformat())),
                        importance=result[0]["metadata"].get("importance", 0.5),
                        memory_type="long_term"
                    )
                    self._add_to_cache(cache_key, entry)
                    return entry
            
            elif memory_type == "medium_term":
                result = self.medium_term.get_objective(memory_id)
                if result:
                    entry = MemoryEntry(
                        content=result["description"],
                        metadata=result["metadata"],
                        timestamp=datetime.fromisoformat(result["metadata"].get("created_at", datetime.now().isoformat())),
                        importance=result["metadata"].get("priority", 3) / 5.0,
                        memory_type="medium_term"
                    )
                    self._add_to_cache(cache_key, entry)
                    return entry
            
            else:  # short_term
                result = self.short_term.get_interaction(memory_id)
                if result:
                    entry = MemoryEntry(
                        content=result["text"],
                        metadata=result["metadata"],
                        timestamp=datetime.fromisoformat(result["timestamp"]),
                        importance=0.5,  # Short-term memories have default importance
                        memory_type="short_term"
                    )
                    self._add_to_cache(cache_key, entry)
                    return entry
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {e}")
            return None
    
    def search_memory(self, query: str, memory_types: Optional[List[str]] = None, k_per_type: int = 5, time_decay_factor: float = 0.95) -> List[MemoryEntry]:
        """Search across memory systems with relevance scoring and time decay."""
        try:
            results_with_scores = []
            now = datetime.now()

            # Determine which memory types to search
            types_to_search = memory_types or ["long_term", "medium_term", "short_term"]

            # Search long-term memory (assuming retrieve_knowledge returns relevance scores)
            if "long_term" in types_to_search:
                # Assuming retrieve_knowledge returns list of tuples: (content, metadata, score)
                # Or adapt based on actual LTM implementation
                ltm_results = []
                if hasattr(self.long_term, 'search_knowledge_with_scores'):
                    ltm_results = self.long_term.search_knowledge_with_scores(query, k=k_per_type)
                else: # Fallback if only basic retrieval exists
                    raw_results = self.long_term.retrieve_knowledge(query)
                    # Assign placeholder score if LTM doesn't provide one
                    ltm_results = [(r.get('content'), r.get('metadata', {}), 0.7) for r in raw_results[:k_per_type]]

                for content, metadata, relevance_score in ltm_results:
                    timestamp_str = metadata.get("timestamp", now.isoformat())
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                    except ValueError:
                        timestamp = now
                    importance = metadata.get("importance", 0.5)
                    age_hours = (now - timestamp).total_seconds() / 3600
                    time_decay = time_decay_factor ** age_hours
                    final_score = (relevance_score * 0.5 + importance * 0.3 + time_decay * 0.2) # Example weighting

                    entry = MemoryEntry(
                        content=content,
                        metadata=metadata,
                        timestamp=timestamp,
                        importance=importance,
                        memory_type="long_term"
                    )
                    results_with_scores.append((entry, final_score))

            # Search medium-term memory (assuming search_objectives returns relevance scores)
            if "medium_term" in types_to_search:
                 # Assuming search_objectives returns list of tuples: (objective_data, score)
                 mtm_results = []
                 if hasattr(self.medium_term, 'search_objectives_with_scores'):
                     mtm_results = self.medium_term.search_objectives_with_scores(query, k=k_per_type)
                 else: # Fallback
                     raw_results = self.medium_term.search_objectives(query)
                     mtm_results = [(r, 0.7) for r in raw_results[:k_per_type]]
                 
                 for objective_data, relevance_score in mtm_results:
                     metadata = objective_data.get("metadata", {})
                     timestamp_str = metadata.get("created_at", now.isoformat())
                     try:
                         timestamp = datetime.fromisoformat(timestamp_str)
                     except ValueError:
                         timestamp = now
                     importance = metadata.get("priority", 3) / 5.0 # Normalize priority
                     age_hours = (now - timestamp).total_seconds() / 3600
                     time_decay = time_decay_factor ** age_hours
                     final_score = (relevance_score * 0.5 + importance * 0.3 + time_decay * 0.2)

                     entry = MemoryEntry(
                         content=objective_data.get("description", ""),
                         metadata=metadata,
                         timestamp=timestamp,
                         importance=importance,
                         memory_type="medium_term"
                     )
                     results_with_scores.append((entry, final_score))

            # Search short-term memory (Semantic if possible, else keyword + recency)
            if "short_term" in types_to_search:
                stm_results = []
                if hasattr(self.short_term, 'search_interactions_with_scores'):
                    stm_results = self.short_term.search_interactions_with_scores(query, k=k_per_type)
                else: # Fallback to recency + basic keyword match
                    recent_interactions = self.short_term.get_recent_interactions(k_per_type * 2) # Get more for filtering
                    count = 0
                    for interaction in recent_interactions:
                         if count >= k_per_type: break
                         relevance_score = 0.7 if query.lower() in interaction.text.lower() else 0.5 # Basic relevance
                         stm_results.append((interaction, relevance_score))
                         count += 1
                
                for interaction, relevance_score in stm_results:
                     timestamp = interaction.timestamp
                     importance = 0.5 # STM importance is typically lower unless marked
                     age_hours = (now - timestamp).total_seconds() / 3600
                     time_decay = time_decay_factor ** age_hours
                     # STM relevance might be weighted more by recency
                     final_score = (relevance_score * 0.4 + importance * 0.1 + time_decay * 0.5) 

                     entry = MemoryEntry(
                         content=interaction.text,
                         metadata=interaction.metadata,
                         timestamp=timestamp,
                         importance=importance,
                         memory_type="short_term"
                     )
                     results_with_scores.append((entry, final_score))

            # Sort results by the calculated final score (descending)
            results_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Return only the MemoryEntry objects
            final_results = [entry for entry, score in results_with_scores]
            self.logger.debug(f"Memory search for '{query}' returned {len(final_results)} results.")
            return final_results

        except Exception as e:
            self.logger.error(f"Error searching memory: {e}", exc_info=True)
            return []
    
    def update_memory(self, memory_id: str, memory_type: str,
                     content: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory entry"""
        try:
            # Update in appropriate memory system
            if memory_type == "long_term":
                if content:
                    self.long_term.update_knowledge(memory_id, content)
                if metadata:
                    self.long_term.update_knowledge_metadata(memory_id, metadata)
            
            elif memory_type == "medium_term":
                if content:
                    self.medium_term.update_objective(memory_id, content)
                if metadata:
                    self.medium_term.update_objective_metadata(memory_id, metadata)
            
            else:  # short_term
                # Short-term memories are immutable
                return False
            
            # Update cache
            cache_key = f"{memory_type}:{memory_id}"
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if content:
                    entry.content = content
                if metadata:
                    entry.metadata.update(metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
            return False
    
    def delete_memory(self, memory_id: str, memory_type: str) -> bool:
        """Delete a memory entry"""
        try:
            # Delete from appropriate memory system
            if memory_type == "long_term":
                self.long_term.delete_knowledge(memory_id)
            elif memory_type == "medium_term":
                self.medium_term.delete_objective(memory_id)
            else:  # short_term
                # Short-term memories are immutable
                return False
            
            # Remove from cache
            cache_key = f"{memory_type}:{memory_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False
    
    def _add_to_cache(self, key: str, entry: MemoryEntry):
        """Add an entry to the memory cache"""
        # Remove oldest entry if cache is full
        if len(self._cache) >= self._cache_size:
            oldest_key = min(self._cache.items(), key=lambda x: x[1].timestamp)[0]
            del self._cache[oldest_key]
        
        self._cache[key] = entry
    
    def clear_cache(self):
        """Clear the memory cache"""
        self._cache.clear()
    
    def consolidate_memories(self):
        """Consolidate memories based on rules."""
        self.logger.info("Starting memory consolidation check...")
        consolidation_actions = []
        # Process each consolidation rule
        for rule in self.consolidation_rules:
            moved_count = 0
            try:
                if rule.source_type == "short_term" and rule.target_type == "medium_term":
                    # Get interactions older than age threshold (implementation depends on ShortTermMemory)
                    old_interactions = []
                    if hasattr(self.short_term, 'get_interactions_older_than'):
                        old_interactions = self.short_term.get_interactions_older_than(rule.age_threshold)
                    else:
                        all_interactions = self.short_term.get_recent_interactions(9999)
                        now_dt = datetime.now()
                        old_interactions = [i for i in all_interactions if (now_dt - i.timestamp).total_seconds() > rule.age_threshold]
                    
                    # Filter by importance (assuming importance is in metadata or needs calculation)
                    important_interactions = [
                        interaction for interaction in old_interactions
                        if interaction.metadata.get("importance", 0.5) >= rule.importance_threshold
                        # Add repetition logic if needed based on ShortTermMemory capabilities
                    ]
                    
                    for interaction in important_interactions:
                        # Avoid duplicates if possible (e.g., check MTM for existing based on content/timestamp)
                        self.add_memory(
                            content=interaction.text,
                            memory_type="medium_term",
                            metadata={
                                **interaction.metadata,
                                "type": "consolidated_interaction",
                                "source": "short_term_consolidation",
                                "original_timestamp": interaction.timestamp.isoformat()
                            },
                            importance=interaction.metadata.get("importance", 0.6)
                        )
                        moved_count += 1
                        # Optionally delete from STM
                        # try: self.short_term.delete_interaction(interaction.interaction_id)
                        # except Exception as del_err: self.logger.warning(f"Could not delete from STM: {del_err}")

                elif rule.source_type == "medium_term" and rule.target_type == "long_term":
                    medium_term_objectives = self.medium_term.search_objectives("")
                    now_dt = datetime.now()
                    for objective in medium_term_objectives:
                        metadata = objective.get('metadata', {})
                        created_at_str = metadata.get("created_at")
                        if not created_at_str: continue
                        
                        try: created_at = datetime.fromisoformat(created_at_str)
                        except ValueError: continue
                        
                        age_in_seconds = (now_dt - created_at).total_seconds()
                        if age_in_seconds > rule.age_threshold:
                            # Importance scoring (as previously defined)
                            completion_status = metadata.get("status", "active")
                            priority = metadata.get("priority", 3)
                            importance = 0.5
                            if completion_status == "completed": importance = 0.9
                            elif completion_status == "failed": importance = 0.6
                            importance += (priority - 3) / 10.0
                            importance = max(0.1, min(1.0, importance))
                            
                            if importance >= rule.importance_threshold:
                                # Check repetition based on progress updates if rule requires it
                                # progress_updates = len(objective.get('progress', []))
                                # if progress_updates < rule.repetition_threshold: continue
                                
                                ltm_metadata = {
                                    **metadata,
                                    "type": "consolidated_objective",
                                    "source": "medium_term_consolidation",
                                    "original_id": objective.get("objective_id", "unknown"),
                                    "consolidated_at": now_dt.isoformat()
                                }
                                self.add_memory(
                                    content=objective.get("description", ""),
                                    memory_type="long_term",
                                    metadata=ltm_metadata,
                                    importance=importance
                                )
                                moved_count += 1
                                # Optionally delete from MTM
                                # try: self.medium_term.delete_objective(objective.get("objective_id"))
                                # except Exception as del_err: self.logger.warning(f"Could not delete from MTM: {del_err}")

            except Exception as e:
                self.logger.error(f"Error processing consolidation rule ({rule.source_type}->{rule.target_type}): {e}", exc_info=True)

            if moved_count > 0:
                 action_str = f"Consolidated {moved_count} items from {rule.source_type} to {rule.target_type}"
                 self.logger.info(action_str)
                 consolidation_actions.append(action_str)
                 
        if not consolidation_actions:
             self.logger.info("No memories met criteria for consolidation during this check.")
        else:
             self.logger.info(f"Memory consolidation check completed. Actions: {consolidation_actions}")

    def reflect_on_memories(self, query: str, max_memories: int = 20) -> Dict[str, Any]:
        """
        Use LLM to reflect on and extract insights from memories.
        Requires the llm_client to be set during initialization or later.

        Args:
            query: Query to focus the reflection.
            max_memories: Maximum number of memories to include in the context.

        Returns:
            Dictionary with insights and extracted knowledge, or an error dictionary.
        """
        if not self.llm_client:
            self.logger.error("LLM client not available for memory reflection.")
            return {"error": "LLM client not configured for memory reflection."}

        try:
            # Get relevant memories from all systems using the refined search
            relevant_memories = self.search_memory(query=query, k_per_type=max_memories // 2)

            if not relevant_memories:
                self.logger.info(f"No relevant memories found for reflection query: '{query}'")
                return {"insights": [], "patterns": [], "knowledge": [], "connections": [], "suggested_actions": [], "summary": "No relevant memories found.", "raw_llm_response": ""}

            # Format memories for the LLM, respecting max_memories overall
            memory_text = ""
            memory_count = 0
            # Token counting helper
            def _count_tokens_helper(text: str) -> int:
                 if self.llm_client and hasattr(self.llm_client, 'tokenizer') and self.llm_client.tokenizer:
                     try: return len(self.llm_client.tokenizer.encode(text))
                     except Exception: return len(text) // 3
                 else: return len(text) // 3

            current_tokens = 0
            max_context_tokens = 3000 # Estimate max tokens for memory context

            # Sort by timestamp (most recent first) for reflection context
            relevant_memories.sort(key=lambda x: x.timestamp, reverse=True)

            for memory in relevant_memories:
                if memory_count >= max_memories:
                     break

                memory_source = f"[{memory.memory_type.replace('_', ' ').upper()}]"
                memory_time = memory.timestamp.strftime("%Y-%m-%d %H:%M")
                content_str = str(memory.content)
                if isinstance(memory.content, dict): content_str = json.dumps(memory.content, indent=None)

                part = f"{memory_source} ({memory_time}):\n{content_str}\n---\n"
                part_tokens = _count_tokens_helper(part)

                if current_tokens + part_tokens < max_context_tokens:
                    memory_text += part
                    current_tokens += part_tokens
                    memory_count += 1
                else:
                     self.logger.warning("Reached token limit while formatting memories for reflection prompt.")
                     break

            if not memory_text:
                 return {"error": "Could not format any memories within token limits for reflection."}

            # Create system prompt for reflection (as defined previously)
            system_prompt = """
            You are an AI assistant specialized in analyzing and synthesizing information from memory logs.
            Review the provided memories carefully. Your goal is to extract meaningful insights, identify recurring patterns or themes, summarize the key information, establish connections between memories, and suggest potential next steps or areas for further investigation based ONLY on the provided memories.

            Focus on:
            1.  **Insights:** What significant conclusions or understandings can be drawn?
            2.  **Patterns:** Are there recurring topics, behaviors, or outcomes?
            3.  **Knowledge:** What factual information or key takeaways should be retained?
            4.  **Connections:** How do different memories relate to each other?
            5.  **Suggestions:** Based on the analysis, what actions, questions, or objectives seem relevant?
            6.  **Summary:** Provide a concise overall summary of the analyzed memories.

            Present your analysis in a structured JSON format. Ensure the JSON is valid.
            Example structure:
            {
                "summary": "A brief summary of the key themes in the memories.",
                "insights": ["Insight 1", "Insight 2"],
                "patterns": ["Pattern A identified", "Pattern B recurring"],
                "knowledge": ["Key fact 1", "Important detail 2"],
                "connections": ["Memory 1 relates to Memory 5 because...", "Theme X connects memories 2, 4, 7"],
                "suggested_actions": ["Follow up on topic Y", "Verify fact Z"]
            }
            """

            reflection_prompt = f"""
            Analyze the following memories related to the query: "{query}"
            Memory Log:
            ---
            {memory_text.strip()}
            ---
            Provide your analysis in the specified JSON format.
            """

            self.logger.debug(f"Sending reflection prompt to LLM. Query: '{query}', Memories: {memory_count}, Tokens: {current_tokens}")

            # Get reflection from LLM (as before)
            response_content = self.llm_client.process_with_llm(
                prompt=reflection_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=1500
            )

            # Parse the JSON response (as before)
            try:
                if response_content.strip().startswith("```json"):
                    response_content = response_content.strip()[7:-3].strip()
                elif response_content.strip().startswith("```"):
                     response_content = response_content.strip()[3:-3].strip()

                result = json.loads(response_content)
                required_keys = {"summary", "insights", "patterns", "knowledge", "connections", "suggested_actions"}
                if not required_keys.issubset(result.keys()):
                     self.logger.warning(f"LLM reflection response missing expected keys. Got: {result.keys()}")
                     for key in required_keys: result.setdefault(key, [] if key != 'summary' else "")

                result["raw_llm_response"] = response_content
                self.logger.info(f"Successfully processed memory reflection for query: '{query}'")
                return result

            except json.JSONDecodeError as json_err:
                self.logger.error(f"Failed to parse LLM reflection response as JSON: {json_err}. Raw response:\n{response_content}")
                return {"error": f"LLM response was not valid JSON: {json_err}", "raw_llm_response": response_content}

        except LLMTokenLimitError as token_err:
            self.logger.error(f"Memory reflection failed due to token limit: {token_err}")
            return {"error": f"Token limit exceeded during reflection: {token_err}", "raw_llm_response": ""}
        except LLMCommunicationError as comm_err:
            self.logger.error(f"Memory reflection failed due to LLM communication error: {comm_err}")
            return {"error": f"LLM communication failed: {comm_err}", "raw_llm_response": ""}
        except Exception as e:
            self.logger.exception(f"Unexpected error during memory reflection: {e}")
            return {"error": f"An unexpected error occurred: {e}", "raw_llm_response": ""}

    def reset_context(self):
        """Reset short-term memory, optionally after consolidation."""
        self.logger.info("Resetting context (clearing short-term memory after consolidation check)." )
        # First, consolidate any important memories
        try:
            self.consolidate_memories()
        except Exception as e:
             self.logger.error(f"Error during consolidation before context reset: {e}")

        # Then clear short-term memory
        try:
             if hasattr(self.short_term, 'reset'):
                 self.short_term.reset()
                 # Also clear the unified memory cache as STM is gone
                 self.clear_cache()
                 self.logger.info("Short-term memory reset and cache cleared.")
             else:
                  self.logger.warning("Short-term memory object does not have a 'reset' method.")
        except Exception as e:
             self.logger.error(f"Error resetting short-term memory: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        stats = {"cache_size": len(self._cache)}
        try:
             stats["long_term_count"] = self.long_term.count()
        except Exception as e:
            self.logger.warning(f"Could not get long_term count: {e}")
            stats["long_term_count"] = "Error"
        try:
             stats["medium_term_count"] = len(self.medium_term.search_objectives(""))
        except Exception as e:
             self.logger.warning(f"Could not get medium_term count: {e}")
             stats["medium_term_count"] = "Error"
        try:
             stats["short_term_count"] = self.short_term.count()
        except Exception as e:
             self.logger.warning(f"Could not get short_term count: {e}")
             stats["short_term_count"] = "Error"
        return stats
    
    def organize_knowledge(self, summarize: bool = True, rebuild_indices: bool = False, deduplicate: bool = False):
        """Organizes long-term knowledge: Summarization, Indexing, Deduplication."""
        self.logger.info("Starting knowledge organization...")
        actions_taken = []

        # 1. Rebuild Indices
        if rebuild_indices and hasattr(self.long_term, 'rebuild_index'):
            try:
                self.logger.info("Rebuilding long-term memory index...")
                start_time = time.time()
                self.long_term.rebuild_index()
                duration = time.time() - start_time
                self.logger.info(f"Index rebuild completed in {duration:.2f} seconds.")
                actions_taken.append("rebuilt_index")
            except NotImplementedError:
                self.logger.warning("Long-term memory does not support 'rebuild_index'. Skipping.")
            except Exception as e:
                self.logger.error(f"Error rebuilding long-term memory index: {e}", exc_info=True)

        # 2. Summarization
        if summarize and self.llm_client:
            try:
                self.logger.info("Identifying old/low-importance memories for summarization...")
                # Define criteria for summarization (e.g., older than 1 month, importance < 0.4)
                # This requires the LTM implementation to support querying by date/metadata
                # Placeholder: Assume get_all_knowledge() returns all, then filter
                all_knowledge = self.long_term.get_all_knowledge() # Assuming this exists and returns list of dicts
                
                now = datetime.now()
                summarization_candidates = []
                for item in all_knowledge:
                     metadata = item.get("metadata", {})
                     timestamp_str = metadata.get("timestamp")
                     importance = metadata.get("importance", 0.5)
                     is_summary = metadata.get("type") == "knowledge_summary"

                     if timestamp_str and not is_summary:
                         try:
                             timestamp = datetime.fromisoformat(timestamp_str)
                             age_days = (now - timestamp).days
                             # Criteria: Older than 30 days and importance < 0.4
                             if age_days > 30 and importance < 0.4:
                                 summarization_candidates.append(item)
                         except ValueError:
                             continue # Skip items with invalid timestamp format
                
                if summarization_candidates:
                    self.logger.info(f"Found {len(summarization_candidates)} candidates for summarization.")
                    # Summarize in batches to avoid huge prompts
                    batch_size = 5
                    summarized_ids = set()
                    for i in range(0, len(summarization_candidates), batch_size):
                        batch = summarization_candidates[i:i+batch_size]
                        batch_content = ""
                        batch_ids = []
                        for item in batch:
                            batch_content += f"ID: {item.get('id', 'N/A')} | Content: {item.get('content', '')}\n---\n"
                            batch_ids.append(item.get('id'))
                        if not batch_content.strip(): continue

                        system_prompt = """
                        You are an AI assistant specializing in knowledge consolidation. 
                        Analyze the provided batch of low-importance or old memory entries.
                        Generate a concise, synthesized summary capturing the key information or themes present in the batch.
                        Focus on retaining factual information and core concepts.
                        The summary will replace the individual entries.
                        Return ONLY the summary text.
                        """
                        prompt = f"Summarize the following memory entries:\n\n{batch_content}"
                        
                        try:
                            summary = self.llm_client.process_with_llm(prompt, system_prompt, temperature=0.3, max_tokens=500)
                            
                            # Add the summary as a new knowledge item
                            summary_metadata = {
                                "type": "knowledge_summary",
                                "timestamp": now.isoformat(),
                                "source_ids": batch_ids,
                                "importance": 0.6 # Summaries are moderately important
                            }
                            self.long_term.add_knowledge(summary, "Synthesized summary of older knowledge", metadata=summary_metadata)
                            
                            # Optionally: Delete the original items
                            # Be cautious with deletion!
                            # for item_id in batch_ids:
                            #    if item_id: self.long_term.delete_knowledge(item_id)
                            self.logger.info(f"Summarized batch of {len(batch_ids)} items.")
                            summarized_ids.update(batch_ids)
                        except Exception as llm_err:
                            self.logger.error(f"LLM summarization failed for batch starting at index {i}: {llm_err}")
                    if summarized_ids:
                        actions_taken.append(f"summarized_{len(summarized_ids)}_items")
                else:
                    self.logger.info("No suitable memories found for summarization.")
            except NotImplementedError:
                 self.logger.warning("Long-term memory does not support 'get_all_knowledge'. Skipping summarization.")
            except Exception as e:
                self.logger.error(f"Error during memory summarization: {e}", exc_info=True)
        elif summarize and not self.llm_client:
             self.logger.warning("Summarization requested but LLM client is not available.")

        # 3. Deduplication (Optional, potentially complex)
        if deduplicate:
             self.logger.warning("Deduplication logic is not implemented yet.")
             # Requires comparing embeddings or content of potentially all items.
             # Could involve finding items with very high similarity scores (>0.95?) and merging/deleting.
             # actions_taken.append("deduplication_attempted")

        self.logger.info(f"Knowledge organization finished. Actions taken: {actions_taken or 'None'}")

    def assemble_context(self, query: str, max_tokens: int = 4000, include_recent: int = 5, k_per_type: int = 5) -> str:
        """
        Assembles relevant context from memory systems for LLM prompts,
        respecting token limits.

        Args:
            query: The query or topic to focus the context on.
            max_tokens: The maximum number of tokens allowed for the assembled context string.
            include_recent: Number of most recent short-term interactions to always include.
            k_per_type: Number of semantically relevant items to fetch per memory type.

        Returns:
            A formatted string containing the assembled context, truncated if necessary.
        """
        self.logger.debug(f"Assembling context for query '{query}' with max_tokens={max_tokens}")
        context_parts = []
        total_tokens = 0

        # Define token counting helper inside or ensure it's accessible
        def _count_tokens_helper(text: str) -> int:
            if self.llm_client and hasattr(self.llm_client, 'tokenizer') and self.llm_client.tokenizer:
                try:
                    return len(self.llm_client.tokenizer.encode(text))
                except Exception as e:
                    self.logger.warning(f"Tiktoken encoding failed: {e}, falling back to basic count.")
                    return len(text) // 3 # Fallback
            else:
                # Fallback if no LLM client or tokenizer
                return len(text) // 3

        # Estimate overhead tokens for formatting, prompts etc.
        token_buffer = 150 # Leave buffer for final prompt structure and separators
        effective_max_tokens = max(0, max_tokens - token_buffer)

        # 1. Always include the N most recent short-term interactions
        recent_interactions_added = []
        if include_recent > 0:
            try:
                recent_interactions = self.short_term.get_recent_interactions(include_recent)
                # Add most recent first for context assembly
                for interaction in reversed(recent_interactions):
                    speaker = interaction.metadata.get("speaker", "unknown")
                    timestamp = interaction.timestamp.strftime("%Y-%m-%d %H:%M")
                    part = f"[Recent Interaction @ {timestamp}] {speaker.upper()}: {interaction.text}\n"
                    part_tokens = _count_tokens_helper(part)

                    if total_tokens + part_tokens <= effective_max_tokens:
                        recent_interactions_added.insert(0, part) # Prepend to keep order
                        total_tokens += part_tokens
                    else:
                        self.logger.warning(f"Reached token limit ({total_tokens}+{part_tokens} > {effective_max_tokens}) while adding recent interactions.")
                        break
            except Exception as e:
                self.logger.error(f"Error retrieving recent interactions for context: {e}", exc_info=True)

        # 2. Retrieve semantically relevant memories
        relevant_memories_added = []
        try:
            relevant_memories = self.search_memory(
                query=f"Context related to: {query}",
                k_per_type=k_per_type,
                memory_types=["long_term", "medium_term", "short_term"]
            )
            self.logger.debug(f"Retrieved {len(relevant_memories)} potentially relevant memories based on refined search.")

            # Filter out recent interactions already added (by content matching)
            recent_interaction_texts = {p.split(": ", 1)[1].strip() for p in recent_interactions_added}
            filtered_memories = [m for m in relevant_memories if not (m.memory_type == "short_term" and m.content in recent_interaction_texts)]

            # Add relevant memories until token limit is reached
            for memory in filtered_memories:
                timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
                mem_type_label = memory.memory_type.replace('_', ' ').upper()
                content_str = str(memory.content)

                part_prefix = f"[{mem_type_label} @ {timestamp}] "
                if memory.memory_type == "medium_term":
                    status = memory.metadata.get('status', 'unknown')
                    part_prefix = f"[OBJECTIVE ({status.upper()}) @ {timestamp}] "

                prefix_tokens = _count_tokens_helper(part_prefix)
                content_tokens = _count_tokens_helper(content_str)
                newline_token = 1
                available_content_tokens = effective_max_tokens - total_tokens - prefix_tokens - newline_token

                if available_content_tokens <= 20:
                    self.logger.debug("Token limit reached before adding next relevant memory.")
                    break

                if content_tokens > available_content_tokens:
                    estimated_chars_per_token = 3
                    max_chars = max(0, available_content_tokens * estimated_chars_per_token)
                    content_str = content_str[:max_chars] + "... [Truncated]"
                    part = part_prefix + content_str + "\n"
                    part_tokens = _count_tokens_helper(part)
                else:
                    part = part_prefix + content_str + "\n"
                    part_tokens = prefix_tokens + content_tokens + newline_token

                if total_tokens + part_tokens <= effective_max_tokens:
                    relevant_memories_added.append(part)
                    total_tokens += part_tokens
                else:
                    self.logger.debug(f"Could not fit memory {mem_type_label} even after potential truncation. Stopping.")
                    break
        except Exception as e:
            self.logger.error(f"Error searching/processing relevant memories for context: {e}", exc_info=True)

        # Assemble the final context string
        final_parts = recent_interactions_added + relevant_memories_added
        if not final_parts:
             self.logger.warning(f"Context assembly for query '{query}' resulted in empty context.")
             return "(No relevant context found)"

        final_context = "--- Relevant Context Start ---\n"
        final_context += "".join(final_parts)
        final_context += "--- Relevant Context End ---"

        final_tokens = _count_tokens_helper(final_context)
        self.logger.info(f"Assembled context for query '{query}': {final_tokens} tokens (Effective Max: {effective_max_tokens}, Original Max: {max_tokens}).")

        return final_context

# Final check: Ensure class closing brace is present and indentation is correct 