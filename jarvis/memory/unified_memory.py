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
    
    def search_memory(self, query: str, memory_types: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Search across memory systems"""
        try:
            results = []
            
            # Determine which memory types to search
            types_to_search = memory_types or ["long_term", "medium_term", "short_term"]
            
            # Search long-term memory
            if "long_term" in types_to_search:
                lt_results = self.long_term.retrieve_knowledge(query)
                for result in lt_results:
                    entry = MemoryEntry(
                        content=result["content"],
                        metadata=result["metadata"],
                        timestamp=datetime.fromisoformat(result["metadata"].get("timestamp", datetime.now().isoformat())),
                        importance=result["metadata"].get("importance", 0.5),
                        memory_type="long_term"
                    )
                    results.append(entry)
            
            # Search medium-term memory
            if "medium_term" in types_to_search:
                mt_results = self.medium_term.search_objectives(query)
                for result in mt_results:
                    entry = MemoryEntry(
                        content=result["description"],
                        metadata=result["metadata"],
                        timestamp=datetime.fromisoformat(result["metadata"].get("created_at", datetime.now().isoformat())),
                        importance=result["metadata"].get("priority", 3) / 5.0,
                        memory_type="medium_term"
                    )
                    results.append(entry)
            
            # Search short-term memory
            if "short_term" in types_to_search:
                st_results = self.short_term.get_recent_interactions(10)  # Get last 10 interactions
                for interaction in st_results:
                    if query.lower() in interaction.text.lower():
                        entry = MemoryEntry(
                            content=interaction.text,
                            metadata=interaction.metadata,
                            timestamp=interaction.timestamp,
                            importance=0.5,
                            memory_type="short_term"
                        )
                        results.append(entry)
            
            # Sort by importance and timestamp
            results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching memory: {e}")
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
        """
        Consolidate memories based on rules.
        Move important short-term memories to medium-term, and
        important medium-term memories to long-term.
        """
        # Process each consolidation rule
        for rule in self.consolidation_rules:
            if rule.source_type == "short_term" and rule.target_type == "medium_term":
                # Get interactions older than age threshold
                old_interactions = [
                    interaction for interaction in self.short_term.interactions
                    if (datetime.now() - interaction.timestamp).total_seconds() > rule.age_threshold
                ]
                
                # Filter by importance
                important_interactions = [
                    interaction for interaction in old_interactions
                    if interaction.metadata.get("importance", 0) >= rule.importance_threshold
                ]
                
                # Move to medium-term memory
                for interaction in important_interactions:
                    self.add_memory(
                        content=interaction.text,
                        memory_type="medium_term",
                        metadata={
                            **interaction.metadata,
                            "source": "short_term_consolidation",
                            "original_timestamp": interaction.timestamp.isoformat()
                        },
                        importance=interaction.metadata.get("importance", 0.5)
                    )
            
            elif rule.source_type == "medium_term" and rule.target_type == "long_term":
                # Find medium-term objectives older than threshold
                medium_term_objectives = self.medium_term.search_objectives("")
                for objective in medium_term_objectives:
                    # Skip if it doesn't have required metadata
                    if "created_at" not in objective["metadata"]:
                        continue
                    
                    # Check if it's old enough
                    created_at = datetime.fromisoformat(objective["metadata"]["created_at"])
                    age_in_seconds = (datetime.now() - created_at).total_seconds()
                    
                    if age_in_seconds > rule.age_threshold:
                        # Calculate importance based on progress or completion
                        progress = objective.get("progress", [])
                        completion_status = objective["metadata"].get("status", "active")
                        priority = objective["metadata"].get("priority", 3)
                        
                        # Compute importance score
                        importance = 0.5  # Default
                        if completion_status == "completed":
                            importance = 0.9  # Completed objectives are important
                        elif completion_status == "failed":
                            importance = 0.6  # Failed objectives have some value
                        elif len(progress) >= rule.repetition_threshold:
                            importance = 0.8  # Objectives with substantial progress
                        
                        # Apply priority modifier
                        importance += (priority / 10)
                        importance = min(1.0, importance)  # Cap at 1.0
                        
                        # Only consolidate if importance is above threshold
                        if importance >= rule.importance_threshold:
                            # Add to long-term memory
                            self.add_memory(
                                content=objective["description"],
                                memory_type="long_term",
                                metadata={
                                    **objective["metadata"],
                                    "source": "medium_term_consolidation",
                                    "original_id": objective["id"],
                                    "progress": progress,
                                    "consolidated_at": datetime.now().isoformat()
                                },
                                importance=importance
                            )
        
        # Automatically run consolidation based on time
        self.logger.info("Memory consolidation completed")
    
    def reflect_on_memories(self, query: str, llm_client) -> Dict[str, Any]:
        """
        Use LLM to reflect on and extract insights from memories
        
        Args:
            query: Query to focus the reflection
            llm_client: The LLM client to use for reflection
            
        Returns:
            Dictionary with insights and extracted knowledge
        """
        try:
            # Get relevant memories from all systems
            relevant_memories = self.search_memory(query)
            
            if not relevant_memories:
                return {"error": "No relevant memories found"}
            
            # Format memories for the LLM
            memory_text = ""
            for i, memory in enumerate(relevant_memories[:20]):  # Limit to top 20 memories
                memory_source = f"[{memory.memory_type}]"
                memory_time = memory.timestamp.strftime("%Y-%m-%d %H:%M")
                
                # Format content based on type
                if isinstance(memory.content, str):
                    content = memory.content
                elif isinstance(memory.content, dict):
                    content = json.dumps(memory.content, indent=2)
                else:
                    content = str(memory.content)
                
                memory_text += f"Memory {i+1} {memory_source} ({memory_time}):\n{content}\n\n"
            
            # Create system prompt for reflection
            system_prompt = """
            You are Jarvis, an intelligent assistant with access to memory systems.
            Analyze the provided memories and extract insights, patterns, and knowledge.
            Focus on identifying:
            1. Key themes and patterns across memories
            2. Important information to retain
            3. Connections between different memories
            4. Potential actions or objectives based on these memories
            
            Format your response as JSON with the following structure:
            {
                "insights": [list of key insights],
                "patterns": [identified patterns],
                "knowledge": [important information to retain],
                "connections": [connections between memories],
                "suggested_actions": [potential actions based on analysis]
            }
            """
            
            reflection_prompt = f"""
            I need you to analyze and reflect on the following memories related to: "{query}"
            
            {memory_text}
            
            Extract insights, patterns, and knowledge from these memories.
            """
            
            # Get reflection from LLM
            response = llm_client.process_with_llm(reflection_prompt, system_prompt)
            
            # Parse the JSON response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract structured information
                insights = []
                patterns = []
                knowledge = []
                
                # Simple extraction of bulleted lists
                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        if 'insight' in line.lower():
                            insights.append(line[2:])
                        elif 'pattern' in line.lower():
                            patterns.append(line[2:])
                        else:
                            knowledge.append(line[2:])
                
                return {
                    "insights": insights,
                    "patterns": patterns,
                    "knowledge": knowledge,
                    "connections": [],
                    "suggested_actions": []
                }
                
        except Exception as e:
            self.logger.error(f"Error in memory reflection: {e}")
            return {"error": str(e)}
    
    def reset_context(self):
        """Reset short-term memory while preserving important information"""
        # First, consolidate any important memories
        self.consolidate_memories()
        
        # Then clear short-term memory
        self.short_term.reset()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            "cache_size": len(self._cache),
            "long_term_count": len(self.long_term.get_all_knowledge()),
            "medium_term_count": len(self.medium_term.search_objectives("")),
            "short_term_count": len(self.short_term.get_recent_interactions(1000))
        } 