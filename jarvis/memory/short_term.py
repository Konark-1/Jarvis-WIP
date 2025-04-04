from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from pydantic import BaseModel, Field

class Interaction(BaseModel):
    """A single interaction between the user and Jarvis"""
    speaker: str  # "user" or "jarvis"
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ShortTermMemory(BaseModel):
    """
    Short-term memory for Jarvis (context-relevant)
    This memory is used to maintain context during conversations
    """
    
    # Recent interactions (conversation history)
    interactions: List[Interaction] = Field(default_factory=list)
    
    # Maximum number of interactions to keep in memory
    max_interactions: int = 20
    
    # Context variables (for maintaining state during a conversation)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def add_interaction(self, speaker: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an interaction to short-term memory"""
        if metadata is None:
            metadata = {}
        
        interaction = Interaction(
            speaker=speaker,
            text=text,
            metadata=metadata
        )
        
        self.interactions.append(interaction)
        
        # Trim if needed
        if len(self.interactions) > self.max_interactions:
            self.interactions = self.interactions[-self.max_interactions:]
    
    def get_recent_interactions(self, n: Optional[int] = None) -> List[Interaction]:
        """Get the n most recent interactions"""
        if n is None:
            return self.interactions
        return self.interactions[-n:]
    
    def get_conversation_history(self, n: Optional[int] = None) -> str:
        """Get conversation history as a formatted string"""
        interactions = self.get_recent_interactions(n)
        
        history = []
        for interaction in interactions:
            speaker = "User" if interaction.speaker == "user" else "Jarvis"
            timestamp = interaction.timestamp.strftime("%H:%M:%S")
            history.append(f"[{timestamp}] {speaker}: {interaction.text}")
        
        return "\n".join(history)
    
    def set_context_var(self, key: str, value: Any):
        """Set a context variable"""
        self.context[key] = value
    
    def get_context_var(self, key: str, default: Any = None) -> Any:
        """Get a context variable"""
        return self.context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear all context variables"""
        self.context = {}
    
    def reset(self) -> None:
        """Reset short-term memory (clear interactions and context)"""
        self.interactions = []
        self.clear_context() 