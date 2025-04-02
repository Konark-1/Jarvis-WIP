import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field
import chromadb

class LongTermMemory(BaseModel):
    """
    Long-term memory for Jarvis (core programming knowledge)
    This memory persists across restarts and contains fundamental knowledge
    """
    
    db_path: str = Field(default="memory/db/long_term")
    collection_name: str = Field(default="core_knowledge")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._setup_db()
    
    def _setup_db(self):
        """Set up the vector database for long-term memory"""
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize the client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except ValueError:
            self.collection = self.client.create_collection(self.collection_name)
    
    def add_knowledge(self, knowledge_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add knowledge to long-term memory"""
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Add to collection
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[knowledge_id]
        )
    
    def retrieve_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve knowledge from long-term memory using similarity search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "id": results["ids"][0][i]
            })
        
        return formatted_results
    
    def delete_knowledge(self, knowledge_id: str):
        """Delete knowledge from long-term memory"""
        self.collection.delete(ids=[knowledge_id])
    
    def update_knowledge(self, knowledge_id: str, content: str, metadata: Dict[str, Any] = None):
        """Update knowledge in long-term memory"""
        # Delete existing knowledge
        try:
            self.delete_knowledge(knowledge_id)
        except:
            pass
        
        # Add updated knowledge
        self.add_knowledge(knowledge_id, content, metadata) 