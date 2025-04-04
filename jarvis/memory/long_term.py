import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from pydantic import BaseModel, Field, ConfigDict
import chromadb
from chromadb import Client, Collection

class LongTermMemory(BaseModel):
    """
    Long-term memory for Jarvis (core programming knowledge)
    This memory persists across restarts and contains fundamental knowledge
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    db_path: str = Field(default="memory/db/long_term")
    collection_name: str = Field(default="core_knowledge")
    client: Optional[Client] = None
    collection: Optional[Collection] = None
    logger: Optional[logging.Logger] = None
    table: Optional[Any] = None
    table_name: str = "knowledge"
    
    def __init__(self, **data) -> None:
        # Prepare data for Pydantic initialization
        init_data = data.copy()

        # Initialize logger if not provided
        if 'logger' not in init_data:
            # Assign default logger
            init_data['logger'] = logging.getLogger("long_term_memory")
            # Basic configuration if no handlers exist (optional, depends on project setup)
            if not init_data['logger'].hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                init_data['logger'].addHandler(handler)
                init_data['logger'].setLevel(logging.INFO) # Or your desired default level

        # Call Pydantic's __init__ with potentially added logger
        super().__init__(**init_data)
        
        # Setup DB after Pydantic has initialized fields like db_path
        try:
            self._setup_db()
            self.logger.info("LongTermMemory initialized successfully.")
        except Exception as e:
            # Use the logger initialized by Pydantic
            self.logger.error(f"CRITICAL ERROR during LongTermMemory DB setup: {e}", exc_info=True)
            # raise # Optionally re-raise
    
    def _setup_db(self) -> None:
        """Set up the vector database for long-term memory"""
        self.logger.debug(f"Setting up ChromaDB for LongTermMemory at path: {self.db_path}")
        # Create directory if it doesn't exist
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error creating directory {self.db_path}: {e}", exc_info=True)
            raise # Re-raise critical error

        # Initialize the client
        try:
            self.logger.debug("Initializing ChromaDB PersistentClient...")
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.logger.debug(f"ChromaDB PersistentClient initialized: {self.client}")
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB PersistentClient at path {self.db_path}: {e}", exc_info=True)
            raise # Re-raise critical error

        # Get or create collection
        try:
            self.logger.debug(f"Attempting to get collection: {self.collection_name}")
            self.collection = self.client.get_collection(self.collection_name)
            self.logger.info(f"Successfully got existing collection: {self.collection_name}")
        except Exception as get_err: # More specific exception handling might be needed depending on chromadb version
            self.logger.warning(f"Failed to get collection '{self.collection_name}': {get_err}. Attempting to create it.")
            try:
                self.logger.debug(f"Attempting to create collection: {self.collection_name}")
                # Potential point of failure for KeyError: '_type' if schema/metadata has issues?
                self.collection = self.client.create_collection(self.collection_name)
                self.logger.info(f"Successfully created collection: {self.collection_name}")
            except Exception as create_err:
                self.logger.error(f"CRITICAL ERROR: Failed to create collection '{self.collection_name}': {create_err}", exc_info=True)
                raise # Re-raise critical error
    
    def add_knowledge(self, knowledge_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
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
    
    def delete_knowledge(self, knowledge_id: str) -> None:
        """Delete knowledge from long-term memory"""
        self.collection.delete(ids=[knowledge_id])
    
    def update_knowledge(self, knowledge_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update knowledge in long-term memory"""
        # Delete existing knowledge
        try:
            self.delete_knowledge(knowledge_id)
        except:
            pass
        
        # Add updated knowledge
        self.add_knowledge(knowledge_id, content, metadata) 