import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import time

from pydantic import BaseModel, Field, ConfigDict
import chromadb
from chromadb import Client, Collection

class Objective(BaseModel):
    """A user objective to be tracked in medium-term memory"""
    objective_id: str
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "active"  # active, completed, failed
    progress: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_progress(self, step: str, status: str = "completed", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a progress step to the objective"""
        if metadata is None:
            metadata = {}
        
        progress_entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        self.progress.append(progress_entry)
    
    def complete(self) -> None:
        """Mark the objective as completed"""
        self.status = "completed"
        self.completed_at = datetime.now()
    
    def fail(self, reason: str) -> None:
        """Mark the objective as failed"""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.add_progress("Objective failed", "failed", {"reason": reason})

class MediumTermMemory(BaseModel):
    """
    Medium-term memory for Jarvis (objective-centric)
    This memory tracks user objectives and persists across sessions
    """
    
    db_path: str = Field(default="memory/db/medium_term")
    objectives: Dict[str, Objective] = Field(default_factory=dict)
    current_objective_id: Optional[str] = None
    client: Optional[Client] = None
    collection: Optional[Collection] = None
    logger: logging.Logger
    memory_file: str = "memory/medium_term_memory.json"
    auto_save: bool = True
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data) -> None:
        # 1. Initialize logger first if not passed in data
        if 'logger' not in data:
            data['logger'] = logging.getLogger("medium_term_memory") # Assign to data dict

        # 2. Call Pydantic's init with the logger included
        super().__init__(**data)

        # 3. Perform the rest of the initialization that might fail
        # The logger is now available via self.logger as Pydantic initialized it
        try:
            # Now self.logger definitely exists before these calls
            self.logger.info(f"Initializing MediumTermMemory with db_path: {self.db_path}")
            self._setup_db()
            self._load_objectives() # This can now safely log errors using self.logger
            self.logger.info("MediumTermMemory initialized successfully.")
        except Exception as e:
            # Logging the error here is fine
            self.logger.error(f"CRITICAL ERROR during MediumTermMemory initialization: {e}", exc_info=True)
            # raise # Optionally re-raise if this failure should halt execution
    
    def _setup_db(self) -> None:
        """Set up the vector database for medium-term memory"""
        self.logger.debug(f"Setting up ChromaDB for MediumTermMemory at path: {self.db_path}")
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
        collection_name = "objectives"
        try:
            self.logger.debug(f"Attempting to get collection: {collection_name}")
            self.collection = self.client.get_collection(collection_name)
            self.logger.info(f"Successfully got existing collection: {collection_name}")
        except Exception as get_err: # More specific exception handling might be needed depending on chromadb version
            self.logger.warning(f"Failed to get collection '{collection_name}': {get_err}. Attempting to create it.")
            try:
                self.logger.debug(f"Attempting to create collection: {collection_name}")
                # Potential point of failure for KeyError: '_type' if schema/metadata has issues?
                self.collection = self.client.create_collection(collection_name)
                self.logger.info(f"Successfully created collection: {collection_name}")
            except Exception as create_err:
                self.logger.error(f"CRITICAL ERROR: Failed to create collection '{collection_name}': {create_err}", exc_info=True)
                raise # Re-raise critical error
    
    def _load_objectives(self) -> None:
        """Load objectives from the database"""
        # Load from objectives.json if it exists
        objectives_file = os.path.join(self.db_path, "objectives.json")
        if os.path.exists(objectives_file):
            try:
                with open(objectives_file, "r") as f:
                    objectives_data = json.load(f)
                
                for obj_id, obj_data in objectives_data.items():
                    # Convert string timestamps to datetime objects
                    if "created_at" in obj_data:
                        obj_data["created_at"] = datetime.fromisoformat(obj_data["created_at"])
                    if "completed_at" in obj_data and obj_data["completed_at"]:
                        obj_data["completed_at"] = datetime.fromisoformat(obj_data["completed_at"])
                    
                    self.objectives[obj_id] = Objective(**obj_data)
            except Exception as e:
                # <<< Use self.logger here, which should now be defined >>>
                # print(f"Error loading objectives: {e}") # Keep print for immediate feedback if logger fails?
                self.logger.error(f"Error loading objectives from file {objectives_file}: {e}", exc_info=True)
    
    def _save_objectives(self) -> None:
        """Save objectives to the database"""
        # Save to objectives.json
        objectives_file = os.path.join(self.db_path, "objectives.json")
        os.makedirs(os.path.dirname(objectives_file), exist_ok=True)
        
        # Convert to dict and serialize datetime objects
        objectives_dict = {}
        for obj_id, objective in self.objectives.items():
            obj_dict = objective.model_dump()
            obj_dict["created_at"] = objective.created_at.isoformat()
            if objective.completed_at:
                obj_dict["completed_at"] = objective.completed_at.isoformat()
            
            objectives_dict[obj_id] = obj_dict
        
        with open(objectives_file, "w") as f:
            json.dump(objectives_dict, f, indent=2)
    
    def create_objective(self, description: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new objective and return its ID"""
        # Generate unique ID
        objective_id = f"obj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create objective
        objective = Objective(
            objective_id=objective_id,
            description=description
        )
        
        # Store objective
        self.objectives[objective_id] = objective
        
        # Save to disk
        self._save_objectives()
        
        # Prepare metadata for vector DB
        vector_metadata = {"status": "active", "created_at": objective.created_at.isoformat()}
        if metadata:
            # Add additional metadata but don't overwrite our required fields
            for key, value in metadata.items():
                if key not in ["status", "created_at"]:
                    vector_metadata[key] = value
        
        # Add to vector database for semantic search
        self.collection.add(
            documents=[description],
            metadatas=[vector_metadata],
            ids=[objective_id]
        )
        
        # Set as current objective
        self.current_objective_id = objective_id
        
        return objective_id
    
    def get_objective(self, objective_id: str) -> Optional[Objective]:
        """Get an objective by ID"""
        return self.objectives.get(objective_id)
    
    def get_current_objective(self) -> Optional[Objective]:
        """Get the current objective"""
        if self.current_objective_id:
            return self.get_objective(self.current_objective_id)
        return None
    
    def set_current_objective(self, objective_id: str) -> bool:
        """Set the current objective"""
        if objective_id in self.objectives:
            self.current_objective_id = objective_id
            return True
        return False
    
    def add_progress(self, step: str, status: str = "completed", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add progress to the current objective"""
        if not self.current_objective_id:
            return False
        
        objective = self.objectives[self.current_objective_id]
        objective.add_progress(step, status, metadata)
        self._save_objectives()
        return True
    
    def complete_objective(self, objective_id: Optional[str] = None) -> bool:
        """Complete an objective"""
        if objective_id is None:
            objective_id = self.current_objective_id
        
        if objective_id and objective_id in self.objectives:
            objective = self.objectives[objective_id]
            objective.complete()
            
            # Update in vector database
            self.collection.update(
                ids=[objective_id],
                metadatas=[{"status": "completed", "completed_at": objective.completed_at.isoformat()}]
            )
            
            self._save_objectives()
            return True
        
        return False
    
    def search_objectives(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search objectives by description using semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            obj_id = results["ids"][0][i]
            objective = self.get_objective(obj_id)
            
            formatted_results.append({
                "description": doc,
                "metadata": results["metadatas"][0][i],
                "id": obj_id,
                "objective": objective
            })
        
        return formatted_results 