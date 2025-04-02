import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
import chromadb

class Objective(BaseModel):
    """A user objective to be tracked in medium-term memory"""
    objective_id: str
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "active"  # active, completed, failed
    progress: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_progress(self, step: str, status: str = "completed", metadata: Dict[str, Any] = None):
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
    
    def complete(self):
        """Mark the objective as completed"""
        self.status = "completed"
        self.completed_at = datetime.now()
    
    def fail(self, reason: str):
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
    
    def __init__(self, **data):
        super().__init__(**data)
        self._setup_db()
        self._load_objectives()
    
    def _setup_db(self):
        """Set up the vector database for medium-term memory"""
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize the client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("objectives")
        except ValueError:
            self.collection = self.client.create_collection("objectives")
    
    def _load_objectives(self):
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
                print(f"Error loading objectives: {e}")
    
    def _save_objectives(self):
        """Save objectives to the database"""
        # Save to objectives.json
        objectives_file = os.path.join(self.db_path, "objectives.json")
        os.makedirs(os.path.dirname(objectives_file), exist_ok=True)
        
        # Convert to dict and serialize datetime objects
        objectives_dict = {}
        for obj_id, objective in self.objectives.items():
            obj_dict = objective.dict()
            obj_dict["created_at"] = objective.created_at.isoformat()
            if objective.completed_at:
                obj_dict["completed_at"] = objective.completed_at.isoformat()
            
            objectives_dict[obj_id] = obj_dict
        
        with open(objectives_file, "w") as f:
            json.dump(objectives_dict, f, indent=2)
    
    def create_objective(self, description: str) -> str:
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
        
        # Add to vector database for semantic search
        self.collection.add(
            documents=[description],
            metadatas=[{"status": "active", "created_at": objective.created_at.isoformat()}],
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
    
    def add_progress(self, step: str, status: str = "completed", metadata: Dict[str, Any] = None):
        """Add progress to the current objective"""
        if not self.current_objective_id:
            return False
        
        objective = self.objectives[self.current_objective_id]
        objective.add_progress(step, status, metadata)
        self._save_objectives()
        return True
    
    def complete_objective(self, objective_id: Optional[str] = None):
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