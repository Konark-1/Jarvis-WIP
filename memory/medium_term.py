import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

import lancedb # Added
import pyarrow as pa # Added for schema
from pydantic import BaseModel, Field, PrivateAttr, validator # Added validator
# import chromadb # Removed
# import chromadb.config # Removed

logger = logging.getLogger(__name__) # Added

class Objective(BaseModel):
    """A user objective to be tracked in medium-term memory"""
    objective_id: str
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "active"  # active, completed, failed
    progress: List[Dict[str, Any]] = Field(default_factory=list)

    # Keep methods like add_progress, complete, fail - they modify the Pydantic object
    # Persistence logic will be handled in MediumTermMemory

    def add_progress(self, step: str, status: str = "completed", metadata: Optional[Dict[str, Any]] = None):
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

    # Add a method to convert progress to JSON string for LanceDB
    def _progress_to_json(self) -> str:
        return json.dumps(self.progress)

    # Add a class method to create from LanceDB dict (parsing JSON)
    @classmethod
    def from_lancedb(cls, data: Dict[str, Any]) -> 'Objective':
        data['progress'] = json.loads(data.get('progress_json', '[]'))
        # Ensure datetime conversion if necessary (LanceDB might return native)
        for key in ['created_at', 'completed_at']:
            if key in data and isinstance(data[key], (int, float)):
                 # LanceDB might store timestamps as microseconds/nanoseconds since epoch
                 # This needs testing based on actual LanceDB behavior
                 try:
                     # Assuming microseconds - adjust if needed
                     data[key] = datetime.fromtimestamp(data[key] / 1_000_000)
                 except Exception:
                     logger.warning(f"Could not parse timestamp {data[key]} for key {key}")
                     data[key] = None # Or handle differently
            elif key in data and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])

        # Remove the json field before creating the model
        data.pop('progress_json', None)
        return cls(**data)

class MediumTermMemory(BaseModel):
    """
    Medium-term memory for Jarvis (objective-centric) using LanceDB.
    This memory tracks user objectives and persists across sessions.
    Vector search functionality is pending embedding model integration.
    """
    
    db_path: str = Field(default="memory/db/medium_term")
    table_name: str = Field(default="objectives") # Renamed from implicit/collection
    # objectives: Dict[str, Objective] = Field(default_factory=dict) # Removed - fetch from DB
    current_objective_id: Optional[str] = None

    # LanceDB connection and table
    _db: Optional[lancedb.LanceDBConnection] = PrivateAttr(default=None)
    _table: Optional[lancedb.table.Table] = PrivateAttr(default=None)
    
    def __init__(self, **data):
        logger.info("MTM: Starting __init__")
        super().__init__(**data)
        logger.info("MTM: Finished super().__init__")
        try:
            logger.info("MTM: Initializing LanceDB setup...") # Changed log
            self._setup_db()
            logger.info("MTM: LanceDB setup finished.") # Changed log
        except Exception as e:
            logger.error(f"MTM: Error during LanceDB setup: {e}", exc_info=True) # Changed log
        # Remove _load_objectives call
        # try:
        #     print("MTM: Calling _load_objectives")
        #     self._load_objectives()
        #     print("MTM: Finished _load_objectives")
        # except Exception as e:
        #     print(f"MTM: Error during _load_objectives: {e}")
        #     raise
        logger.info("MTM: Finished __init__")
    
    def _setup_db(self):
        """Set up the LanceDB database for medium-term memory."""
        logger.info(f"MTM: Setting up DB at path: {self.db_path}")
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            logger.info(f"MTM: Connecting to LanceDB at {self.db_path}")
            self._db = lancedb.connect(self.db_path)
            
            # Define schema (PyArrow)
            pa_schema = pa.schema([
                pa.field("objective_id", pa.string(), nullable=False),
                pa.field("description", pa.string()),
                pa.field("status", pa.string()),
                pa.field("created_at", pa.timestamp('us')), # Use timestamp type
                pa.field("completed_at", pa.timestamp('us'), nullable=True),
                pa.field("progress_json", pa.string()) # Store progress list as JSON string
                # Add vector field later: pa.field("vector", pa.list_(pa.float32()))
            ])

            if self.table_name in self._db.table_names():
                logger.info(f"MTM: Opening existing table '{self.table_name}'")
                self._table = self._db.open_table(self.table_name)
                # TODO: Add schema validation/migration if needed
            else:
                 logger.info(f"MTM: Creating new table '{self.table_name}'")
                 self._table = self._db.create_table(self.table_name, schema=pa_schema, mode="create")
                 logger.info(f"MTM: Table '{self.table_name}' created successfully.")

        except ImportError:
             logger.error("MTM: 'pyarrow' is required for LanceDB schema definition. Please install it.")
             self._db = None
             self._table = None
        except Exception as e:
            logger.error(f"MTM: Failed to setup LanceDB: {e}", exc_info=True)
            self._db = None
            self._table = None
    
    # Removed _load_objectives and _save_objectives

    def _objective_to_lancedb_dict(self, objective: Objective) -> Dict[str, Any]:
        """Convert Objective Pydantic model to a dict suitable for LanceDB table."""
        data = objective.dict()
        data['progress_json'] = objective._progress_to_json()
        data.pop('progress', None) # Remove original list
        # Ensure datetimes are native for LanceDB (or compatible format)
        # LanceDB PyArrow schema expects native datetime, Pydantic serializes by default.
        # Let LanceDB handle the conversion based on schema.
        data['created_at'] = objective.created_at
        data['completed_at'] = objective.completed_at
        return data

    def create_objective(self, description: str) -> str:
        """Create a new objective, save it to LanceDB, and return its ID."""
        if not self._table:
            logger.error("MTM: LanceDB table is not available. Cannot create objective.")
            raise ConnectionError("MediumTermMemory database not initialized.")

        # Generate unique ID
        objective_id = f"obj_{datetime.now().strftime('%Y%m%d%H%M%S%f')}" # Added microsec for uniqueness
        
        # Create objective model instance
        objective = Objective(
            objective_id=objective_id,
            description=description
        )
        
        # Convert to LanceDB format
        data_dict = self._objective_to_lancedb_dict(objective)
        
        # Add to LanceDB table
        try:
            logger.debug(f"MTM: Adding objective to LanceDB: {objective_id}")
            self._table.add([data_dict])
            logger.info(f"MTM: Objective '{objective_id}' created successfully.")
            # Set as current objective
            self.current_objective_id = objective_id
            return objective_id
        except Exception as e:
            logger.error(f"MTM: Failed to add objective '{objective_id}' to LanceDB: {e}", exc_info=True)
            raise # Re-raise to indicate failure

    def get_objective(self, objective_id: str) -> Optional[Objective]:
        """Get an objective by ID from LanceDB."""
        if not self._table:
            logger.error("MTM: LanceDB table is not available. Cannot get objective.")
            return None
        
        try:
            logger.debug(f"MTM: Querying LanceDB for objective ID: {objective_id}")
            # Query LanceDB table
            result = self._table.search()
                           .where(f'objective_id = "{objective_id}"')
                           .limit(1)
                           .to_list() # Returns a list of dicts
            
            if result:
                logger.debug(f"MTM: Objective '{objective_id}' found.")
                # Convert back to Objective model
                return Objective.from_lancedb(result[0])
            else:
                logger.warning(f"MTM: Objective '{objective_id}' not found in LanceDB.")
                return None
        except Exception as e:
            logger.error(f"MTM: Failed to query objective '{objective_id}' from LanceDB: {e}", exc_info=True)
            return None

    def get_current_objective(self) -> Optional[Objective]:
        """Get the current objective from LanceDB."""
        if self.current_objective_id:
            return self.get_objective(self.current_objective_id)
        logger.debug("MTM: No current objective ID set.")
        return None

    def set_current_objective(self, objective_id: str) -> bool:
        """Set the current objective ID after verifying it exists in LanceDB."""
        if self.get_objective(objective_id): # Verify existence
            self.current_objective_id = objective_id
            logger.info(f"MTM: Current objective set to: {objective_id}")
            return True
        logger.warning(f"MTM: Attempted to set non-existent objective '{objective_id}' as current.")
        return False

    def _update_objective_in_db(self, objective: Objective):
        """Helper to update an existing objective in LanceDB (delete + add)."""
        if not self._table:
             logger.error(f"MTM: LanceDB table not available. Cannot update objective {objective.objective_id}.")
             raise ConnectionError("MediumTermMemory database not initialized.")
        
        try:
            logger.debug(f"MTM: Updating objective {objective.objective_id} in LanceDB.")
            # Delete existing record
            self._table.delete(f'objective_id = "{objective.objective_id}"')
            # Add updated record
            data_dict = self._objective_to_lancedb_dict(objective)
            self._table.add([data_dict])
            logger.info(f"MTM: Objective {objective.objective_id} updated successfully.")
        except Exception as e:
            logger.error(f"MTM: Failed to update objective {objective.objective_id} in LanceDB: {e}", exc_info=True)
            raise # Re-raise to signal update failure

    def add_progress(self, step: str, status: str = "completed", metadata: Optional[Dict[str, Any]] = None):
        """Add progress to the current objective and update it in LanceDB."""
        objective = self.get_current_objective()
        if not objective:
            logger.warning("MTM: Cannot add progress, no current objective found.")
            return False
        
        try:
            objective.add_progress(step, status, metadata)
            self._update_objective_in_db(objective)
            return True
        except Exception as e:
            # Error logged in _update_objective_in_db
            return False

    def complete_objective(self, objective_id: Optional[str] = None):
        """Complete an objective and update it in LanceDB."""
        obj_id_to_complete = objective_id or self.current_objective_id
        
        if not obj_id_to_complete:
            logger.warning("MTM: Cannot complete objective, no ID specified or current objective set.")
            return False
            
        objective = self.get_objective(obj_id_to_complete)
        if not objective:
            logger.warning(f"MTM: Cannot complete objective, objective ID '{obj_id_to_complete}' not found.")
            return False
            
        try:
            objective.complete()
            self._update_objective_in_db(objective)
            # If completing the current objective, maybe clear it?
            # if obj_id_to_complete == self.current_objective_id:
            #     self.current_objective_id = None
            return True
        except Exception as e:
             # Error logged in _update_objective_in_db
            return False
            
    def fail_objective(self, reason: str, objective_id: Optional[str] = None):
        """Fail an objective and update it in LanceDB."""
        obj_id_to_fail = objective_id or self.current_objective_id

        if not obj_id_to_fail:
            logger.warning("MTM: Cannot fail objective, no ID specified or current objective set.")
            return False

        objective = self.get_objective(obj_id_to_fail)
        if not objective:
            logger.warning(f"MTM: Cannot fail objective, objective ID '{obj_id_to_fail}' not found.")
            return False

        try:
            objective.fail(reason)
            self._update_objective_in_db(objective)
            # If failing the current objective, maybe clear it?
            # if obj_id_to_fail == self.current_objective_id:
            #     self.current_objective_id = None
            return True
        except Exception as e:
            # Error logged in _update_objective_in_db
            return False

    def search_objectives(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search objectives by semantic description (Currently Disabled - Requires Embeddings)."""
        if not self._table:
            logger.error("MTM: LanceDB table is not available. Cannot search objectives.")
            return []

        logger.warning("MTM: search_objectives called, but vector search is not implemented yet. Returning empty list.")
        # TODO: Implement vector generation for query and use self._table.search(...)
        # Example placeholder:
        # query_vector = generate_embedding(query)
        # results = self._table.search(query_vector).limit(n_results).to_list()
        # # Format results (convert back to Objective models or dicts)
        return [] 