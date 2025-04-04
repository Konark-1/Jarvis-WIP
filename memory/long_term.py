import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging # Added

import lancedb # Added
from pydantic import BaseModel, Field, PrivateAttr
# import chromadb # Removed

logger = logging.getLogger(__name__) # Added

class LongTermMemory(BaseModel):
    """
    Long-term memory for Jarvis (core programming knowledge) using LanceDB.
    This memory persists across restarts and contains fundamental knowledge.
    Vector search functionality is pending embedding model integration.
    """
    
    db_path: str = Field(default="memory/db/long_term")
    table_name: str = Field(default="core_knowledge") # Renamed from collection_name
    
    # LanceDB connection and table
    _db: Optional[lancedb.LanceDBConnection] = PrivateAttr(default=None) # Changed
    _table: Optional[lancedb.table.Table] = PrivateAttr(default=None) # Changed
    
    def __init__(self, **data):
        super().__init__(**data)
        try:
             logger.info("LTM: Initializing LanceDB setup...") # Changed log
             self._setup_db()
             logger.info("LTM: LanceDB setup finished.") # Changed log
        except Exception as e:
            logger.error(f"LTM: Error during LanceDB setup: {e}", exc_info=True) # Changed log
            # Keep _db and _table as None if setup fails
    
    def _setup_db(self):
        """Set up the LanceDB database for long-term memory.""" # Modified docstring
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            logger.info(f"LTM: Connecting to LanceDB at {self.db_path}") # Added log
            self._db = lancedb.connect(self.db_path)
            
            # Define schema (without vector for now)
            # Store metadata as a string (JSON dump) as LanceDB prefers simple types or Arrow Structs
            schema = {
                "id": str,
                "content": str,
                "metadata_json": str, # Store metadata dict as JSON string
                "timestamp": datetime # LanceDB supports datetime
            }

            if self.table_name in self._db.table_names():
                logger.info(f"LTM: Opening existing table '{self.table_name}'")
                self._table = self._db.open_table(self.table_name)
                # TODO: Add schema validation/migration if needed in future
            else:
                 logger.info(f"LTM: Creating new table '{self.table_name}'")
                 # LanceDB requires data for schema inference or an explicit schema (pyarrow Schema)
                 # For simplicity, create with dummy data or define PyArrow schema.
                 # Let's try creating with an explicit PyArrow schema. Requires pyarrow installed.
                 try:
                     import pyarrow as pa
                     pa_schema = pa.schema([
                         pa.field("id", pa.string(), nullable=False),
                         pa.field("content", pa.string()),
                         pa.field("metadata_json", pa.string()),
                         pa.field("timestamp", pa.timestamp('us')) # Example timestamp type
                         # Add vector field here later: pa.field("vector", pa.list_(pa.float32()))
                     ])
                     self._table = self._db.create_table(self.table_name, schema=pa_schema, mode="overwrite") # Use overwrite for safety on init
                     logger.info(f"LTM: Table '{self.table_name}' created successfully.")
                 except ImportError:
                     logger.error("LTM: 'pyarrow' is required for explicit schema creation. Please install it.")
                     self._table = None # Ensure table is None if creation fails
                 except Exception as e:
                      logger.error(f"LTM: Failed to create table '{self.table_name}': {e}", exc_info=True)
                      self._table = None


        except Exception as e:
            logger.error(f"LTM: Failed to setup LanceDB: {e}", exc_info=True)
            self._db = None
            self._table = None

    def add_knowledge(self, knowledge_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add or update knowledge in long-term memory (overwrites if ID exists).""" # Modified docstring
        if not self._table:
            logger.error("LTM: LanceDB table is not available. Cannot add knowledge.")
            return

        if metadata is None:
            metadata = {}
        
        # Add timestamp to metadata if not present (LanceDB schema has dedicated field)
        timestamp = datetime.now()
        metadata["timestamp"] = timestamp.isoformat() # Keep in metadata for now, also use dedicated field

        try:
            # Prepare data according to schema
            data = {
                "id": knowledge_id,
                "content": content,
                "metadata_json": json.dumps(metadata), # Serialize metadata
                "timestamp": timestamp
                # Add vector field later
            }
            logger.debug(f"LTM: Adding/updating knowledge with ID: {knowledge_id}")
            # LanceDB's add typically merges/updates based on schema, but let's use delete + add for explicit update
            # Revisit: LanceDB `add` with schema might handle updates implicitly or use merge_insert
            try:
                 self.delete_knowledge(knowledge_id) # Ensure clean insert/update
            except Exception as del_e:
                 logger.warning(f"LTM: Pre-delete before add failed (may not exist yet): {del_e}")

            self._table.add([data])
            logger.info(f"LTM: Knowledge '{knowledge_id}' added/updated successfully.")

        except Exception as e:
            logger.error(f"LTM: Failed to add/update knowledge '{knowledge_id}': {e}", exc_info=True)

    def retrieve_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve knowledge by semantic search (Currently Disabled - Requires Embeddings).""" # Modified docstring
        if not self._table:
            logger.error("LTM: LanceDB table is not available. Cannot retrieve knowledge.")
            return []

        logger.warning("LTM: retrieve_knowledge called, but vector search is not implemented yet. Returning empty list.")
        # TODO: Implement vector generation for query and use self._table.search(...)
        # Example placeholder:
        # query_vector = generate_embedding(query) # Replace with actual embedding call
        # results = self._table.search(query_vector).limit(n_results).to_list()
        # # Format results (parse metadata_json, etc.)
        return []

    def delete_knowledge(self, knowledge_id: str):
        """Delete knowledge by its ID.""" # Modified docstring
        if not self._table:
            logger.error("LTM: LanceDB table is not available. Cannot delete knowledge.")
            return

        try:
            logger.debug(f"LTM: Deleting knowledge with ID: {knowledge_id}")
            # Construct SQL-like where clause for deletion
            where_clause = f'id = "{knowledge_id}"'
            self._table.delete(where_clause)
            logger.info(f"LTM: Knowledge '{knowledge_id}' deleted successfully (if it existed).")
        except Exception as e:
             # Catch potential errors if the delete operation fails (e.g., table issues)
             logger.error(f"LTM: Failed to delete knowledge '{knowledge_id}': {e}", exc_info=True)
             raise # Re-raise after logging? Or just log? Let's just log for now.

    def update_knowledge(self, knowledge_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Update knowledge. Simply calls add_knowledge which now handles overwrite.""" # Modified docstring
        logger.debug(f"LTM: update_knowledge called for ID: {knowledge_id}. Delegating to add_knowledge.")
        # The add_knowledge method now handles the update logic (delete + add)
        self.add_knowledge(knowledge_id, content, metadata) 