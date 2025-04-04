import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger with the given name and log level"""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler for logging to a file
    log_file = os.path.join(logs_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # --- ADDED: Configure library log levels ---
    # Reduce noise from libraries unless in debug mode
    if log_level > logging.DEBUG:
        for lib_name in ["httpx", "chromadb", "pydantic", "httpcore", "openai", "groq", "langchain", "crewai"]:
             lib_logger = logging.getLogger(lib_name)
             # Check if handlers already exist to avoid duplication if libraries also configure
             if not lib_logger.hasHandlers(): 
                 # Add a NullHandler to prevent library logs from propagating to the root logger
                 # if they don't configure their own handlers.
                 # Or set a specific level if we *want* to see their warnings/errors.
                 # lib_logger.addHandler(logging.NullHandler())
                 lib_logger.setLevel(logging.WARNING) # Only show WARNING and above
                 # Optionally add our handlers if we want their logs in our files/console
                 # lib_logger.addHandler(file_handler)
                 # lib_logger.addHandler(console_handler)
                 # lib_logger.propagate = False # Prevent double logging if handlers added
             else:
                 lib_logger.setLevel(logging.WARNING) # Set level even if handlers exist
    # --- END ADDED SECTION ---
    
    return logger 