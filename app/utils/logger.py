# -*- coding: utf-8 -*-
import logging
import logging.handlers
import os
import sys
from datetime import datetime

def setup_logger(app_name="manga_translator"):
    """
    Sets up the root logger with rotation and console output.
    Logs are saved to the 'logs' directory in the project root.
    """
    # 1. Ensure logs directory exists
    # We assume CWD is the project root when running the app
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{app_name}.log")
    
    # 2. Get Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication during reloads/restarts
    if root_logger.handlers:
        root_logger.handlers = []
    
    # 3. Define Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 4. File Handler (Rotating)
    # 5 MB max size, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 5. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 6. Global Exception Hook to catch crashes
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))
        
    sys.excepthook = handle_exception
    
    root_logger.info(f"Logging initialized. Log file: {log_file}")
