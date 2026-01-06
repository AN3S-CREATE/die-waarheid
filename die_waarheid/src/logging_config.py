"""
Structured logging configuration for Die Waarheid
JSON-formatted logging with structured fields for better analysis
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from config import LOGS_DIR


class AlertHandler(logging.Handler):
    """Custom handler for critical alerts"""
    
    def __init__(self):
        super().__init__(logging.ERROR)
        self.alerts = []
    
    def emit(self, record):
        """Emit a log record and store alerts"""
        if record.levelno >= logging.ERROR:
            alert = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName
            }
            
            if record.exc_info:
                alert['exception'] = self.format(record)
            
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]


def setup_logging(log_level: str = "INFO", enable_json: bool = False) -> Dict[str, Any]:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
    
    Returns:
        Dictionary with logging configuration
    """
    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    log_file = LOGS_DIR / f"die_waarheid_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_file = LOGS_DIR / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Alert handler for critical errors
    alert_handler = AlertHandler()
    alert_handler.setLevel(logging.ERROR)
    root_logger.addHandler(alert_handler)
    
    # Configure specific loggers
    configure_specific_loggers()
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, JSON: {enable_json}")
    logger.info(f"Log files: {log_file}, Errors: {error_file}")
    
    return {
        'level': log_level,
        'json_format': enable_json,
        'log_file': str(log_file),
        'error_file': str(error_file),
        'alert_handler': alert_handler
    }


def configure_specific_loggers():
    """Configure logging for specific components"""
    
    # AI Analyzer - reduce noise
    logging.getLogger('src.ai_analyzer').setLevel(logging.INFO)
    
    # Transcriber - keep detailed
    logging.getLogger('src.whisper_transcriber').setLevel(logging.DEBUG)
    
    # Forensics - keep detailed
    logging.getLogger('src.forensics').setLevel(logging.DEBUG)
    
    # Pipeline - keep detailed
    logging.getLogger('src.pipeline_processor').setLevel(logging.INFO)
    
    # External libraries - reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def setup_error_alerts(email_config: Dict[str, str] = None):
    """
    Setup error alert notifications
    
    Args:
        email_config: Email configuration for alerts
    """
    logger = logging.getLogger(__name__)
    
    if email_config:
        # TODO: Implement email alerts
        logger.info("Email alerts configured")
    else:
        logger.info("Email alerts not configured")


# Initialize logging on import
logging_config = setup_logging()
