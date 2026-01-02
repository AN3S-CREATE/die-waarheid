"""
Structured logging configuration for Die Waarheid
JSON-formatted logging with structured fields for better analysis
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from config import LOG_DIR, LOG_LEVEL


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger wrapper for Die Waarheid
    Provides convenient methods for logging with structured fields
    """

    def __init__(self, name: str):
        """
        Initialize structured logger

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)

    def _log_with_fields(self, level: int, message: str, **fields):
        """
        Log message with structured fields

        Args:
            level: Log level
            message: Log message
            **fields: Additional structured fields
        """
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            None
        )
        record.extra_fields = fields
        self.logger.handle(record)

    def debug(self, message: str, **fields):
        """Log debug message with fields"""
        self._log_with_fields(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields):
        """Log info message with fields"""
        self._log_with_fields(logging.INFO, message, **fields)

    def warning(self, message: str, **fields):
        """Log warning message with fields"""
        self._log_with_fields(logging.WARNING, message, **fields)

    def error(self, message: str, **fields):
        """Log error message with fields"""
        self._log_with_fields(logging.ERROR, message, **fields)

    def critical(self, message: str, **fields):
        """Log critical message with fields"""
        self._log_with_fields(logging.CRITICAL, message, **fields)

    def analysis_started(self, case_id: str, filename: str, analysis_type: str):
        """Log analysis start"""
        self.info(
            f"Analysis started: {analysis_type}",
            case_id=case_id,
            filename=filename,
            analysis_type=analysis_type,
            event_type="analysis_start"
        )

    def analysis_completed(self, case_id: str, filename: str, duration: float, success: bool):
        """Log analysis completion"""
        self.info(
            f"Analysis completed",
            case_id=case_id,
            filename=filename,
            duration_seconds=duration,
            success=success,
            event_type="analysis_complete"
        )

    def api_call(self, api_name: str, method: str, duration: float, success: bool, status_code: Optional[int] = None):
        """Log API call"""
        self.info(
            f"API call: {api_name}.{method}",
            api_name=api_name,
            method=method,
            duration_seconds=duration,
            success=success,
            status_code=status_code,
            event_type="api_call"
        )

    def cache_operation(self, operation: str, filename: str, hit: bool):
        """Log cache operation"""
        self.debug(
            f"Cache {operation}",
            operation=operation,
            filename=filename,
            cache_hit=hit,
            event_type="cache_operation"
        )

    def database_operation(self, operation: str, table: str, success: bool, duration: float):
        """Log database operation"""
        self.info(
            f"Database {operation}",
            operation=operation,
            table=table,
            success=success,
            duration_seconds=duration,
            event_type="database_operation"
        )

    def validation_error(self, validation_type: str, field: str, error: str):
        """Log validation error"""
        self.warning(
            f"Validation error: {validation_type}",
            validation_type=validation_type,
            field=field,
            error=error,
            event_type="validation_error"
        )

    def performance_metric(self, metric_name: str, value: float, unit: str):
        """Log performance metric"""
        self.debug(
            f"Performance metric: {metric_name}",
            metric_name=metric_name,
            value=value,
            unit=unit,
            event_type="performance_metric"
        )


def setup_logging(log_dir: Optional[Path] = None, log_level: str = LOG_LEVEL) -> None:
    """
    Setup logging configuration for Die Waarheid

    Args:
        log_dir: Directory for log files (uses config if None)
        log_level: Logging level
    """
    if log_dir is None:
        log_dir = LOG_DIR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = JSONFormatter()

    file_handler = logging.FileHandler(log_dir / "die_waarheid.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    root_logger.info("Logging initialized", extra={'log_dir': str(log_dir), 'log_level': log_level})


def get_logger(name: str) -> StructuredLogger:
    """
    Get structured logger instance

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Logging system initialized")
    logger.analysis_started("CASE_001", "test.wav", "forensics")
    logger.analysis_completed("CASE_001", "test.wav", 5.2, True)
    logger.api_call("Gemini", "analyze_message", 1.5, True, 200)
    logger.cache_operation("set", "test.wav", False)
    logger.database_operation("insert", "analysis_results", True, 0.1)
