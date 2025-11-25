"""
ENAHOPY - Centralized Logging System
===================================

Unified logging configuration for all ENAHOPY operations.
Provides structured logging, performance tracking, and error reporting.

Author: ENAHOPY Team
Date: 2024-12-09
"""

import functools
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .exceptions import format_exception_for_logging


class ENAHOFormatter(logging.Formatter):
    """Custom formatter for ENAHOPY logs with structured output."""

    def __init__(self, structured: bool = False):
        """
        Initialize formatter.

        Args:
            structured: If True, output JSON-structured logs
        """
        self.structured = structured

        if structured:
            super().__init__()
        else:
            # Human-readable format
            super().__init__(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        if not self.structured:
            return super().format(record)

        # Structured JSON output
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add thread info if available
        if hasattr(record, "thread"):
            log_entry["thread"] = record.thread

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            }
        }

        if extra_fields:
            log_entry["extra"] = extra_fields

        import json

        return json.dumps(log_entry, ensure_ascii=False)


class ENAHOLogManager:
    """Centralized log manager for ENAHOPY."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.loggers = {}
            self.handlers = {}
            self.formatters = {}
            self._setup_default_configuration()
            self._initialized = True

    def _setup_default_configuration(self):
        """Setup default logging configuration."""
        # Create formatters
        self.formatters["human"] = ENAHOFormatter(structured=False)
        self.formatters["structured"] = ENAHOFormatter(structured=True)

        # Create handlers
        self.handlers["console"] = logging.StreamHandler(sys.stdout)
        self.handlers["console"].setFormatter(self.formatters["human"])

    def setup_logging(
        self,
        verbose: bool = True,
        structured: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        enable_performance: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ) -> logging.Logger:
        """
        Setup logging configuration for ENAHOPY.

        Args:
            verbose: Enable verbose logging
            structured: Use structured (JSON) logging
            log_file: Path to log file (optional)
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_performance: Enable performance logging
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup log files to keep

        Returns:
            Configured logger
        """
        # Determine log level
        if verbose and log_level == "INFO":
            log_level = "DEBUG"
        elif not verbose and log_level == "DEBUG":
            log_level = "INFO"

        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Get root ENAHOPY logger
        logger = logging.getLogger("enahopy")
        logger.setLevel(numeric_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter_type = "structured" if structured else "human"
        console_handler.setFormatter(self.formatters[formatter_type])
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)

        # File handler if requested
        if log_file:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
            )

            # Always use structured format for file logs
            file_handler.setFormatter(self.formatters["structured"])
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)

        # Performance logger
        if enable_performance:
            perf_logger = logging.getLogger("enahopy.performance")
            perf_logger.setLevel(logging.DEBUG)

            # Create separate performance log file if log_file is specified
            if log_file:
                perf_file = log_file_path.with_name(
                    f"{log_file_path.stem}_performance{log_file_path.suffix}"
                )
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
                )
                perf_handler.setFormatter(self.formatters["structured"])
                perf_logger.addHandler(perf_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        # Store configuration
        self.loggers["main"] = logger

        logger.info(
            "ENAHOPY logging initialized",
            extra={
                "log_level": log_level,
                "structured": structured,
                "log_file": log_file,
                "performance_logging": enable_performance,
            },
        )

        return logger

    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance.

        Args:
            name: Logger name (will be prefixed with 'enahopy.')

        Returns:
            Logger instance
        """
        if name:
            full_name = f"enahopy.{name}"
        else:
            full_name = "enahopy"

        return logging.getLogger(full_name)

    def log_exception(
        self,
        logger: logging.Logger,
        exception: Exception,
        operation_context: Optional[Dict[str, Any]] = None,
        log_level: str = "ERROR",
    ):
        """
        Log an exception with full context.

        Args:
            logger: Logger to use
            exception: Exception to log
            operation_context: Additional operation context
            log_level: Log level to use
        """
        exc_details = format_exception_for_logging(exception)

        extra_data = {
            "exception_details": exc_details,
            "operation_context": operation_context or {},
        }

        log_method = getattr(logger, log_level.lower(), logger.error)
        log_method(f"Exception occurred: {str(exception)}", extra=extra_data, exc_info=True)


# Global log manager instance
_log_manager = ENAHOLogManager()


def setup_logging(
    verbose: bool = True,
    structured: bool = False,
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    enable_performance: bool = False,
) -> logging.Logger:
    """
    Setup ENAHOPY logging (convenience function).

    Args:
        verbose: Enable verbose logging
        structured: Use structured (JSON) logging
        log_file: Path to log file (optional)
        log_level: Log level string
        enable_performance: Enable performance logging

    Returns:
        Configured logger
    """
    return _log_manager.setup_logging(
        verbose=verbose,
        structured=structured,
        log_file=log_file,
        log_level=log_level,
        enable_performance=enable_performance,
    )


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance (convenience function).

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return _log_manager.get_logger(name)


def log_exception(
    exception: Exception,
    operation_context: Optional[Dict[str, Any]] = None,
    log_level: str = "ERROR",
    logger_name: str = None,
):
    """
    Log an exception with context (convenience function).

    Args:
        exception: Exception to log
        operation_context: Additional context
        log_level: Log level to use
        logger_name: Logger name to use
    """
    logger = get_logger(logger_name)
    _log_manager.log_exception(logger, exception, operation_context, log_level)


def log_performance(func: Callable = None, *, logger_name: str = None, threshold: float = 1.0):
    """
    Decorator to log function performance.

    Args:
        func: Function to wrap (when used as @log_performance)
        logger_name: Logger name to use
        threshold: Only log if execution time > threshold seconds

    Returns:
        Decorated function
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or f"performance.{f.__module__}.{f.__name__}")

            start_time = time.time()
            start_perf = time.perf_counter()

            try:
                result = f(*args, **kwargs)
                success = True
                error_info = None
            except Exception as e:
                success = False
                error_info = {"exception_type": type(e).__name__, "exception_message": str(e)}
                raise
            finally:
                end_time = time.time()
                end_perf = time.perf_counter()

                execution_time = end_perf - start_perf

                if execution_time >= threshold:
                    extra_data = {
                        "function": f"{f.__module__}.{f.__name__}",
                        "execution_time": round(execution_time, 4),
                        "success": success,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                        "start_time": datetime.fromtimestamp(start_time).isoformat(),
                        "end_time": datetime.fromtimestamp(end_time).isoformat(),
                    }

                    if error_info:
                        extra_data["error"] = error_info

                    logger.info(
                        f"Function {f.__name__} executed in {execution_time:.4f}s", extra=extra_data
                    )

            return result

        return wrapper

    if func is None:
        # Called with arguments: @log_performance(threshold=2.0)
        return decorator
    else:
        # Called without arguments: @log_performance
        return decorator(func)


# Export public interface
__all__ = [
    "setup_logging",
    "get_logger",
    "log_exception",
    "log_performance",
    "ENAHOFormatter",
    "ENAHOLogManager",
]
