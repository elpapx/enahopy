"""
ENAHO Logging Module
===================

Sistema de logging estructurado y configuración de handlers.
Soporte para logging JSON estructurado y logging tradicional
con configuración flexible.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Formatter para logging estructurado"""

    def format(self, record):
        log_obj = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Agregar contexto adicional si está disponible
        if hasattr(record, 'context'):
            log_obj['context'] = record.context

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(verbose: bool = True, structured: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura el sistema de logging mejorado

    Args:
        verbose: Si mostrar logs detallados (INFO) o solo warnings
        structured: Si usar formato JSON estructurado
        log_file: Archivo opcional para guardar logs

    Returns:
        Logger configurado
    """
    logger = logging.getLogger('enaho_downloader')

    if logger.handlers:
        return logger

    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)

    handler = logging.StreamHandler()

    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Agregar handler de archivo si se especifica
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_performance(func):
    """Decorator para logging de performance"""
    import time
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log success
            logger = logging.getLogger('enaho_downloader')
            logger.debug(f"{func.__name__} executed successfully in {execution_time:.2f}s")

            return result
        except Exception as e:
            execution_time = time.time() - start_time

            # Log error
            logger = logging.getLogger('enaho_downloader')
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")

            raise

    return wrapper


__all__ = [
    'StructuredFormatter',
    'setup_logging',
    'log_performance'
]