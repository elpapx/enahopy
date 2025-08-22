"""
ENAHO Exceptions Module
======================

Jerarquía completa de excepciones personalizadas para enaho-analyzer.
Todas las excepciones incluyen contexto, códigos de error y timestamps
para facilitar debugging y logging.
"""

from datetime import datetime
from typing import Any, Optional


class ENAHOError(Exception):
    """Excepción base para errores de ENAHO"""

    def __init__(self, message: str, error_code: Optional[str] = None, **context):
        super().__init__(message)
        self.error_code = error_code
        self.context = context
        self.timestamp = datetime.now().isoformat()


class ENAHODownloadError(ENAHOError):
    """Error durante la descarga de archivos"""

    pass


class ENAHOValidationError(ENAHOError):
    """Error de validación de parámetros"""

    pass


class ENAHOIntegrityError(ENAHOError):
    """Error de integridad de archivos"""

    pass


class ENAHOTimeoutError(ENAHOError):
    """Error de timeout"""

    pass


class FileReaderError(ENAHOError):
    """Errores específicos de la lectura de archivos."""

    pass


class UnsupportedFormatError(FileReaderError):
    """Error para formatos de archivo no soportados."""

    pass


__all__ = [
    "ENAHOError",
    "ENAHODownloadError",
    "ENAHOValidationError",
    "ENAHOIntegrityError",
    "ENAHOTimeoutError",
    "FileReaderError",
    "UnsupportedFormatError",
]
