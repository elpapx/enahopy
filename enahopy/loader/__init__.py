"""

ENAHO Loader Package

====================



Sistema completo de descarga, lectura y procesamiento de datos ENAHO.

Punto de entrada principal para todas las funcionalidades del loader.

"""

# Imports desde core

from .core import (
    CacheManager,
    ENAHOConfig,
    ENAHODownloadError,
    ENAHOError,
    ENAHOIntegrityError,
    ENAHOTimeoutError,
    ENAHOValidationError,
    FileReaderError,
    UnsupportedFormatError,
    setup_logging,
)
from .io import ENAHODataDownloader  # Esta es la clase principal
from .io import (
    DASK_AVAILABLE,
    PYREADSTAT_AVAILABLE,
    BaseReader,
    ColumnValidationResult,
    ColumnValidator,
    CSVReader,
    ENAHOExtractor,
    ENAHOLocalReader,
    ENAHOValidator,
    IReader,
    NetworkUtils,
    ParquetReader,
    ReaderFactory,
    SPSSReader,
    StataReader,
)
from .utils import (
    ENAHOUtils,
    download_enaho_data,
    find_enaho_files,
    get_available_data,
    get_file_info,
    read_enaho_file,
    validate_download_request,
)

# Imports desde io - AQUÍ ESTÁ LA CLAVE


# Imports desde utils


# Definir qué se exporta cuando se hace "from enahopy.loader import *"

__all__ = [
    # Clases principales
    "ENAHODataDownloader",
    "ENAHOLocalReader",
    "ENAHOUtils",
    # Configuración
    "ENAHOConfig",
    # Funciones de conveniencia
    "download_enaho_data",
    "read_enaho_file",
    "get_file_info",
    "find_enaho_files",
    "get_available_data",
    "validate_download_request",
    # Excepciones
    "ENAHOError",
    "ENAHODownloadError",
    "ENAHOValidationError",
    "ENAHOIntegrityError",
    "ENAHOTimeoutError",
    "FileReaderError",
    "UnsupportedFormatError",
    # Validación
    "ENAHOValidator",
    "ColumnValidationResult",
    "ColumnValidator",
    # Readers
    "ReaderFactory",
    "IReader",
    "BaseReader",
    "SPSSReader",
    "StataReader",
    "ParquetReader",
    "CSVReader",
    # Utilidades
    "setup_logging",
    "CacheManager",
    "NetworkUtils",
    "ENAHOExtractor",
    # Flags de disponibilidad
    "DASK_AVAILABLE",
    "PYREADSTAT_AVAILABLE",
]


# Versión del módulo

__version__ = "0.1.0"
