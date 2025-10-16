"""
ENAHO Main Module
================

Main ENAHODataDownloader class that orchestrates all functionality:
download, extraction, local reading, and cache management.
Primary entry point for the enahopy library.
"""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.cache import CacheManager
from ..core.config import ENAHOConfig
from ..core.exceptions import ENAHOError
from ..core.logging import log_performance, setup_logging
from .downloaders.downloader import ENAHODownloader
from .downloaders.extractor import ENAHOExtractor
from .local_reader import ENAHOLocalReader
from .validators.enaho import ENAHOValidator


class ENAHODataDownloader:
    """Main ENAHO data downloader with advanced functionality.

    Primary class that orchestrates all download, extraction, and local reading
    operations for ENAHO files. Provides a unified interface for working with
    INEI data from both remote servers and local files.

    This class integrates:
    - Parallel or sequential download of multiple modules/years
    - Automatic ZIP file extraction
    - Reading and validation of local files (.dta, .sav, .csv, .parquet)
    - Intelligent caching system with TTL (Time-To-Live)
    - Module and year availability validation
    - Metadata and session management

    The ENAHODataDownloader serves as the main entry point for enahopy,
    handling everything from fetching data from INEI servers to managing
    local file operations with built-in validation and error recovery.

    Attributes:
        config (ENAHOConfig): ENAHO configuration including URLs, timeouts,
            cache settings, and download parameters.
        logger (logging.Logger): Configured logger for operation tracking
            and debugging.
        cache_manager (CacheManager): Cache manager for metadata, checksums,
            and download history management.
        validator (ENAHOValidator): Validator for modules, years, and file paths
            to ensure data integrity.
        downloader (ENAHODownloader): HTTP download handler with retry logic
            and progress tracking.
        extractor (ENAHOExtractor): ZIP file extraction handler supporting
            selective extraction.

    Examples:
        Basic usage to download and load data:

        >>> from enahopy import ENAHODataDownloader
        >>> loader = ENAHODataDownloader(verbose=True)
        >>> # Download summary module (sumaria) for 2023
        >>> loader.download(
        ...     modules=["34"],
        ...     years=["2023"],
        ...     output_dir="./data",
        ...     decompress=True
        ... )
        === Starting ENAHO cross-sectional download ===
        Modules: ['34']
        Years: ['2023']
        Downloading module 34 year 2023
        Download completed: modulo_34_2023.zip (15.2 MB)

        Parallel download with direct memory loading:

        >>> loader = ENAHODataDownloader()
        >>> data = loader.download(
        ...     modules=["01", "02", "34"],
        ...     years=["2023", "2022"],
        ...     decompress=True,
        ...     load_dta=True,
        ...     parallel=True,
        ...     max_workers=4
        ... )
        >>> # Access loaded data
        >>> df_sumaria_2023 = data[("2023", "34")]["enaho01-2023-34"]
        >>> print(df_sumaria_2023.shape)
        (35000, 150)

        Reading existing local files:

        >>> reader = loader.read_local_file("data/sumaria-2023.dta")
        >>> df, validation = reader.read_data(
        ...     columns=["conglome", "vivienda", "hogar", "gashog2d"]
        ... )
        >>> print(validation.get_summary())
        Columns found: 4/4 (100.0%)

        Validating availability before download:

        >>> result = loader.validate_availability(
        ...     modules=["01", "34"],
        ...     years=["2023", "2022"]
        ... )
        >>> if result["status"] == "valid":
        ...     print(f"Ready to download {result['estimated_downloads']} files")
        Ready to download 4 files

    Note:
        - Cache is automatically cleaned based on configured TTL
        - Uses default configuration if none provided
        - Downloaded files are validated automatically
        - Supports download interruption and resumption
        - Thread-safe for parallel downloads
        - Network connection required for downloads (offline mode for local files)

    See Also:
        - :class:`~enahopy.loader.io.local_reader.ENAHOLocalReader`: Local file reader
        - :class:`~enahopy.loader.core.config.ENAHOConfig`: Configuration class
        - :func:`~enahopy.loader.utils.io_utils.download_enaho_data`: Convenience function
    """

    def __init__(
        self,
        verbose: bool = True,
        structured_logging: bool = False,
        config: Optional[ENAHOConfig] = None,
    ):
        """Initialize the main ENAHO data downloader.

        Sets up the downloader with logging, caching, validation, and download
        infrastructure. Automatically cleans expired cache entries on initialization.

        Args:
            verbose: If True, displays detailed progress information including
                downloads, validations, and all operations. Useful for debugging
                and monitoring long-running operations. Defaults to True.
            structured_logging: If True, uses structured JSON format for logs,
                ideal for monitoring systems and log aggregation platforms.
                If False, uses simple text format for human readability.
                Defaults to False.
            config: Custom ENAHO configuration. If None, uses default configuration
                with standard URLs, timeouts, and cache settings. Allows customization
                of base_url, cache_dir, timeouts, retry logic, and more.
                Defaults to None (uses ENAHOConfig()).

        Examples:
            Basic configuration:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> # Uses default settings, verbose=True

            With structured logging for production:

            >>> loader = ENAHODataDownloader(
            ...     verbose=True,
            ...     structured_logging=True
            ... )
            >>> # Logs in JSON format for monitoring systems

            With custom configuration:

            >>> from enahopy.loader.core.config import ENAHOConfig
            >>> custom_config = ENAHOConfig(
            ...     cache_dir="/tmp/enaho_cache",
            ...     timeout=60,
            ...     max_retries=5
            ... )
            >>> loader = ENAHODataDownloader(config=custom_config)
            >>> # Uses custom cache location and timeout

            Quiet mode for batch processing:

            >>> loader = ENAHODataDownloader(verbose=False)
            >>> # No console output, useful for scripts

        Note:
            - Cache is automatically cleaned of expired entries on initialization
            - Configuration is immutable (dataclass frozen) after creation
            - Logger is configured based on verbose and structured_logging parameters
            - Default cache directory is ~/.enaho_cache/
            - Default TTL (Time-To-Live) for cache is 24 hours

        See Also:
            - :class:`~enahopy.loader.core.config.ENAHOConfig`: Configuration options
            - :class:`~enahopy.loader.core.cache.CacheManager`: Cache management
            - :func:`~enahopy.loader.core.logging.setup_logging`: Logging setup
        """
        self.config = config or ENAHOConfig()
        self.logger = setup_logging(verbose, structured_logging)
        self.cache_manager = CacheManager(self.config.cache_dir, self.config.cache_ttl_hours)
        self.validator = ENAHOValidator(self.config)
        self.downloader = ENAHODownloader(self.config, self.logger, self.cache_manager)
        self.extractor = ENAHOExtractor(self.logger)

        # Limpiar cache expirado al inicio
        self.cache_manager.clean_expired()

    def get_available_years(self, is_panel: bool = False) -> List[str]:
        """Get list of available years for ENAHO download.

        Retrieves the list of survey years available for download from INEI servers.
        Years are returned sorted from most recent to oldest.

        Args:
            is_panel: If True, returns years for panel longitudinal data
                (same households tracked over time). If False, returns years
                for cross-sectional data (independent samples per year).
                Defaults to False.

        Returns:
            List of available years as strings, sorted from most recent to oldest.
            Example: ["2023", "2022", "2021", ...]

        Examples:
            Get cross-sectional years (default):

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> years = loader.get_available_years()
            >>> print(years[:3])
            ['2023', '2022', '2021']
            >>> print(f"Total years available: {len(years)}")
            Total years available: 20

            Get panel longitudinal years:

            >>> panel_years = loader.get_available_years(is_panel=True)
            >>> print(panel_years)
            ['2023', '2022', '2021', '2020', ...]

            Check if specific year is available:

            >>> years = loader.get_available_years()
            >>> if "2023" in years:
            ...     print("2023 data is available for download")
            2023 data is available for download

        Note:
            - Cross-sectional data: Independent samples, most commonly used
            - Panel data: Same households tracked over multiple years
            - Year availability may vary by dataset type
            - Most recent year may have data updates throughout the year

        See Also:
            - :meth:`get_available_modules`: Get available survey modules
            - :meth:`validate_availability`: Validate before downloading
        """
        year_map = self.config.YEAR_MAP_PANEL if is_panel else self.config.YEAR_MAP_TRANSVERSAL
        return sorted(year_map.keys(), reverse=True)

    def get_available_modules(self) -> Dict[str, str]:
        """Get dictionary of available ENAHO modules with descriptions.

        Retrieves the complete list of ENAHO survey modules that can be downloaded,
        with Spanish descriptions from INEI's official module catalog.

        Returns:
            Dictionary where keys are module codes (strings) and values are
            Spanish descriptions. Returns a copy to prevent accidental modification.

            Example structure:
            {
                "01": "Características de la Vivienda y del Hogar",
                "02": "Características de los Miembros del Hogar",
                "34": "Sumarias ( Variables Calculadas )",
                ...
            }

        Examples:
            Get all available modules:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> modules = loader.get_available_modules()
            >>> print(modules["34"])
            Sumarias ( Variables Calculadas )

            List all modules with descriptions:

            >>> for code, desc in modules.items():
            ...     print(f"Module {code}: {desc}")
            Module 01: Características de la Vivienda y del Hogar
            Module 02: Características de los Miembros del Hogar
            Module 34: Sumarias ( Variables Calculadas )
            Module 37: Salud
            ...

            Check if specific module exists:

            >>> modules = loader.get_available_modules()
            >>> if "34" in modules:
            ...     print(f"Sumaria module available: {modules['34']}")
            Sumaria module available: Sumarias ( Variables Calculadas )

            Get module codes for download:

            >>> module_codes = list(modules.keys())
            >>> print(f"Total modules available: {len(module_codes)}")
            Total modules available: 15

        Note:
            - Returns a copy to prevent accidental modification of internal config
            - Module codes can have leading zeros (e.g., "01", "02")
            - Most commonly used modules: "34" (Sumaria), "01" (Housing), "02" (Members)
            - Descriptions are in Spanish as provided by INEI
            - Not all modules are available for all years

        See Also:
            - :meth:`get_available_years`: Get available years
            - :meth:`validate_availability`: Validate module/year combinations
            - :meth:`download`: Download specific modules
        """
        return self.config.AVAILABLE_MODULES.copy()

    def validate_availability(
        self, modules: List[str], years: List[str], is_panel: bool = False
    ) -> Dict[str, Any]:
        """Validate module and year availability without downloading files.

        Verifies that requested modules and years exist in the ENAHO database
        and returns a detailed validation report. Useful for checking requests
        before initiating long downloads.

        This method performs fast metadata-only validation, making it ideal
        for pre-flight checks in batch processing or user-facing applications.

        Args:
            modules: List of module codes to validate. Accepts codes with or
                without leading zeros (e.g., ["1", "34"] or ["01", "34"]).
                Module codes are automatically normalized.
            years: List of years to validate as strings (e.g., ["2023", "2022"]).
                Must match available years for the specified dataset type.
            is_panel: If True, validates against panel longitudinal years.
                If False, validates against cross-sectional years.
                Defaults to False.

        Returns:
            Dictionary with validation results:

            If valid:
                - status: "valid"
                - modules: Normalized module list (with leading zeros)
                - years: Validated year list
                - dataset_type: "panel" or "transversal"
                - estimated_downloads: Total downloads needed (modules × years)

            If invalid:
                - status: "invalid"
                - error: Descriptive error message
                - error_code: Error code for programmatic handling
                - context: Additional error context information

        Examples:
            Successful validation:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> result = loader.validate_availability(
            ...     modules=["1", "34"],
            ...     years=["2023", "2022"]
            ... )
            >>> print(result["status"])
            valid
            >>> print(result["estimated_downloads"])
            4
            >>> print(result["modules"])
            ['01', '34']

            Failed validation with invalid module:

            >>> result = loader.validate_availability(
            ...     modules=["99"],  # invalid module
            ...     years=["2023"]
            ... )
            >>> print(result["status"])
            invalid
            >>> print(result["error"])
            Module '99' not found in available modules

            Validation before batch download:

            >>> modules_to_download = ["01", "02", "34"]
            >>> years_to_download = ["2023", "2022", "2021"]
            >>> result = loader.validate_availability(modules_to_download, years_to_download)
            >>> if result["status"] == "valid":
            ...     print(f"Ready to download {result['estimated_downloads']} files")
            ...     loader.download(modules_to_download, years_to_download, output_dir="./data")
            ... else:
            ...     print(f"Validation failed: {result['error']}")
            Ready to download 9 files

            Programmatic error handling:

            >>> result = loader.validate_availability(["34"], ["1990"])  # year too old
            >>> if result["status"] == "invalid":
            ...     if result.get("error_code") == "INVALID_YEAR":
            ...         print("Year not available, trying more recent years")

        Note:
            - This method does NOT download files, only validates metadata
            - Fast operation, typically completes in <100ms
            - Recommended to use before large batch downloads
            - Module codes are automatically normalized (e.g., "1" becomes "01")
            - Returned error codes enable programmatic error handling

        See Also:
            - :meth:`get_available_modules`: View available modules
            - :meth:`get_available_years`: View available years
            - :meth:`download`: Main download method
        """
        try:
            normalized_modules = self.validator.validate_modules(modules)
            self.validator.validate_years(years, is_panel)

            return {
                "status": "valid",
                "modules": normalized_modules,
                "years": years,
                "dataset_type": "panel" if is_panel else "transversal",
                "estimated_downloads": len(normalized_modules) * len(years),
            }
        except Exception as e:
            return {
                "status": "invalid",
                "error": str(e),
                "error_code": getattr(e, "error_code", None),
                "context": getattr(e, "context", {}),
            }

    def read_local_file(self, file_path: str, **kwargs) -> ENAHOLocalReader:
        """Create a reader for a local ENAHO file.

        Factory method that creates and configures an ENAHOLocalReader instance
        for reading local ENAHO files. Inherits configuration from the downloader.

        Args:
            file_path: Path to the local file. Supports .dta, .sav, .csv, .parquet formats.
            **kwargs: Additional arguments passed to ENAHOLocalReader:
                - verbose (bool): Enable verbose logging. Default: True
                - structured_logging (bool): Use JSON format. Default: False
                - log_file (str): Path to log file. Default: None

        Returns:
            Configured ENAHOLocalReader instance ready to read the file.

        Raises:
            FileNotFoundError: If file_path does not exist.
            UnsupportedFormatError: If file format is not supported.

        Examples:
            Read a downloaded Stata file:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> reader = loader.read_local_file("data/sumaria-2023.dta")
            >>> df, validation = reader.read_data(
            ...     columns=["conglome", "vivienda", "gashog2d"]
            ... )

            Read with custom logging:

            >>> reader = loader.read_local_file(
            ...     "data/enaho01.dta",
            ...     verbose=True,
            ...     log_file="./logs/read.log"
            ... )

        Note:
            - Inherits config from parent ENAHODataDownloader
            - Automatically detects file format from extension
            - More convenient than creating ENAHOLocalReader directly

        See Also:
            - :class:`~enahopy.loader.io.local_reader.ENAHOLocalReader`: Reader class
            - :meth:`find_local_files`: Find multiple files in directory
            - :meth:`batch_read_local_files`: Read multiple files at once
        """
        return ENAHOLocalReader(
            file_path=file_path,
            config=self.config,
            verbose=kwargs.get("verbose", True),
            structured_logging=kwargs.get("structured_logging", False),
            log_file=kwargs.get("log_file", None),
        )

    def find_local_files(
        self, directory: Union[str, Path], pattern: str = "*.dta", recursive: bool = True
    ) -> List[Path]:
        """Find local ENAHO files in a directory.

        Searches for files matching a pattern in the specified directory,
        with optional recursive subdirectory search.

        Args:
            directory: Directory to search in. Can be string or Path object.
            pattern: File pattern to match. Supports glob patterns like:
                - "*.dta": All Stata files
                - "*sumaria*": Files containing "sumaria"
                - "enaho01-202*.dta": Specific pattern
                Defaults to "*.dta".
            recursive: If True, searches recursively in all subdirectories.
                If False, searches only in the specified directory.
                Defaults to True.

        Returns:
            List of Path objects for all matching files found.

        Raises:
            FileNotFoundError: If the specified directory does not exist.

        Examples:
            Find all Stata files recursively:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> files = loader.find_local_files("./data", pattern="*.dta")
            >>> print(f"Found {len(files)} .dta files")
            Found 15 .dta files

            Find only sumaria files:

            >>> sumaria_files = loader.find_local_files(
            ...     "./data",
            ...     pattern="*sumaria*.dta",
            ...     recursive=True
            ... )
            >>> for file in sumaria_files:
            ...     print(file.name)
            sumaria-2023.dta
            sumaria-2022.dta

            Non-recursive search:

            >>> files = loader.find_local_files(
            ...     "./data",
            ...     pattern="*.csv",
            ...     recursive=False
            ... )
            >>> # Only finds CSV files in ./data, not in subdirectories

        Note:
            - Supports glob patterns (wildcards)
            - Returns Path objects for easy manipulation
            - Recursive search can be slow for large directory trees
            - Case-sensitive on Linux, case-insensitive on Windows

        See Also:
            - :meth:`read_local_file`: Read a single file
            - :meth:`batch_read_local_files`: Read multiple files at once
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory}")

        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        self.logger.info(f"Encontrados {len(files)} archivos con patrón '{pattern}' en {directory}")
        return files

    def batch_read_local_files(
        self, file_paths: List[Union[str, Path]], columns: Optional[List[str]] = None, **read_kwargs
    ) -> Dict[str, Tuple]:
        """Read multiple local files in batch.

        Reads multiple ENAHO files sequentially, returning a dictionary with
        results and validation information for each file. Useful for processing
        multiple downloaded modules or years at once.

        Args:
            file_paths: List of file paths to read. Can be strings or Path objects.
            columns: Columns to read from each file. If None, reads all columns.
                Same columns are used for all files. Defaults to None.
            **read_kwargs: Additional keyword arguments passed to read_data():
                - use_chunks (bool): Enable chunked reading
                - chunk_size (int): Rows per chunk
                - ignore_missing_columns (bool): Ignore missing columns
                - case_sensitive (bool): Case-sensitive column matching

        Returns:
            Dictionary where keys are file stems (filename without extension)
            and values are tuples of (DataFrame, ColumnValidationResult).

            Example structure:
            {
                "sumaria-2023": (df_2023, validation_2023),
                "sumaria-2022": (df_2022, validation_2022),
                ...
            }

        Examples:
            Read multiple sumaria files:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> files = [
            ...     "data/sumaria-2023.dta",
            ...     "data/sumaria-2022.dta",
            ...     "data/sumaria-2021.dta"
            ... ]
            >>> results = loader.batch_read_local_files(
            ...     files,
            ...     columns=["conglome", "gashog2d", "pobreza"]
            ... )
            >>> for filename, (df, validation) in results.items():
            ...     print(f"{filename}: {df.shape}")
            sumaria-2023: (35000, 3)
            sumaria-2022: (34500, 3)
            sumaria-2021: (34000, 3)

            Combine results from multiple years:

            >>> import pandas as pd
            >>> results = loader.batch_read_local_files(
            ...     ["data/sumaria-2023.dta", "data/sumaria-2022.dta"],
            ...     columns=["gashog2d", "pobreza"]
            ... )
            >>> combined = pd.concat([df for df, _ in results.values()], ignore_index=True)
            >>> print(combined.shape)

            With error handling:

            >>> files = loader.find_local_files("./data", pattern="*2023*.dta")
            >>> results = loader.batch_read_local_files(
            ...     files,
            ...     columns=["conglome", "vivienda"],
            ...     ignore_missing_columns=True
            ... )
            >>> # Files with errors are skipped, logged, and not included in results

        Note:
            - Files are read sequentially, not in parallel
            - Files with errors are skipped and logged
            - Success rate is logged at the end
            - Returns only successfully read files
            - File stem (name without extension) used as dictionary key

        See Also:
            - :meth:`find_local_files`: Find files to read
            - :meth:`read_local_file`: Read single file
            - :class:`~enahopy.loader.io.local_reader.ENAHOLocalReader`: Reader class
        """
        results = {}

        for file_path in file_paths:
            file_path = Path(file_path)
            self.logger.info(f"Procesando archivo: {file_path.name}")

            try:
                reader = self.read_local_file(str(file_path))
                data, validation = reader.read_data(columns=columns, **read_kwargs)
                results[file_path.stem] = (data, validation)

            except Exception as e:
                self.logger.error(f"Error procesando {file_path.name}: {str(e)}")
                continue

        self.logger.info(f"Procesados exitosamente {len(results)}/{len(file_paths)} archivos")
        return results

    @log_performance
    def download(
        self,
        modules: List[str],
        years: List[str],
        output_dir: str = ".",
        is_panel: bool = False,
        decompress: bool = False,
        only_dta: bool = False,
        load_dta: bool = False,
        overwrite: bool = False,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        low_memory: bool = True,
        chunksize: Optional[int] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Optional[Dict]:
        """Descarga datos ENAHO con opciones avanzadas de procesamiento.

        Metodo principal que descarga módulos ENAHO desde servidores INEI,
        con soporte para descarga paralela, extracción automática, y carga
        directa en memoria. Incluye validación, retry automático, y gestión
        de metadatos de sesión.

        Args:
            modules: Lista de códigos de módulos a descargar. Acepta formatos
                con o sin ceros iniciales. Ejemplos: ["1", "2", "34"] o
                ["01", "02", "34"]. Ver get_available_modules() para opciones.
            years: Lista de años a descargar como strings. Ejemplo:
                ["2023", "2022", "2021"]. Ver get_available_years() para
                años disponibles según tipo de dataset.
            output_dir: Directorio donde guardar archivos descargados.
                Se crea automáticamente si no existe. Defaults to "." (actual).
            is_panel: Si True, descarga datos panel longitudinal. Si False,
                descarga corte transversal. Defaults to False.
            decompress: Si True, extrae automáticamente archivos ZIP después
                de descargar. Los archivos extraídos se guardan en subdirectorios
                nombrados como "modulo_{module}_{year}". Defaults to False.
            only_dta: Si True, extrae solo archivos .dta del ZIP (ignora .doc,
                .pdf, etc.). Requiere decompress=True. Defaults to False.
            load_dta: Si True, carga archivos .dta extraídos directamente en
                memoria como DataFrames. Requiere decompress=True.
                Defaults to False.
            overwrite: Si True, descarga archivos aunque ya existan localmente.
                Si False, reutiliza archivos válidos existentes. Defaults to False.
            parallel: Si True, descarga múltiples archivos en paralelo usando
                ThreadPoolExecutor. Recomendado para >3 archivos. Defaults to False.
            max_workers: Número máximo de workers paralelos. Si None, usa valor
                configurado (default: 4). Solo aplica si parallel=True.
                Defaults to None.
            verbose: Si True, muestra progreso detallado de cada operación.
                Defaults to True.
            low_memory: Si True, usa modo de bajo consumo de memoria al cargar
                .dta. Solo aplica si load_dta=True. Defaults to True.
            chunksize: Tamaño de chunks para lectura de .dta en modo streaming.
                Solo aplica si load_dta=True. Si None, carga archivo completo.
                Defaults to None.
            progress_callback: Función callback para monitoreo de progreso.
                Recibe (task_id: str, completed: int, total: int).
                Defaults to None.

        Returns:
            Si load_dta=True, retorna Dictionary con estructura:
            {(año, módulo): {nombre_archivo: DataFrame, ...}, ...}
            Ejemplo: {("2023", "34"): {"enaho01-2023-34": DataFrame}}

            Si load_dta=False, retorna None (archivos guardados en disco).

        Raises:
            ENAHOValidationError: Si módulos o años no son válidos.
            ENAHODownloadError: Si hay errores de red durante descarga.
            ENAHOError: Para errores inesperados.

        Examples:
            Descarga simple sin extracción:

            >>> loader = ENAHODataDownloader()
            >>> loader.download(
            ...     modules=["34"],
            ...     years=["2023"],
            ...     output_dir="./datos"
            ... )
            === Iniciando descarga ENAHO corte transversal ===
            Módulos: ['34']
            Años: ['2023']
            Total de descargas programadas: 1
            Descargando módulo 34 año 2023
            Descarga completada: modulo_34_2023.zip (15.2 MB)

            Descarga con extracción y carga en memoria:

            >>> data = loader.download(
            ...     modules=["01", "34"],
            ...     years=["2023", "2022"],
            ...     output_dir="./datos",
            ...     decompress=True,
            ...     only_dta=True,
            ...     load_dta=True
            ... )
            >>> # Acceder a datos cargados
            >>> df_sumaria = data[("2023", "34")]["enaho01-2023-34"]
            >>> print(f"Shape: {df_sumaria.shape}")
            Shape: (35000, 150)

            Descarga paralela de múltiples años:

            >>> loader.download(
            ...     modules=["34"],
            ...     years=["2023", "2022", "2021", "2020"],
            ...     output_dir="./datos",
            ...     decompress=True,
            ...     parallel=True,
            ...     max_workers=4
            ... )
            Descarga paralela con 4 workers
            Total de descargas programadas: 4
            [████████████████████] 4/4 completas
            === Resumen de descarga ===
            Tareas completadas: 4/4
            Tasa de éxito: 100.0%
            Tiempo total: 45.3 segundos

            Con callback de progreso personalizado:

            >>> def mi_callback(task_id, completed, total):
            ...     print(f"Progreso: {completed}/{total} - {task_id}")
            >>> loader.download(
            ...     modules=["34"],
            ...     years=["2023", "2022"],
            ...     progress_callback=mi_callback
            ... )

        Note:
            - Los archivos se validan automáticamente post-descarga
            - El cache reutiliza archivos válidos si overwrite=False
            - Descarga paralela es más rápida pero usa más recursos
            - Los metadatos de sesión se guardan en cache automáticamente
            - Para datasets >500MB, considere desactivar load_dta

        See Also:
            - :meth:`validate_availability`: Validar antes de descargar
            - :meth:`get_download_history`: Ver historial de descargas
            - :meth:`read_local_file`: Leer archivos descargados previamente
        """

        start_time = time.time()

        try:
            # Validaciones
            normalized_modules = self.validator.validate_modules(modules)
            self.validator.validate_years(years, is_panel)
            output_path = self.validator.validate_output_dir(output_dir)

            # Configuración
            year_mapping = (
                self.config.YEAR_MAP_PANEL if is_panel else self.config.YEAR_MAP_TRANSVERSAL
            )
            dataset_type = "panel" if is_panel else "corte transversal"

            if verbose:
                self.logger.info(f"=== Iniciando descarga ENAHO {dataset_type} ===")
                self.logger.info(f"Módulos: {normalized_modules}")
                self.logger.info(f"Años: {years}")
                self.logger.info(f"Directorio: {output_path}")

            # Preparar tareas
            tasks = [
                (year, module, year_mapping[year])
                for year in years
                for module in normalized_modules
            ]
            total_tasks = len(tasks)

            if verbose:
                self.logger.info(f"Total de descargas programadas: {total_tasks}")

            # Resultados
            all_results = {} if load_dta else None
            completed_tasks = 0
            failed_tasks = []

            # Función auxiliar para procesar una tarea
            def process_task(year: str, module: str, code: int) -> Tuple[str, str, Optional[Dict]]:
                try:
                    result = self._process_single_download(
                        year,
                        module,
                        code,
                        output_path,
                        overwrite,
                        decompress,
                        only_dta,
                        load_dta,
                        verbose,
                        low_memory,
                        chunksize,
                    )
                    return year, module, result
                except Exception as e:
                    self.logger.error(f"Error procesando {year}-{module}: {str(e)}")
                    return year, module, None

            # Ejecutar descargas
            if parallel and total_tasks > 1:
                workers = min(max_workers or self.config.default_max_workers, total_tasks)
                if verbose:
                    self.logger.info(f"Descarga paralela con {workers} workers")

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_task = {
                        executor.submit(process_task, year, module, code): (year, module)
                        for year, module, code in tasks
                    }

                    for future in as_completed(future_to_task):
                        year, module = future_to_task[future]
                        try:
                            task_year, task_module, result = future.result()
                            completed_tasks += 1

                            if result is not None and load_dta:
                                all_results[(task_year, task_module)] = result
                            elif result is None:
                                failed_tasks.append((task_year, task_module))

                            # Callback de progreso
                            if progress_callback:
                                progress_callback(
                                    f"{task_year}-{task_module}", completed_tasks, total_tasks
                                )

                        except Exception as e:
                            failed_tasks.append((year, module))
                            self.logger.error(f"Error en future para {year}-{module}: {str(e)}")
            else:
                # Descarga secuencial
                for year, module, code in tasks:
                    task_year, task_module, result = process_task(year, module, code)
                    completed_tasks += 1

                    if result is not None and load_dta:
                        all_results[(task_year, task_module)] = result
                    elif result is None:
                        failed_tasks.append((task_year, task_module))

                    # Callback de progreso
                    if progress_callback:
                        progress_callback(
                            f"{task_year}-{task_module}", completed_tasks, total_tasks
                        )

            # Resumen final
            elapsed_time = time.time() - start_time
            success_rate = ((total_tasks - len(failed_tasks)) / total_tasks) * 100

            if verbose:
                self.logger.info("=== Resumen de descarga ===")
                self.logger.info(f"Tareas completadas: {completed_tasks}/{total_tasks}")
                self.logger.info(f"Tasa de éxito: {success_rate:.1f}%")
                self.logger.info(f"Tiempo total: {elapsed_time:.1f} segundos")

                if failed_tasks:
                    self.logger.warning(f"Tareas fallidas: {failed_tasks}")

            # Guardar metadatos de la sesión
            session_metadata = {
                "timestamp": datetime.now().isoformat(),
                "modules": normalized_modules,
                "years": years,
                "dataset_type": dataset_type,
                "total_tasks": total_tasks,
                "successful_tasks": total_tasks - len(failed_tasks),
                "failed_tasks": failed_tasks,
                "elapsed_time": elapsed_time,
                "success_rate": success_rate,
            }
            self.cache_manager.set_metadata("last_download_session", session_metadata)

            return all_results

        except ENAHOError:
            raise
        except Exception as e:
            self.logger.error(f"Error inesperado durante la descarga: {str(e)}")
            raise ENAHOError(f"Error inesperado: {str(e)}", "UNEXPECTED_ERROR")

    def _process_single_download(
        self,
        year: str,
        module: str,
        code: int,
        output_dir: Path,
        overwrite: bool,
        decompress: bool,
        only_dta: bool,
        load_dta: bool,
        verbose: bool,
        low_memory: bool,
        chunksize: Optional[int],
    ) -> Optional[Dict]:
        """Procesa una descarga individual"""
        try:
            # Descargar archivo
            zip_path = self.downloader.download_file(
                year, module, code, output_dir, overwrite, verbose
            )

            if not decompress:
                return None

            # Crear directorio de extracción
            extract_dir = output_dir / f"modulo_{module}_{year}"
            extract_dir.mkdir(exist_ok=True)

            # Extraer archivo
            extracted_files = self.extractor.extract_zip(
                zip_path, extract_dir, only_dta, flatten=True
            )

            if verbose:
                self.logger.info(f"Extraídos {len(extracted_files)} archivos en: {extract_dir}")

            # Eliminar ZIP después de extraer (opcional)
            if decompress:  # Solo eliminar si se extrajo exitosamente
                zip_path.unlink()
                if verbose:
                    self.logger.info(f"Archivo ZIP eliminado: {zip_path.name}")

            # Cargar archivos .dta si se solicita
            if load_dta:
                return self.extractor.load_dta_files(
                    extract_dir, low_memory=low_memory, chunksize=chunksize
                )

            return None

        except Exception as e:
            self.logger.error(f"Error procesando {year}-{module}: {str(e)}")
            raise

    def get_download_history(self) -> Optional[Dict]:
        """Get download history from the last session.

        Retrieves metadata from the most recent download operation, including
        success rates, timing information, and file details.

        Returns:
            Dictionary with last session metadata, or None if no history exists.

            Structure includes:
            - timestamp: ISO format timestamp of session
            - modules: List of modules downloaded
            - years: List of years downloaded
            - dataset_type: "panel" or "transversal"
            - total_tasks: Total downloads attempted
            - successful_tasks: Number of successful downloads
            - failed_tasks: List of failed (year, module) tuples
            - elapsed_time: Total time in seconds
            - success_rate: Percentage of successful downloads

        Examples:
            Check last download session:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> history = loader.get_download_history()
            >>> if history:
            ...     print(f"Last download: {history['timestamp']}")
            ...     print(f"Success rate: {history['success_rate']:.1f}%")
            Last download: 2024-01-15T10:30:45
            Success rate: 100.0%

        Note:
            - Returns None if no downloads have been performed
            - History is stored in cache and survives between sessions
            - Only tracks the most recent download session

        See Also:
            - :meth:`download`: Main download method that creates history
            - :meth:`export_metadata`: Export complete metadata including history
        """
        return self.cache_manager.get_metadata("last_download_session")

    def clean_cache(self) -> None:
        """Clean the cache completely.

        Removes all cached data including download history, metadata, and
        temporary files. This operation cannot be undone.

        Examples:
            Clear cache to force fresh downloads:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> loader.clean_cache()
            >>> # Next download will fetch fresh data from servers

        Warning:
            This will delete all cached data and download history.
            Use with caution.

        See Also:
            - :class:`~enahopy.loader.core.cache.CacheManager`: Cache management
        """
        if self.cache_manager.cache_dir.exists():
            shutil.rmtree(self.cache_manager.cache_dir)
            self.cache_manager.cache_dir.mkdir(exist_ok=True)
        self.logger.info("Cache limpiado")

    def export_metadata(self, output_file: str) -> None:
        """Export configuration and availability metadata to JSON file.

        Exports complete metadata including available years, modules, configuration
        settings, and download history to a JSON file for documentation or sharing.

        Args:
            output_file: Path to output JSON file. Directory will be created if needed.

        Examples:
            Export metadata for documentation:

            >>> from enahopy import ENAHODataDownloader
            >>> loader = ENAHODataDownloader()
            >>> loader.export_metadata("metadata/enaho_info.json")
            >>> # Creates JSON file with all available modules, years, etc.

        Note:
            - Output file is UTF-8 encoded JSON with indentation
            - Includes timestamp of export
            - Useful for documenting available data

        See Also:
            - :meth:`get_available_modules`: Module information
            - :meth:`get_available_years`: Year information
            - :meth:`get_download_history`: Download history
        """
        metadata = {
            "config": {
                "available_years_transversal": self.get_available_years(False),
                "available_years_panel": self.get_available_years(True),
                "available_modules": self.get_available_modules(),
                "base_url": self.config.base_url,
                "cache_dir": self.config.cache_dir,
            },
            "last_session": self.get_download_history(),
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Metadatos exportados a: {output_file}")


__all__ = ["ENAHODataDownloader"]
