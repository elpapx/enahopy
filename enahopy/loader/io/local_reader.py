"""
ENAHO Local Reader Module
========================

Advanced local file reader for ENAHO data files.
Supports multiple formats, column validation,
metadata extraction, and data export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.config import ENAHOConfig
from ..core.exceptions import UnsupportedFormatError
from ..core.logging import log_performance, setup_logging
from ..io.base import DASK_AVAILABLE, IReader
from .readers.factory import ReaderFactory
from .validators.columns import ColumnValidator
from .validators.enaho import ENAHOValidator
from .validators.results import ColumnValidationResult

if DASK_AVAILABLE:
    import dask.dataframe as dd


class ENAHOLocalReader:
    """Advanced local file reader for ENAHO data with comprehensive features.

    Provides a unified interface for reading ENAHO files in multiple formats
    (.dta, .sav, .csv, .parquet) with column validation, metadata extraction,
    and flexible data export options.

    The ENAHOLocalReader is designed for working with already-downloaded ENAHO
    files, offering advanced features like selective column reading, chunked
    processing for large files, and complete metadata extraction including
    variable labels and value labels from Stata/SPSS files.

    Key Features:
    - Selective column reading with automatic validation
    - Chunked reading for memory-efficient processing of large files
    - Complete metadata extraction (labels, value labels, formats)
    - Multi-format export (CSV, Parquet, Stata DTA, Excel)
    - Optional Dask integration for distributed processing
    - Automatic format detection from file extension

    Attributes:
        config (ENAHOConfig): ENAHO configuration with default settings
        logger (logging.Logger): Logger for operation tracking
        validator (ENAHOValidator): Validator for files and modules
        file_path (Path): Path object pointing to the local file
        reader (IReader): Format-specific reader instance (CSV, Stata, etc.)
        column_validator (ColumnValidator): Validator for column names

    Examples:
        Basic reading of Stata file:

        >>> from enahopy import ENAHOLocalReader
        >>> reader = ENAHOLocalReader("data/sumaria-2023.dta")
        >>> df, validation = reader.read_data(
        ...     columns=["conglome", "vivienda", "hogar", "gashog2d"]
        ... )
        >>> print(validation.get_summary())
        Columns found: 4/4 (100.0%)
        >>> print(df.shape)
        (35000, 4)

        Chunked reading for large files:

        >>> reader = ENAHOLocalReader("data/enaho01-2023.dta")
        >>> data_chunks, validation = reader.read_data(
        ...     use_chunks=True,
        ...     chunk_size=5000
        ... )
        >>> for chunk in data_chunks:
        ...     # Process each chunk
        ...     result = process_chunk(chunk)
        ...     print(f"Processed chunk: {chunk.shape}")

        Complete metadata extraction:

        >>> reader = ENAHOLocalReader("data/modulo34.dta")
        >>> metadata = reader.extract_metadata()
        >>> print(metadata["file_info"]["file_format"])
        Stata DTA
        >>> print(metadata["variables"]["column_names"][:5])
        ['conglome', 'vivienda', 'hogar', 'ubigeo', 'dominio']
        >>> # Access value labels
        >>> print(metadata["value_labels"]["value_labels"].keys())

        Export to multiple formats:

        >>> reader = ENAHOLocalReader("data/sumaria.dta")
        >>> df, _ = reader.read_data()
        >>> # Save as CSV
        >>> reader.save_data(df, "output/sumaria.csv")
        >>> # Save as Parquet (more efficient)
        >>> reader.save_data(df, "output/sumaria.parquet")
        >>> # Save as Excel
        >>> reader.save_data(df, "output/sumaria.xlsx")

    Note:
        - Stata (.dta) and SPSS (.sav) files require pyreadstat package
        - Column matching is case-insensitive by default for flexibility
        - Metadata is automatically cached after first extraction
        - Compatible with Dask for datasets that don't fit in memory
        - Format is automatically detected from file extension

    See Also:
        - :class:`~enahopy.loader.io.main.ENAHODataDownloader`: For downloading files
        - :func:`~enahopy.loader.utils.io_utils.read_enaho_file`: Convenience function
        - :class:`~enahopy.loader.io.readers.factory.ReaderFactory`: Reader factory
    """

    def __init__(
        self,
        file_path: str,
        config: Optional[ENAHOConfig] = None,
        verbose: bool = True,
        structured_logging: bool = False,
        log_file: Optional[str] = None,
    ):
        """Initialize the ENAHO local file reader.

        Creates a reader instance configured for the specified file, automatically
        detecting the file format and setting up the appropriate specialized reader.

        Args:
            file_path: Path to the local file. Supported formats:
                - .dta (Stata binary format)
                - .sav (SPSS binary format)
                - .csv (Comma-separated values)
                - .txt (Text file, treated as CSV)
                - .parquet (Apache Parquet columnar format)
            config: Custom ENAHO configuration. If None, uses default configuration
                with standard settings. Defaults to None.
            verbose: If True, displays detailed logs of operations including file
                reading progress, validation results, and errors. Defaults to True.
            structured_logging: If True, uses JSON format for logs, ideal for log
                aggregation systems. If False, uses human-readable text format.
                Defaults to False.
            log_file: Path to file for saving logs. If None, logs only to console.
                Parent directories are created if needed. Defaults to None.

        Raises:
            FileNotFoundError: If file_path does not exist.
            UnsupportedFormatError: If file format is not supported or extension
                is not recognized.

        Examples:
            Basic configuration:

            >>> from enahopy import ENAHOLocalReader
            >>> reader = ENAHOLocalReader("data/sumaria.dta")
            >>> # Ready to read Stata file

            With custom configuration:

            >>> from enahopy.loader.core.config import ENAHOConfig
            >>> config = ENAHOConfig(chunk_size_default=20000)
            >>> reader = ENAHOLocalReader(
            ...     "data/sumaria.dta",
            ...     config=config,
            ...     verbose=True
            ... )
            >>> # Uses larger chunk size for reading

            With file logging:

            >>> reader = ENAHOLocalReader(
            ...     "data/sumaria.dta",
            ...     verbose=True,
            ...     log_file="./logs/reader.log"
            ... )
            >>> # Logs are written to both console and file

            Quiet mode for batch processing:

            >>> reader = ENAHOLocalReader("data/sumaria.dta", verbose=False)
            >>> # No console output

        Note:
            - File format is automatically detected from extension
            - Specialized readers are created via ReaderFactory
            - File existence is validated during initialization
            - For .dta and .sav files, pyreadstat must be installed

        See Also:
            - :class:`~enahopy.loader.io.readers.factory.ReaderFactory`: Reader creation
            - :class:`~enahopy.loader.core.config.ENAHOConfig`: Configuration options
            - :func:`~enahopy.loader.core.logging.setup_logging`: Logging configuration
        """
        self.config = config or ENAHOConfig()
        self.logger = setup_logging(verbose, structured_logging, log_file)
        self.validator = ENAHOValidator(self.config)
        self.file_path = self.validator.validate_file_exists(file_path)

        try:
            self.reader: IReader = ReaderFactory.create_reader(self.file_path, self.logger)
        except UnsupportedFormatError as e:
            self.logger.error(e)
            raise

        self.column_validator = ColumnValidator(self.logger)
        self._metadata_cache = None
        self.logger.info(f"Lector inicializado para archivo: {self.file_path}")

    def get_available_columns(self) -> List[str]:
        """Get list of available column names in the file.

        Retrieves column names without loading the entire dataset, making this
        operation very fast even for large files.

        Returns:
            List of column names as strings, in the order they appear in the file.

        Examples:
            List all columns in a file:

            >>> from enahopy import ENAHOLocalReader
            >>> reader = ENAHOLocalReader("data/sumaria-2023.dta")
            >>> columns = reader.get_available_columns()
            >>> print(f"Total columns: {len(columns)}")
            Total columns: 150
            >>> print(columns[:5])
            ['conglome', 'vivienda', 'hogar', 'ubigeo', 'dominio']

            Check if specific columns exist:

            >>> columns = reader.get_available_columns()
            >>> required_cols = ["gashog2d", "pobreza", "ingmo"]
            >>> missing = [col for col in required_cols if col not in columns]
            >>> if missing:
            ...     print(f"Missing columns: {missing}")

        Note:
            - Very fast operation, reads only file metadata
            - Does not load actual data into memory
            - Column order matches file structure

        See Also:
            - :meth:`read_data`: Read specific columns
            - :meth:`extract_metadata`: Get complete metadata including labels
        """
        return self.reader.get_available_columns()

    @log_performance
    def read_data(
        self,
        columns: Optional[List[str]] = None,
        use_chunks: bool = False,
        chunk_size: Optional[int] = None,
        ignore_missing_columns: bool = True,
        case_sensitive: bool = False,
    ) -> Tuple[Union[pd.DataFrame, dd.DataFrame, Iterator[pd.DataFrame]], ColumnValidationResult]:
        """Lee datos del archivo local con validación de columnas.

        Metodo principal para cargar datos con soporte para lectura selectiva,
        streaming por chunks, y validación automática de nombres de columnas.

        Args:
            columns: Lista de columnas a leer. Si None, lee todas las columnas
                disponibles. Los nombres se validan contra columnas disponibles.
                Defaults to None.
            use_chunks: Si True, retorna iterator o Dask DataFrame para lectura
                incremental. Útil para archivos >500MB. Defaults to False.
            chunk_size: Número de filas por chunk. Solo aplica si use_chunks=True.
                Si None, usa valor configurado (default: 10000). Defaults to None.
            ignore_missing_columns: Si True, lee columnas disponibles e ignora
                faltantes. Si False, lanza ValueError si hay columnas faltantes.
                Defaults to True.
            case_sensitive: Si True, búsqueda de columnas distingue mayúsculas.
                Si False, búsqueda case-insensitive. Defaults to False.

        Returns:
            Tupla con dos elementos:
            - data: DataFrame con datos (pd.DataFrame, dd.DataFrame si Dask disponible
                y use_chunks=True, o Iterator[pd.DataFrame] si use_chunks=True)
            - validation: ColumnValidationResult con información de validación:
                - found_columns: Columnas encontradas
                - missing_columns: Columnas no encontradas
                - get_summary(): Resumen textual

        Raises:
            ValueError: Si columnas faltantes y ignore_missing_columns=False.
            FileReaderError: Si hay errores leyendo el archivo.

        Examples:
            Lectura de columnas específicas:

            >>> reader = ENAHOLocalReader("datos/sumaria-2023.dta")
            >>> df, validation = reader.read_data(
            ...     columns=["conglome", "vivienda", "hogar", "gashog2d"]
            ... )
            >>> print(validation.get_summary())
            Columnas encontradas: 4/4 (100.0%)
            >>> print(df.shape)
            (35000, 4)

            Lectura por chunks:

            >>> reader = ENAHOLocalReader("datos/large_file.dta")
            >>> data_iter, validation = reader.read_data(
            ...     use_chunks=True,
            ...     chunk_size=5000
            ... )
            >>> for chunk in data_iter:
            ...     # Procesar cada chunk
            ...     print(f"Chunk shape: {chunk.shape}")
            ...     process(chunk)

            Manejo de columnas faltantes:

            >>> reader = ENAHOLocalReader("datos/modulo01.dta")
            >>> df, validation = reader.read_data(
            ...     columns=["conglome", "vivienda", "column_inexistente"],
            ...     ignore_missing_columns=True
            ... )
            >>> print(validation.found_columns)
            ['conglome', 'vivienda']
            >>> print(validation.missing_columns)
            ['column_inexistente']
            >>> print(df.shape[1])  # Solo 2 columnas cargadas
            2

            Búsqueda case-sensitive:

            >>> df, validation = reader.read_data(
            ...     columns=["CONGLOME", "vivienda"],  # CONGLOME en mayúsculas
            ...     case_sensitive=False  # Encuentra 'conglome'
            ... )
            >>> print(validation.found_columns)
            ['conglome', 'vivienda']

        Note:
            - La validación de columnas es case-insensitive por defecto
            - Para archivos grandes (>500MB), use use_chunks=True
            - El DataFrame vacío se retorna si no hay columnas válidas
            - Los metadatos de validación siempre se retornan

        See Also:
            - :meth:`get_available_columns`: Ver columnas disponibles
            - :meth:`extract_metadata`: Obtener metadatos completos
        """
        try:
            available_columns = self.reader.get_available_columns()

            # Si no se especifican columnas, usar todas las disponibles
            if columns is None:
                columns = available_columns

            validation_result = self.column_validator.validate_columns(
                columns, available_columns, case_sensitive
            )
            self.logger.info(
                f"Resultado de validación de columnas:\n{validation_result.get_summary()}"
            )

            if validation_result.missing_columns and not ignore_missing_columns:
                raise ValueError(
                    f"Columnas faltantes: {', '.join(validation_result.missing_columns)}"
                )

            columns_to_read = validation_result.found_columns
            if not columns_to_read:
                self.logger.warning(
                    "No se encontraron columnas válidas para leer. Retornando DataFrame vacío."
                )
                return pd.DataFrame(), validation_result

            if use_chunks:
                data = self.reader.read_in_chunks(
                    columns_to_read, chunk_size or self.config.chunk_size_default
                )
            else:
                data = self.reader.read_columns(columns_to_read)

            return data, validation_result

        except Exception as e:
            self.logger.error(f"Error al leer datos: {e}")
            raise

    def extract_metadata(self) -> Dict:
        """Extract complete metadata from the file.

        Retrieves comprehensive metadata including file information, variable labels,
        value labels (for Stata/SPSS), column names, and data types. Results are
        automatically cached for subsequent calls.

        Returns:
            Dictionary with complete metadata structure:
            - file_info: File details (path, size, format, timestamps)
            - dataset_info: Dataset characteristics (rows, columns, encoding)
            - variables: Variable information (names, labels, types, formats)
            - value_labels: Value label mappings (for Stata/SPSS files)

        Examples:
            Extract and explore metadata:

            >>> from enahopy import ENAHOLocalReader
            >>> reader = ENAHOLocalReader("data/sumaria-2023.dta")
            >>> metadata = reader.extract_metadata()
            >>> # Check file information
            >>> print(metadata["file_info"]["file_format"])
            Stata DTA
            >>> print(f"File size: {metadata['file_info']['file_size_mb']:.2f} MB")
            File size: 15.25 MB

            Access variable information:

            >>> # Get column names and labels
            >>> columns = metadata["variables"]["column_names"]
            >>> labels = metadata["variables"]["column_labels"]
            >>> for col in columns[:5]:
            ...     print(f"{col}: {labels.get(col, 'No label')}")
            conglome: Conglomerado
            vivienda: Vivienda
            hogar: Hogar
            ubigeo: Código de ubicación geográfica
            dominio: Dominio

            Access value labels (Stata/SPSS):

            >>> # For coded variables
            >>> value_labels = metadata["value_labels"]["value_labels"]
            >>> if "pobreza" in value_labels:
            ...     print(value_labels["pobreza"])
            {1: 'Pobre extremo', 2: 'Pobre no extremo', 3: 'No pobre'}

        Note:
            - First call extracts and caches metadata
            - Subsequent calls return cached results instantly
            - For Stata/SPSS files, includes value labels and variable labels
            - For CSV/Parquet files, metadata is more basic
            - Cache persists for the life of the reader instance

        See Also:
            - :meth:`get_available_columns`: Quick column name retrieval
            - :meth:`save_metadata`: Export metadata to file
            - :meth:`get_summary_info`: Get simplified summary
        """
        if self._metadata_cache:
            return self._metadata_cache

        self.logger.info("Extrayendo metadatos completos...")
        self._metadata_cache = self.reader.extract_metadata()
        self.logger.info("Metadatos extraídos exitosamente.")
        return self._metadata_cache

    def save_data(
        self, data: Union[pd.DataFrame, dd.DataFrame], output_path: str, **kwargs
    ) -> None:
        """Save DataFrame to file in various formats.

        Exports data to different file formats with automatic format detection
        from extension. Supports both pandas and Dask DataFrames.

        Args:
            data: DataFrame to save. Can be pandas DataFrame or Dask DataFrame.
            output_path: Output file path. Format is detected from extension:
                - .csv: Comma-separated values
                - .parquet: Apache Parquet (columnar, compressed)
                - .dta: Stata binary format
                - .xlsx: Excel format
            **kwargs: Additional arguments passed to the underlying save function:
                - For CSV: encoding, sep, compression, etc.
                - For Parquet: compression, engine, etc.
                - For Stata: write_index, version, etc.
                - For Excel: sheet_name, engine, etc.

        Raises:
            ValueError: If output format is not supported.

        Examples:
            Save as CSV:

            >>> from enahopy import ENAHOLocalReader
            >>> reader = ENAHOLocalReader("data/sumaria-2023.dta")
            >>> df, _ = reader.read_data(columns=["conglome", "gashog2d"])
            >>> reader.save_data(df, "output/sumaria.csv")

            Save as Parquet with compression:

            >>> reader.save_data(
            ...     df,
            ...     "output/sumaria.parquet",
            ...     compression='gzip'
            ... )

            Save back to Stata format:

            >>> reader.save_data(df, "output/sumaria_modified.dta")

            Save to Excel with specific sheet name:

            >>> reader.save_data(
            ...     df,
            ...     "output/sumaria.xlsx",
            ...     sheet_name="ENAHO 2023"
            ... )

        Note:
            - Parent directories are created automatically
            - Dask DataFrames are computed before saving (except CSV/Parquet)
            - Stata export handles data type conversions automatically
            - Parquet is recommended for large files (faster, smaller)
            - CSV is most portable but larger file size

        See Also:
            - :meth:`read_data`: Read data from file
            - :meth:`save_metadata`: Save metadata separately
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        format_type = output_path_obj.suffix.lower().replace(".", "")

        save_handlers = {
            "csv": lambda df, p, **k: df.to_csv(p, index=False, **k),
            "parquet": lambda df, p, **k: df.to_parquet(p, **k),
            "dta": lambda df, p, **k: self._prepare_data_for_stata(df).to_stata(p, **k),
            "xlsx": lambda df, p, **k: df.to_excel(p, index=False, **k),
        }

        handler = save_handlers.get(format_type)
        if not handler:
            raise ValueError(f"Formato de guardado no soportado: {format_type}")

        self.logger.info(f"Guardando datos como {format_type.upper()} en {output_path_obj}...")

        # Si es Dask, decide si computar o usar métodos nativos
        if DASK_AVAILABLE and isinstance(data, dd.DataFrame):
            if format_type in ["csv", "parquet"]:
                if format_type == "parquet":
                    data.to_parquet(str(output_path_obj))
                else:
                    data.to_csv(f"{output_path_obj.with_suffix('')}-*.csv", index=False)
            else:
                self.logger.info("Convirtiendo Dask DataFrame a Pandas para guardado...")
                data = data.compute()
                handler(data, str(output_path_obj), **kwargs)
        else:  # Si ya es Pandas o no hay Dask
            if hasattr(data, "compute"):  # Es Dask pero no tenemos dd disponible
                data = data.compute()
            handler(data, str(output_path_obj), **kwargs)

        self.logger.info("Datos guardados exitosamente.")

    def save_metadata(self, output_path: str, **kwargs) -> None:
        """Save metadata to file in JSON or CSV format.

        Exports file metadata including variable information, labels, and
        value labels to a human-readable format for documentation or sharing.

        Args:
            output_path: Output file path. Format detected from extension:
                - .json: Complete metadata in JSON format (recommended)
                - .csv: Variable dictionary as CSV table
            **kwargs: Additional arguments (format-specific, reserved for future use).

        Raises:
            ValueError: If output format is not supported (only JSON and CSV allowed).

        Examples:
            Export complete metadata as JSON:

            >>> from enahopy import ENAHOLocalReader
            >>> reader = ENAHOLocalReader("data/sumaria-2023.dta")
            >>> reader.save_metadata("output/sumaria_metadata.json")
            >>> # Creates JSON with full metadata structure

            Export variable dictionary as CSV:

            >>> reader.save_metadata("output/variable_dictionary.csv")
            >>> # Creates CSV with columns: variable_name, variable_label, type, etc.

        Note:
            - JSON format includes complete metadata structure
            - CSV format focuses on variable dictionary
            - Parent directories are created automatically
            - Metadata is extracted automatically if not cached
            - UTF-8 encoding is used for proper character support

        See Also:
            - :meth:`extract_metadata`: Get metadata dictionary
            - :meth:`get_summary_info`: Get simplified summary
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        format_type = output_path_obj.suffix.lower().replace(".", "")

        if format_type == "json":
            self._save_metadata_as_json(output_path_obj, **kwargs)
        elif format_type == "csv":
            self._save_metadata_as_csv(output_path_obj, **kwargs)
        else:
            raise ValueError(f"Formato de guardado de metadatos no soportado: {format_type}")

        self.logger.info(f"Metadatos guardados como {format_type.upper()} en {output_path_obj}")

    def _save_metadata_as_json(self, output_path: Path, **kwargs):
        """Guarda metadatos como JSON"""
        metadata = self.extract_metadata()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _prepare_variables_df(self) -> pd.DataFrame:
        """Prepara un DataFrame con el diccionario de variables."""
        metadata = self.extract_metadata()
        variables_info = metadata.get("variables", {})
        value_labels_info = metadata.get("value_labels", {})

        rows = []
        for name in variables_info.get("column_names", []):
            label_name = value_labels_info.get("variable_value_labels", {}).get(name)
            value_labels = value_labels_info.get("value_labels", {}).get(label_name, {})

            rows.append(
                {
                    "variable_name": name,
                    "variable_label": variables_info.get("column_labels", {}).get(name, ""),
                    "variable_type": variables_info.get("readstat_variable_types", {}).get(
                        name, ""
                    ),
                    "variable_format": variables_info.get("variable_format", {}).get(name, ""),
                    "has_value_labels": bool(value_labels),
                    "value_labels": (
                        json.dumps(value_labels, ensure_ascii=False) if value_labels else ""
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _save_metadata_as_csv(self, output_path: Path, **kwargs):
        """Guarda metadatos como CSV"""
        df_variables = self._prepare_variables_df()
        df_variables.to_csv(output_path, index=False, encoding="utf-8")

    def _prepare_data_for_stata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara el DataFrame para exportación a Stata, manejando tipos de datos problemáticos."""
        data_copy = data.copy()

        for col in data_copy.columns:
            if data_copy[col].dtype == "object":
                # Convertir a string, manejando valores nulos
                data_copy[col] = data_copy[col].astype(str)
                # Reemplazar 'nan' y 'None' por valores vacíos
                data_copy[col] = data_copy[col].replace(["nan", "None"], "")
                # Si toda la columna está vacía después de la limpieza, convertir a float
                if data_copy[col].str.strip().eq("").all():
                    data_copy[col] = np.nan
                    data_copy[col] = data_copy[col].astype(float)
            elif data_copy[col].dtype == "bool":
                # Convertir booleanos a enteros
                data_copy[col] = data_copy[col].astype(int)

        return data_copy

    def get_summary_info(self) -> Dict[str, Any]:
        """Get a summary of file information.

        Returns a simplified summary of key file characteristics without
        extracting complete metadata. Useful for quick file inspection.

        Returns:
            Dictionary containing:
            - file_info: Basic file details (path, size, format)
            - total_columns: Number of columns
            - sample_columns: First 10 column names as preview
            - has_labels: Whether value labels are present
            - dataset_info: Dataset characteristics

        Examples:
            Quick file inspection:

            >>> from enahopy import ENAHOLocalReader
            >>> reader = ENAHOLocalReader("data/sumaria-2023.dta")
            >>> summary = reader.get_summary_info()
            >>> print(f"Columns: {summary['total_columns']}")
            Columns: 150
            >>> print(f"Has labels: {summary['has_labels']}")
            Has labels: True
            >>> print(f"Preview: {summary['sample_columns']}")
            Preview: ['conglome', 'vivienda', 'hogar', ...]

        Note:
            - Faster than extract_metadata() for quick checks
            - Still extracts full metadata internally for completeness
            - Results include sample of column names for preview

        See Also:
            - :meth:`extract_metadata`: Get complete metadata
            - :meth:`get_available_columns`: Get all column names
        """
        metadata = self.extract_metadata()
        available_columns = self.get_available_columns()

        return {
            "file_info": metadata.get("file_info", {}),
            "total_columns": len(available_columns),
            "sample_columns": available_columns[:10],  # Primeras 10 columnas como muestra
            "has_labels": bool(metadata.get("value_labels", {}).get("value_labels", {})),
            "dataset_info": metadata.get("dataset_info", {}),
        }


__all__ = ["ENAHOLocalReader"]
