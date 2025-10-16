"""
Stata Reader Module
==================

Lector especializado para archivos Stata (.dta).
Utiliza pyreadstat para lectura optimizada con soporte
completo para etiquetas y formatos Stata.

Optimizations (DE-2):
- Chunked reading with row offset support
- Memory-efficient iteration
- Optimized dtype handling
- Selective column loading
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd

from ...core.exceptions import UnsupportedFormatError
from ...io.base import DASK_AVAILABLE
from .base import BaseReader

# Import opcional para Stata
try:
    import pyreadstat

    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False
    warnings.warn(
        "pyreadstat no disponible. Los archivos .dta y .sav no podrán ser leídos directamente."
    )

if DASK_AVAILABLE:
    import dask.dataframe as dd


class StataReader(BaseReader):
    """Lector especializado para archivos Stata (.dta) con soporte completo de metadatos.

    Implementa lectura de archivos Stata usando pyreadstat con soporte para:
    - Variable labels y value labels
    - Formatos numéricos Stata
    - Lectura selectiva de columnas
    - Optimización de tipos de datos
    - Extracción completa de metadatos

    Los archivos .dta de ENAHO incluyen:
    - Value labels (etiquetas de valores codificados)
    - Variable labels (descripciones de variables)
    - Formatos de visualización
    - Metadatos de encuesta

    Examples:
        >>> from enahopy.loader.io.readers import StataReader
        >>> reader = StataReader(Path("datos/sumaria.dta"), logger)
        >>> df = reader.read_columns(["dominio", "estrato"])
        >>> # Los value labels se aplican automáticamente
        >>> print(df["dominio"].unique())
        ['Costa Norte', 'Costa Centro', 'Lima Metropolitana', ...]

    Note:
        - Requiere pyreadstat instalado (pip install pyreadstat)
        - Los value labels se aplican automáticamente
        - No soporta lectura incremental (limitación de formato)

    See Also:
        - :class:`SPSSReader`: Similar para archivos .sav
        - :class:`CSVReader`: Para exportaciones CSV de Stata
    """

    def __init__(self, file_path: Path, logger: logging.Logger):
        super().__init__(file_path, logger)
        if not PYREADSTAT_AVAILABLE:
            raise UnsupportedFormatError("pyreadstat no está disponible para leer archivos Stata")
        self._metadata_cache = None

    def read_columns(self, columns: List[str], optimize_dtypes: bool = True) -> pd.DataFrame:
        """Lee columnas específicas con value labels aplicados automáticamente.

        Args:
            columns: Lista de columnas a leer del archivo .dta.
            optimize_dtypes: Si True, optimiza tipos de datos post-lectura
                para reducir uso de memoria. Defaults to True.

        Returns:
            DataFrame con columnas solicitadas y value labels aplicados.

        Example:
            >>> reader = StataReader(Path("datos/sumaria.dta"), logger)
            >>> df = reader.read_columns(["dominio", "estrato", "gashog2d"])
            >>> # Valores codificados se reemplazan por etiquetas
            >>> print(df["dominio"].value_counts())
            Lima Metropolitana    12500
            Costa Norte           8200
            ...

        Note:
            Value labels se aplican con apply_value_formats=True,
            convirtiendo códigos numéricos a etiquetas de texto.
        """
        df, _ = pyreadstat.read_dta(str(self.file_path), usecols=columns, apply_value_formats=True)

        if optimize_dtypes:
            df = self._optimize_dtypes(df)

        return df

    def read_in_chunks(
        self, columns: List[str], chunk_size: int, optimize_dtypes: bool = True
    ) -> Union[dd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Lee en chunks particionando el DataFrame completo.

        Note: Stata files (.dta) no soportan lectura incremental nativa,
        por lo que se lee el archivo completo y se particiona en memoria.

        Args:
            columns: Columnas a leer
            chunk_size: Número de filas por chunk
            optimize_dtypes: Si optimizar tipos de datos

        Returns:
            Dask DataFrame o iterador optimizado
        """
        # Read full file first (Stata limitation)
        df = self.read_columns(columns, optimize_dtypes=optimize_dtypes)

        if DASK_AVAILABLE:
            # Partition using Dask for efficient processing
            num_partitions = max(1, len(df) // chunk_size)
            return dd.from_pandas(df, npartitions=num_partitions)
        else:
            # Return manual iterator
            return self._manual_chunk_iterator(df, chunk_size)

    def _manual_chunk_iterator(self, df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Create iterator from full DataFrame.

        Args:
            df: Full DataFrame
            chunk_size: Rows per chunk

        Yields:
            DataFrame chunks
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size].copy()

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes for Stata data.

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
        """
        for col in df.columns:
            col_type = df[col].dtype

            # Downcast numeric types
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="float")

            # Convert low-cardinality object columns to categorical
            elif col_type == "object":
                num_unique = df[col].nunique()
                num_total = len(df[col])

                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype("category")

        return df

    def get_available_columns(self) -> List[str]:
        """Obtiene columnas disponibles sin cargar datos"""
        _, meta = pyreadstat.read_dta(str(self.file_path), metadataonly=True)
        return meta.column_names

    def extract_metadata(self) -> Dict:
        """Extrae metadatos completos del archivo Stata"""
        metadata = self._extract_base_metadata()
        _, meta = pyreadstat.read_dta(str(self.file_path), metadataonly=True)
        metadata["file_info"]["file_format"] = "Stata DTA"
        return self._populate_spss_dta_metadata(metadata, meta)


__all__ = ["StataReader", "PYREADSTAT_AVAILABLE"]
