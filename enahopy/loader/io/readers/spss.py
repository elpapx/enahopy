"""
SPSS Reader Module
=================

Lector especializado para archivos SPSS (.sav, .por).
Utiliza pyreadstat para lectura optimizada con soporte
para etiquetas de valores y metadatos.

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

# Import opcional para SPSS
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


class SPSSReader(BaseReader):
    """Lector especializado para archivos SPSS (.sav, .por) con metadatos completos.

    Implementa lectura de archivos SPSS usando pyreadstat con soporte para:
    - Variable labels y value labels
    - Formatos SPSS (.sav y .por)
    - Lectura selectiva de columnas
    - Optimización de tipos de datos
    - Extracción de metadatos de encuestas

    Similar a StataReader pero para formato SPSS. Útil cuando los datos
    ENAHO se distribuyen en formato SPSS o para compatibilidad con
    análisis existentes en SPSS.

    Examples:
        >>> from enahopy.loader.io.readers import SPSSReader
        >>> reader = SPSSReader(Path("datos/enaho.sav"), logger)
        >>> df = reader.read_columns(["p208a", "p209"])
        >>> # Value labels aplicados automáticamente
        >>> print(df["p208a"].unique())
        ['Sí', 'No']

    Note:
        - Requiere pyreadstat instalado
        - Soporta .sav (binario) y .por (portable)
        - Los value labels se aplican automáticamente
        - No soporta lectura incremental (limitación del formato)

    See Also:
        - :class:`StataReader`: Similar para archivos .dta
        - :class:`CSVReader`: Para exportaciones CSV de SPSS
    """

    def __init__(self, file_path: Path, logger: logging.Logger):
        super().__init__(file_path, logger)
        if not PYREADSTAT_AVAILABLE:
            raise UnsupportedFormatError("pyreadstat no está disponible para leer archivos SPSS")
        self._metadata_cache = None

    def read_columns(self, columns: List[str], optimize_dtypes: bool = True) -> pd.DataFrame:
        """Lee columnas específicas con value labels aplicados automáticamente.

        Args:
            columns: Lista de columnas a leer del archivo .sav o .por.
            optimize_dtypes: Si True, optimiza tipos de datos post-lectura.
                Defaults to True.

        Returns:
            DataFrame con columnas solicitadas y value labels aplicados.

        Example:
            >>> reader = SPSSReader(Path("datos/enaho.sav"), logger)
            >>> df = reader.read_columns(["p208a", "p209", "p210"])
            >>> print(df.dtypes)

        Note:
            Value labels se aplican con apply_value_formats=True.
        """
        df, _ = pyreadstat.read_sav(str(self.file_path), usecols=columns, apply_value_formats=True)

        if optimize_dtypes:
            df = self._optimize_dtypes(df)

        return df

    def read_in_chunks(
        self, columns: List[str], chunk_size: int, optimize_dtypes: bool = True
    ) -> Union[dd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Lee en chunks particionando el DataFrame completo.

        Note: SPSS files (.sav) no soportan lectura incremental nativa,
        por lo que se lee el archivo completo y se particiona en memoria.

        Args:
            columns: Columnas a leer
            chunk_size: Número de filas por chunk
            optimize_dtypes: Si optimizar tipos de datos

        Returns:
            Dask DataFrame o iterador optimizado
        """
        # Read full file first (SPSS limitation)
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
        Optimize DataFrame dtypes for SPSS data.

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
        _, meta = pyreadstat.read_sav(str(self.file_path), metadataonly=True)
        return meta.column_names

    def extract_metadata(self) -> Dict:
        """Extrae metadatos completos del archivo SPSS"""
        metadata = self._extract_base_metadata()
        _, meta = pyreadstat.read_sav(str(self.file_path), metadataonly=True)
        metadata["file_info"]["file_format"] = "SPSS"
        return self._populate_spss_dta_metadata(metadata, meta)


__all__ = ["SPSSReader", "PYREADSTAT_AVAILABLE"]
