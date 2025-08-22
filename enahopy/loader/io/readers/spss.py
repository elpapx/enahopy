"""
SPSS Reader Module
=================

Lector especializado para archivos SPSS (.sav, .por).
Utiliza pyreadstat para lectura optimizada con soporte
para etiquetas de valores y metadatos.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Union

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
    """Lector especializado para archivos SPSS (.sav, .por)."""

    def __init__(self, file_path: Path, logger: logging.Logger):
        super().__init__(file_path, logger)
        if not PYREADSTAT_AVAILABLE:
            raise UnsupportedFormatError("pyreadstat no está disponible para leer archivos SPSS")

    def read_columns(self, columns: List[str]) -> pd.DataFrame:
        """Lee columnas específicas del archivo SPSS"""
        df, _ = pyreadstat.read_sav(str(self.file_path), usecols=columns, apply_value_formats=True)
        return df

    def read_in_chunks(
        self, columns: List[str], chunk_size: int
    ) -> Union[dd.DataFrame, Iterator[pd.DataFrame]]:
        """Lee en chunks (limitado para SPSS)"""
        self.logger.warning(
            "Lectura por chunks no optimizada para SPSS. Leyendo completo y particionando."
        )
        df = self.read_columns(columns)

        if DASK_AVAILABLE:
            num_partitions = max(1, len(df) // chunk_size)
            return dd.from_pandas(df, npartitions=num_partitions)
        else:
            # Crear iterador manual si Dask no está disponible
            def chunk_iterator():
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i : i + chunk_size]

            return chunk_iterator()

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
