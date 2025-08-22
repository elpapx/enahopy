"""
CSV Reader Module
================

Lector especializado para archivos CSV y TXT.
Soporte para lectura por chunks nativa de pandas
y configuración flexible de delimitadores.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Union

import pandas as pd

from ...io.base import DASK_AVAILABLE
from .base import BaseReader

if DASK_AVAILABLE:
    import dask.dataframe as dd


class CSVReader(BaseReader):
    """Lector especializado para archivos CSV (.csv, .txt)."""

    def read_columns(self, columns: List[str]) -> pd.DataFrame:
        """Lee columnas específicas del archivo CSV"""
        return pd.read_csv(str(self.file_path), usecols=columns, low_memory=False)

    def read_in_chunks(
        self, columns: List[str], chunk_size: int
    ) -> Union[dd.DataFrame, Iterator[pd.DataFrame]]:
        """Lee en chunks nativo para CSV"""
        if DASK_AVAILABLE:
            return dd.read_csv(
                str(self.file_path), usecols=columns, blocksize=f"{chunk_size * 100}B"
            )
        else:
            # Usar pandas chunk reader
            return pd.read_csv(str(self.file_path), usecols=columns, chunksize=chunk_size)

    def get_available_columns(self) -> List[str]:
        """Obtiene columnas leyendo solo headers"""
        return pd.read_csv(str(self.file_path), nrows=0).columns.tolist()

    def extract_metadata(self) -> Dict:
        """Extrae metadatos del archivo CSV"""
        metadata = self._extract_base_metadata()
        available_columns = self.get_available_columns()
        metadata["file_info"]["file_format"] = "CSV"
        metadata["dataset_info"].update({"number_columns": len(available_columns)})
        metadata["variables"] = {"column_names": available_columns}
        return metadata


__all__ = ["CSVReader"]
