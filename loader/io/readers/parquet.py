"""
Parquet Reader Module
====================

Lector especializado para archivos Parquet (.parquet).
Optimizado para lectura columnar eficiente con soporte
nativo para Dask.
"""

from pathlib import Path
from typing import List, Union, Iterator, Dict
import logging
import pandas as pd

from ...io.base import DASK_AVAILABLE
from .base import BaseReader

if DASK_AVAILABLE:
    import dask.dataframe as dd


class ParquetReader(BaseReader):
    """Lector especializado para archivos Parquet (.parquet)."""

    def read_columns(self, columns: List[str]) -> pd.DataFrame:
        """Lee columnas específicas del archivo Parquet"""
        return pd.read_parquet(str(self.file_path), columns=columns)

    def read_in_chunks(self, columns: List[str], chunk_size: int) -> Union[dd.DataFrame, Iterator[pd.DataFrame]]:
        """Lee en chunks optimizado para Parquet"""
        if DASK_AVAILABLE:
            return dd.read_parquet(str(self.file_path), columns=columns)
        else:
            # Leer completo y crear iterador manual
            df = self.read_columns(columns)

            def chunk_iterator():
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i + chunk_size]

            return chunk_iterator()

    def get_available_columns(self) -> List[str]:
        """Obtiene columnas disponibles eficientemente"""
        # Leer metadatos sin cargar datos
        return pd.read_parquet(str(self.file_path), columns=[]).columns.tolist()

    def extract_metadata(self) -> Dict:
        """Extrae metadatos del archivo Parquet"""
        metadata = self._extract_base_metadata()
        available_columns = self.get_available_columns()
        metadata['file_info']['file_format'] = 'Parquet'
        metadata['dataset_info'].update({'number_columns': len(available_columns)})
        metadata['variables'] = {'column_names': available_columns}
        return metadata


__all__ = [
    'ParquetReader'
]