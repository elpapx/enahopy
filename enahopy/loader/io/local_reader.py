"""
ENAHO Local Reader Module
========================

Lector de archivos locales ENAHO con funcionalidades avanzadas.
Soporte para múltiples formatos, validación de columnas,
extracción de metadatos y exportación de datos.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union, Iterator, Dict, Any
import pandas as pd
import numpy as np

from ..core.config import ENAHOConfig
from ..core.exceptions import UnsupportedFormatError
from ..core.logging import setup_logging, log_performance
from ..io.base import DASK_AVAILABLE, IReader
from .readers.factory import ReaderFactory
from .validators.columns import ColumnValidator
from .validators.results import ColumnValidationResult
from .validators.enaho import ENAHOValidator

if DASK_AVAILABLE:
    import dask.dataframe as dd


class ENAHOLocalReader:
    """Lector de archivos locales ENAHO con funcionalidades avanzadas"""

    def __init__(self,
                 file_path: str,
                 config: Optional[ENAHOConfig] = None,
                 verbose: bool = True,
                 structured_logging: bool = False,
                 log_file: Optional[str] = None):
        """
        Inicializa el lector de archivos locales

        Args:
            file_path: Ruta al archivo local
            config: Configuración ENAHO
            verbose: Si mostrar logs detallados
            structured_logging: Si usar logging estructurado
            log_file: Archivo para logs
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
        """Obtiene la lista de columnas disponibles en el archivo"""
        return self.reader.get_available_columns()

    @log_performance
    def read_data(self,
                  columns: Optional[List[str]] = None,
                  use_chunks: bool = False,
                  chunk_size: Optional[int] = None,
                  ignore_missing_columns: bool = True,
                  case_sensitive: bool = False) -> Tuple[
        Union[pd.DataFrame, dd.DataFrame, Iterator[pd.DataFrame]], ColumnValidationResult]:
        """
        Lee datos del archivo local

        Args:
            columns: Lista de columnas a leer. Si es None, lee todas.
            use_chunks: Si usar lectura por chunks
            chunk_size: Tamaño de los chunks
            ignore_missing_columns: Si ignorar columnas faltantes
            case_sensitive: Si la búsqueda de columnas es case-sensitive

        Returns:
            Tupla con los datos y el resultado de validación de columnas
        """
        try:
            available_columns = self.reader.get_available_columns()

            # Si no se especifican columnas, usar todas las disponibles
            if columns is None:
                columns = available_columns

            validation_result = self.column_validator.validate_columns(
                columns, available_columns, case_sensitive
            )
            self.logger.info(f"Resultado de validación de columnas:\n{validation_result.get_summary()}")

            if validation_result.missing_columns and not ignore_missing_columns:
                raise ValueError(f"Columnas faltantes: {', '.join(validation_result.missing_columns)}")

            columns_to_read = validation_result.found_columns
            if not columns_to_read:
                self.logger.warning("No se encontraron columnas válidas para leer. Retornando DataFrame vacío.")
                return pd.DataFrame(), validation_result

            if use_chunks:
                data = self.reader.read_in_chunks(
                    columns_to_read,
                    chunk_size or self.config.chunk_size_default
                )
            else:
                data = self.reader.read_columns(columns_to_read)

            return data, validation_result

        except Exception as e:
            self.logger.error(f"Error al leer datos: {e}")
            raise

    def extract_metadata(self) -> Dict:
        """Extrae metadatos completos del archivo"""
        if self._metadata_cache:
            return self._metadata_cache

        self.logger.info("Extrayendo metadatos completos...")
        self._metadata_cache = self.reader.extract_metadata()
        self.logger.info("Metadatos extraídos exitosamente.")
        return self._metadata_cache

    def save_data(self,
                  data: Union[pd.DataFrame, dd.DataFrame],
                  output_path: str,
                  **kwargs) -> None:
        """
        Guarda datos en diferentes formatos

        Args:
            data: DataFrame a guardar
            output_path: Ruta de salida
            **kwargs: args para guardar
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        format_type = output_path_obj.suffix.lower().replace('.', '')

        save_handlers = {
            'csv': lambda df, p, **k: df.to_csv(p, index=False, **k),
            'parquet': lambda df, p, **k: df.to_parquet(p, **k),
            'dta': lambda df, p, **k: self._prepare_data_for_stata(df).to_stata(p, **k),
            'xlsx': lambda df, p, **k: df.to_excel(p, index=False, **k)
        }

        handler = save_handlers.get(format_type)
        if not handler:
            raise ValueError(f"Formato de guardado no soportado: {format_type}")

        self.logger.info(f"Guardando datos como {format_type.upper()} en {output_path_obj}...")

        # Si es Dask, decide si computar o usar métodos nativos
        if DASK_AVAILABLE and isinstance(data, dd.DataFrame):
            if format_type in ['csv', 'parquet']:
                if format_type == 'parquet':
                    data.to_parquet(str(output_path_obj))
                else:
                    data.to_csv(f"{output_path_obj.with_suffix('')}-*.csv", index=False)
            else:
                self.logger.info("Convirtiendo Dask DataFrame a Pandas para guardado...")
                data = data.compute()
                handler(data, str(output_path_obj), **kwargs)
        else:  # Si ya es Pandas o no hay Dask
            if hasattr(data, 'compute'):  # Es Dask pero no tenemos dd disponible
                data = data.compute()
            handler(data, str(output_path_obj), **kwargs)

        self.logger.info("Datos guardados exitosamente.")

    def save_metadata(self, output_path: str, **kwargs) -> None:
        """
        Guarda metadatos en diferentes formatos

        Args:
            output_path: Ruta de salida
            **kwargs: Argumentos adicionales
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        format_type = output_path_obj.suffix.lower().replace('.', '')

        if format_type == 'json':
            self._save_metadata_as_json(output_path_obj, **kwargs)
        elif format_type == 'csv':
            self._save_metadata_as_csv(output_path_obj, **kwargs)
        else:
            raise ValueError(f"Formato de guardado de metadatos no soportado: {format_type}")

        self.logger.info(f"Metadatos guardados como {format_type.upper()} en {output_path_obj}")

    def _save_metadata_as_json(self, output_path: Path, **kwargs):
        """Guarda metadatos como JSON"""
        metadata = self.extract_metadata()
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _prepare_variables_df(self) -> pd.DataFrame:
        """Prepara un DataFrame con el diccionario de variables."""
        metadata = self.extract_metadata()
        variables_info = metadata.get('variables', {})
        value_labels_info = metadata.get('value_labels', {})

        rows = []
        for name in variables_info.get('column_names', []):
            label_name = value_labels_info.get('variable_value_labels', {}).get(name)
            value_labels = value_labels_info.get('value_labels', {}).get(label_name, {})

            rows.append({
                "variable_name": name,
                "variable_label": variables_info.get('column_labels', {}).get(name, ""),
                "variable_type": variables_info.get('readstat_variable_types', {}).get(name, ""),
                "variable_format": variables_info.get('variable_format', {}).get(name, ""),
                "has_value_labels": bool(value_labels),
                "value_labels": json.dumps(value_labels, ensure_ascii=False) if value_labels else ""
            })
        return pd.DataFrame(rows)

    def _save_metadata_as_csv(self, output_path: Path, **kwargs):
        """Guarda metadatos como CSV"""
        df_variables = self._prepare_variables_df()
        df_variables.to_csv(output_path, index=False, encoding='utf-8')

    def _prepare_data_for_stata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara el DataFrame para exportación a Stata, manejando tipos de datos problemáticos."""
        data_copy = data.copy()

        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                # Convertir a string, manejando valores nulos
                data_copy[col] = data_copy[col].astype(str)
                # Reemplazar 'nan' y 'None' por valores vacíos
                data_copy[col] = data_copy[col].replace(['nan', 'None'], '')
                # Si toda la columna está vacía después de la limpieza, convertir a float
                if data_copy[col].str.strip().eq('').all():
                    data_copy[col] = np.nan
                    data_copy[col] = data_copy[col].astype(float)
            elif data_copy[col].dtype == 'bool':
                # Convertir booleanos a enteros
                data_copy[col] = data_copy[col].astype(int)

        return data_copy

    def get_summary_info(self) -> Dict[str, Any]:
        """Obtiene un resumen de información del archivo"""
        metadata = self.extract_metadata()
        available_columns = self.get_available_columns()

        return {
            "file_info": metadata.get('file_info', {}),
            "total_columns": len(available_columns),
            "sample_columns": available_columns[:10],  # Primeras 10 columnas como muestra
            "has_labels": bool(metadata.get('value_labels', {}).get('value_labels', {})),
            "dataset_info": metadata.get('dataset_info', {})
        }


__all__ = [
    'ENAHOLocalReader'
]