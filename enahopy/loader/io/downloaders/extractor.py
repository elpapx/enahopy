"""
File Extraction Module
======================

Extractor especializado para archivos ZIP de ENAHO.
Soporte para filtros personalizados, validación de integridad
y optimización de memoria para archivos grandes.
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ...core.exceptions import ENAHOError

# Import opcional para lectura de archivos
try:
    import pyreadstat

    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False


class ENAHOExtractor:
    """Manejador de extracción con soporte para múltiples formatos"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def extract_zip(
        self,
        zip_path: Path,
        extract_dir: Path,
        only_dta: bool = False,
        flatten: bool = True,
        filter_func: Optional[Callable[[str], bool]] = None,
    ) -> List[Path]:
        """
        Extrae archivos de un ZIP con filtros personalizables

        Args:
            zip_path: Ruta al archivo ZIP
            extract_dir: Directorio de extracción
            only_dta: Si extraer solo archivos .dta
            flatten: Si aplanar la estructura de directorios
            filter_func: Función personalizada de filtrado

        Returns:
            Lista de archivos extraídos

        Raises:
            ENAHOError: Si hay problemas con la extracción
        """
        extracted_files = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.is_dir():
                        continue

                    filename = os.path.basename(zip_info.filename)
                    if not filename:
                        continue

                    # Aplicar filtros
                    if only_dta and not filename.lower().endswith(".dta"):
                        continue

                    if filter_func and not filter_func(filename):
                        continue

                    # Determinar ruta de destino
                    if flatten:
                        target_path = extract_dir / filename
                    else:
                        target_path = extract_dir / zip_info.filename
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extraer archivo
                    with zip_ref.open(zip_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

                    extracted_files.append(target_path)

        except zipfile.BadZipFile:
            raise ENAHOError(f"Archivo ZIP corrupto: {zip_path}")

        return extracted_files

    def load_dta_files(
        self, directory: Path, low_memory: bool = True, chunksize: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Carga archivos .dta de un directorio

        Args:
            directory: Directorio con archivos .dta
            low_memory: Si optimizar uso de memoria
            chunksize: Tamaño de chunks (no usado actualmente)

        Returns:
            Diccionario con DataFrames cargados
        """
        dta_files = list(directory.glob("*.dta"))
        loaded_data = {}

        for dta_file in dta_files:
            try:
                if PYREADSTAT_AVAILABLE:
                    df, _ = pyreadstat.read_dta(str(dta_file), apply_value_formats=True)
                else:
                    df = pd.read_stata(str(dta_file))

                if low_memory:
                    df = self._optimize_dtypes(df)

                loaded_data[dta_file.stem] = df
                self.logger.info(f"Archivo cargado: {dta_file.name} ({len(df)} filas)")

            except Exception as e:
                self.logger.warning(f"Error cargando {dta_file.name}: {str(e)}")
                continue

        return loaded_data

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza los tipos de datos para reducir uso de memoria"""
        for col in df.select_dtypes(include=["int64"]).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype("int8")
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        return df

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


__all__ = ["ENAHOExtractor", "PYREADSTAT_AVAILABLE"]
