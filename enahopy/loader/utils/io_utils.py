"""
ENAHO I/O Utilities Module
=========================

Funciones de conveniencia para uso común de la librería.
Incluye shortcuts para descarga, lectura, validación y
utilidades de estimación y recomendaciones.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.config import ENAHOConfig
from ..core.exceptions import ENAHOValidationError
from ..io.base import DASK_AVAILABLE
from ..io.main import ENAHODataDownloader
from ..io.validators.results import ColumnValidationResult

if DASK_AVAILABLE:
    import dask.dataframe as dd


# =====================================================
# FUNCIONES DE CONVENIENCIA PRINCIPALES
# =====================================================


def download_enaho_data(
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
) -> Optional[
    Union[pd.DataFrame, Dict[str, pd.DataFrame], Dict[Tuple[str, str], Dict[str, pd.DataFrame]]]
]:
    """
    Función de conveniencia para descargar datos ENAHO

    Args:
        modules: Lista de módulos a descargar (ej: ["01", "02", "34"])
        years: Lista de años a descargar (ej: ["2023", "2022"])
        output_dir: Directorio de salida
        is_panel: True para datos panel, False para corte transversal
        decompress: Si descomprimir los archivos ZIP
        only_dta: Si extraer solo archivos .dta (requiere decompress=True)
        load_dta: Si cargar los archivos .dta en memoria (requiere decompress=True)
        overwrite: Si sobrescribir archivos existentes
        parallel: Si usar descarga paralela
        max_workers: Número máximo de workers para descarga paralela
        verbose: Si mostrar información detallada
        low_memory: Si optimizar el uso de memoria al cargar DataFrames
        chunksize: Tamaño de chunks para archivos grandes (None para cargar completo)
        progress_callback: Función callback para reportar progreso (task_name, completed, total)

    Returns:
        - Si load_dta=False: None
        - Si 1 año, 1 módulo: pd.DataFrame directamente (siempre)
        - Si múltiples años/módulos: Dict[(año, módulo), Dict[str, pd.DataFrame]]

    Raises:
        ENAHOValidationError: Si los parámetros no son válidos
        ENAHODownloadError: Si hay errores durante la descarga
        ENAHOError: Para otros errores relacionados con ENAHO
    """
    downloader = ENAHODataDownloader(verbose=verbose)
    result = downloader.download(
        modules=modules,
        years=years,
        output_dir=output_dir,
        is_panel=is_panel,
        decompress=decompress,
        only_dta=only_dta,
        load_dta=load_dta,
        overwrite=overwrite,
        parallel=parallel,
        max_workers=max_workers,
        verbose=verbose,
        low_memory=low_memory,
        chunksize=chunksize,
        progress_callback=progress_callback,
    )

    # Si no se cargaron datos, devolver None
    if not result or not load_dta:
        return None

    # Simplificar el resultado para facilitar el uso
    # Si solo hay 1 año y 1 módulo, SIEMPRE devolver un DataFrame
    if len(years) == 1 and len(modules) == 1:
        # Buscar la clave correcta en el resultado
        # Las claves pueden estar normalizadas (ej: '02' en lugar de '2')
        found_key = None
        for key in result.keys():
            year_key, module_key = key
            # Comparar año exacto y módulo normalizado
            if year_key == years[0] and module_key.zfill(2) == modules[0].zfill(2):
                found_key = key
                break

        if found_key:
            files_dict = result[found_key]

            # Si solo hay 1 archivo, devolver el DataFrame directamente
            if len(files_dict) == 1:
                return list(files_dict.values())[0]

            # Si hay múltiples archivos, buscar el principal o devolver el primero
            file_names = list(files_dict.keys())

            # Buscar archivo principal (el que contiene el número del módulo)
            module_normalized = modules[0].zfill(2)
            for fname in file_names:
                if module_normalized in fname.lower():
                    return files_dict[fname]

            # Si no se encuentra, devolver el primer DataFrame
            return list(files_dict.values())[0]

    # Para múltiples años/módulos, devolver estructura completa
    return result


def read_enaho_file(
    file_path: str,
    columns: Optional[List[str]] = None,
    use_chunks: bool = False,
    chunk_size: Optional[int] = None,
    ignore_missing_columns: bool = True,
    case_sensitive: bool = False,
    verbose: bool = True,
) -> Tuple[Union[pd.DataFrame, dd.DataFrame, Iterator[pd.DataFrame]], ColumnValidationResult]:
    """
    Función de conveniencia para leer un archivo ENAHO local

    Args:
        file_path: Ruta al archivo
        columns: Lista de columnas a leer (None para todas)
        use_chunks: Si usar lectura por chunks
        chunk_size: Tamaño de chunks
        ignore_missing_columns: Si ignorar columnas faltantes
        case_sensitive: Si la búsqueda es case-sensitive
        verbose: Si mostrar información detallada

    Returns:
        Tupla con (datos, resultado_validacion)

    Raises:
        FileNotFoundError: Si el archivo no existe
        UnsupportedFormatError: Si el formato no es soportado
        FileReaderError: Si hay errores de lectura
    """
    downloader = ENAHODataDownloader(verbose=verbose)
    reader = downloader.read_local_file(file_path, verbose=verbose)

    return reader.read_data(
        columns=columns,
        use_chunks=use_chunks,
        chunk_size=chunk_size,
        ignore_missing_columns=ignore_missing_columns,
        case_sensitive=case_sensitive,
    )


def get_file_info(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener información de un archivo ENAHO

    Args:
        file_path: Ruta al archivo
        verbose: Si mostrar logs detallados

    Returns:
        Diccionario con información del archivo

    Raises:
        FileNotFoundError: Si el archivo no existe
        UnsupportedFormatError: Si el formato no es soportado
    """
    downloader = ENAHODataDownloader(verbose=verbose)
    reader = downloader.read_local_file(file_path, verbose=verbose)

    return reader.get_summary_info()


def find_enaho_files(
    directory: str, pattern: str = "*.dta", recursive: bool = True, verbose: bool = False
) -> List[Path]:
    """
    Función de conveniencia para encontrar archivos ENAHO en un directorio

    Args:
        directory: Directorio donde buscar
        pattern: Patrón de archivos a buscar
        recursive: Si buscar recursivamente
        verbose: Si mostrar logs detallados

    Returns:
        Lista de rutas de archivos encontrados

    Raises:
        FileNotFoundError: Si el directorio no existe
    """
    downloader = ENAHODataDownloader(verbose=verbose)
    return downloader.find_local_files(directory, pattern, recursive)


def get_available_data(is_panel: bool = False) -> Dict[str, Any]:
    """
    Obtiene información sobre datos disponibles

    Args:
        is_panel: True para datos panel, False para corte transversal

    Returns:
        Diccionario con años y módulos disponibles
    """
    downloader = ENAHODataDownloader(verbose=False)
    return {
        "years": downloader.get_available_years(is_panel),
        "modules": downloader.get_available_modules(),
        "dataset_type": "panel" if is_panel else "transversal",
    }


def validate_download_request(
    modules: List[str], years: List[str], is_panel: bool = False
) -> Dict[str, Any]:
    """
    Valida una solicitud de descarga sin ejecutarla

    Args:
        modules: Lista de módulos
        years: Lista de años
        is_panel: Tipo de dataset

    Returns:
        Diccionario con el resultado de la validación
    """
    downloader = ENAHODataDownloader(verbose=False)
    return downloader.validate_availability(modules, years, is_panel)


# =====================================================
# CLASE DE UTILIDADES ADICIONALES
# =====================================================


class ENAHOUtils:
    """Utilidades adicionales para trabajar con datos ENAHO"""

    @staticmethod
    def estimate_download_size(modules: List[str], years: List[str]) -> Dict[str, float]:
        """
        Estima el tamaño de descarga basado en promedios históricos

        Returns:
            Diccionario con estimaciones en MB
        """
        # Tamaños promedio por módulo (en MB)
        avg_sizes = {
            "01": 15,
            "02": 25,
            "03": 20,
            "04": 18,
            "05": 45,
            "07": 8,
            "08": 12,
            "09": 5,
            "34": 10,
            "37": 3,
        }

        total_estimated = 0
        detailed_estimate = {}

        for module in modules:
            module_size = avg_sizes.get(module.zfill(2), 15)  # Default 15MB
            total_size = module_size * len(years)
            detailed_estimate[module] = total_size
            total_estimated += total_size

        return {
            "total_mb": total_estimated,
            "total_gb": total_estimated / 1024,
            "by_module": detailed_estimate,
            "compressed_size": total_estimated * 0.3,  # Estimación de compresión
            "note": "Estimaciones basadas en tamaños promedio históricos",
        }

    @staticmethod
    def get_module_description(module: str) -> str:
        """Obtiene la descripción de un módulo"""
        config = ENAHOConfig()
        return config.AVAILABLE_MODULES.get(module.zfill(2), "Módulo desconocido")

    @staticmethod
    def recommend_parallel_settings(num_downloads: int) -> Dict[str, Any]:
        """Recomienda configuraciones para descarga paralela"""
        if num_downloads <= 2:
            return {"parallel": False, "max_workers": 1, "reason": "Pocos archivos"}
        elif num_downloads <= 8:
            return {"parallel": True, "max_workers": 2, "reason": "Cantidad moderada"}
        elif num_downloads <= 20:
            return {"parallel": True, "max_workers": 4, "reason": "Cantidad alta"}
        else:
            return {"parallel": True, "max_workers": 6, "reason": "Cantidad muy alta"}

    @staticmethod
    def merge_enaho_dataframes(
        dataframes: Dict[str, pd.DataFrame], on: List[str] = None, how: str = "inner"
    ) -> pd.DataFrame:
        """
        Une múltiples DataFrames ENAHO usando llaves comunes

        Args:
            dataframes: Diccionario con DataFrames a unir
            on: Columnas para hacer el join (por defecto usa llaves ENAHO comunes)
            how: Tipo de join ('inner', 'outer', 'left', 'right')

        Returns:
            DataFrame unido
        """
        if not dataframes:
            return pd.DataFrame()

        # Llaves comunes de ENAHO si no se especifican
        if on is None:
            on = ["conglome", "vivienda", "hogar"]
            # Agregar codperso si está disponible en todos los DataFrames
            if all("codperso" in df.columns for df in dataframes.values()):
                on.append("codperso")

        # Verificar que las llaves existen en todos los DataFrames
        for name, df in dataframes.items():
            missing_keys = [key for key in on if key not in df.columns]
            if missing_keys:
                warnings.warn(f"DataFrame '{name}' no tiene las llaves: {missing_keys}")

        # Realizar joins secuenciales
        result = None
        for name, df in dataframes.items():
            available_keys = [key for key in on if key in df.columns]
            if result is None:
                result = df.copy()
            else:
                result = result.merge(df, on=available_keys, how=how, suffixes=("", f"_{name}"))

        return result

    @staticmethod
    def validate_enaho_keys(df: pd.DataFrame, level: str = "hogar") -> Dict[str, Any]:
        """
        Valida las llaves primarias de un DataFrame ENAHO

        Args:
            df: DataFrame a validar
            level: Nivel de validación ("hogar", "persona", "vivienda")

        Returns:
            Diccionario con resultado de validación
        """
        key_configs = {
            "hogar": ["conglome", "vivienda", "hogar"],
            "persona": ["conglome", "vivienda", "hogar", "codperso"],
            "vivienda": ["conglome", "vivienda"],
        }

        if level not in key_configs:
            raise ValueError(f"Nivel no soportado: {level}")

        required_keys = key_configs[level]

        # Verificar columnas existentes
        missing_keys = [key for key in required_keys if key not in df.columns]

        if missing_keys:
            return {
                "is_valid": False,
                "missing_keys": missing_keys,
                "level": level,
                "error": f"Columnas faltantes: {missing_keys}",
            }

        # Verificar duplicados
        duplicates = df.duplicated(subset=required_keys).sum()
        unique_combinations = df[required_keys].drop_duplicates().shape[0]

        return {
            "is_valid": duplicates == 0,
            "level": level,
            "total_records": len(df),
            "unique_combinations": unique_combinations,
            "duplicates": duplicates,
            "completeness": (df[required_keys].notna().all(axis=1).sum() / len(df)) * 100,
        }


# =====================================================
# FUNCIONES DE COMPATIBILIDAD
# =====================================================


def main():
    """Función de ejemplo de uso con todas las nuevas funcionalidades"""
    print("=== ENAHO Data Downloader - Versión Refactorizada ===\n")

    try:
        # 1. Crear instancia del downloader
        downloader = ENAHODataDownloader(verbose=True)

        # 2. Mostrar información disponible
        print("Años disponibles (transversal):", downloader.get_available_years(False)[:5])
        print("Módulos disponibles:", list(downloader.get_available_modules().keys()))

        # 3. Validar una solicitud de descarga
        validation = validate_download_request(["01", "34"], ["2023", "2022"])
        print(f"\nValidación de descarga: {validation['status']}")

        # 4. Ejemplo de descarga (comentado para evitar descargas reales)
        """
        result = download_enaho_data(
            modules=["34"],  # Solo sumaria para ejemplo
            years=["2023"],
            output_dir="./test_output",
            decompress=True,
            only_dta=True,
            verbose=True
        )
        """

        # 5. Ejemplo de lectura de archivo local (ajustar ruta)
        test_file_path = "./test_data/ejemplo.dta"  # Ajustar esta ruta

        if Path(test_file_path).exists():
            print(f"\n--- Leyendo archivo local: {test_file_path} ---")

            # Obtener información del archivo
            file_info = get_file_info(test_file_path, verbose=False)
            print(f"Formato: {file_info['file_info'].get('file_format', 'N/A')}")
            print(f"Total columnas: {file_info['total_columns']}")
            print(f"Columnas muestra: {file_info['sample_columns']}")

            # Leer datos específicos
            data, validation = read_enaho_file(
                file_path=test_file_path,
                columns=["conglome", "vivienda", "hogar"],
                ignore_missing_columns=True,
                case_sensitive=False,
                verbose=True,
            )

            print(f"\nDatos leídos: {len(data)} filas")
            print("Validación de columnas:")
            print(validation.get_summary())

            if len(data) > 0:
                print("\nPrimeras filas:")
                print(data.head())

        else:
            print(f"\nArchivo de ejemplo no encontrado: {test_file_path}")
            print("Para probar la lectura local, ajusta la variable 'test_file_path'")

        # 6. Buscar archivos en directorio
        search_dir = "./test_data"
        if Path(search_dir).exists():
            found_files = find_enaho_files(search_dir, "*.dta", recursive=True)
            print(f"\nArchivos .dta encontrados en {search_dir}: {len(found_files)}")
            for file in found_files[:3]:  # Mostrar solo los primeros 3
                print(f"  - {file}")

        # 7. Utilidades adicionales
        print("\n--- Utilidades ---")
        estimate = ENAHOUtils.estimate_download_size(["01", "34"], ["2023", "2022"])
        print(f"Estimación de descarga: {estimate['total_mb']:.1f} MB")

        module_desc = ENAHOUtils.get_module_description("34")
        print(f"Descripción módulo 34: {module_desc}")

        parallel_rec = ENAHOUtils.recommend_parallel_settings(8)
        print(f"Recomendación paralelo para 8 descargas: {parallel_rec}")

        print("\n✅ Ejemplos completados exitosamente!")

    except Exception as e:
        print(f"\n❌ Error en la demostración: {str(e)}")
        import traceback

        traceback.print_exc()


__all__ = [
    # Funciones principales
    "download_enaho_data",
    "read_enaho_file",
    "get_file_info",
    "find_enaho_files",
    "get_available_data",
    "validate_download_request",
    # Clases de utilidades
    "ENAHOUtils",
    # Función de demostración
    "main",
]
