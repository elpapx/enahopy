"""
ENAHO Main Module
================

Clase principal ENAHODataDownloader que orquesta toda la funcionalidad:
descarga, extracción, lectura local y gestión de cache.
Punto de entrada principal para la librería.
"""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

from ..core.config import ENAHOConfig
from ..core.cache import CacheManager
from ..core.logging import setup_logging, log_performance
from ..core.exceptions import ENAHOError
from .validators.enaho import ENAHOValidator
from .downloaders.downloader import ENAHODownloader
from .downloaders.extractor import ENAHOExtractor
from .local_reader import ENAHOLocalReader


class ENAHODataDownloader:
    """Descargador principal con funcionalidades avanzadas y lectura local"""

    def __init__(self,
                 verbose: bool = True,
                 structured_logging: bool = False,
                 config: Optional[ENAHOConfig] = None):
        """
        Inicializa el descargador principal

        Args:
            verbose: Si mostrar información detallada
            structured_logging: Si usar logging estructurado
            config: Configuración personalizada
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
        """Obtiene los años disponibles"""
        year_map = self.config.YEAR_MAP_PANEL if is_panel else self.config.YEAR_MAP_TRANSVERSAL
        return sorted(year_map.keys(), reverse=True)

    def get_available_modules(self) -> Dict[str, str]:
        """Obtiene los módulos disponibles con descripciones"""
        return self.config.AVAILABLE_MODULES.copy()

    def validate_availability(self, modules: List[str], years: List[str], is_panel: bool = False) -> Dict[str, Any]:
        """Valida la disponibilidad de módulos y años, retorna reporte detallado"""
        try:
            normalized_modules = self.validator.validate_modules(modules)
            self.validator.validate_years(years, is_panel)

            return {
                "status": "valid",
                "modules": normalized_modules,
                "years": years,
                "dataset_type": "panel" if is_panel else "transversal",
                "estimated_downloads": len(normalized_modules) * len(years)
            }
        except Exception as e:
            return {
                "status": "invalid",
                "error": str(e),
                "error_code": getattr(e, 'error_code', None),
                "context": getattr(e, 'context', {})
            }

    def read_local_file(self, file_path: str, **kwargs) -> ENAHOLocalReader:
        """
        Crea un lector para un archivo local

        Args:
            file_path: Ruta al archivo local
            **kwargs: Argumentos adicionales para ENAHOLocalReader

        Returns:
            Instancia de ENAHOLocalReader configurada
        """
        return ENAHOLocalReader(
            file_path=file_path,
            config=self.config,
            verbose=kwargs.get('verbose', True),
            structured_logging=kwargs.get('structured_logging', False),
            log_file=kwargs.get('log_file', None)
        )

    def find_local_files(self,
                         directory: Union[str, Path],
                         pattern: str = "*.dta",
                         recursive: bool = True) -> List[Path]:
        """
        Encuentra archivos locales en un directorio

        Args:
            directory: Directorio donde buscar
            pattern: Patrón de archivos (ej: "*.dta", "*modulo*")
            recursive: Si buscar recursivamente en subdirectorios

        Returns:
            Lista de rutas de archivos encontrados
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

    def batch_read_local_files(self,
                               file_paths: List[Union[str, Path]],
                               columns: Optional[List[str]] = None,
                               **read_kwargs) -> Dict[str, Tuple]:
        """
        Lee múltiples archivos locales en lote

        Args:
            file_paths: Lista de rutas de archivos
            columns: Columnas a leer de cada archivo
            **read_kwargs: Argumentos para read_data()

        Returns:
            Diccionario con nombre de archivo como clave y (datos, validación) como valor
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
    def download(self,
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
                 progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Optional[Dict]:
        """Metodo principal mejorado para descarga de datos ENAHO"""

        start_time = time.time()

        try:
            # Validaciones
            normalized_modules = self.validator.validate_modules(modules)
            self.validator.validate_years(years, is_panel)
            output_path = self.validator.validate_output_dir(output_dir)

            # Configuración
            year_mapping = self.config.YEAR_MAP_PANEL if is_panel else self.config.YEAR_MAP_TRANSVERSAL
            dataset_type = "panel" if is_panel else "corte transversal"

            if verbose:
                self.logger.info(f"=== Iniciando descarga ENAHO {dataset_type} ===")
                self.logger.info(f"Módulos: {normalized_modules}")
                self.logger.info(f"Años: {years}")
                self.logger.info(f"Directorio: {output_path}")

            # Preparar tareas
            tasks = [(year, module, year_mapping[year]) for year in years for module in normalized_modules]
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
                        year, module, code, output_path, overwrite,
                        decompress, only_dta, load_dta, verbose,
                        low_memory, chunksize
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
                                progress_callback(f"{task_year}-{task_module}", completed_tasks, total_tasks)

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
                        progress_callback(f"{task_year}-{task_module}", completed_tasks, total_tasks)

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
                "success_rate": success_rate
            }
            self.cache_manager.set_metadata("last_download_session", session_metadata)

            return all_results

        except ENAHOError:
            raise
        except Exception as e:
            self.logger.error(f"Error inesperado durante la descarga: {str(e)}")
            raise ENAHOError(f"Error inesperado: {str(e)}", "UNEXPECTED_ERROR")

    def _process_single_download(self, year: str, module: str, code: int,
                                 output_dir: Path, overwrite: bool, decompress: bool,
                                 only_dta: bool, load_dta: bool, verbose: bool,
                                 low_memory: bool, chunksize: Optional[int]) -> Optional[Dict]:
        """Procesa una descarga individual"""
        try:
            # Descargar archivo
            zip_path = self.downloader.download_file(year, module, code, output_dir, overwrite, verbose)

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
                return self.extractor.load_dta_files(extract_dir, low_memory=low_memory, chunksize=chunksize)

            return None

        except Exception as e:
            self.logger.error(f"Error procesando {year}-{module}: {str(e)}")
            raise

    def get_download_history(self) -> Optional[Dict]:
        """Obtiene el historial de la última sesión de descarga"""
        return self.cache_manager.get_metadata("last_download_session")

    def clean_cache(self) -> None:
        """Limpia el cache completamente"""
        if self.cache_manager.cache_dir.exists():
            shutil.rmtree(self.cache_manager.cache_dir)
            self.cache_manager.cache_dir.mkdir(exist_ok=True)
        self.logger.info("Cache limpiado")

    def export_metadata(self, output_file: str) -> None:
        """Exporta metadatos de configuración y disponibilidad"""
        metadata = {
            "config": {
                "available_years_transversal": self.get_available_years(False),
                "available_years_panel": self.get_available_years(True),
                "available_modules": self.get_available_modules(),
                "base_url": self.config.base_url,
                "cache_dir": self.config.cache_dir
            },
            "last_session": self.get_download_history(),
            "export_timestamp": datetime.now().isoformat()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Metadatos exportados a: {output_file}")


__all__ = [
    'ENAHODataDownloader'
]