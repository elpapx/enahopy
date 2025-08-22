"""
ENAHO Downloader Module
======================

Descargador principal con retry automático, validación de integridad
y barras de progreso. Maneja la descarga de archivos ZIP desde
servidores INEI con validación de checksums.
"""

import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from ...core.cache import CacheManager
from ...core.config import ENAHOConfig
from ...core.exceptions import ENAHODownloadError, ENAHOIntegrityError, ENAHOTimeoutError
from .network import NetworkUtils


class ENAHODownloader:
    """Manejador de descargas con retry y validación"""

    def __init__(self, config: ENAHOConfig, logger: logging.Logger, cache_manager: CacheManager):
        self.config = config
        self.logger = logger
        self.cache_manager = cache_manager
        self.network = NetworkUtils(config, logger)

    def _build_url(self, code: int, module: str) -> str:
        """Construye la URL de descarga"""
        filename = f"{code}-Modulo{module}.zip"
        return urljoin(self.config.base_url, filename)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcula checksum SHA256 de un archivo"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _validate_zip_integrity(self, file_path: Path) -> bool:
        """Valida la integridad de un archivo ZIP"""
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Test the zip file
                zip_ref.testzip()
                return True
        except (zipfile.BadZipFile, zipfile.LargeZipFile):
            return False

    def _download_with_progress(self, url: str, file_path: Path, verbose: bool) -> None:
        """Descarga un archivo con barra de progreso y validación"""
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

        try:
            # Verificar que la URL existe
            if not self.network.check_url_exists(url):
                raise ENAHODownloadError(f"URL no encontrada: {url}", "URL_NOT_FOUND")

            # Obtener tamaño del archivo
            file_size = self.network.get_file_size(url)

            with self.network.session.get(
                url, stream=True, timeout=self.config.timeout
            ) as response:
                response.raise_for_status()

                total_size = file_size or int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as file, tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Descargando {file_path.name}",
                    disable=not verbose,
                ) as progress_bar:

                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            file.write(chunk)
                            progress_bar.update(len(chunk))

            # Validar integridad del archivo descargado
            if not self._validate_zip_integrity(temp_path):
                temp_path.unlink(missing_ok=True)
                raise ENAHOIntegrityError(f"Archivo ZIP corrupto: {file_path.name}")

            # Mover archivo temporal al destino final
            temp_path.rename(file_path)

            # Calcular y guardar checksum si está habilitado
            if self.config.verify_checksums:
                checksum = self._calculate_checksum(file_path)
                self.cache_manager.set_metadata(
                    f"checksum_{file_path.name}",
                    {"checksum": checksum, "size": file_path.stat().st_size},
                )

        except requests.exceptions.RequestException as e:
            temp_path.unlink(missing_ok=True)
            if "timeout" in str(e).lower():
                raise ENAHOTimeoutError(f"Timeout descargando {url}")
            raise ENAHODownloadError(f"Error de red descargando {url}: {str(e)}")
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise

    def download_file(
        self, year: str, module: str, code: int, output_dir: Path, overwrite: bool, verbose: bool
    ) -> Path:
        """
        Descarga un archivo individual con manejo mejorado de errores

        Args:
            year: Año de la encuesta
            module: Código del módulo
            code: Código interno para la URL
            output_dir: Directorio de destino
            overwrite: Si sobrescribir archivos existentes
            verbose: Si mostrar información detallada

        Returns:
            Path al archivo descargado

        Raises:
            ENAHODownloadError: Si hay errores en la descarga
        """
        url = self._build_url(code, module)
        filename = f"modulo_{module}_{year}.zip"
        file_path = output_dir / filename

        # Verificar si el archivo ya existe y es válido
        if file_path.exists() and not overwrite:
            if self._validate_zip_integrity(file_path):
                if verbose:
                    self.logger.info(f"Archivo válido encontrado: {filename}")
                return file_path
            else:
                self.logger.warning(f"Archivo corrupto encontrado, re-descargando: {filename}")
                file_path.unlink()

        if verbose:
            self.logger.info(f"Descargando módulo {module} año {year}")

        try:
            self._download_with_progress(url, file_path, verbose)

            if verbose:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Descarga completada: {filename} ({file_size_mb:.1f} MB)")

            return file_path

        except Exception as e:
            # Log detallado del error
            self.logger.error(
                f"Error descargando {filename}: {str(e)}",
                extra={
                    "context": {
                        "url": url,
                        "year": year,
                        "module": module,
                        "output_path": str(file_path),
                    }
                },
            )
            raise


__all__ = ["ENAHODownloader"]
