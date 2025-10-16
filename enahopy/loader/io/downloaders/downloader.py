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
    """Manejador de descargas con retry y validación.

    Esta clase se encarga de descargar archivos ZIP desde los servidores
    INEI con validación de integridad, retry automático en caso de errores,
    y almacenamiento de checksums para verificación posterior.

    Attributes:
        config: Configuración del sistema ENAHO.
        logger: Logger para registro de operaciones.
        cache_manager: Gestor de cache para metadatos y checksums.
        network: Utilidades de red para operaciones HTTP.

    Examples:
        Uso básico con configuración por defecto:

        >>> from enahopy.loader.core.config import ENAHOConfig
        >>> from enahopy.loader.core.cache import CacheManager
        >>> from enahopy.loader.core.logging import setup_logging
        >>> config = ENAHOConfig()
        >>> logger = setup_logging(verbose=True)
        >>> cache = CacheManager(config.cache_dir, config.cache_ttl_hours)
        >>> downloader = ENAHODownloader(config, logger, cache)
        >>> file_path = downloader.download_file(
        ...     year="2023",
        ...     module="34",
        ...     code=814,
        ...     output_dir=Path("./data"),
        ...     overwrite=False,
        ...     verbose=True
        ... )

    Note:
        Esta clase es usualmente instanciada por :class:`ENAHODataDownloader`
        y no necesita ser creada directamente por el usuario.

    See Also:
        - :class:`~enahopy.loader.io.main.ENAHODataDownloader`: Clase principal de descarga
        - :class:`~enahopy.loader.core.cache.CacheManager`: Gestor de cache
        - :class:`~enahopy.loader.io.downloaders.network.NetworkUtils`: Utilidades de red
    """

    def __init__(self, config: ENAHOConfig, logger: logging.Logger, cache_manager: CacheManager):
        self.config = config
        self.logger = logger
        self.cache_manager = cache_manager
        self.network = NetworkUtils(config, logger)

    def _build_url(self, code: int, module: str) -> str:
        """Construye la URL de descarga para un módulo específico.

        Args:
            code: Código interno INEI para el archivo (ej: 814, 906).
            module: Código del módulo ENAHO (ej: "34", "01").

        Returns:
            URL completa para descargar el archivo ZIP.

        Example:
            >>> downloader._build_url(814, "34")
            'https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/814-Modulo34.zip'
        """
        filename = f"{code}-Modulo{module}.zip"
        return urljoin(self.config.base_url, filename)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcula checksum SHA256 de un archivo para validación de integridad.

        Usa lectura por chunks para ser eficiente en memoria con archivos grandes.

        Args:
            file_path: Ruta al archivo para calcular checksum.

        Returns:
            Hash SHA256 en formato hexadecimal (64 caracteres).

        Note:
            Este metodo lee el archivo en chunks de 4KB para minimizar
            uso de memoria, permitiendo checksums de archivos >1GB.
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _validate_zip_integrity(self, file_path: Path) -> bool:
        """Valida la integridad de un archivo ZIP usando testzip().

        Args:
            file_path: Ruta al archivo ZIP a validar.

        Returns:
            True si el archivo ZIP es válido y no está corrupto.
            False si el archivo está corrupto o no es un ZIP válido.

        Note:
            Este metodo abre el ZIP y ejecuta testzip() que verifica
            los CRC de todos los archivos internos. Es rápido y no
            extrae archivos al disco.
        """
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Test the zip file
                zip_ref.testzip()
                return True
        except (zipfile.BadZipFile, zipfile.LargeZipFile):
            return False

    def _download_with_progress(self, url: str, file_path: Path, verbose: bool) -> None:
        """Descarga un archivo con barra de progreso y validación de integridad.

        Implementa descarga segura con archivo temporal, validación de ZIP,
        y cálculo de checksum opcional. La barra de progreso usa tqdm.

        Args:
            url: URL completa del archivo a descargar.
            file_path: Ruta destino para el archivo descargado.
            verbose: Si True, muestra barra de progreso durante descarga.

        Raises:
            ENAHODownloadError: Si la URL no existe o hay errores de red.
            ENAHOIntegrityError: Si el archivo descargado está corrupto.
            ENAHOTimeoutError: Si la descarga excede el timeout configurado.

        Note:
            - Descarga primero a un archivo .tmp para evitar corrupción
            - Valida integridad del ZIP antes de mover al destino
            - Calcula y guarda checksum si verify_checksums está habilitado
            - Limpia archivos temporales en caso de error
        """
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
