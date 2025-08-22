"""
Network Utilities Module
=======================

Utilidades de red con retry automático y session management.
Incluye verificación de URLs, obtención de tamaños de archivo
y configuración de headers apropiados.
"""

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...core.config import ENAHOConfig


class NetworkUtils:
    """Utilidades de red con retry automático"""

    def __init__(self, config: ENAHOConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Crea una sesión HTTP con retry automático"""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Headers personalizados
        session.headers.update(
            {
                "User-Agent": "ENAHO-Analyzer/1.0 (Python Data Analysis Tool)",
                "Accept": "application/zip, application/octet-stream, */*",
                "Accept-Encoding": "gzip, deflate",
            }
        )

        return session

    def check_url_exists(self, url: str) -> bool:
        """
        Verifica si una URL existe

        Args:
            url: URL a verificar

        Returns:
            True si la URL existe y es accesible
        """
        try:
            response = self.session.head(url, timeout=self.config.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_file_size(self, url: str) -> Optional[int]:
        """
        Obtiene el tamaño de un archivo remoto

        Args:
            url: URL del archivo

        Returns:
            Tamaño en bytes o None si no se puede obtener
        """
        try:
            response = self.session.head(url, timeout=self.config.timeout)
            if response.status_code == 200:
                return int(response.headers.get("content-length", 0))
        except (requests.exceptions.RequestException, ValueError):
            pass
        return None


__all__ = ["NetworkUtils"]
