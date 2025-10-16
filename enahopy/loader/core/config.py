"""
ENAHO Configuration Module
=========================

Configuración centralizada para la librería enaho-analyzer.
Contiene dataclasses inmutables con configuraciones por defecto,
mapeos de años, módulos disponibles y settings de performance.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class ENAHOConfig:
    """Configuración inmutable para la descarga de datos ENAHO"""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "ENAHO_BASE_URL", "https://proyectos.inei.gob.pe/iinei/srienaho/descarga/STATA/"
        )
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("ENAHO_CHUNK_SIZE", "8192")))
    default_max_workers: int = field(
        default_factory=lambda: int(os.getenv("ENAHO_MAX_WORKERS", "4"))
    )
    timeout: int = field(default_factory=lambda: int(os.getenv("ENAHO_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ENAHO_MAX_RETRIES", "3")))
    backoff_factor: float = field(
        default_factory=lambda: float(os.getenv("ENAHO_BACKOFF_FACTOR", "0.5"))
    )

    # Configuración de lectura local
    chunk_size_default: int = field(
        default_factory=lambda: int(os.getenv("ENAHO_READ_CHUNK_SIZE", "10000"))
    )

    # Cache settings
    cache_dir: str = field(default_factory=lambda: os.getenv("ENAHO_CACHE_DIR", ".enaho_cache"))
    cache_ttl_hours: int = field(default_factory=lambda: int(os.getenv("ENAHO_CACHE_TTL", "24")))

    # Validación de integridad
    verify_checksums: bool = field(
        default_factory=lambda: os.getenv("ENAHO_VERIFY_CHECKSUMS", "false").lower() == "true"
    )

    # Mapeos actualizados con más años
    YEAR_MAP_TRANSVERSAL: Dict[str, int] = field(
        default_factory=lambda: {
            "2024": 966,
            "2023": 906,
            "2022": 784,
            "2021": 759,
            "2020": 737,
            "2019": 687,
            "2018": 634,
            "2017": 603,
            "2016": 546,
            "2015": 498,
            "2014": 440,
            "2013": 404,
            "2012": 324,
            "2011": 291,
            "2010": 279,
            "2009": 285,
            "2008": 284,
            "2007": 283,
            "2006": 282,
            "2005": 281,
            "2004": 280,
            "2003": 279,
            "2002": 278,
            "2001": 277,  # Años adicionales
        }
    )

    YEAR_MAP_PANEL: Dict[str, int] = field(
        default_factory=lambda: {
            "2024": 978,
            "2023": 912,
            "2022": 845,
            "2021": 763,
            "2020": 743,
            "2019": 699,
            "2018": 651,
            "2017": 612,
            "2016": 614,
            "2015": 529,
            "2011": 302,
        }
    )

    # Módulos disponibles con descripciones
    AVAILABLE_MODULES: Dict[str, str] = field(
        default_factory=lambda: {
            "01": "Características de la Vivienda y del Hogar",
            "02": "Características de los Miembros del Hogar",
            "03": "Educación",
            "04": "Salud",
            "05": "Empleo e Ingresos",
            "07": "Gastos en Alimentos y bebidas",
            "08": "Instituciones Beneficas",
            "09": "Mantenimiento de Viviendas",
            "34": "Sumarias ( Variables Calculadas )",
            "37": "Programas Sociales",
        }
    )

    def __post_init__(self):
        """Validación post-inicialización"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size debe ser positivo")
        if self.default_max_workers <= 0:
            raise ValueError("default_max_workers debe ser positivo")


__all__ = ["ENAHOConfig"]
