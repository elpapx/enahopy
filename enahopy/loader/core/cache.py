"""
ENAHO Cache Module
=================

Sistema de cache para metadatos y archivos temporales.
Gestión de TTL, limpieza automática y persistencia JSON.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CacheManager:
    """Gestor de cache para metadatos y archivos temporales"""

    def __init__(self, cache_dir: str, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.ttl_seconds = ttl_hours * 3600
        self.metadata_file = self.cache_dir / "metadata.json"

    def _is_valid(self, timestamp: float) -> bool:
        """Verifica si un elemento del cache es válido"""
        return time.time() - timestamp < self.ttl_seconds

    def get_metadata(self, key: str) -> Optional[Dict]:
        """Obtiene metadatos del cache"""
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            if key in cache_data and self._is_valid(cache_data[key]["timestamp"]):
                return cache_data[key]["data"]
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    def set_metadata(self, key: str, data: Dict) -> None:
        """Guarda metadatos en el cache"""
        cache_data = {}

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except json.JSONDecodeError:
                cache_data = {}

        cache_data[key] = {"timestamp": time.time(), "data": data}

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def clean_expired(self) -> None:
        """Limpia elementos expirados del cache"""
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            valid_data = {
                key: value
                for key, value in cache_data.items()
                if self._is_valid(value["timestamp"])
            }

            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(valid_data, f, ensure_ascii=False, indent=2)

        except (json.JSONDecodeError, KeyError):
            pass


__all__ = ["CacheManager"]
