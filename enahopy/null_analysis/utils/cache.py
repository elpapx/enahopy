# null_analysis/utils/cache.py
"""Sistema de caché para análisis de nulos"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class NullAnalysisCache:
    """Cache específico para análisis de nulos"""

    def __init__(self, cache_dir: str = ".null_analysis_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        """Guarda valor en cache"""
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(value, f, default=str)

    def clear(self) -> None:
        """Limpia cache"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
