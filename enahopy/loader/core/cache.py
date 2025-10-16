"""
ENAHO Cache Module
=================

Sistema de cache para metadatos y archivos temporales.
Gestión de TTL, limpieza automática, persistencia JSON, LRU eviction,
compresión y analytics.

Optimizations (DE-1):
- LRU cache eviction with configurable size limits
- Gzip compression for reduced disk usage
- Hit/miss rate tracking and analytics
"""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ...exceptions import ENAHOCacheError


class CacheManager:
    """Gestor de cache para metadatos y archivos temporales"""

    def __init__(
        self,
        cache_dir: str,
        ttl_hours: int = 24,
        max_size_mb: Optional[int] = None,
        enable_compression: bool = True,
    ) -> None:
        """
        Initialize cache manager with LRU eviction and compression support.

        Args:
            cache_dir: Directory for cache storage. Will be created if it doesn't exist.
            ttl_hours: Time-to-live in hours for cache entries. Default is 24 hours.
            max_size_mb: Maximum cache size in MB. None means unlimited.
                        When limit is reached, least recently used entries are evicted.
            enable_compression: Enable gzip compression for cache storage.
                               Reduces disk usage by ~40% with minimal performance impact.

        Raises:
            ENAHOCacheError: If cache directory cannot be created due to permissions
                             or other OS-level errors.

        Example:
            >>> from enahopy.loader.core import CacheManager
            >>> # Basic usage
            >>> cache = CacheManager(cache_dir=".enaho_cache", ttl_hours=48)
            >>> # With LRU eviction and compression
            >>> cache = CacheManager(cache_dir=".enaho_cache", ttl_hours=48,
            ...                      max_size_mb=100, enable_compression=True)
            >>> cache.set_metadata("my_key", {"data": "value"})
        """
        self.cache_dir: Path = Path(cache_dir)
        self.ttl_seconds: int = ttl_hours * 3600
        self.max_size_bytes: Optional[int] = max_size_mb * 1024 * 1024 if max_size_mb else None
        self.enable_compression: bool = enable_compression

        # File paths
        self.metadata_file: Path = self.cache_dir / (
            "metadata.json.gz" if enable_compression else "metadata.json"
        )
        self.analytics_file: Path = self.cache_dir / "cache_analytics.json"

        self.logger: logging.Logger = logging.getLogger(__name__)

        # Analytics tracking
        self._analytics: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "writes": 0,
            "total_accesses": 0,
            "start_time": time.time(),
        }
        self._load_analytics()

        try:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        except (OSError, PermissionError) as e:
            raise ENAHOCacheError(
                f"Failed to create cache directory: {cache_dir}",
                error_code="CACHE_DIR_CREATE_FAILED",
                operation="cache_init",
                cache_dir=cache_dir,
                original_error=str(e),
            ) from e

    def _validate_cache_key(self, key: str) -> None:
        """
        Validate cache key format and constraints.

        Args:
            key: Cache key to validate.

        Raises:
            ENAHOCacheError: If key is invalid.
        """
        if not key or not key.strip():
            raise ENAHOCacheError(
                "Cache key cannot be empty or whitespace only",
                error_code="INVALID_CACHE_KEY",
                operation="validate_key",
                cache_key=key,
            )

        if len(key) > 255:
            raise ENAHOCacheError(
                f"Cache key too long: {len(key)} characters (max: 255)",
                error_code="INVALID_CACHE_KEY",
                operation="validate_key",
                cache_key=key[:50] + "...",
            )

        # Caracteres problemáticos en sistemas de archivos
        invalid_chars = set('<>:"|?*\\/\0')
        if any(char in key for char in invalid_chars):
            found_invalid = invalid_chars & set(key)
            raise ENAHOCacheError(
                f"Cache key contains invalid characters: {found_invalid}",
                error_code="INVALID_CACHE_KEY",
                operation="validate_key",
                cache_key=key,
            )

    def _validate_cache_data(self, data: Dict[str, Any]) -> None:
        """
        Validate cache data is JSON-serializable.

        Args:
            data: Data to validate.

        Raises:
            ENAHOCacheError: If data is not serializable.
        """
        if not isinstance(data, dict):
            raise ENAHOCacheError(
                f"Cache data must be a dictionary, got {type(data).__name__}",
                error_code="INVALID_CACHE_DATA",
                operation="validate_data",
            )

        try:
            # Intento rápido de serialización
            json.dumps(data)
        except (TypeError, ValueError, OverflowError) as e:
            raise ENAHOCacheError(
                f"Cache data is not JSON-serializable: {str(e)}",
                error_code="INVALID_CACHE_DATA",
                operation="validate_data",
                original_error=str(e),
            ) from e

    def _is_valid(self, timestamp: float) -> bool:
        """
        Verifica si un elemento del cache es válido según su TTL.

        Un elemento se considera válido si el tiempo transcurrido desde
        su timestamp no excede el TTL (Time-To-Live) configurado.

        Args:
            timestamp: Timestamp UNIX (segundos desde epoch) del elemento a verificar.
                       Debe ser un float positivo generado por time.time().

        Returns:
            True si el elemento está dentro del TTL y es válido.
            False si ha expirado y debe ser removido del cache.

        Example:
            >>> import time
            >>> cache = CacheManager("/tmp/cache", ttl_hours=24)
            >>> now = time.time()
            >>> cache._is_valid(now - 3600)  # 1 hora atrás
            True
            >>> cache._is_valid(now - 86400*2)  # 2 días atrás (excede 24h TTL)
            False

        Note:
            Este metodo es interno (_is_valid) y se utiliza automáticamente
            por get_metadata() y clean_expired(). No debe ser llamado directamente
            por usuarios de la librería.
        """
        return time.time() - timestamp < self.ttl_seconds

    def _load_cache_file(self) -> Dict[str, Any]:
        """
        Load cache file with compression support.

        Returns:
            Dictionary with cache data.
        """
        if not self.metadata_file.exists():
            return {}

        try:
            if self.enable_compression:
                with gzip.open(self.metadata_file, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Failed to load cache file: {e}")
            return {}

    def _save_cache_file(self, cache_data: Dict[str, Any]) -> None:
        """
        Save cache file with compression support and atomic writes.

        Args:
            cache_data: Cache data to save.
        """
        temp_file = self.metadata_file.with_suffix(".tmp")

        try:
            if self.enable_compression:
                with gzip.open(temp_file, "wt", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            else:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)

            # Atomic move
            temp_file.replace(self.metadata_file)

        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def _update_access_time(self, cache_data: Dict[str, Any], key: str) -> None:
        """
        Update last access time for LRU tracking.

        Args:
            cache_data: Cache data dictionary.
            key: Key to update.
        """
        if key in cache_data:
            cache_data[key]["last_access"] = time.time()

    def _enforce_size_limit(self, cache_data: Dict[str, Any]) -> int:
        """
        Enforce cache size limit using LRU eviction.

        Args:
            cache_data: Cache data dictionary.

        Returns:
            Number of entries evicted.
        """
        if not self.max_size_bytes:
            return 0

        # Calculate current cache size
        current_size = len(json.dumps(cache_data).encode("utf-8"))

        if current_size <= self.max_size_bytes:
            return 0

        # Sort by last access time (LRU)
        sorted_keys = sorted(
            cache_data.keys(),
            key=lambda k: cache_data[k].get("last_access", cache_data[k].get("timestamp", 0)),
        )

        evictions = 0
        for key in sorted_keys:
            if current_size <= self.max_size_bytes:
                break

            # Remove entry
            entry_size = len(json.dumps(cache_data[key]).encode("utf-8"))
            del cache_data[key]
            current_size -= entry_size
            evictions += 1

        if evictions > 0:
            self._analytics["evictions"] += evictions
            self.logger.info(f"LRU evicted {evictions} cache entries to enforce size limit")

        return evictions

    def _load_analytics(self) -> None:
        """Load analytics data from file."""
        if self.analytics_file.exists():
            try:
                with open(self.analytics_file, "r") as f:
                    saved_analytics = json.load(f)
                    self._analytics.update(saved_analytics)
            except Exception as e:
                self.logger.debug(f"Could not load analytics: {e}")

    def _save_analytics(self) -> None:
        """Save analytics data to file."""
        try:
            with open(self.analytics_file, "w") as f:
                json.dump(self._analytics, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not save analytics: {e}")

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get cache analytics and performance metrics.

        Returns:
            Dictionary with analytics data including:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0-1)
            - evictions: Number of LRU evictions
            - writes: Number of write operations
            - total_accesses: Total cache accesses
            - uptime_hours: Hours since cache initialization

        Example:
            >>> cache = CacheManager(".cache")
            >>> # ... perform operations ...
            >>> analytics = cache.get_analytics()
            >>> print(f"Hit rate: {analytics['hit_rate']*100:.1f}%")
        """
        total_accesses = self._analytics["hits"] + self._analytics["misses"]
        hit_rate = self._analytics["hits"] / total_accesses if total_accesses > 0 else 0
        uptime = (time.time() - self._analytics["start_time"]) / 3600

        return {**self._analytics, "hit_rate": hit_rate, "uptime_hours": uptime}

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata from cache with LRU tracking and analytics.

        Args:
            key: Cache key to retrieve.

        Returns:
            Cached data if found and valid, None otherwise.

        Raises:
            ENAHOCacheError: If cache file is corrupted beyond recovery or key is invalid.

        Example:
            >>> cache = CacheManager(".cache")
            >>> data = cache.get_metadata("my_data")
            >>> if data is not None:
            ...     print(f"Found cached data: {data}")
        """
        # Validate key antes de buscar
        self._validate_cache_key(key)

        # Update analytics
        self._analytics["total_accesses"] += 1

        try:
            cache_data = self._load_cache_file()

            if not cache_data:
                self._analytics["misses"] += 1
                self._save_analytics()
                self.logger.debug(f"Cache file not found: {self.metadata_file}")
                return None

            if key in cache_data and self._is_valid(cache_data[key]["timestamp"]):
                # Cache hit
                self._analytics["hits"] += 1
                self._save_analytics()

                # Update access time for LRU
                self._update_access_time(cache_data, key)
                self._save_cache_file(cache_data)

                self.logger.debug(f"Cache hit for key: {key}")
                return cache_data[key]["data"]
            else:
                # Cache miss or expired
                self._analytics["misses"] += 1
                self._save_analytics()
                self.logger.debug(f"Cache miss or expired for key: {key}")
                return None

        except (json.JSONDecodeError, OSError) as e:
            self._analytics["misses"] += 1
            self._save_analytics()
            self.logger.warning(f"Cache error: {e}")
            return None

    def set_metadata(self, key: str, data: Dict[str, Any]) -> None:
        """
        Store metadata in cache with compression, LRU eviction, and analytics.

        Args:
            key: Cache key for storage. Must be a non-empty string.
            data: Data to cache. Must be JSON-serializable.

        Raises:
            ENAHOCacheError: If cache write operation fails or inputs are invalid.

        Example:
            >>> cache = CacheManager(".cache")
            >>> cache.set_metadata("downloads", {"count": 5, "last_date": "2024-01-01"})
        """
        # Validate inputs antes de procesar
        self._validate_cache_key(key)
        self._validate_cache_data(data)

        try:
            # Load existing cache
            cache_data = self._load_cache_file()

            # Add/update entry
            current_time = time.time()
            cache_data[key] = {"timestamp": current_time, "last_access": current_time, "data": data}

            # Enforce size limit with LRU eviction
            self._enforce_size_limit(cache_data)

            # Save with compression
            self._save_cache_file(cache_data)

            # Update analytics
            self._analytics["writes"] += 1
            self._save_analytics()

            self.logger.debug(f"Cache updated for key: {key}")

        except (OSError, PermissionError) as e:
            raise ENAHOCacheError(
                f"Failed to write to cache file: {self.metadata_file}",
                error_code="CACHE_WRITE_FAILED",
                operation="set_metadata",
                cache_key=key,
                original_error=str(e),
            ) from e

    def clean_expired(self) -> None:
        """
        Elimina entradas expiradas del cache según su TTL.

        Este metodo es seguro y tolerante a fallos:
        - Maneja archivos de cache corruptos recreándolos
        - No lanza excepciones (solo loggea warnings)
        - Usa escritura atómica para prevenir corrupción durante limpieza
        - Puede llamarse múltiples veces sin efectos secundarios

        El metodo compara el timestamp de cada entrada con el TTL configurado
        y elimina aquellas que hayan expirado. Si no hay cambios necesarios,
        no modifica el archivo de cache.

        Returns:
            None. Los resultados se loggean al nivel INFO/DEBUG.

        Example:
            >>> cache = CacheManager(".cache", ttl_hours=24)
            >>> cache.set_metadata("old_data", {"value": 1})
            >>> # ... 25 horas después ...
            >>> cache.clean_expired()  # Elimina "old_data"
            INFO: Cleaned 1 expired cache entries

        Note:
            Se recomienda ejecutar este metodo periódicamente (ej: al inicio
            de la aplicación) para mantener el cache limpio y optimizado.

        See Also:
            - get_cache_stats(): Para ver estadísticas antes de limpiar
            - clear_all(): Para eliminar cache
        """
        if not self.metadata_file.exists():
            self.logger.debug("Cache file does not exist, nothing to clean")
            return

        try:
            cache_data = self._load_cache_file()

            original_count = len(cache_data)
            valid_data = {
                key: value
                for key, value in cache_data.items()
                if self._is_valid(value.get("timestamp", 0))
            }

            # Only write if there were changes
            if len(valid_data) != original_count:
                self._save_cache_file(valid_data)
                cleaned_count = original_count - len(valid_data)
                self.logger.info(f"Cleaned {cleaned_count} expired cache entries")
            else:
                self.logger.debug("No expired cache entries to clean")

        except json.JSONDecodeError as e:
            self.logger.warning(f"Corrupted cache file during cleanup, recreating: {e}")
            try:
                self._create_empty_cache()
                self.logger.info("Cache file recreated during cleanup")
            except Exception as recovery_error:
                self.logger.error(f"Failed to recover cache during cleanup: {recovery_error}")
                # Don't raise here - cleanup should be tolerant

        except (KeyError, TypeError) as e:
            self.logger.warning(f"Invalid cache structure during cleanup: {e}")
            # Try to recreate cache
            try:
                self._create_empty_cache()
                self.logger.info("Cache recreated due to invalid structure")
            except Exception:
                self.logger.error("Failed to recover invalid cache structure")

        except (OSError, PermissionError) as e:
            self.logger.error(f"Failed to clean cache due to file system error: {e}")
            # Don't raise - cleanup should be tolerant

    def _create_empty_cache(self) -> None:
        """Create an empty cache file with compression support."""
        self._save_cache_file({})

    def clear_all(self) -> None:
        """
        Elimina TODAS las entradas del cache de forma permanente.

        Este metodo elimina físicamente el archivo de cache del disco.
        Es útil para debugging, testing, o cuando se necesita resetear
        completamente el estado del cache.

        ADVERTENCIA: Esta operación es irreversible. Todos los datos
        cacheados se perderán permanentemente.

        Raises:
            ENAHOCacheError: Si el archivo de cache no puede ser eliminado
                             debido a permisos o errores del sistema operativo.

        Example:
            >>> cache = CacheManager(".cache")
            >>> cache.set_metadata("key1", {"data": "value1"})
            >>> cache.set_metadata("key2", {"data": "value2"})
            >>> stats = cache.get_cache_stats()
            >>> print(stats["total_entries"])  # 2
            >>> cache.clear_all()
            >>> stats = cache.get_cache_stats()
            >>> print(stats["total_entries"])  # 0

        Note:
            El directorio de cache NO se elimina, solo el archivo metadata.json.
            Para diferencia con clean_expired():
            - clear_all(): Elimina all
            - clean_expired(): Solo elimina entradas vencidas

        See Also:
            - clean_expired(): Para eliminar solo entradas expiradas
            - get_cache_stats(): Para ver qué se eliminará antes de limpiar
        """
        try:
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                self.logger.info("Cache cleared successfully")
            else:
                self.logger.debug("Cache already empty")
        except (OSError, PermissionError) as e:
            raise ENAHOCacheError(
                f"Failed to clear cache file: {self.metadata_file}",
                error_code="CACHE_CLEAR_FAILED",
                operation="clear_all",
                original_error=str(e),
            ) from e

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del estado del cache.

        Proporciona métricas útiles para monitoreo, debugging y optimización
        del cache. Las estadísticas incluyen conteos de entradas, tamaño en
        disco, y estado de validez.

        Returns:
            Dictionary con las siguientes claves:
            - total_entries (int): Número total de entradas en cache
            - valid_entries (int): Entradas aún válidas (dentro de TTL)
            - expired_entries (int): Entradas expiradas que pueden limpiarse
            - cache_file_size (int): Tamaño del archivo en bytes (comprimido si enabled)
            - cache_directory (str): Ruta absoluta del directorio de cache
            - compression_enabled (bool): Si la compresión está habilitada
            - error (str, opcional): Mensaje de error si las stats fallan

        Example:
            >>> cache = CacheManager(".cache", ttl_hours=24)
            >>> cache.set_metadata("key1", {"data": 1})
            >>> cache.set_metadata("key2", {"data": 2})
            >>> stats = cache.get_cache_stats()
            >>> print(f"Cache tiene {stats['total_entries']} entradas")
            Cache tiene 2 entradas
            >>> print(f"Tamaño: {stats['cache_file_size'] / 1024:.2f} KB")
            Tamaño: 0.45 KB

        Note:
            Este metodo nunca lanza excepciones. Si hay errores leyendo
            el cache, retorna un diccionario con la clave 'error'.

        See Also:
            - clean_expired(): Para limpiar las expired_entries
            - clear_all(): Para resetear el cache completamente
        """
        if not self.metadata_file.exists():
            return {
                "total_entries": 0,
                "valid_entries": 0,
                "expired_entries": 0,
                "compression_enabled": self.enable_compression,
            }

        try:
            cache_data = self._load_cache_file()

            total = len(cache_data)
            valid = sum(
                1 for item in cache_data.values() if self._is_valid(item.get("timestamp", 0))
            )

            return {
                "total_entries": total,
                "valid_entries": valid,
                "expired_entries": total - valid,
                "cache_file_size": self.metadata_file.stat().st_size,
                "cache_directory": str(self.cache_dir),
                "compression_enabled": self.enable_compression,
            }

        except Exception as e:
            self.logger.warning(f"Failed to get cache stats: {e}")
            return {"error": str(e), "compression_enabled": self.enable_compression}


__all__ = ["CacheManager"]
