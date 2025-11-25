"""
ENAHO Async Downloader Module - MEASURE Phase
=============================================

High-performance asynchronous downloader with concurrent downloads,
advanced retry logic, memory profiling, and streaming capabilities.
Optimized for large ENAHO datasets with significant performance improvements.
"""

import asyncio
import hashlib
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiofiles

try:
    import aiofiles.os
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    pass

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from tqdm.asyncio import tqdm

from ...core.cache import CacheManager
from ...core.config import ENAHOConfig
from ...core.exceptions import ENAHODownloadError, ENAHOIntegrityError, ENAHOTimeoutError


class PerformanceMonitor:
    """Monitor performance metrics during downloads"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
        self.download_stats = {
            "total_bytes": 0,
            "total_files": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "retry_attempts": 0,
            "concurrent_connections": 0,
            "peak_memory_mb": 0,
            "total_duration": 0.0,
        }

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        if PSUTIL_AVAILABLE:
            self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    def record_download(self, file_size: int, duration: float, success: bool = True):
        """Record download completion"""
        self.download_stats["total_bytes"] += file_size
        self.download_stats["total_files"] += 1

        if success:
            self.download_stats["successful_downloads"] += 1
        else:
            self.download_stats["failed_downloads"] += 1

        # Update peak memory usage
        if PSUTIL_AVAILABLE:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.download_stats["peak_memory_mb"] = max(
                self.download_stats["peak_memory_mb"], current_memory
            )

    def record_retry(self):
        """Record retry attempt"""
        self.download_stats["retry_attempts"] += 1

    def finish_monitoring(self) -> Dict[str, Any]:
        """Finish monitoring and return stats"""
        if self.start_time:
            self.download_stats["total_duration"] = time.time() - self.start_time

        # Calculate throughput
        if self.download_stats["total_duration"] > 0:
            throughput_mbps = (
                self.download_stats["total_bytes"] / 1024 / 1024
            ) / self.download_stats["total_duration"]
            self.download_stats["throughput_mbps"] = throughput_mbps

        return self.download_stats

    def get_summary(self) -> str:
        """Get performance summary"""
        self.download_stats
        return """
Performance Summary:
  • Files Downloaded: {stats['successful_downloads']}/{stats['total_files']}
  • Total Data: {stats['total_bytes'] / (1024*1024):.1f} MB
  • Duration: {stats['total_duration']:.1f}s
  • Throughput: {stats.get('throughput_mbps', 0):.1f} MB/s
  • Peak Memory: {stats['peak_memory_mb']:.1f} MB
  • Retry Attempts: {stats['retry_attempts']}
        """.strip()


class AsyncNetworkUtils:
    """Async network utilities with connection pooling"""

    def __init__(self, config: ENAHOConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._session = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()

    async def create_session(self) -> aiohttp.ClientSession:
        """Create optimized aiohttp session"""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for async downloading")

        # Optimized connector settings
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent_downloads * 2,  # Connection pool size
            limit_per_host=self.config.max_concurrent_downloads,
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )

        # Timeout configuration
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout * 3,  # Total timeout
            connect=self.config.timeout,  # Connection timeout
            sock_read=self.config.timeout,  # Socket read timeout
        )

        # Custom headers
        headers = {
            "User-Agent": "ENAHO-Analyzer-Async/2.0 (High-Performance Python Data Analysis Tool)",
            "Accept": "application/zip, application/octet-stream, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers)

        return self._session

    async def close_session(self):
        """Close session and cleanup"""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get current session"""
        if not self._session:
            raise RuntimeError("Session not created. Use async context manager.")
        return self._session

    async def check_url_exists(self, url: str) -> bool:
        """Check if URL exists asynchronously"""
        try:
            async with self.session.head(url) as response:
                return response.status == 200
        except Exception:
            return False

    async def get_file_info(self, url: str) -> Tuple[Optional[int], Dict[str, str]]:
        """Get file size and headers asynchronously"""
        try:
            async with self.session.head(url) as response:
                if response.status == 200:
                    size = int(response.headers.get("content-length", 0))
                    headers = dict(response.headers)
                    return size, headers
                return None, {}
        except Exception:
            return None, {}


class StreamingDownloader:
    """High-performance streaming downloader with memory optimization"""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        config: ENAHOConfig,
        logger: logging.Logger,
        monitor: PerformanceMonitor,
    ):
        self.session = session
        self.config = config
        self.logger = logger
        self.monitor = monitor

    async def download_with_streaming(
        self, url: str, file_path: Path, progress_bar: Optional[tqdm] = None
    ) -> Dict[str, Any]:
        """
        Download file with streaming and memory optimization

        Args:
            url: URL to download
            file_path: Destination file path
            progress_bar: Optional progress bar

        Returns:
            Download statistics
        """
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        start_time = time.time()
        downloaded_bytes = 0

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise ENAHODownloadError(f"HTTP {response.status} for {url}")

                int(response.headers.get("content-length", 0))

                # Use aiofiles for async file I/O
                async with aiofiles.open(temp_path, "wb") as file:
                    async for chunk in response.content.iter_chunked(self.config.chunk_size):
                        await file.write(chunk)
                        downloaded_bytes += len(chunk)

                        if progress_bar:
                            progress_bar.update(len(chunk))

                # Atomic move to final location
                await aiofiles.os.rename(temp_path, file_path)

                duration = time.time() - start_time
                stats = {
                    "url": url,
                    "file_path": str(file_path),
                    "bytes_downloaded": downloaded_bytes,
                    "duration": duration,
                    "speed_mbps": (
                        (downloaded_bytes / 1024 / 1024) / duration if duration > 0 else 0
                    ),
                    "status": "success",
                }

                self.monitor.record_download(downloaded_bytes, duration, True)
                return stats

        except Exception as e:
            # Cleanup temp file
            try:
                await aiofiles.os.remove(temp_path)
            except Exception:
                pass

            self.monitor.record_download(downloaded_bytes, time.time() - start_time, False)

            if isinstance(e, asyncio.TimeoutError):
                raise ENAHOTimeoutError(f"Timeout downloading {url}")
            else:
                raise ENAHODownloadError(f"Failed to download {url}: {str(e)}")


class AsyncRetryHandler:
    """Advanced async retry handler with exponential backof"""

    def __init__(self, config: ENAHOConfig, logger: logging.Logger, monitor: PerformanceMonitor):
        self.config = config
        self.logger = logger
        self.monitor = monitor

    async def retry_download(self, download_func, *args, **kwargs) -> Any:
        """Retry download with exponential backof"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await download_func(*args, **kwargs)

            except (aiohttp.ClientError, asyncio.TimeoutError, ENAHODownloadError) as e:
                last_exception = e
                self.monitor.record_retry()

                if attempt < self.config.max_retries:
                    # Exponential backoff with jitter
                    delay = self.config.backoff_factor * (2**attempt) + (
                        0.1 * attempt
                    )  # Add small jitter

                    self.logger.warning(
                        f"Download failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {str(e)}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    await asyncio.sleep(delay)
                else:
                    self.logger.error("All retry attempts failed for download")

        raise last_exception


class ENAHOAsyncDownloader:
    """
    High-performance async ENAHO downloader with concurrent downloads,
    streaming, memory optimization, and comprehensive performance monitoring.
    """

    def __init__(self, config: ENAHOConfig, logger: logging.Logger, cache_manager: CacheManager):
        self.config = config
        self.logger = logger
        self.cache_manager = cache_manager

        # Performance monitoring
        self.monitor = PerformanceMonitor(logger)

        # Semaphore for controlling concurrent downloads
        self.download_semaphore = asyncio.Semaphore(self.config.max_concurrent_downloads)

        # Thread pool for CPU-intensive operations (checksum, zip validation)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _build_url(self, code: int, module: str) -> str:
        """Build download URL"""
        filename = f"{code}-Modulo{module}.zip"
        return urljoin(self.config.base_url, filename)

    def _calculate_checksum_sync(self, file_path: Path) -> str:
        """Calculate checksum synchronously (for thread pool)"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):  # Larger chunks for better performance
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _validate_zip_integrity_sync(self, file_path: Path) -> bool:
        """Validate ZIP integrity synchronously (for thread pool)"""
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.testzip()
                return True
        except (zipfile.BadZipFile, zipfile.LargeZipFile):
            return False

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate checksum asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._calculate_checksum_sync, file_path)

    async def _validate_zip_integrity(self, file_path: Path) -> bool:
        """Validate ZIP integrity asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._validate_zip_integrity_sync, file_path
        )

    async def download_single_file(
        self,
        year: str,
        module: str,
        code: int,
        output_dir: Path,
        overwrite: bool = False,
        network_utils: AsyncNetworkUtils = None,
        progress_bar: Optional[tqdm] = None,
    ) -> Dict[str, Any]:
        """
        Download a single file asynchronously

        Args:
            year: Survey year
            module: Module code
            code: Internal code for URL
            output_dir: Output directory
            overwrite: Whether to overwrite existing files
            network_utils: Shared network utilities
            progress_bar: Optional progress bar

        Returns:
            Download result dictionary
        """
        url = self._build_url(code, module)
        filename = f"modulo_{module}_{year}.zip"
        file_path = output_dir / filename

        # Check if file exists and is valid
        if file_path.exists() and not overwrite:
            if await self._validate_zip_integrity(file_path):
                self.logger.info(f"Valid file found: {filename}")
                return {
                    "file": filename,
                    "status": "cached",
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                }
            else:
                self.logger.warning(f"Corrupted file found, re-downloading: {filename}")
                await aiofiles.os.remove(file_path)

        # Use semaphore to control concurrent downloads
        async with self.download_semaphore:
            try:
                # Check if URL exists
                if not await network_utils.check_url_exists(url):
                    raise ENAHODownloadError(f"URL not found: {url}", "URL_NOT_FOUND")

                # Get file info
                file_size, headers = await network_utils.get_file_info(url)

                # Create streaming downloader
                downloader = StreamingDownloader(
                    network_utils.session, self.config, self.logger, self.monitor
                )

                # Create retry handler
                retry_handler = AsyncRetryHandler(self.config, self.logger, self.monitor)

                # Download with retry logic
                download_stats = await retry_handler.retry_download(
                    downloader.download_with_streaming, url, file_path, progress_bar
                )

                # Validate integrity
                if not await self._validate_zip_integrity(file_path):
                    await aiofiles.os.remove(file_path)
                    raise ENAHOIntegrityError(f"Corrupted ZIP file: {filename}")

                # Calculate and cache checksum if enabled
                if self.config.verify_checksums:
                    checksum = await self._calculate_checksum(file_path)
                    await self.cache_manager.set_metadata_async(
                        f"checksum_{filename}",
                        {
                            "checksum": checksum,
                            "size": file_path.stat().st_size,
                            "download_stats": download_stats,
                        },
                    )

                result = {
                    "file": filename,
                    "status": "downloaded",
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "stats": download_stats,
                }

                self.logger.info(
                    f"Downloaded {filename} "
                    f"({download_stats['bytes_downloaded'] / (1024*1024):.1f} MB, "
                    f"{download_stats['speed_mbps']:.1f} MB/s)"
                )

                return result

            except Exception as e:
                self.logger.error(
                    f"Failed to download {filename}: {str(e)}",
                    extra={
                        "context": {
                            "url": url,
                            "year": year,
                            "module": module,
                            "output_path": str(file_path),
                        }
                    },
                )
                return {"file": filename, "status": "failed", "error": str(e)}

    async def download_multiple_files(
        self,
        download_requests: List[Dict[str, Any]],
        output_dir: Path,
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Download multiple files concurrently with optimized performance

        Args:
            download_requests: List of download request dictionaries
            output_dir: Output directory
            overwrite: Whether to overwrite existing files
            show_progress: Whether to show progress bars

        Returns:
            Comprehensive download results
        """
        self.logger.info(f"Starting concurrent download of {len(download_requests)} files")
        self.monitor.start_monitoring()

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "successful": [],
            "failed": [],
            "cached": [],
            "total_files": len(download_requests),
            "total_size": 0,
            "performance_stats": {},
        }

        # Create shared network utilities
        async with AsyncNetworkUtils(self.config, self.logger) as network_utils:
            # Create overall progress bar
            overall_progress = None
            if show_progress:
                overall_progress = tqdm(
                    total=len(download_requests), desc="Downloading ENAHO files", unit="file"
                )

            # Create tasks for concurrent downloads
            tasks = []
            for req in download_requests:
                task = self.download_single_file(
                    year=req["year"],
                    module=req["module"],
                    code=req["code"],
                    output_dir=output_dir,
                    overwrite=overwrite,
                    network_utils=network_utils,
                )
                tasks.append(task)

            # Execute downloads concurrently
            download_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(download_results):
                if isinstance(result, Exception):
                    results["failed"].append(
                        {"request": download_requests[i], "error": str(result)}
                    )
                else:
                    if result["status"] == "downloaded":
                        results["successful"].append(result)
                        results["total_size"] += result.get("size", 0)
                    elif result["status"] == "cached":
                        results["cached"].append(result)
                        results["total_size"] += result.get("size", 0)
                    elif result["status"] == "failed":
                        results["failed"].append(result)

                if overall_progress:
                    overall_progress.update(1)

            if overall_progress:
                overall_progress.close()

        # Finalize performance monitoring
        perf_stats = self.monitor.finish_monitoring()
        results["performance_stats"] = perf_stats

        # Log summary
        self.logger.info(
            f"Download completed: {len(results['successful'])} successful, "
            f"{len(results['cached'])} cached, {len(results['failed'])} failed. "
            f"Total: {results['total_size'] / (1024*1024):.1f} MB"
        )

        if show_progress:
            print(self.monitor.get_summary())

        return results

    async def download_year_modules(
        self,
        year: str,
        modules: List[str],
        output_dir: Path,
        code_mapping: Dict[str, int],
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Download all modules for a specific year concurrently

        Args:
            year: Survey year
            modules: List of module codes
            output_dir: Output directory
            code_mapping: Mapping of modules to internal codes
            overwrite: Whether to overwrite existing files
            show_progress: Whether to show progress

        Returns:
            Download results
        """
        download_requests = []

        for module in modules:
            if module in code_mapping:
                download_requests.append(
                    {"year": year, "module": module, "code": code_mapping[module]}
                )

        return await self.download_multiple_files(
            download_requests, output_dir, overwrite, show_progress
        )

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


# Enhanced cache manager with async support
class AsyncCacheManager(CacheManager):
    """Async-enabled cache manager"""

    async def set_metadata_async(self, key: str, metadata: Dict[str, Any]):
        """Set metadata asynchronously"""
        # Use thread pool for disk I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.set_metadata, key, metadata)

    async def get_metadata_async(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_metadata, key)


def create_async_downloader(
    config: Optional[ENAHOConfig] = None,
    logger: Optional[logging.Logger] = None,
    cache_manager: Optional[CacheManager] = None,
) -> ENAHOAsyncDownloader:
    """
    Factory function to create async downloader with optimized configuration

    Args:
        config: Optional ENAHO configuration
        logger: Optional logger
        cache_manager: Optional cache manager

    Returns:
        Configured async downloader
    """
    if config is None:
        config = ENAHOConfig()
        # Optimize for async downloads
        config.max_concurrent_downloads = min(8, config.max_concurrent_downloads * 2)
        config.chunk_size = max(
            config.chunk_size, 64 * 1024
        )  # Larger chunks for better performance

    if logger is None:
        from ...core.logging import setup_logging

        logger = setup_logging()

    if cache_manager is None:
        cache_manager = AsyncCacheManager()

    return ENAHOAsyncDownloader(config, logger, cache_manager)


__all__ = [
    "ENAHOAsyncDownloader",
    "AsyncNetworkUtils",
    "StreamingDownloader",
    "PerformanceMonitor",
    "AsyncRetryHandler",
    "create_async_downloader",
]
