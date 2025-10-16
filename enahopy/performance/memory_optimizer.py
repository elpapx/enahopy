"""
ENAHO Memory Optimizer - MEASURE Phase
======================================

Advanced memory profiling, monitoring, and optimization system for large ENAHO datasets.
Includes intelligent memory management, streaming capabilities, and performance analytics.
"""

import gc
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from memory_profiler import LineProfiler, profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""

    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    process_threads: int
    gc_collections: Dict[str, int]


@dataclass
class MemoryProfile:
    """Memory profiling result"""

    function_name: str
    initial_memory: float
    peak_memory: float
    final_memory: float
    memory_delta: float
    duration: float
    snapshots: List[MemorySnapshot]
    recommendations: List[str]


class MemoryMonitor:
    """Real-time memory monitoring with alerts and optimization suggestions"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        warning_threshold: float = 80.0,
        critical_threshold: float = 95.0,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = []
        self.alerts = []

        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, memory monitoring will be limited")

    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Memory monitoring stopped")

    def _monitoring_loop(self, interval: float):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self.capture_snapshot()
                self.snapshots.append(snapshot)

                # Check for alerts
                if snapshot.percent > self.critical_threshold:
                    alert = f"CRITICAL: Memory usage {snapshot.percent:.1f}% > {self.critical_threshold}%"
                    self.alerts.append(alert)
                    self.logger.critical(alert)
                elif snapshot.percent > self.warning_threshold:
                    alert = (
                        f"WARNING: Memory usage {snapshot.percent:.1f}% > {self.warning_threshold}%"
                    )
                    self.alerts.append(alert)
                    self.logger.warning(alert)

                # Keep only last 1000 snapshots to prevent memory buildup
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-1000:]

                time.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)

    def capture_snapshot(self) -> MemorySnapshot:
        """Capture current memory snapshot"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # System memory
            system_memory = psutil.virtual_memory()

            # GC stats
            gc_stats = {}
            for i in range(3):
                gc_stats[f"gen_{i}"] = gc.get_count()[i]

            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=memory_percent,
                available_mb=system_memory.available / (1024 * 1024),
                process_threads=process.num_threads(),
                gc_collections=gc_stats,
            )
        else:
            # Fallback without psutil
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0.0,
                vms_mb=0.0,
                percent=0.0,
                available_mb=0.0,
                process_threads=1,
                gc_collections={},
            )

    def get_peak_memory(self) -> float:
        """Get peak memory usage from snapshots"""
        if not self.snapshots:
            return 0.0
        return max(snapshot.rss_mb for snapshot in self.snapshots)

    def get_memory_trend(self, window_size: int = 10) -> str:
        """Analyze memory usage trend"""
        if len(self.snapshots) < window_size:
            return "insufficient_data"

        recent = self.snapshots[-window_size:]
        first_half = recent[: window_size // 2]
        second_half = recent[window_size // 2 :]

        avg_first = sum(s.rss_mb for s in first_half) / len(first_half)
        avg_second = sum(s.rss_mb for s in second_half) / len(second_half)

        if avg_second > avg_first * 1.1:
            return "increasing"
        elif avg_second < avg_first * 0.9:
            return "decreasing"
        else:
            return "stable"

    def generate_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []

        if not self.snapshots:
            return recommendations

        current = self.snapshots[-1]
        peak = self.get_peak_memory()
        trend = self.get_memory_trend()

        # High memory usage
        if current.percent > 70:
            recommendations.append(
                "Consider using chunked processing or streaming for large datasets"
            )
            recommendations.append("Enable garbage collection more frequently")

        # Memory trend analysis
        if trend == "increasing":
            recommendations.append("Memory usage is increasing - check for memory leaks")
            recommendations.append("Consider using more memory-efficient data types")

        # Low available memory
        if current.available_mb < 1000:
            recommendations.append("System memory is low - consider processing in smaller chunks")
            recommendations.append("Close unnecessary applications")

        # High peak memory
        if peak > current.rss_mb * 2:
            recommendations.append(
                "Peak memory usage is high - implement streaming or chunked processing"
            )

        return recommendations


class DataFrameOptimizer:
    """Optimize pandas DataFrames for memory efficiency"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def optimize_dtypes(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency

        Args:
            df: DataFrame to optimize
            inplace: Whether to modify DataFrame in place

        Returns:
            Optimized DataFrame
        """
        if not inplace:
            df = df.copy()

        initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Optimize numeric columns
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Optimize object columns
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                df[col] = df[col].astype("category")

        final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_saved = initial_memory - final_memory

        self.logger.info(
            f"DataFrame optimized: {initial_memory:.1f} MB → {final_memory:.1f} MB "
            f"(saved {memory_saved:.1f} MB, {memory_saved/initial_memory*100:.1f}%)"
        )

        return df

    def suggest_chunking_strategy(
        self, file_size_mb: float, available_memory_mb: float
    ) -> Dict[str, Any]:
        """
        Suggest chunking strategy based on file size and available memory

        Args:
            file_size_mb: File size in MB
            available_memory_mb: Available memory in MB

        Returns:
            Chunking strategy recommendations
        """
        # Rule of thumb: use 1/4 of available memory for data processing
        safe_memory_mb = available_memory_mb * 0.25

        if file_size_mb <= safe_memory_mb:
            return {
                "strategy": "load_full",
                "chunk_size": None,
                "estimated_chunks": 1,
                "reason": "File fits comfortably in memory",
            }

        # Calculate optimal chunk size
        # Estimate that pandas uses ~3x the file size in memory
        pandas_memory_factor = 3.0
        estimated_memory_per_mb = pandas_memory_factor

        chunk_size_mb = safe_memory_mb / estimated_memory_per_mb
        estimated_chunks = int(np.ceil(file_size_mb / chunk_size_mb))

        # Convert to rows (rough estimate)
        # Assume average of 100 bytes per row (very rough estimate)
        estimated_rows_per_chunk = int((chunk_size_mb * 1024 * 1024) / 100)

        return {
            "strategy": "chunked_processing",
            "chunk_size": max(1000, estimated_rows_per_chunk),  # Minimum 1000 rows
            "estimated_chunks": estimated_chunks,
            "chunk_size_mb": chunk_size_mb,
            "reason": f"File too large for memory, process in {estimated_chunks} chunks",
        }

    def create_memory_efficient_reader(
        self, file_path: Path, chunk_strategy: Dict[str, Any]
    ) -> Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader]:
        """
        Create memory-efficient file reader based on strategy

        Args:
            file_path: Path to file
            chunk_strategy: Chunking strategy from suggest_chunking_strategy

        Returns:
            DataFrame or chunked reader
        """
        if chunk_strategy["strategy"] == "load_full":
            return pd.read_csv(file_path)
        else:
            return pd.read_csv(file_path, chunksize=chunk_strategy["chunk_size"])


class StreamingProcessor:
    """Process large datasets with streaming to minimize memory usage"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.optimizer = DataFrameOptimizer(logger)

    def process_csv_streaming(
        self,
        file_path: Path,
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[Path] = None,
        chunk_size: int = 10000,
    ) -> Dict[str, Any]:
        """
        Process large CSV file in streaming fashion

        Args:
            file_path: Input CSV file
            processing_function: Function to apply to each chunk
            output_path: Optional output file path
            chunk_size: Size of each chunk

        Returns:
            Processing statistics
        """
        stats = {
            "total_rows": 0,
            "chunks_processed": 0,
            "processing_time": 0,
            "peak_memory_mb": 0,
            "errors": [],
        }

        start_time = time.time()
        monitor = MemoryMonitor(self.logger)
        monitor.start_monitoring(interval=0.5)

        try:
            first_chunk = True

            for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                chunk_start = time.time()

                try:
                    # Optimize chunk memory usage
                    chunk = self.optimizer.optimize_dtypes(chunk)

                    # Apply processing function
                    processed_chunk = processing_function(chunk)

                    # Write to output file
                    if output_path:
                        mode = "w" if first_chunk else "a"
                        header = first_chunk
                        processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                        first_chunk = False

                    # Update statistics
                    stats["total_rows"] += len(chunk)
                    stats["chunks_processed"] += 1

                    # Force garbage collection after each chunk
                    del chunk, processed_chunk
                    gc.collect()

                    chunk_time = time.time() - chunk_start
                    self.logger.debug(f"Processed chunk {chunk_num + 1} in {chunk_time:.2f}s")

                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_num + 1}: {str(e)}"
                    stats["errors"].append(error_msg)
                    self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error reading file {file_path}: {str(e)}"
            stats["errors"].append(error_msg)
            self.logger.error(error_msg)

        finally:
            monitor.stop_monitoring()
            stats["processing_time"] = time.time() - start_time
            stats["peak_memory_mb"] = monitor.get_peak_memory()

        return stats

    def aggregate_streaming(
        self,
        file_path: Path,
        group_columns: List[str],
        agg_functions: Dict[str, str],
        chunk_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Perform aggregation on large dataset using streaming

        Args:
            file_path: Input CSV file
            group_columns: Columns to group by
            agg_functions: Aggregation functions
            chunk_size: Chunk size for processing

        Returns:
            Aggregated DataFrame
        """
        aggregated_data = {}

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Optimize chunk
            chunk = self.optimizer.optimize_dtypes(chunk)

            # Perform groupby aggregation on chunk
            chunk_agg = chunk.groupby(group_columns).agg(agg_functions)

            # Merge with existing aggregated data
            for index, row in chunk_agg.iterrows():
                if index in aggregated_data:
                    # Combine with existing data (assuming sum for now)
                    for col in chunk_agg.columns:
                        aggregated_data[index][col] += row[col]
                else:
                    aggregated_data[index] = row.to_dict()

            # Clean up
            del chunk, chunk_agg
            gc.collect()

        # Convert back to DataFrame
        result = pd.DataFrame.from_dict(aggregated_data, orient="index")
        result.index.names = group_columns
        return result.reset_index()


def memory_profile_decorator(monitor_interval: float = 0.1):
    """
    Decorator for memory profiling functions

    Args:
        monitor_interval: Monitoring interval in seconds

    Returns:
        Decorated function with memory profiling
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = MemoryMonitor()

            # Capture initial state
            initial_snapshot = monitor.capture_snapshot()
            monitor.start_monitoring(monitor_interval)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop monitoring and generate profile
                monitor.stop_monitoring()
                end_time = time.time()

                final_snapshot = monitor.capture_snapshot()
                peak_memory = monitor.get_peak_memory()

                profile = MemoryProfile(
                    function_name=func.__name__,
                    initial_memory=initial_snapshot.rss_mb,
                    peak_memory=peak_memory,
                    final_memory=final_snapshot.rss_mb,
                    memory_delta=final_snapshot.rss_mb - initial_snapshot.rss_mb,
                    duration=end_time - start_time,
                    snapshots=monitor.snapshots,
                    recommendations=monitor.generate_recommendations(),
                )

                # Log profile summary
                logger = logging.getLogger(func.__module__)
                logger.info(
                    f"Memory profile for {func.__name__}: "
                    f"Peak: {peak_memory:.1f}MB, "
                    f"Delta: {profile.memory_delta:+.1f}MB, "
                    f"Duration: {profile.duration:.2f}s"
                )

                # Store profile for later analysis
                if not hasattr(wrapper, "_memory_profiles"):
                    wrapper._memory_profiles = []
                wrapper._memory_profiles.append(profile)

        return wrapper

    return decorator


@contextmanager
def memory_optimized_context(
    auto_gc: bool = True, gc_threshold: Optional[Tuple[int, int, int]] = None
):
    """
    Context manager for memory-optimized operations

    Args:
        auto_gc: Enable automatic garbage collection
        gc_threshold: Custom GC threshold
    """
    # Store original GC settings
    original_gc_enabled = gc.isenabled()
    original_thresholds = gc.get_threshold()

    try:
        if auto_gc:
            gc.enable()
            if gc_threshold:
                gc.set_threshold(*gc_threshold)
            else:
                # More aggressive GC for memory optimization
                gc.set_threshold(100, 10, 10)

        # Force initial cleanup
        gc.collect()

        yield

    finally:
        # Restore original settings
        if original_gc_enabled:
            gc.enable()
        else:
            gc.disable()
        gc.set_threshold(*original_thresholds)

        # Final cleanup
        gc.collect()


def optimize_pandas_settings():
    """Configure pandas for memory-efficient operations"""
    # Configure pandas for memory efficiency
    pd.set_option("compute.use_bottleneck", True)
    pd.set_option("compute.use_numexpr", True)

    # Set reasonable display options to prevent memory issues with large DataFrames
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 50)


def create_memory_optimizer(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Factory function to create memory optimization tools

    Args:
        logger: Optional logger instance

    Returns:
        Dictionary of memory optimization tools
    """
    logger = logger or logging.getLogger(__name__)

    return {
        "monitor": MemoryMonitor(logger),
        "dataframe_optimizer": DataFrameOptimizer(logger),
        "streaming_processor": StreamingProcessor(logger),
        "profile_decorator": memory_profile_decorator,
        "optimized_context": memory_optimized_context,
    }


__all__ = [
    "MemoryMonitor",
    "DataFrameOptimizer",
    "StreamingProcessor",
    "MemoryProfile",
    "MemorySnapshot",
    "memory_profile_decorator",
    "memory_optimized_context",
    "optimize_pandas_settings",
    "create_memory_optimizer",
]
