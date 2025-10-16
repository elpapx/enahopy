"""
ENAHO Performance Benchmarking & Monitoring System - MEASURE Phase
==================================================================

Comprehensive benchmarking and performance monitoring system for ENAHO operations.
Tracks download speeds, processing performance, memory usage, and provides automated
performance regression detection and optimization recommendations.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from .memory_optimizer import MemoryMonitor, create_memory_optimizer
from .streaming import StreamingConfig, StreamingProcessor


@dataclass
class SystemInfo:
    """System information for benchmarking context"""

    python_version: str
    platform: str
    cpu_count: int
    cpu_freq_mhz: float
    memory_total_gb: float
    disk_free_gb: float
    enaho_version: str
    timestamp: datetime

    @classmethod
    def capture(cls) -> "SystemInfo":
        """Capture current system information"""
        system_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or 1,
            "cpu_freq_mhz": 0.0,
            "memory_total_gb": 0.0,
            "disk_free_gb": 0.0,
            "enaho_version": "0.1.1",  # Should be imported from package
            "timestamp": datetime.now(),
        }

        if PSUTIL_AVAILABLE:
            try:
                cpu_freq = psutil.cpu_freq()
                system_info["cpu_freq_mhz"] = cpu_freq.current if cpu_freq else 0.0

                memory = psutil.virtual_memory()
                system_info["memory_total_gb"] = memory.total / (1024**3)

                disk = psutil.disk_usage(".")
                system_info["disk_free_gb"] = disk.free / (1024**3)
            except:
                pass

        return cls(**system_info)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark operation"""

    operation: str
    duration_seconds: float
    throughput_mb_per_sec: float
    memory_peak_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""

    suite_name: str
    system_info: SystemInfo
    results: List[BenchmarkResult]
    total_duration: float
    timestamp: datetime

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the benchmark suite"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if successful:
            avg_throughput = np.mean([r.throughput_mb_per_sec for r in successful])
            avg_memory = np.mean([r.memory_peak_mb for r in successful])
            avg_cpu = np.mean([r.cpu_percent for r in successful])
        else:
            avg_throughput = avg_memory = avg_cpu = 0.0

        return {
            "total_operations": len(self.results),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0.0,
            "average_throughput_mb_per_sec": avg_throughput,
            "average_memory_peak_mb": avg_memory,
            "average_cpu_percent": avg_cpu,
            "total_duration": self.total_duration,
        }


class PerformanceProfiler:
    """Advanced performance profiler for ENAHO operations"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.memory_tools = create_memory_optimizer(logger)
        self.active_profiles = {}

    def profile_operation(self, operation_name: str, data_size_mb: float = 0.0):
        """
        Context manager for profiling operations

        Args:
            operation_name: Name of the operation being profiled
            data_size_mb: Size of data being processed in MB
        """
        return OperationProfiler(operation_name, data_size_mb, self.logger)

    async def profile_async_operation(
        self, operation_name: str, async_func: Callable, *args, **kwargs
    ) -> BenchmarkResult:
        """
        Profile an async operation

        Args:
            operation_name: Name of the operation
            async_func: Async function to profile
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Benchmark result
        """
        start_time = time.time()
        memory_monitor = MemoryMonitor(self.logger)
        cpu_monitor = CPUMonitor()

        memory_monitor.start_monitoring(interval=0.1)
        cpu_monitor.start()

        error_message = None
        success = True

        try:
            result = await async_func(*args, **kwargs)
            return result
        except Exception as e:
            error_message = str(e)
            success = False
            self.logger.error(f"Error in {operation_name}: {error_message}")
            return None
        finally:
            end_time = time.time()
            memory_monitor.stop_monitoring()
            cpu_monitor.stop()

            duration = end_time - start_time
            peak_memory = memory_monitor.get_peak_memory()
            avg_cpu = cpu_monitor.get_average_cpu()

            # Estimate throughput (placeholder logic)
            throughput = 0.0
            if "data_size_mb" in kwargs:
                data_size = kwargs["data_size_mb"]
                throughput = data_size / duration if duration > 0 else 0.0

            # Create benchmark result
            benchmark_result = BenchmarkResult(
                operation=operation_name,
                duration_seconds=duration,
                throughput_mb_per_sec=throughput,
                memory_peak_mb=peak_memory,
                cpu_percent=avg_cpu,
                success=success,
                error_message=error_message,
                metadata={"async": True},
            )

            self.logger.info(
                f"Profile {operation_name}: {duration:.2f}s, "
                f"{throughput:.1f} MB/s, Peak Memory: {peak_memory:.1f}MB"
            )


class OperationProfiler:
    """Context manager for profiling individual operations"""

    def __init__(self, operation_name: str, data_size_mb: float, logger: logging.Logger):
        self.operation_name = operation_name
        self.data_size_mb = data_size_mb
        self.logger = logger

        self.start_time = None
        self.memory_monitor = None
        self.cpu_monitor = None

    def __enter__(self) -> "OperationProfiler":
        self.start_time = time.time()
        self.memory_monitor = MemoryMonitor(self.logger)
        self.cpu_monitor = CPUMonitor()

        self.memory_monitor.start_monitoring(interval=0.1)
        self.cpu_monitor.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()

        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
        if self.cpu_monitor:
            self.cpu_monitor.stop()

        duration = end_time - self.start_time
        peak_memory = self.memory_monitor.get_peak_memory() if self.memory_monitor else 0.0
        avg_cpu = self.cpu_monitor.get_average_cpu() if self.cpu_monitor else 0.0

        throughput = self.data_size_mb / duration if duration > 0 else 0.0

        success = exc_type is None
        error_message = str(exc_val) if exc_val else None

        result = BenchmarkResult(
            operation=self.operation_name,
            duration_seconds=duration,
            throughput_mb_per_sec=throughput,
            memory_peak_mb=peak_memory,
            cpu_percent=avg_cpu,
            success=success,
            error_message=error_message,
        )

        self.logger.info(
            f"Profile {self.operation_name}: {duration:.2f}s, "
            f"{throughput:.1f} MB/s, Peak Memory: {peak_memory:.1f}MB"
        )

        # Store result for later retrieval
        self.result = result


class CPUMonitor:
    """Monitor CPU usage during operations"""

    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.monitor_thread = None

    def start(self):
        """Start CPU monitoring"""
        if not PSUTIL_AVAILABLE:
            return

        self.monitoring = True
        self.cpu_samples = []

        import threading

        self.monitor_thread = threading.Thread(target=self._monitor_cpu, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop CPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_cpu(self):
        """Internal CPU monitoring loop"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
            except:
                time.sleep(0.1)

    def get_average_cpu(self) -> float:
        """Get average CPU usage"""
        return np.mean(self.cpu_samples) if self.cpu_samples else 0.0


class ENAHOBenchmarkSuite:
    """Comprehensive benchmark suite for ENAHO operations"""

    def __init__(self, logger: Optional[logging.Logger] = None, output_dir: Optional[Path] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = output_dir or Path("./benchmarks")
        self.output_dir.mkdir(exist_ok=True)

        self.profiler = PerformanceProfiler(logger)
        self.results = []

    def benchmark_download_performance(
        self, test_urls: List[str], concurrent_levels: List[int] = [1, 2, 4, 8]
    ) -> List[BenchmarkResult]:
        """
        Benchmark download performance with different concurrency levels

        Args:
            test_urls: URLs to test (can be mock URLs for testing)
            concurrent_levels: Different concurrency levels to test

        Returns:
            List of benchmark results
        """
        self.logger.info("Starting download performance benchmark")
        results = []

        for concurrency in concurrent_levels:
            with self.profiler.profile_operation(
                f"download_concurrent_{concurrency}", data_size_mb=100.0  # Estimate
            ) as profiler:
                try:
                    # Mock download simulation (replace with actual download logic)
                    self._simulate_concurrent_downloads(test_urls, concurrency)
                    results.append(profiler.result)
                except Exception as e:
                    self.logger.error(
                        f"Download benchmark failed for concurrency {concurrency}: {e}"
                    )

        return results

    def benchmark_data_processing(
        self, file_sizes_mb: List[float] = [10, 50, 100, 500]
    ) -> List[BenchmarkResult]:
        """
        Benchmark data processing performance with different file sizes

        Args:
            file_sizes_mb: Different file sizes to test in MB

        Returns:
            List of benchmark results
        """
        self.logger.info("Starting data processing benchmark")
        results = []

        for size_mb in file_sizes_mb:
            with self.profiler.profile_operation(
                f"data_processing_{size_mb}mb", data_size_mb=size_mb
            ) as profiler:
                try:
                    # Create synthetic data for testing
                    test_df = self._create_test_dataframe(size_mb)

                    # Simulate processing operations
                    self._simulate_data_processing(test_df)

                    results.append(profiler.result)
                except Exception as e:
                    self.logger.error(f"Data processing benchmark failed for {size_mb}MB: {e}")

        return results

    def benchmark_streaming_performance(
        self, data_sizes_mb: List[float] = [100, 500, 1000]
    ) -> List[BenchmarkResult]:
        """
        Benchmark streaming processing performance

        Args:
            data_sizes_mb: Different data sizes to test

        Returns:
            List of benchmark results
        """
        self.logger.info("Starting streaming performance benchmark")
        results = []

        streaming_config = StreamingConfig(chunk_size=10000)
        processor = StreamingProcessor(streaming_config, self.logger)

        for size_mb in data_sizes_mb:
            with self.profiler.profile_operation(
                f"streaming_processing_{size_mb}mb", data_size_mb=size_mb
            ) as profiler:
                try:
                    # Create test file
                    test_file = self._create_test_csv_file(size_mb)

                    # Create streaming reader
                    from .streaming import CSVStreamingReader

                    reader = CSVStreamingReader(test_file)

                    # Process with streaming
                    def simple_processing(df):
                        return df.copy()  # Simple processing

                    stats = processor.process_streaming(reader, simple_processing)

                    # Clean up test file
                    test_file.unlink()

                    results.append(profiler.result)
                except Exception as e:
                    self.logger.error(f"Streaming benchmark failed for {size_mb}MB: {e}")

        return results

    def benchmark_memory_efficiency(
        self, operations: List[str] = ["load", "process", "aggregate"]
    ) -> List[BenchmarkResult]:
        """
        Benchmark memory efficiency of different operations

        Args:
            operations: List of operations to benchmark

        Returns:
            List of benchmark results
        """
        self.logger.info("Starting memory efficiency benchmark")
        results = []

        test_data_mb = 100  # Standard test size

        for operation in operations:
            with self.profiler.profile_operation(
                f"memory_efficiency_{operation}", data_size_mb=test_data_mb
            ) as profiler:
                try:
                    test_df = self._create_test_dataframe(test_data_mb)

                    if operation == "load":
                        # Test data loading efficiency
                        _ = test_df.copy()
                    elif operation == "process":
                        # Test data processing efficiency
                        _ = test_df.groupby(test_df.columns[0]).sum()
                    elif operation == "aggregate":
                        # Test aggregation efficiency
                        _ = test_df.agg(["mean", "std", "min", "max"])

                    results.append(profiler.result)
                except Exception as e:
                    self.logger.error(f"Memory benchmark failed for {operation}: {e}")

        return results

    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """
        Run comprehensive benchmark suite

        Returns:
            Complete benchmark suite results
        """
        self.logger.info("Starting comprehensive ENAHO benchmark suite")
        start_time = time.time()

        all_results = []
        system_info = SystemInfo.capture()

        # Download performance
        try:
            mock_urls = [f"http://example.com/test_{i}.zip" for i in range(5)]
            download_results = self.benchmark_download_performance(mock_urls)
            all_results.extend(download_results)
        except Exception as e:
            self.logger.error(f"Download benchmark failed: {e}")

        # Data processing performance
        try:
            processing_results = self.benchmark_data_processing()
            all_results.extend(processing_results)
        except Exception as e:
            self.logger.error(f"Data processing benchmark failed: {e}")

        # Streaming performance
        try:
            streaming_results = self.benchmark_streaming_performance()
            all_results.extend(streaming_results)
        except Exception as e:
            self.logger.error(f"Streaming benchmark failed: {e}")

        # Memory efficiency
        try:
            memory_results = self.benchmark_memory_efficiency()
            all_results.extend(memory_results)
        except Exception as e:
            self.logger.error(f"Memory benchmark failed: {e}")

        total_duration = time.time() - start_time

        benchmark_suite = BenchmarkSuite(
            suite_name="ENAHO_Comprehensive",
            system_info=system_info,
            results=all_results,
            total_duration=total_duration,
            timestamp=datetime.now(),
        )

        # Save results
        self.save_benchmark_results(benchmark_suite)

        # Generate report
        self.generate_benchmark_report(benchmark_suite)

        return benchmark_suite

    def save_benchmark_results(self, suite: BenchmarkSuite):
        """Save benchmark results to JSON file"""
        timestamp_str = suite.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp_str}.json"
        filepath = self.output_dir / filename

        # Convert to JSON serializable format
        data = asdict(suite)
        data["timestamp"] = suite.timestamp.isoformat()
        data["system_info"]["timestamp"] = suite.system_info.timestamp.isoformat()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Benchmark results saved to {filepath}")

    def generate_benchmark_report(self, suite: BenchmarkSuite):
        """Generate human-readable benchmark report"""
        timestamp_str = suite.timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"benchmark_report_{timestamp_str}.md"

        with open(report_path, "w") as f:
            f.write(f"# ENAHO Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {suite.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # System information
            f.write("## System Information\n\n")
            f.write(f"- **Platform:** {suite.system_info.platform}\n")
            f.write(f"- **Python:** {suite.system_info.python_version.split()[0]}\n")
            f.write(f"- **CPUs:** {suite.system_info.cpu_count}\n")
            f.write(f"- **Memory:** {suite.system_info.memory_total_gb:.1f} GB\n")
            f.write(f"- **ENAHO Version:** {suite.system_info.enaho_version}\n\n")

            # Summary statistics
            summary = suite.get_summary_stats()
            f.write("## Summary\n\n")
            f.write(f"- **Total Operations:** {summary['total_operations']}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']:.1%}\n")
            f.write(
                f"- **Average Throughput:** {summary['average_throughput_mb_per_sec']:.1f} MB/s\n"
            )
            f.write(f"- **Average Peak Memory:** {summary['average_memory_peak_mb']:.1f} MB\n")
            f.write(f"- **Total Duration:** {summary['total_duration']:.1f}s\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            f.write(
                "| Operation | Duration (s) | Throughput (MB/s) | Peak Memory (MB) | CPU (%) | Status |\n"
            )
            f.write(
                "|-----------|--------------|-------------------|------------------|---------|--------|\n"
            )

            for result in suite.results:
                status = "✅" if result.success else "❌"
                f.write(
                    f"| {result.operation} | {result.duration_seconds:.2f} | "
                    f"{result.throughput_mb_per_sec:.1f} | {result.memory_peak_mb:.1f} | "
                    f"{result.cpu_percent:.1f} | {status} |\n"
                )

            # Recommendations
            f.write("\n## Recommendations\n\n")
            recommendations = self._generate_recommendations(suite)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

        self.logger.info(f"Benchmark report generated: {report_path}")

    def compare_benchmarks(self, benchmark_files: List[Path]) -> Dict[str, Any]:
        """
        Compare multiple benchmark results for performance regression detection

        Args:
            benchmark_files: List of benchmark JSON files

        Returns:
            Comparison analysis
        """
        benchmarks = []
        for file_path in benchmark_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    benchmarks.append(data)
            except Exception as e:
                self.logger.error(f"Error loading benchmark file {file_path}: {e}")

        if len(benchmarks) < 2:
            return {"error": "Need at least 2 benchmark files for comparison"}

        # Analyze trends
        comparison = {
            "benchmark_count": len(benchmarks),
            "date_range": {
                "earliest": min(b["timestamp"] for b in benchmarks),
                "latest": max(b["timestamp"] for b in benchmarks),
            },
            "performance_trends": {},
            "regressions_detected": [],
        }

        # Group results by operation
        operations = {}
        for benchmark in benchmarks:
            for result in benchmark["results"]:
                op_name = result["operation"]
                if op_name not in operations:
                    operations[op_name] = []
                operations[op_name].append(
                    {
                        "timestamp": benchmark["timestamp"],
                        "throughput": result["throughput_mb_per_sec"],
                        "memory": result["memory_peak_mb"],
                        "duration": result["duration_seconds"],
                    }
                )

        # Analyze trends for each operation
        for op_name, results in operations.items():
            if len(results) < 2:
                continue

            # Sort by timestamp
            results.sort(key=lambda x: x["timestamp"])

            # Calculate trends
            throughputs = [r["throughput"] for r in results]
            memories = [r["memory"] for r in results]

            throughput_trend = (
                (throughputs[-1] - throughputs[0]) / throughputs[0] if throughputs[0] > 0 else 0
            )
            memory_trend = (memories[-1] - memories[0]) / memories[0] if memories[0] > 0 else 0

            comparison["performance_trends"][op_name] = {
                "throughput_change_percent": throughput_trend * 100,
                "memory_change_percent": memory_trend * 100,
                "latest_throughput": throughputs[-1],
                "latest_memory": memories[-1],
            }

            # Detect regressions (>20% performance decrease or >50% memory increase)
            if throughput_trend < -0.20:
                comparison["regressions_detected"].append(
                    f"Performance regression in {op_name}: {throughput_trend*100:.1f}% throughput decrease"
                )

            if memory_trend > 0.50:
                comparison["regressions_detected"].append(
                    f"Memory regression in {op_name}: {memory_trend*100:.1f}% memory increase"
                )

        return comparison

    def _simulate_concurrent_downloads(self, urls: List[str], concurrency: int):
        """Simulate concurrent downloads for benchmarking"""
        # Mock simulation - replace with actual download logic
        time.sleep(0.5 * len(urls) / concurrency)  # Simulate download time

    def _simulate_data_processing(self, df: pd.DataFrame):
        """Simulate data processing operations"""
        # Perform various operations to simulate real processing
        _ = df.describe()
        _ = df.groupby(df.columns[0]).agg(["mean", "count"])
        _ = df.fillna(df.mean())

    def _create_test_dataframe(self, size_mb: float) -> pd.DataFrame:
        """Create test DataFrame of approximate size"""
        # Estimate rows needed (rough approximation)
        rows_needed = int(size_mb * 1024 * 1024 / 100)  # ~100 bytes per row

        np.random.seed(42)  # For reproducible results
        data = {
            "id": range(rows_needed),
            "value1": np.random.randn(rows_needed),
            "value2": np.random.randn(rows_needed),
            "category": np.random.choice(["A", "B", "C"], rows_needed),
            "date": pd.date_range("2020-01-01", periods=rows_needed, freq="D"),
        }

        return pd.DataFrame(data)

    def _create_test_csv_file(self, size_mb: float) -> Path:
        """Create test CSV file of approximate size"""
        df = self._create_test_dataframe(size_mb)
        temp_file = self.output_dir / f"test_data_{size_mb}mb.csv"
        df.to_csv(temp_file, index=False)
        return temp_file

    def _generate_recommendations(self, suite: BenchmarkSuite) -> List[str]:
        """Generate performance recommendations based on benchmark results"""
        recommendations = []
        summary = suite.get_summary_stats()

        # Performance recommendations
        if summary["average_throughput_mb_per_sec"] < 50:
            recommendations.append("Consider optimizing I/O operations or using faster storage")

        if summary["average_memory_peak_mb"] > 2000:
            recommendations.append(
                "High memory usage detected - consider streaming processing for large datasets"
            )

        if summary["success_rate"] < 0.95:
            recommendations.append("Low success rate - investigate error handling and retry logic")

        # System-specific recommendations
        if suite.system_info.memory_total_gb < 8:
            recommendations.append(
                "System has limited RAM - use chunked processing for large files"
            )

        if suite.system_info.cpu_count < 4:
            recommendations.append(
                "Limited CPU cores - consider optimizing single-threaded performance"
            )

        return recommendations


def run_quick_benchmark(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Run a quick performance benchmark for immediate feedback

    Args:
        logger: Optional logger

    Returns:
        Quick benchmark results
    """
    logger = logger or logging.getLogger(__name__)

    benchmark_suite = ENAHOBenchmarkSuite(logger)

    # Run subset of benchmarks
    results = []

    # Quick data processing test
    with benchmark_suite.profiler.profile_operation("quick_data_processing", 10.0) as profiler:
        test_df = benchmark_suite._create_test_dataframe(10.0)  # 10MB test
        benchmark_suite._simulate_data_processing(test_df)
        results.append(profiler.result)

    # Quick memory test
    with benchmark_suite.profiler.profile_operation("quick_memory_test", 5.0) as profiler:
        test_df = benchmark_suite._create_test_dataframe(5.0)  # 5MB test
        _ = test_df.copy()  # Simple memory operation
        results.append(profiler.result)

    # Calculate summary
    successful = [r for r in results if r.success]
    summary = {
        "total_tests": len(results),
        "successful_tests": len(successful),
        "average_throughput": (
            np.mean([r.throughput_mb_per_sec for r in successful]) if successful else 0
        ),
        "peak_memory": max([r.memory_peak_mb for r in successful]) if successful else 0,
        "system_info": asdict(SystemInfo.capture()),
    }

    return summary


__all__ = [
    "SystemInfo",
    "BenchmarkResult",
    "BenchmarkSuite",
    "PerformanceProfiler",
    "ENAHOBenchmarkSuite",
    "run_quick_benchmark",
]
