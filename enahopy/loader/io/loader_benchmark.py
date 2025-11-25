"""
ENAHO Loader Performance Benchmark
===================================

Comprehensive benchmarking suite for data loading performance.
Measures memory usage, loading time, and throughput for different file formats.

Created for Task DE-2: Data Loading Performance Optimization
"""

import gc
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Memory profiling (optional dependency)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with 'pip install psutil' for memory profiling.")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    operation: str
    file_format: str
    file_size_mb: float
    rows: int
    columns: int
    elapsed_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_delta_mb: float
    throughput_mb_per_sec: float
    throughput_rows_per_sec: float
    use_chunks: bool = False
    chunk_size: Optional[int] = None
    optimization_level: str = "baseline"  # baseline, optimized
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "operation": self.operation,
            "file_format": self.file_format,
            "file_size_mb": round(self.file_size_mb, 2),
            "rows": self.rows,
            "columns": self.columns,
            "elapsed_time": round(self.elapsed_time, 3),
            "memory_before_mb": round(self.memory_before_mb, 2),
            "memory_after_mb": round(self.memory_after_mb, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "memory_delta_mb": round(self.memory_delta_mb, 2),
            "throughput_mb_per_sec": round(self.throughput_mb_per_sec, 2),
            "throughput_rows_per_sec": round(self.throughput_rows_per_sec, 2),
            "use_chunks": self.use_chunks,
            "chunk_size": self.chunk_size,
            "optimization_level": self.optimization_level,
            "metadata": self.metadata,
        }


class MemoryMonitor:
    """Monitor memory usage during operations"""

    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.peak_memory = 0.0

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        if not self.process:
            return 0.0
        return self.process.memory_info().rss / (1024 * 1024)

    def reset_peak(self):
        """Reset peak memory tracking"""
        self.peak_memory = self.get_memory_mb()

    def update_peak(self):
        """Update peak memory if current is higher"""
        current = self.get_memory_mb()
        if current > self.peak_memory:
            self.peak_memory = current

    def get_peak_mb(self) -> float:
        """Get peak memory usage in MB"""
        return self.peak_memory


class LoaderBenchmark:
    """Benchmark suite for ENAHO data loading"""

    def __init__(self, temp_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize benchmark suite

        Args:
            temp_dir: Directory for temporary test files
            verbose: Whether to print detailed logs
        """
        self.temp_dir = (
            Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "enahopy_bench"
        )
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.monitor = MemoryMonitor() if PSUTIL_AVAILABLE else None
        self.results: List[BenchmarkResult] = []

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for benchmarks"""
        logger = logging.getLogger("LoaderBenchmark")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _generate_test_data(self, rows: int, columns: int) -> pd.DataFrame:
        """
        Generate synthetic test data mimicking ENAHO structure

        Args:
            rows: Number of rows
            columns: Number of columns

        Returns:
            DataFrame with test data
        """
        np.random.seed(42)

        data = {
            # Identifiers (strings)
            "conglome": [f"{np.random.randint(1, 1000):06d}" for _ in range(rows)],
            "vivienda": [f"{np.random.randint(1, 100):02d}" for _ in range(rows)],
            "hogar": [f"{np.random.randint(1, 10)}" for _ in range(rows)],
            "codperso": [f"{np.random.randint(1, 20):02d}" for _ in range(rows)],
        }

        # Add numeric columns
        for i in range(columns - 4):
            col_name = f"var_{i:03d}"
            if i % 3 == 0:
                # Integer column
                data[col_name] = np.random.randint(0, 100, size=rows)
            elif i % 3 == 1:
                # Float column
                data[col_name] = np.random.randn(rows) * 100
            else:
                # Categorical-like column (small number of unique values)
                data[col_name] = np.random.choice(["A", "B", "C", "D", "E"], size=rows)

        return pd.DataFrame(data)

    def _create_test_file(self, file_path: Path, df: pd.DataFrame, format: str) -> float:
        """
        Create test file in specified format

        Args:
            file_path: Path for the file
            df: DataFrame to save
            format: File format (csv, parquet, dta)

        Returns:
            File size in MB
        """
        if format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "dta":
            # Stata has column name limitations
            df_copy = df.copy()
            df_copy.columns = [f"v{i}" for i in range(len(df.columns))]
            # Convert string columns to avoid Stata errors
            for col in df_copy.select_dtypes(include=["object"]).columns:
                df_copy[col] = df_copy[col].astype(str).str[:244]  # Stata limit
            df_copy.to_stata(file_path, write_index=False, version=118)
        else:
            raise ValueError(f"Unsupported format: {format}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        return file_size_mb

    def _measure_memory(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Measure memory usage of a function

        Args:
            func: Function to measure
            *args, **kwargs: Arguments for the function

        Returns:
            Tuple of (function result, memory stats dict)
        """
        if not self.monitor:
            # If psutil not available, return dummy metrics
            result = func(*args, **kwargs)
            return result, {"before_mb": 0.0, "after_mb": 0.0, "peak_mb": 0.0, "delta_mb": 0.0}

        # Force garbage collection
        gc.collect()

        # Measure before
        self.monitor.reset_peak()
        mem_before = self.monitor.get_memory_mb()

        # Execute function
        result = func(*args, **kwargs)

        # Measure after
        self.monitor.update_peak()
        mem_after = self.monitor.get_memory_mb()
        mem_peak = self.monitor.get_peak_mb()
        mem_delta = mem_after - mem_before

        return result, {
            "before_mb": mem_before,
            "after_mb": mem_after,
            "peak_mb": mem_peak,
            "delta_mb": mem_delta,
        }

    def benchmark_read_full(
        self,
        file_path: Path,
        file_format: str,
        rows: int,
        columns: int,
        optimization_level: str = "baseline",
    ) -> BenchmarkResult:
        """
        Benchmark full file read (no chunks)

        Args:
            file_path: Path to test file
            file_format: File format
            rows: Number of rows in file
            columns: Number of columns in file
            optimization_level: baseline or optimized

        Returns:
            BenchmarkResult with metrics
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        def read_func():
            if file_format == "csv":
                return pd.read_csv(file_path)
            elif file_format == "parquet":
                return pd.read_parquet(file_path)
            elif file_format == "dta":
                return pd.read_stata(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")

        # Measure execution
        start_time = time.time()
        df, mem_stats = self._measure_memory(read_func)
        elapsed_time = time.time() - start_time

        # Calculate throughput
        throughput_mb = file_size_mb / elapsed_time if elapsed_time > 0 else 0
        throughput_rows = rows / elapsed_time if elapsed_time > 0 else 0

        result = BenchmarkResult(
            operation="read_full",
            file_format=file_format,
            file_size_mb=file_size_mb,
            rows=rows,
            columns=columns,
            elapsed_time=elapsed_time,
            memory_before_mb=mem_stats["before_mb"],
            memory_after_mb=mem_stats["after_mb"],
            memory_peak_mb=mem_stats["peak_mb"],
            memory_delta_mb=mem_stats["delta_mb"],
            throughput_mb_per_sec=throughput_mb,
            throughput_rows_per_sec=throughput_rows,
            use_chunks=False,
            optimization_level=optimization_level,
        )

        self.results.append(result)

        if self.verbose:
            self.logger.info(
                f"Read {file_format}: {elapsed_time:.2f}s, "
                f"{mem_stats['delta_mb']:.1f}MB memory, "
                f"{throughput_mb:.1f} MB/s"
            )

        # Clean up
        del df
        gc.collect()

        return result

    def benchmark_read_chunks(
        self,
        file_path: Path,
        file_format: str,
        rows: int,
        columns: int,
        chunk_size: int = 10000,
        optimization_level: str = "baseline",
    ) -> BenchmarkResult:
        """
        Benchmark chunked file read

        Args:
            file_path: Path to test file
            file_format: File format
            rows: Number of rows in file
            columns: Number of columns in file
            chunk_size: Chunk size for reading
            optimization_level: baseline or optimized

        Returns:
            BenchmarkResult with metrics
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        def read_chunks_func():
            total_rows = 0
            if file_format == "csv":
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    total_rows += len(chunk)
                    # Simulate processing
                    _ = chunk.shape
            elif file_format == "parquet":
                # Parquet doesn't have native chunking in pandas, read full
                df = pd.read_parquet(file_path)
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i : i + chunk_size]
                    total_rows += len(chunk)
            elif file_format == "dta":
                # Stata doesn't have native chunking, read full then iterate
                df = pd.read_stata(file_path)
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i : i + chunk_size]
                    total_rows += len(chunk)
            return total_rows

        # Measure execution
        start_time = time.time()
        total_rows, mem_stats = self._measure_memory(read_chunks_func)
        elapsed_time = time.time() - start_time

        # Calculate throughput
        throughput_mb = file_size_mb / elapsed_time if elapsed_time > 0 else 0
        throughput_rows = total_rows / elapsed_time if elapsed_time > 0 else 0

        result = BenchmarkResult(
            operation="read_chunks",
            file_format=file_format,
            file_size_mb=file_size_mb,
            rows=total_rows,
            columns=columns,
            elapsed_time=elapsed_time,
            memory_before_mb=mem_stats["before_mb"],
            memory_after_mb=mem_stats["after_mb"],
            memory_peak_mb=mem_stats["peak_mb"],
            memory_delta_mb=mem_stats["delta_mb"],
            throughput_mb_per_sec=throughput_mb,
            throughput_rows_per_sec=throughput_rows,
            use_chunks=True,
            chunk_size=chunk_size,
            optimization_level=optimization_level,
        )

        self.results.append(result)

        if self.verbose:
            self.logger.info(
                f"Read chunks {file_format}: {elapsed_time:.2f}s, "
                f"{mem_stats['delta_mb']:.1f}MB memory, "
                f"{throughput_rows:.0f} rows/s"
            )

        gc.collect()

        return result

    def run_comprehensive_suite(
        self, test_configs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite

        Args:
            test_configs: List of test configurations. Each config is a dict with:
                - rows: number of rows
                - columns: number of columns
                - formats: list of formats to test
                - chunk_sizes: list of chunk sizes to test

        Returns:
            Summary statistics dictionary
        """
        if test_configs is None:
            # Default test configurations
            test_configs = [
                {
                    "name": "Small (10MB)",
                    "rows": 10000,
                    "columns": 50,
                    "formats": ["csv", "parquet"],
                    "chunk_sizes": [5000, 10000],
                },
                {
                    "name": "Medium (100MB)",
                    "rows": 100000,
                    "columns": 50,
                    "formats": ["csv", "parquet"],
                    "chunk_sizes": [10000, 50000],
                },
                {
                    "name": "Large (500MB+)",
                    "rows": 500000,
                    "columns": 50,
                    "formats": ["csv", "parquet"],
                    "chunk_sizes": [50000, 100000],
                },
            ]

        self.logger.info("=" * 70)
        self.logger.info("ENAHO Loader Benchmark Suite - BASELINE")
        self.logger.info("=" * 70)

        for config in test_configs:
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Test: {config['name']}")
            self.logger.info(f"Rows: {config['rows']:,}, Columns: {config['columns']}")
            self.logger.info(f"{'='*70}")

            # Generate test data
            self.logger.info("Generating test data...")
            df = self._generate_test_data(config["rows"], config["columns"])

            for fmt in config["formats"]:
                # Create test file
                file_path = self.temp_dir / f"test_{config['rows']}_{fmt}.{fmt}"
                if fmt == "dta":
                    file_path = file_path.with_suffix(".dta")

                self.logger.info(f"\nCreating {fmt.upper()} test file...")
                file_size_mb = self._create_test_file(file_path, df, fmt)
                self.logger.info(f"File size: {file_size_mb:.2f} MB")

                # Benchmark full read
                self.logger.info(f"Benchmarking full read ({fmt})...")
                self.benchmark_read_full(file_path, fmt, config["rows"], config["columns"])

                # Benchmark chunked reads
                for chunk_size in config["chunk_sizes"]:
                    self.logger.info(
                        f"Benchmarking chunked read ({fmt}, chunk_size={chunk_size})..."
                    )
                    self.benchmark_read_chunks(
                        file_path, fmt, config["rows"], config["columns"], chunk_size
                    )

                # Clean up file
                file_path.unlink()

            # Clean up test data
            del df
            gc.collect()

        # Generate summary
        summary = self._generate_summary()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total tests: {summary['total_tests']}")
        self.logger.info(f"Average memory delta: {summary['avg_memory_delta_mb']:.2f} MB")
        self.logger.info(f"Peak memory usage: {summary['peak_memory_mb']:.2f} MB")
        self.logger.info(f"Average throughput: {summary['avg_throughput_mb_s']:.2f} MB/s")

        return summary

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        if not self.results:
            return {}

        memory_deltas = [r.memory_delta_mb for r in self.results]
        peak_memories = [r.memory_peak_mb for r in self.results]
        throughputs = [r.throughput_mb_per_sec for r in self.results]

        return {
            "total_tests": len(self.results),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta_mb": max(memory_deltas),
            "peak_memory_mb": max(peak_memories),
            "avg_throughput_mb_s": sum(throughputs) / len(throughputs),
            "max_throughput_mb_s": max(throughputs),
            "results": [r.to_dict() for r in self.results],
        }

    def save_results(self, output_path: str):
        """
        Save benchmark results to JSON file

        Args:
            output_path: Path for output JSON file
        """
        summary = self._generate_summary()

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Results saved to: {output_path}")

    def cleanup(self):
        """Clean up temporary files"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.logger.info("Cleanup complete")


def run_baseline_benchmark(output_file: str = "loader_benchmark_baseline.json") -> Dict[str, Any]:
    """
    Run baseline benchmark and save results

    Args:
        output_file: Path for output JSON file

    Returns:
        Summary statistics dictionary
    """
    benchmark = LoaderBenchmark(verbose=True)

    try:
        summary = benchmark.run_comprehensive_suite()
        benchmark.save_results(output_file)
        return summary
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    if not PSUTIL_AVAILABLE:
        print("ERROR: psutil is required for benchmarking")
        print("Install with: pip install psutil")
        sys.exit(1)

    print("Running ENAHO Loader Baseline Benchmark...")
    print("This may take several minutes...\n")

    results = run_baseline_benchmark()

    print("\n" + "=" * 70)
    print("BASELINE BENCHMARK COMPLETE")
    print("=" * 70)
    print("Results saved to: loader_benchmark_baseline.json")
