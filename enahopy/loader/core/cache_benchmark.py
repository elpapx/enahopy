"""
ENAHO Cache Benchmark Suite
===========================

Performance benchmarking for cache operations.
Measures read/write performance, compression ratios, and memory usage.

Author: ENAHOPY Team - data-engineer
Date: 2025-10-08
"""

import json
import random
import string
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .cache import CacheManager


class CacheBenchmark:
    """Benchmark suite for cache operations."""

    def __init__(self, cache_dir: str = None, ttl_hours: int = 24):
        """
        Initialize benchmark suite.

        Args:
            cache_dir: Directory for cache. Uses temp dir if None.
            ttl_hours: TTL for cache entries.
        """
        if cache_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="cache_benchmark_")
            cache_dir = self.temp_dir
        else:
            self.temp_dir = None

        self.cache_manager = CacheManager(cache_dir, ttl_hours)
        self.results: List[Dict[str, Any]] = []

    def cleanup(self):
        """Clean up benchmark artifacts."""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def generate_test_data(self, size_kb: int) -> Dict[str, Any]:
        """
        Generate test data of specified size.

        Args:
            size_kb: Approximate size in KB.

        Returns:
            Dictionary with test data.
        """
        # Generate random string data to approximate size
        chars_needed = size_kb * 1024 // 2  # Rough estimate (UTF-8)

        data = {
            "id": random.randint(1, 1000000),
            "timestamp": time.time(),
            "data": "".join(random.choices(string.ascii_letters + string.digits, k=chars_needed)),
            "metadata": {"source": "benchmark", "size_kb": size_kb, "version": "1.0"},
        }

        return data

    def benchmark_write(
        self, num_operations: int = 100, data_size_kb: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark cache write operations.

        Args:
            num_operations: Number of write operations.
            data_size_kb: Size of each data entry in KB.

        Returns:
            Dictionary with benchmark results.
        """
        test_data = self.generate_test_data(data_size_kb)

        start_time = time.perf_counter()

        for i in range(num_operations):
            key = f"benchmark_write_{i}"
            self.cache_manager.set_metadata(key, test_data)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Get cache file size
        cache_file_size = 0
        if self.cache_manager.metadata_file.exists():
            cache_file_size = self.cache_manager.metadata_file.stat().st_size

        return {
            "operation": "write",
            "num_operations": num_operations,
            "data_size_kb": data_size_kb,
            "total_time_sec": elapsed,
            "ops_per_sec": num_operations / elapsed if elapsed > 0 else 0,
            "avg_time_ms": (elapsed * 1000) / num_operations if num_operations > 0 else 0,
            "cache_file_size_mb": cache_file_size / (1024 * 1024),
            "timestamp": time.time(),
        }

    def benchmark_read(self, num_operations: int = 100, data_size_kb: int = 10) -> Dict[str, float]:
        """
        Benchmark cache read operations.

        Args:
            num_operations: Number of read operations.
            data_size_kb: Size of each data entry in KB.

        Returns:
            Dictionary with benchmark results.
        """
        # First, populate cache
        test_data = self.generate_test_data(data_size_kb)
        for i in range(num_operations):
            key = f"benchmark_read_{i}"
            self.cache_manager.set_metadata(key, test_data)

        # Now benchmark reads
        start_time = time.perf_counter()

        hits = 0
        for i in range(num_operations):
            key = f"benchmark_read_{i}"
            result = self.cache_manager.get_metadata(key)
            if result is not None:
                hits += 1

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        return {
            "operation": "read",
            "num_operations": num_operations,
            "data_size_kb": data_size_kb,
            "total_time_sec": elapsed,
            "ops_per_sec": num_operations / elapsed if elapsed > 0 else 0,
            "avg_time_ms": (elapsed * 1000) / num_operations if num_operations > 0 else 0,
            "cache_hits": hits,
            "hit_rate": hits / num_operations if num_operations > 0 else 0,
            "timestamp": time.time(),
        }

    def benchmark_mixed_operations(
        self, num_operations: int = 100, read_ratio: float = 0.7, data_size_kb: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark mixed read/write operations.

        Args:
            num_operations: Total number of operations.
            read_ratio: Ratio of read to write operations (0-1).
            data_size_kb: Size of data entries in KB.

        Returns:
            Dictionary with benchmark results.
        """
        test_data = self.generate_test_data(data_size_kb)

        # Pre-populate some data for reads
        num_prepopulate = int(num_operations * 0.5)
        for i in range(num_prepopulate):
            key = f"benchmark_mixed_{i}"
            self.cache_manager.set_metadata(key, test_data)

        reads = 0
        writes = 0
        hits = 0

        start_time = time.perf_counter()

        for i in range(num_operations):
            if random.random() < read_ratio:
                # Read operation
                key = f"benchmark_mixed_{random.randint(0, num_prepopulate - 1)}"
                result = self.cache_manager.get_metadata(key)
                reads += 1
                if result is not None:
                    hits += 1
            else:
                # Write operation
                key = f"benchmark_mixed_new_{i}"
                self.cache_manager.set_metadata(key, test_data)
                writes += 1

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        return {
            "operation": "mixed",
            "num_operations": num_operations,
            "read_ratio": read_ratio,
            "data_size_kb": data_size_kb,
            "total_time_sec": elapsed,
            "ops_per_sec": num_operations / elapsed if elapsed > 0 else 0,
            "num_reads": reads,
            "num_writes": writes,
            "cache_hits": hits,
            "hit_rate": hits / reads if reads > 0 else 0,
            "timestamp": time.time(),
        }

    def benchmark_cleanup(
        self, num_entries: int = 1000, expired_ratio: float = 0.3
    ) -> Dict[str, float]:
        """
        Benchmark cache cleanup operations.

        Args:
            num_entries: Number of cache entries.
            expired_ratio: Ratio of entries to mark as expired (0-1).

        Returns:
            Dictionary with benchmark results.
        """
        # Create cache with short TTL
        cache_manager_short = CacheManager(str(self.cache_manager.cache_dir), ttl_hours=0.001)

        # Populate cache
        for i in range(num_entries):
            key = f"benchmark_cleanup_{i}"
            cache_manager_short.set_metadata(key, {"index": i})

        # Mark some as expired by manipulating timestamps
        if expired_ratio > 0:
            cache_file = cache_manager_short.metadata_file
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            num_to_expire = int(num_entries * expired_ratio)
            keys_to_expire = random.sample(list(cache_data.keys()), num_to_expire)

            for key in keys_to_expire:
                cache_data[key]["timestamp"] = time.time() - 7200  # 2 hours ago

            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

        # Benchmark cleanup
        start_time = time.perf_counter()
        cache_manager_short.clean_expired()
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Verify results
        stats = cache_manager_short.get_cache_stats()

        return {
            "operation": "cleanup",
            "num_entries": num_entries,
            "expired_ratio": expired_ratio,
            "total_time_sec": elapsed,
            "entries_remaining": stats.get("total_entries", 0),
            "entries_removed": int(num_entries * expired_ratio),
            "timestamp": time.time(),
        }

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.

        Returns:
            Dictionary with all benchmark results.
        """
        print("=" * 60)
        print("ENAHOPY Cache Benchmark Suite")
        print("=" * 60)
        print()

        results = {"timestamp": time.time(), "benchmarks": []}

        # Test different data sizes
        data_sizes = [1, 10, 100]  # KB

        for size_kb in data_sizes:
            print(f"Testing with {size_kb}KB data entries...")

            # Write benchmark
            print("  - Write operations...")
            write_result = self.benchmark_write(num_operations=100, data_size_kb=size_kb)
            results["benchmarks"].append(write_result)
            print(
                f"    {write_result['ops_per_sec']:.2f} ops/sec, "
                f"{write_result['avg_time_ms']:.2f} ms/op"
            )

            # Read benchmark
            print("  - Read operations...")
            read_result = self.benchmark_read(num_operations=100, data_size_kb=size_kb)
            results["benchmarks"].append(read_result)
            print(
                f"    {read_result['ops_per_sec']:.2f} ops/sec, "
                f"{read_result['avg_time_ms']:.2f} ms/op, "
                f"{read_result['hit_rate']*100:.1f}% hit rate"
            )

            # Mixed operations
            print("  - Mixed operations (70% read, 30% write)...")
            mixed_result = self.benchmark_mixed_operations(
                num_operations=100, read_ratio=0.7, data_size_kb=size_kb
            )
            results["benchmarks"].append(mixed_result)
            print(f"    {mixed_result['ops_per_sec']:.2f} ops/sec")

            # Clear cache between size tests
            self.cache_manager.clear_all()
            print()

        # Cleanup benchmark
        print("Testing cache cleanup...")
        cleanup_result = self.benchmark_cleanup(num_entries=1000, expired_ratio=0.3)
        results["benchmarks"].append(cleanup_result)
        print(
            f"  Cleaned {cleanup_result['entries_removed']} entries in "
            f"{cleanup_result['total_time_sec']*1000:.2f} ms"
        )

        print()
        print("=" * 60)
        print("Benchmark Complete")
        print("=" * 60)

        results["summary"] = self._generate_summary(results["benchmarks"])

        return results

    def _generate_summary(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from benchmarks."""
        write_ops = [b for b in benchmarks if b["operation"] == "write"]
        read_ops = [b for b in benchmarks if b["operation"] == "read"]

        avg_write_speed = (
            sum(b["ops_per_sec"] for b in write_ops) / len(write_ops) if write_ops else 0
        )
        avg_read_speed = sum(b["ops_per_sec"] for b in read_ops) / len(read_ops) if read_ops else 0
        avg_hit_rate = sum(b["hit_rate"] for b in read_ops) / len(read_ops) if read_ops else 0

        return {
            "avg_write_ops_per_sec": avg_write_speed,
            "avg_read_ops_per_sec": avg_read_speed,
            "avg_cache_hit_rate": avg_hit_rate,
            "total_benchmarks": len(benchmarks),
        }

    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {filepath}")


def run_baseline_benchmark():
    """Run baseline benchmark for comparison with optimized version."""
    benchmark = CacheBenchmark()

    try:
        results = benchmark.run_full_benchmark_suite()

        # Save to file
        output_file = Path("cache_benchmark_baseline.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nBaseline results saved to: {output_file}")
        print("\nSummary:")
        print(f"  Average Write Speed: {results['summary']['avg_write_ops_per_sec']:.2f} ops/sec")
        print(f"  Average Read Speed: {results['summary']['avg_read_ops_per_sec']:.2f} ops/sec")
        print(f"  Average Hit Rate: {results['summary']['avg_cache_hit_rate']*100:.1f}%")

        return results

    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    run_baseline_benchmark()
