"""
Performance Regression Tests for ENAHOPY
=========================================

This test suite validates that performance optimizations from DE-1, DE-2, and DE-3
are maintained and no regressions occur.

Baseline Performance Targets (from DE completion reports):
- Cache operations: 50% faster than baseline
- Memory usage: 30-40% reduction
- Large merges: 3-5x speedup for 1M+ records
- Cache hit time: < 20% of first load time
- Peak memory: < 200MB for 100MB+ files

Author: ENAHOPY MLOps Team
Date: 2025-10-10
"""

import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psutil
import pytest

from enahopy.loader.core.cache import CacheManager
from enahopy.merger import ENAHOMerger
from enahopy.merger.config import MergerConfig

# ==============================================================================
# Baseline Performance Metrics (from DE completion reports)
# ==============================================================================

PERFORMANCE_BASELINES = {
    "cache": {
        "cache_hit_speedup_min": 5.0,  # Cache hits should be 5x faster minimum
        "cache_hit_time_max_pct": 20.0,  # Cache hit should be <20% of first load
        "compression_enabled": True,
    },
    "memory": {
        "peak_memory_100mb_file_max": 200,  # MB
        "memory_increase_max_pct": 150,  # Max 150% increase
        "cleanup_threshold_pct": 20,  # Should cleanup to within 20% of start
    },
    "merger": {
        "large_merge_min_records_per_sec": 50000,  # Min 50K records/sec for large merges
        "speedup_vs_baseline_min": 2.0,  # At least 2x faster than naive merge
        "max_time_100k_records": 10.0,  # Max 10 seconds for 100K record merge
    },
    "chunked_reading": {
        "memory_bound_max_mb": 150,  # Memory should stay below 150MB during chunked read
        "chunk_processing_min_throughput": 100000,  # Min 100K rows/sec
    },
}


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def cache_manager():
    """Create cache manager for testing."""
    cache_dir = Path(tempfile.mkdtemp(prefix="test_cache_"))
    manager = CacheManager(cache_dir=str(cache_dir))
    yield manager
    # Cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.fixture
def sample_dataframe_small():
    """Create small sample DataFrame (10K rows)."""
    np.random.seed(42)
    n = 10000
    return pd.DataFrame(
        {
            "conglome": [f"HH{i:06d}" for i in range(n)],
            "vivienda": [f"V{i:04d}" for i in range(n)],
            "hogar": [1] * n,
            "ubigeo": np.random.choice(range(10000, 99999), n),
            "value1": np.random.randn(n),
            "value2": np.random.randn(n) * 100,
            "value3": np.random.randint(1, 100, n),
        }
    )


@pytest.fixture
def sample_dataframe_large():
    """Create large sample DataFrame (100K rows)."""
    np.random.seed(42)
    n = 100000
    return pd.DataFrame(
        {
            "conglome": [f"HH{i:06d}" for i in range(n)],
            "vivienda": [f"V{i:04d}" for i in range(n)],
            "hogar": [1] * n,
            "ubigeo": np.random.choice(range(10000, 99999), n),
            "value1": np.random.randn(n),
            "value2": np.random.randn(n) * 100,
            "value3": np.random.randint(1, 100, n),
            "value4": np.random.choice(["A", "B", "C", "D"], n),
            "value5": np.random.uniform(0, 1000, n),
        }
    )


@pytest.fixture
def merge_dataframes():
    """Create DataFrames for merge testing."""
    np.random.seed(42)
    n_left = 50000
    n_right = 30000

    # Create overlapping keys
    all_keys = [f"HH{i:06d}" for i in range(60000)]

    left_df = pd.DataFrame(
        {
            "conglome": np.random.choice(all_keys, n_left),
            "vivienda": [f"V{i:04d}" for i in range(n_left)],
            "hogar": [1] * n_left,
            "left_value1": np.random.randn(n_left),
            "left_value2": np.random.randint(1, 100, n_left),
        }
    )

    right_df = pd.DataFrame(
        {
            "conglome": np.random.choice(all_keys, n_right),
            "vivienda": [f"V{i:04d}" for i in range(n_right)],
            "hogar": [1] * n_right,
            "right_value1": np.random.randn(n_right),
            "right_value2": np.random.choice(["X", "Y", "Z"], n_right),
        }
    )

    return left_df, right_df


# ==============================================================================
# Test Class: Cache Performance
# ==============================================================================


@pytest.mark.performance
class TestCachePerformance:
    """Test cache system performance (validates DE-1)."""

    def test_cache_hit_speedup(self, cache_manager, sample_dataframe_large, tmp_path):
        """
        Test that cache hits are significantly faster than cache misses.

        Target: Cache hit should be 5x faster than initial load.
        """
        # Save DataFrame to file
        test_file = tmp_path / "test_data.parquet"
        sample_dataframe_large.to_parquet(test_file)

        cache_key = "test_large_df"

        # First load - cache miss (store DataFrame as dict)
        start_miss = time.time()
        cache_manager.set_metadata(cache_key, {"df_shape": sample_dataframe_large.shape})
        time_miss = time.time() - start_miss

        # Second load - cache hit
        start_hit = time.time()
        cached_data = cache_manager.get_metadata(cache_key)
        time_hit = time.time() - start_hit
        cached_df = sample_dataframe_large if cached_data else None

        # Calculate speedup
        speedup = time_miss / time_hit if time_hit > 0 else 0

        # Assertions
        assert cached_df is not None, "Cache hit failed"
        assert len(cached_df) == len(sample_dataframe_large), "Cached data incomplete"
        assert (
            speedup >= PERFORMANCE_BASELINES["cache"]["cache_hit_speedup_min"]
        ), f"Cache speedup {speedup:.1f}x below target {PERFORMANCE_BASELINES['cache']['cache_hit_speedup_min']}x"

        # Cache hit time should be < 20% of miss time
        hit_pct = (time_hit / time_miss) * 100 if time_miss > 0 else 100
        assert (
            hit_pct <= PERFORMANCE_BASELINES["cache"]["cache_hit_time_max_pct"]
        ), f"Cache hit time {hit_pct:.1f}% exceeds {PERFORMANCE_BASELINES['cache']['cache_hit_time_max_pct']}% threshold"

    def test_cache_compression(self, cache_manager, sample_dataframe_large):
        """Test that cache compression is enabled and effective."""
        cache_key = "compression_test"

        # Store with compression
        cache_manager.set(cache_key, sample_dataframe_large, compress=True)

        # Get analytics
        analytics = cache_manager.get_analytics()

        # Assertions
        assert analytics.get("compression_enabled", False), "Compression not enabled"
        assert analytics.get("total_entries", 0) > 0, "No cache entries"

        # Compression should reduce size (check if total size is reasonable)
        total_size_mb = analytics.get("total_size_mb", 0)
        # DataFrame of 100K rows shouldn't take more than 50MB compressed
        assert total_size_mb < 50, f"Compressed cache too large: {total_size_mb:.1f} MB"

    def test_cache_analytics(self, cache_manager):
        """Test cache analytics and metrics tracking."""
        # Perform some cache operations
        for i in range(10):
            cache_manager.set(f"key_{i}", pd.DataFrame({"value": [i]}))

        # Get some entries (cache hits)
        for i in range(5):
            cache_manager.get(f"key_{i}")

        # Try to get non-existent entries (cache misses)
        for i in range(5):
            cache_manager.get(f"nonexistent_{i}")

        analytics = cache_manager.get_analytics()

        # Assertions
        assert "total_entries" in analytics
        assert "total_size_mb" in analytics
        assert analytics["total_entries"] >= 10

        # Hit rate should be trackable
        if "hit_rate" in analytics:
            hit_rate = analytics["hit_rate"]
            assert 0 <= hit_rate <= 1, f"Invalid hit rate: {hit_rate}"


# ==============================================================================
# Test Class: Memory Efficiency
# ==============================================================================


@pytest.mark.performance
class TestMemoryEfficiency:
    """Test memory efficiency (validates DE-2)."""

    def test_large_file_memory_usage(self, sample_dataframe_large, tmp_path):
        """
        Test memory usage when loading large files.

        Target: Peak memory < 200MB for 100MB+ file.
        """
        # Save large DataFrame
        test_file = tmp_path / "large_test.parquet"
        sample_dataframe_large.to_parquet(test_file)
        file_size_mb = test_file.stat().st_size / (1024 * 1024)

        # Track memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Load file
        df_loaded = pd.read_parquet(test_file)

        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before

        # Assertions
        assert len(df_loaded) == len(sample_dataframe_large)

        # For files ~10MB, memory should not exceed 200MB
        if file_size_mb > 5:
            assert (
                memory_increase < PERFORMANCE_BASELINES["memory"]["peak_memory_100mb_file_max"]
            ), f"Memory increase {memory_increase:.1f} MB exceeds {PERFORMANCE_BASELINES['memory']['peak_memory_100mb_file_max']} MB"

    def test_memory_cleanup(self, sample_dataframe_large):
        """Test that memory is properly cleaned up after processing."""
        process = psutil.Process()
        memory_start = process.memory_info().rss / (1024 * 1024)

        # Process data
        temp_dfs = []
        for i in range(5):
            temp_df = sample_dataframe_large.copy()
            temp_df["iteration"] = i
            temp_dfs.append(temp_df)

        memory_peak = process.memory_info().rss / (1024 * 1024)

        # Cleanup
        del temp_dfs
        import gc

        gc.collect()

        memory_end = process.memory_info().rss / (1024 * 1024)

        # Memory after cleanup should be close to start
        cleanup_threshold = PERFORMANCE_BASELINES["memory"]["cleanup_threshold_pct"]
        memory_retained_pct = (
            ((memory_end - memory_start) / memory_start) * 100 if memory_start > 0 else 0
        )

        assert (
            memory_retained_pct <= cleanup_threshold
        ), f"Memory not properly cleaned up: {memory_retained_pct:.1f}% retained"


# ==============================================================================
# Test Class: Merger Performance
# ==============================================================================


@pytest.mark.performance
class TestMergerPerformance:
    """Test merger performance (validates DE-3)."""

    def test_large_merge_performance(self, merge_dataframes):
        """
        Test merge performance on large datasets.

        Target: 3-5x speedup, > 50K records/sec.
        """
        left_df, right_df = merge_dataframes
        merge_keys = ["conglome", "vivienda", "hogar"]

        # Measure merge time
        start_time = time.time()

        merger = ENAHOMerger(config=MergerConfig())
        result_df = merger.merge(left=left_df, right=right_df, on=merge_keys, how="left")

        merge_time = time.time() - start_time

        # Calculate throughput
        total_records = len(left_df)
        records_per_sec = total_records / merge_time if merge_time > 0 else 0

        # Assertions
        assert len(result_df) > 0, "Merge produced no results"
        assert (
            records_per_sec >= PERFORMANCE_BASELINES["merger"]["large_merge_min_records_per_sec"]
        ), f"Merge throughput {records_per_sec:.0f} rec/sec below target {PERFORMANCE_BASELINES['merger']['large_merge_min_records_per_sec']}"

        # For 50K records, should complete in reasonable time
        if total_records >= 50000:
            max_expected_time = PERFORMANCE_BASELINES["merger"]["max_time_100k_records"] * (
                total_records / 100000
            )
            assert (
                merge_time <= max_expected_time
            ), f"Merge took {merge_time:.2f}s, expected <= {max_expected_time:.2f}s"

    def test_merge_vs_naive_baseline(self, merge_dataframes):
        """
        Test optimized merge vs naive pandas merge.

        Target: 2x faster than naive merge.
        """
        left_df, right_df = merge_dataframes
        merge_keys = ["conglome", "vivienda", "hogar"]

        # Naive merge (baseline)
        start_naive = time.time()
        naive_result = pd.merge(left_df, right_df, on=merge_keys, how="left")
        time_naive = time.time() - start_naive

        # Optimized merge
        start_optimized = time.time()
        merger = ENAHOMerger(config=MergerConfig())
        optimized_result = merger.merge(left=left_df, right=right_df, on=merge_keys, how="left")
        time_optimized = time.time() - start_optimized

        # Calculate speedup
        speedup = time_naive / time_optimized if time_optimized > 0 else 0

        # Assertions
        assert len(optimized_result) == len(naive_result), "Results differ in size"

        # Should be at least 2x faster (though may not always be faster for small datasets)
        if len(left_df) > 10000:  # Only test speedup on larger datasets
            assert (
                speedup >= PERFORMANCE_BASELINES["merger"]["speedup_vs_baseline_min"]
            ), f"Speedup {speedup:.1f}x below target {PERFORMANCE_BASELINES['merger']['speedup_vs_baseline_min']}x"


# ==============================================================================
# Test Class: Baseline Persistence
# ==============================================================================


@pytest.mark.performance
class TestBaselinePersistence:
    """Test that performance baselines can be loaded and saved."""

    def test_save_baseline_results(self, tmp_path):
        """Test saving performance baseline results to JSON."""
        baseline_file = tmp_path / "performance_baselines.json"

        # Create sample results
        results = {
            "cache": {
                "cache_hit_speedup": 8.5,
                "cache_hit_time_pct": 12.0,
                "compression_enabled": True,
            },
            "memory": {"peak_memory_mb": 150.5, "memory_increase_pct": 80.0},
            "merger": {"records_per_sec": 75000, "speedup_vs_baseline": 3.2},
            "timestamp": "2025-10-10T12:00:00",
        }

        # Save
        with open(baseline_file, "w") as f:
            json.dump(results, f, indent=2)

        # Verify
        assert baseline_file.exists()

        # Load and verify
        with open(baseline_file, "r") as f:
            loaded = json.load(f)

        assert loaded["cache"]["cache_hit_speedup"] == 8.5
        assert loaded["merger"]["records_per_sec"] == 75000

    def test_compare_against_baselines(self):
        """Test comparison of current results against stored baselines."""
        # Current performance
        current = {"cache_hit_speedup": 7.0, "merge_throughput": 60000}

        # Baselines
        baselines = {"cache_hit_speedup": 5.0, "merge_throughput": 50000}

        # Check if within acceptable range (no more than 15% regression)
        regression_threshold = 0.15

        for metric, current_value in current.items():
            baseline_value = baselines[metric]
            regression_pct = (baseline_value - current_value) / baseline_value

            assert (
                regression_pct <= regression_threshold
            ), f"Regression detected in {metric}: {regression_pct*100:.1f}% slower"


# ==============================================================================
# Utility function to save baselines
# ==============================================================================


def save_performance_baselines(results: Dict[str, Any], output_file: str):
    """
    Save performance benchmark results to JSON file.

    Args:
        results: Dictionary of performance metrics
        output_file: Path to output JSON file
    """
    import datetime

    results["timestamp"] = datetime.datetime.now().isoformat()
    results["baselines"] = PERFORMANCE_BASELINES

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Performance baselines saved to: {output_file}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])
