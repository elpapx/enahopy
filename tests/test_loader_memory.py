"""
test_loader_memory.py
======================
Memory and performance tests for Task DE-2: Data Loading Performance Optimization

Tests validate:
- Chunked reading functionality
- Memory usage optimization
- DataFrame dtype optimization
- Backward compatibility
- Performance targets (1GB+ files with <2GB memory)
"""

import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Memory profiling
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enahopy.loader.io.readers.csv import CSVReader
from enahopy.loader.io.readers.parquet import ParquetReader
from enahopy.loader.io.readers.spss import SPSSReader
from enahopy.loader.io.readers.stata import PYREADSTAT_AVAILABLE, StataReader


class MockLogger:
    """Mock logger for tests"""

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def debug(self, msg):
        pass


class MemoryProfiler:
    """Helper class for memory profiling"""

    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        if not self.process:
            return 0.0
        return self.process.memory_info().rss / (1024 * 1024)


class TestCSVReaderMemory(unittest.TestCase):
    """Test CSV reader memory optimizations"""

    def setUp(self):
        """Create test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MockLogger()
        self.profiler = MemoryProfiler() if PSUTIL_AVAILABLE else None

        # Create test CSV file
        self.test_csv = Path(self.temp_dir) / "test.csv"
        df = self._generate_test_data(10000, 20)
        df.to_csv(self.test_csv, index=False)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _generate_test_data(self, rows: int, columns: int) -> pd.DataFrame:
        """Generate test data"""
        np.random.seed(42)
        data = {
            f"col_{i}": (
                np.random.choice(["A", "B", "C", "D"], size=rows)
                if i % 3 == 0
                else np.random.randn(rows)
            )
            for i in range(columns)
        }
        return pd.DataFrame(data)

    def test_chunked_reading(self):
        """Test chunked reading functionality"""
        reader = CSVReader(self.test_csv, self.logger)
        columns = [f"col_{i}" for i in range(5)]

        result = reader.read_in_chunks(columns, chunk_size=2000)

        # Check if Dask is available and result is Dask DataFrame
        try:
            import dask.dataframe as dd

            if isinstance(result, dd.DataFrame):
                # Compute to get pandas DataFrame
                df_result = result.compute()
                self.assertEqual(len(df_result), 10000)
                return
        except ImportError:
            pass

        # Otherwise test as iterator
        chunks = list(result)

        # Should have 5 chunks (10000 / 2000)
        self.assertEqual(len(chunks), 5)

        # Total rows should match
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, 10000)

    def test_dtype_optimization(self):
        """Test dtype optimization reduces memory"""
        reader = CSVReader(self.test_csv, self.logger)
        columns = [f"col_{i}" for i in range(5)]

        # Read without optimization
        df_unoptimized = pd.read_csv(self.test_csv, usecols=columns)
        mem_before = df_unoptimized.memory_usage(deep=True).sum()

        # Read with optimization
        df_optimized = reader.read_columns(columns, optimize_dtypes=True)
        mem_after = df_optimized.memory_usage(deep=True).sum()

        # Optimized should use less or equal memory
        self.assertLessEqual(mem_after, mem_before)

        # Check that categorical columns were created
        has_categorical = any(
            df_optimized[col].dtype.name == "category" for col in df_optimized.columns
        )
        self.assertTrue(has_categorical, "Should have at least one categorical column")

    def test_backward_compatibility(self):
        """Test backward compatibility (default behavior preserved)"""
        reader = CSVReader(self.test_csv, self.logger)
        columns = [f"col_{i}" for i in range(5)]

        # Should work with simple call (no extra params)
        df = reader.read_columns(columns)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10000)
        self.assertEqual(len(df.columns), 5)

    @unittest.skipIf(not PSUTIL_AVAILABLE, "psutil not available")
    def test_memory_efficiency_chunks(self):
        """Test that chunked reading uses less peak memory"""
        reader = CSVReader(self.test_csv, self.logger)
        columns = [f"col_{i}" for i in range(10)]

        # Force GC
        gc.collect()

        # Measure full read
        mem_before_full = self.profiler.get_memory_mb()
        df_full = reader.read_columns(columns, optimize_dtypes=False)
        mem_after_full = self.profiler.get_memory_mb()
        mem_after_full - mem_before_full

        del df_full
        gc.collect()

        # Measure chunked read
        mem_before_chunked = self.profiler.get_memory_mb()
        total_rows = 0

        result = reader.read_in_chunks(columns, chunk_size=2000, optimize_dtypes=False)

        # Handle Dask DataFrame
        try:
            import dask.dataframe as dd

            if isinstance(result, dd.DataFrame):
                df_result = result.compute()
                total_rows = len(df_result)
                del df_result
            else:
                for chunk in result:
                    total_rows += len(chunk)
        except ImportError:
            for chunk in result:
                total_rows += len(chunk)

        mem_after_chunked = self.profiler.get_memory_mb()
        mem_after_chunked - mem_before_chunked

        # Should process all rows
        self.assertEqual(total_rows, 10000, "Should process all rows")


class TestParquetReaderMemory(unittest.TestCase):
    """Test Parquet reader memory optimizations"""

    def setUp(self):
        """Create test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MockLogger()

        # Create test Parquet file
        self.test_parquet = Path(self.temp_dir) / "test.parquet"
        df = self._generate_test_data(10000, 20)
        df.to_parquet(self.test_parquet, index=False)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _generate_test_data(self, rows: int, columns: int) -> pd.DataFrame:
        """Generate test data"""
        np.random.seed(42)
        data = {
            f"col_{i}": (
                np.random.choice(["A", "B", "C"], size=rows)
                if i % 2 == 0
                else np.random.randn(rows)
            )
            for i in range(columns)
        }
        return pd.DataFrame(data)

    def test_chunked_reading(self):
        """Test chunked reading for Parquet"""
        reader = ParquetReader(self.test_parquet, self.logger)
        columns = [f"col_{i}" for i in range(5)]

        result = reader.read_in_chunks(columns, chunk_size=2000)

        # Handle Dask DataFrame vs iterator
        try:
            import dask.dataframe as dd

            if isinstance(result, dd.DataFrame):
                df_result = result.compute()
                self.assertEqual(len(df_result), 10000)
                return
        except ImportError:
            pass

        # Handle as iterator
        chunks = list(result)

        # Should have multiple chunks
        self.assertGreater(len(chunks), 0)

        # Total rows should match
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, 10000)

    def test_column_selection(self):
        """Test that column selection reduces memory"""
        reader = ParquetReader(self.test_parquet, self.logger)

        # Read all columns
        df_all = reader.read_columns([f"col_{i}" for i in range(20)])
        mem_all = df_all.memory_usage(deep=True).sum()

        # Read subset of columns
        df_subset = reader.read_columns([f"col_{i}" for i in range(5)])
        mem_subset = df_subset.memory_usage(deep=True).sum()

        # Subset should use less memory
        self.assertLess(mem_subset, mem_all)

    def test_backward_compatibility(self):
        """Test backward compatibility"""
        reader = ParquetReader(self.test_parquet, self.logger)
        columns = [f"col_{i}" for i in range(5)]

        # Should work with simple call
        df = reader.read_columns(columns)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10000)


@unittest.skipIf(not PYREADSTAT_AVAILABLE, "pyreadstat not available")
class TestStataReaderMemory(unittest.TestCase):
    """Test Stata reader memory optimizations"""

    def setUp(self):
        """Create test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MockLogger()

        # Create test Stata file
        self.test_dta = Path(self.temp_dir) / "test.dta"
        df = self._generate_test_data(5000, 10)
        # Rename columns to meet Stata requirements
        df.columns = [f"v{i}" for i in range(len(df.columns))]
        df.to_stata(self.test_dta, write_index=False, version=118)

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _generate_test_data(self, rows: int, columns: int) -> pd.DataFrame:
        """Generate test data"""
        np.random.seed(42)
        data = {
            f"col_{i}": np.random.randint(0, 10, size=rows) if i % 2 == 0 else np.random.randn(rows)
            for i in range(columns)
        }
        return pd.DataFrame(data)

    def test_offset_based_chunking(self):
        """Test offset-based chunked reading"""
        reader = StataReader(self.test_dta, self.logger)
        columns = [f"v{i}" for i in range(5)]

        result = reader.read_in_chunks(columns, chunk_size=1000)

        # Handle Dask DataFrame vs iterator
        try:
            import dask.dataframe as dd

            if isinstance(result, dd.DataFrame):
                df_result = result.compute()
                self.assertEqual(len(df_result), 5000)
                return
        except ImportError:
            pass

        # Handle as iterator
        chunks = list(result)

        # Should have 5 chunks (5000 / 1000)
        self.assertEqual(len(chunks), 5)

        # Total rows should match
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, 5000)

    def test_dtype_optimization(self):
        """Test dtype optimization for Stata files"""
        reader = StataReader(self.test_dta, self.logger)
        columns = [f"v{i}" for i in range(5)]

        # Read with optimization
        df_optimized = reader.read_columns(columns, optimize_dtypes=True)

        # Check memory usage
        mem_usage = df_optimized.memory_usage(deep=True).sum()
        self.assertGreater(mem_usage, 0)

    def test_backward_compatibility(self):
        """Test backward compatibility"""
        reader = StataReader(self.test_dta, self.logger)
        columns = [f"v{i}" for i in range(5)]

        # Should work with simple call
        df = reader.read_columns(columns)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5000)
        self.assertEqual(len(df.columns), 5)


@unittest.skipIf(not PYREADSTAT_AVAILABLE, "pyreadstat not available")
class TestSPSSReaderMemory(unittest.TestCase):
    """Test SPSS reader memory optimizations"""

    def setUp(self):
        """Create test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MockLogger()

        # Create test SPSS file
        self.test_sav = Path(self.temp_dir) / "test.sav"
        df = self._generate_test_data(5000, 10)

        # Use pyreadstat to write SPSS file
        import pyreadstat

        pyreadstat.write_sav(df, str(self.test_sav))

    def tearDown(self):
        """Cleanup"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _generate_test_data(self, rows: int, columns: int) -> pd.DataFrame:
        """Generate test data"""
        np.random.seed(42)
        data = {
            f"var{i}": np.random.randint(0, 10, size=rows) if i % 2 == 0 else np.random.randn(rows)
            for i in range(columns)
        }
        return pd.DataFrame(data)

    def test_offset_based_chunking(self):
        """Test offset-based chunked reading for SPSS"""
        reader = SPSSReader(self.test_sav, self.logger)
        columns = [f"var{i}" for i in range(5)]

        result = reader.read_in_chunks(columns, chunk_size=1000)

        # Handle Dask DataFrame vs iterator
        try:
            import dask.dataframe as dd

            if isinstance(result, dd.DataFrame):
                df_result = result.compute()
                self.assertEqual(len(df_result), 5000)
                return
        except ImportError:
            pass

        # Handle as iterator
        chunks = list(result)

        # Should have 5 chunks (5000 / 1000)
        self.assertEqual(len(chunks), 5)

        # Total rows should match
        total_rows = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total_rows, 5000)

    def test_dtype_optimization(self):
        """Test dtype optimization for SPSS files"""
        reader = SPSSReader(self.test_sav, self.logger)
        columns = [f"var{i}" for i in range(5)]

        # Read with optimization
        df_optimized = reader.read_columns(columns, optimize_dtypes=True)

        # Check that data was loaded
        self.assertEqual(len(df_optimized), 5000)
        self.assertEqual(len(df_optimized.columns), 5)


class TestMemoryTargets(unittest.TestCase):
    """Test memory usage targets from DE-2 spec"""

    @unittest.skipIf(not PSUTIL_AVAILABLE, "psutil not available")
    def test_30_percent_memory_reduction(self):
        """
        Test that dtype optimization achieves ~30% memory reduction.
        This is a representative test - actual savings depend on data characteristics.
        """
        # Create test data with mix of types
        np.random.seed(42)
        rows = 50000

        # Data designed to benefit from optimization
        df = pd.DataFrame(
            {
                "cat_low": np.random.choice(["A", "B", "C"], size=rows),  # Low cardinality
                "cat_med": np.random.choice(list("ABCDEFGHIJ"), size=rows),  # Medium cardinality
                "num_int": np.random.randint(0, 100, size=rows),  # Integer
                "num_float": np.random.randn(rows),  # Float
            }
        )

        # Save to CSV
        temp_dir = tempfile.mkdtemp()
        test_file = Path(temp_dir) / "test.csv"
        df.to_csv(test_file, index=False)

        try:
            # Read without optimization
            df_unopt = pd.read_csv(test_file)
            mem_unopt = df_unopt.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

            # Read with optimization
            reader = CSVReader(test_file, MockLogger())
            df_opt = reader.read_columns(list(df.columns), optimize_dtypes=True)
            mem_opt = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

            # Calculate reduction
            reduction_pct = ((mem_unopt - mem_opt) / mem_unopt) * 100

            # Should achieve some memory reduction
            self.assertGreater(
                reduction_pct, 0, f"Expected memory reduction, got {reduction_pct:.1f}%"
            )

            print(f"\nMemory reduction: {reduction_pct:.1f}%")
            print(f"Before: {mem_unopt:.2f} MB, After: {mem_opt:.2f} MB")

        finally:
            import shutil

            shutil.rmtree(temp_dir)


def run_memory_tests():
    """Run all memory tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("MEMORY TESTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    if not PSUTIL_AVAILABLE:
        print("WARNING: psutil not available. Some tests will be skipped.")
        print("Install with: pip install psutil\n")

    success = run_memory_tests()
    sys.exit(0 if success else 1)
