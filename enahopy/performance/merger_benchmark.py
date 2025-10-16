"""
ENAHO Merger Performance Benchmark Suite
========================================

Comprehensive performance benchmarking for merger operations:
- Module merging (household + person level)
- Geographic merging
- Large dataset handling
- Memory efficiency

Author: ENAHOPY Team - data-engineer (Task DE-3)
Date: 2025-10-09
Version: 1.0.0
"""

import gc
import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..merger import ENAHOGeoMerger
from ..merger.config import (
    GeoMergeConfiguration,
    ModuleMergeConfig,
    ModuleMergeLevel,
    ModuleMergeStrategy,
)
from ..merger.modules.merger import ENAHOModuleMerger


@dataclass
class MergerBenchmarkResult:
    """Result from a single merger benchmark operation"""

    operation: str
    dataset_size: int  # Number of records
    duration_seconds: float
    records_per_second: float
    memory_peak_mb: float
    memory_delta_mb: float
    cardinality_change: float  # Ratio of output/input rows
    merge_quality_score: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MergerBenchmarkSuite:
    """Complete merger benchmark suite results"""

    suite_name: str
    timestamp: datetime
    results: List[MergerBenchmarkResult]
    total_duration: float
    system_info: Dict[str, Any]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if successful:
            avg_throughput = np.mean([r.records_per_second for r in successful])
            avg_memory = np.mean([r.memory_peak_mb for r in successful])
            avg_quality = np.mean([r.merge_quality_score for r in successful])
        else:
            avg_throughput = avg_memory = avg_quality = 0.0

        return {
            "total_operations": len(self.results),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0.0,
            "average_throughput_recs_per_sec": avg_throughput,
            "average_memory_peak_mb": avg_memory,
            "average_merge_quality": avg_quality,
            "total_duration": self.total_duration,
        }


class MergerPerformanceProfiler:
    """Performance profiler for merger operations"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.baseline_memory_mb = 0.0

    def __enter__(self):
        """Start profiling"""
        gc.collect()  # Clean memory before profiling
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            self.baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling"""
        self.end_time = time.perf_counter()
        gc.collect()

        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_peak_mb = final_memory_mb
            self.memory_delta_mb = final_memory_mb - self.baseline_memory_mb
        else:
            self.memory_peak_mb = 0.0
            self.memory_delta_mb = 0.0

        self.duration_seconds = self.end_time - self.start_time
        self.success = exc_type is None
        self.error_message = str(exc_val) if exc_val else None


class MergerBenchmarkSuiteRunner:
    """Comprehensive benchmark suite for merger operations"""

    def __init__(self, logger: Optional[logging.Logger] = None, output_dir: Optional[Path] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = output_dir or Path("./benchmarks")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []

    def generate_test_modules(
        self, num_records: int, level: str = "hogar"
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic ENAHO modules for testing

        Args:
            num_records: Number of records to generate
            level: "hogar" or "persona"

        Returns:
            Dictionary of test modules
        """
        np.random.seed(42)  # For reproducibility

        # Base keys (household level)
        conglome = [f"{i:06d}" for i in range(1, num_records + 1)]
        vivienda = np.random.randint(1, 10, num_records).astype(str).str.zfill(3)
        hogar = np.random.randint(1, 3, num_records).astype(str)

        # Module 01 (Household characteristics)
        mod_01 = pd.DataFrame(
            {
                "conglome": conglome,
                "vivienda": vivienda,
                "hogar": hogar,
                "result01": np.random.choice([100, 200, 300, 400], num_records),
                "factor07": np.random.uniform(0.8, 1.5, num_records),
                "mieperho": np.random.randint(1, 10, num_records),
                "percepho": np.random.randint(1, 15, num_records),
            }
        )

        # Module 34 (Household income)
        mod_34 = pd.DataFrame(
            {
                "conglome": conglome,
                "vivienda": vivienda,
                "hogar": hogar,
                "inghog1d": np.random.uniform(500, 10000, num_records),
                "inghog2d": np.random.uniform(400, 8000, num_records),
                "gashog1d": np.random.uniform(300, 7000, num_records),
                "gashog2d": np.random.uniform(200, 6000, num_records),
            }
        )

        modules = {"01": mod_01, "34": mod_34}

        if level == "persona":
            # Add person-level module
            # Expand to multiple persons per household
            person_records = []
            for idx in range(num_records):
                num_persons = np.random.randint(1, 6)  # 1-5 persons per household
                for p in range(1, num_persons + 1):
                    person_records.append(
                        {
                            "conglome": conglome[idx],
                            "vivienda": vivienda[idx],
                            "hogar": hogar[idx],
                            "codperso": f"{p:02d}",
                            "p203": np.random.choice([1, 2]),  # Sex
                            "p208a": np.random.randint(0, 100),  # Age
                            "p301a": np.random.choice([1, 2, 3, 4, 5]),  # Education
                        }
                    )

            mod_02 = pd.DataFrame(person_records)
            modules["02"] = mod_02

        return modules

    def generate_geographic_data(self, num_ubigeos: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic geographic data

        Args:
            num_ubigeos: Number of unique UBIGEOs

        Returns:
            DataFrame with geographic data
        """
        np.random.seed(42)

        # Generate valid-looking UBIGEOs
        departments = [f"{i:02d}" for i in range(1, 26)]  # 1-25
        ubigeos = []

        for _ in range(num_ubigeos):
            dept = np.random.choice(departments)
            prov = f"{np.random.randint(1, 20):02d}"
            dist = f"{np.random.randint(1, 50):02d}"
            ubigeos.append(f"{dept}{prov}{dist}")

        df_geo = pd.DataFrame(
            {
                "ubigeo": ubigeos,
                "departamento": [f"Departamento_{u[:2]}" for u in ubigeos],
                "provincia": [f"Provincia_{u[:4]}" for u in ubigeos],
                "distrito": [f"Distrito_{u}" for u in ubigeos],
                "region": np.random.choice(["Costa", "Sierra", "Selva"], num_ubigeos),
                "altitude": np.random.randint(0, 5000, num_ubigeos),
                "latitud": np.random.uniform(-18, -1, num_ubigeos),
                "longitud": np.random.uniform(-81, -68, num_ubigeos),
            }
        )

        return df_geo

    def benchmark_module_merge_hogar(
        self, dataset_sizes: List[int] = [1000, 10000, 100000, 1000000]
    ) -> List[MergerBenchmarkResult]:
        """
        Benchmark module merging at household level

        Args:
            dataset_sizes: Different dataset sizes to test

        Returns:
            List of benchmark results
        """
        self.logger.info("=" * 60)
        self.logger.info("Benchmarking Module Merge (Household Level)")
        self.logger.info("=" * 60)

        results = []

        for size in dataset_sizes:
            self.logger.info(f"\nTesting with {size:,} records...")

            # Generate test data
            modules = self.generate_test_modules(size, level="hogar")

            # Configure merger
            config = ModuleMergeConfig(
                merge_level=ModuleMergeLevel.HOGAR,
                merge_strategy=ModuleMergeStrategy.COALESCE,
                use_validation_cache=True,
                validate_cardinality=True,
            )

            logger_instance = logging.getLogger(f"test_merge_{size}")
            merger = ENAHOModuleMerger(config, logger_instance)

            # Run benchmark
            with MergerPerformanceProfiler(self.logger) as profiler:
                try:
                    result = merger.merge_modules(modules["01"], modules["34"], "01", "34", config)

                    # Extract metrics
                    cardinality_change = len(result.merged_df) / len(modules["01"])
                    quality_score = result.quality_score

                    benchmark_result = MergerBenchmarkResult(
                        operation=f"module_merge_hogar_{size}",
                        dataset_size=size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=(
                            size / profiler.duration_seconds if profiler.duration_seconds > 0 else 0
                        ),
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=cardinality_change,
                        merge_quality_score=quality_score,
                        success=profiler.success,
                        error_message=profiler.error_message,
                        metadata={
                            "rows_input": len(modules["01"]),
                            "rows_output": len(result.merged_df),
                            "conflicts_resolved": result.conflicts_resolved,
                        },
                    )

                    results.append(benchmark_result)

                    self.logger.info(f"  ✅ Duration: {profiler.duration_seconds:.2f}s")
                    self.logger.info(
                        f"  ✅ Throughput: {benchmark_result.records_per_second:,.0f} rec/s"
                    )
                    self.logger.info(
                        f"  ✅ Memory: {profiler.memory_peak_mb:.1f}MB (Δ{profiler.memory_delta_mb:.1f}MB)"
                    )
                    self.logger.info(f"  ✅ Quality: {quality_score:.1f}%")

                except Exception as e:
                    self.logger.error(f"  ❌ Failed: {str(e)}")
                    benchmark_result = MergerBenchmarkResult(
                        operation=f"module_merge_hogar_{size}",
                        dataset_size=size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=0,
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=0,
                        merge_quality_score=0,
                        success=False,
                        error_message=str(e),
                    )
                    results.append(benchmark_result)

            # Clean up memory
            del modules
            gc.collect()

        return results

    def benchmark_module_merge_persona(
        self, dataset_sizes: List[int] = [1000, 10000, 100000]
    ) -> List[MergerBenchmarkResult]:
        """
        Benchmark module merging at person level

        Args:
            dataset_sizes: Different dataset sizes to test (households, will expand to persons)

        Returns:
            List of benchmark results
        """
        self.logger.info("=" * 60)
        self.logger.info("Benchmarking Module Merge (Person Level)")
        self.logger.info("=" * 60)

        results = []

        for size in dataset_sizes:
            self.logger.info(f"\nTesting with ~{size:,} households...")

            # Generate test data (person level)
            modules = self.generate_test_modules(size, level="persona")
            actual_size = len(modules["02"])  # Actual person records

            # Configure merger
            config = ModuleMergeConfig(
                merge_level=ModuleMergeLevel.PERSONA,
                merge_strategy=ModuleMergeStrategy.COALESCE,
                use_validation_cache=True,
            )

            logger_instance = logging.getLogger(f"test_merge_persona_{size}")
            merger = ENAHOModuleMerger(config, logger_instance)

            # Run benchmark
            with MergerPerformanceProfiler(self.logger) as profiler:
                try:
                    # First merge household data
                    result_hogar = merger.merge_modules(
                        modules["01"],
                        modules["34"],
                        "01",
                        "34",
                        ModuleMergeConfig(merge_level=ModuleMergeLevel.HOGAR),
                    )

                    # Then merge with person data
                    result = merger.merge_modules(
                        result_hogar.merged_df, modules["02"], "01+34", "02", config
                    )

                    cardinality_change = len(result.merged_df) / actual_size
                    quality_score = result.quality_score

                    benchmark_result = MergerBenchmarkResult(
                        operation=f"module_merge_persona_{size}",
                        dataset_size=actual_size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=(
                            actual_size / profiler.duration_seconds
                            if profiler.duration_seconds > 0
                            else 0
                        ),
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=cardinality_change,
                        merge_quality_score=quality_score,
                        success=profiler.success,
                        error_message=profiler.error_message,
                        metadata={
                            "households": size,
                            "persons": actual_size,
                            "rows_output": len(result.merged_df),
                        },
                    )

                    results.append(benchmark_result)

                    self.logger.info(f"  ✅ Duration: {profiler.duration_seconds:.2f}s")
                    self.logger.info(
                        f"  ✅ Throughput: {benchmark_result.records_per_second:,.0f} rec/s"
                    )
                    self.logger.info(f"  ✅ Memory: {profiler.memory_peak_mb:.1f}MB")

                except Exception as e:
                    self.logger.error(f"  ❌ Failed: {str(e)}")
                    benchmark_result = MergerBenchmarkResult(
                        operation=f"module_merge_persona_{size}",
                        dataset_size=actual_size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=0,
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=0,
                        merge_quality_score=0,
                        success=False,
                        error_message=str(e),
                    )
                    results.append(benchmark_result)

            del modules
            gc.collect()

        return results

    def benchmark_geographic_merge(
        self, dataset_sizes: List[int] = [1000, 10000, 100000, 1000000]
    ) -> List[MergerBenchmarkResult]:
        """
        Benchmark geographic merging

        Args:
            dataset_sizes: Different dataset sizes to test

        Returns:
            List of benchmark results
        """
        self.logger.info("=" * 60)
        self.logger.info("Benchmarking Geographic Merge")
        self.logger.info("=" * 60)

        results = []

        # Generate geographic data once
        df_geo = self.generate_geographic_data(num_ubigeos=5000)
        ubigeo_list = df_geo["ubigeo"].tolist()

        for size in dataset_sizes:
            self.logger.info(f"\nTesting with {size:,} records...")

            # Generate test data with ubigeos
            np.random.seed(42)
            df_principal = pd.DataFrame(
                {
                    "ubigeo": np.random.choice(ubigeo_list, size),
                    "value1": np.random.randn(size),
                    "value2": np.random.randn(size),
                    "value3": np.random.randint(0, 100, size),
                }
            )

            # Configure merger
            config = GeoMergeConfiguration(
                columna_union="ubigeo",
                validar_formato_ubigeo=False,  # Skip validation for speed
                optimizar_memoria=True,
                chunk_size=50000,
            )

            merger = ENAHOGeoMerger(geo_config=config, verbose=False)

            # Run benchmark
            with MergerPerformanceProfiler(self.logger) as profiler:
                try:
                    # Mock the pattern detector to avoid slow operations
                    from unittest.mock import patch

                    with patch.object(
                        merger.pattern_detector, "detectar_columnas_geograficas"
                    ) as mock_detect:
                        mock_detect.return_value = {
                            "departamento": "departamento",
                            "provincia": "provincia",
                            "distrito": "distrito",
                            "region": "region",
                        }

                        result_df, validation = merger.merge_geographic_data(
                            df_principal,
                            df_geo,
                            validate_before_merge=False,
                        )

                    cardinality_change = len(result_df) / len(df_principal)
                    coverage = validation.coverage_percentage if validation else 100.0

                    benchmark_result = MergerBenchmarkResult(
                        operation=f"geographic_merge_{size}",
                        dataset_size=size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=(
                            size / profiler.duration_seconds if profiler.duration_seconds > 0 else 0
                        ),
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=cardinality_change,
                        merge_quality_score=coverage,
                        success=profiler.success,
                        error_message=profiler.error_message,
                        metadata={
                            "rows_input": len(df_principal),
                            "rows_output": len(result_df),
                            "coverage_percentage": coverage,
                        },
                    )

                    results.append(benchmark_result)

                    self.logger.info(f"  ✅ Duration: {profiler.duration_seconds:.2f}s")
                    self.logger.info(
                        f"  ✅ Throughput: {benchmark_result.records_per_second:,.0f} rec/s"
                    )
                    self.logger.info(f"  ✅ Memory: {profiler.memory_peak_mb:.1f}MB")
                    self.logger.info(f"  ✅ Coverage: {coverage:.1f}%")

                except Exception as e:
                    self.logger.error(f"  ❌ Failed: {str(e)}")
                    benchmark_result = MergerBenchmarkResult(
                        operation=f"geographic_merge_{size}",
                        dataset_size=size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=0,
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=0,
                        merge_quality_score=0,
                        success=False,
                        error_message=str(e),
                    )
                    results.append(benchmark_result)

            del df_principal, result_df
            gc.collect()

        return results

    def benchmark_batch_processing(
        self, total_size: int = 1000000, batch_sizes: List[int] = [10000, 50000, 100000]
    ) -> List[MergerBenchmarkResult]:
        """
        Benchmark batch processing with different batch sizes

        Args:
            total_size: Total dataset size
            batch_sizes: Different batch sizes to test

        Returns:
            List of benchmark results
        """
        self.logger.info("=" * 60)
        self.logger.info("Benchmarking Batch Processing")
        self.logger.info("=" * 60)

        results = []

        # Generate test data once
        modules = self.generate_test_modules(total_size, level="hogar")

        for batch_size in batch_sizes:
            self.logger.info(f"\nTesting with batch size {batch_size:,}...")

            config = ModuleMergeConfig(
                merge_level=ModuleMergeLevel.HOGAR,
                merge_strategy=ModuleMergeStrategy.COALESCE,
                chunk_processing=True,
                chunk_size=batch_size,
            )

            logger_instance = logging.getLogger(f"test_batch_{batch_size}")
            merger = ENAHOModuleMerger(config, logger_instance)

            with MergerPerformanceProfiler(self.logger) as profiler:
                try:
                    result = merger.merge_modules(modules["01"], modules["34"], "01", "34", config)

                    benchmark_result = MergerBenchmarkResult(
                        operation=f"batch_merge_{batch_size}",
                        dataset_size=total_size,
                        duration_seconds=profiler.duration_seconds,
                        records_per_second=(
                            total_size / profiler.duration_seconds
                            if profiler.duration_seconds > 0
                            else 0
                        ),
                        memory_peak_mb=profiler.memory_peak_mb,
                        memory_delta_mb=profiler.memory_delta_mb,
                        cardinality_change=len(result.merged_df) / total_size,
                        merge_quality_score=result.quality_score,
                        success=profiler.success,
                        error_message=profiler.error_message,
                        metadata={"batch_size": batch_size},
                    )

                    results.append(benchmark_result)

                    self.logger.info(f"  ✅ Duration: {profiler.duration_seconds:.2f}s")
                    self.logger.info(
                        f"  ✅ Throughput: {benchmark_result.records_per_second:,.0f} rec/s"
                    )
                    self.logger.info(f"  ✅ Memory: {profiler.memory_peak_mb:.1f}MB")

                except Exception as e:
                    self.logger.error(f"  ❌ Failed: {str(e)}")
                    results.append(
                        MergerBenchmarkResult(
                            operation=f"batch_merge_{batch_size}",
                            dataset_size=total_size,
                            duration_seconds=profiler.duration_seconds,
                            records_per_second=0,
                            memory_peak_mb=profiler.memory_peak_mb,
                            memory_delta_mb=profiler.memory_delta_mb,
                            cardinality_change=0,
                            merge_quality_score=0,
                            success=False,
                            error_message=str(e),
                        )
                    )

            gc.collect()

        return results

    def run_comprehensive_benchmark(self) -> MergerBenchmarkSuite:
        """
        Run comprehensive benchmark suite

        Returns:
            Complete benchmark suite results
        """
        self.logger.info("=" * 70)
        self.logger.info(" " * 15 + "ENAHO MERGER BENCHMARK SUITE")
        self.logger.info("=" * 70)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)

        start_time = time.time()
        all_results = []

        # Capture system info
        system_info = self._capture_system_info()

        # 1. Module merge (household level)
        try:
            hogar_results = self.benchmark_module_merge_hogar(
                dataset_sizes=[1000, 10000, 100000, 1000000]
            )
            all_results.extend(hogar_results)
        except Exception as e:
            self.logger.error(f"Household merge benchmark failed: {e}")

        # 2. Module merge (person level)
        try:
            persona_results = self.benchmark_module_merge_persona(
                dataset_sizes=[1000, 10000, 100000]
            )
            all_results.extend(persona_results)
        except Exception as e:
            self.logger.error(f"Person merge benchmark failed: {e}")

        # 3. Geographic merge
        try:
            geo_results = self.benchmark_geographic_merge(
                dataset_sizes=[1000, 10000, 100000, 1000000]
            )
            all_results.extend(geo_results)
        except Exception as e:
            self.logger.error(f"Geographic merge benchmark failed: {e}")

        # 4. Batch processing
        try:
            batch_results = self.benchmark_batch_processing(
                total_size=1000000, batch_sizes=[10000, 50000, 100000]
            )
            all_results.extend(batch_results)
        except Exception as e:
            self.logger.error(f"Batch processing benchmark failed: {e}")

        total_duration = time.time() - start_time

        benchmark_suite = MergerBenchmarkSuite(
            suite_name="ENAHO_Merger_Comprehensive",
            timestamp=datetime.now(),
            results=all_results,
            total_duration=total_duration,
            system_info=system_info,
        )

        # Save results
        self.save_benchmark_results(benchmark_suite)

        # Generate report
        self.generate_benchmark_report(benchmark_suite)

        self.logger.info("=" * 70)
        self.logger.info(f"Benchmark completed in {total_duration:.1f} seconds")
        self.logger.info("=" * 70)

        return benchmark_suite

    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture system information"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": ".".join(map(str, [3, 11, 0])),  # Placeholder
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }

        if PSUTIL_AVAILABLE:
            info["cpu_count"] = psutil.cpu_count()
            info["memory_total_gb"] = psutil.virtual_memory().total / (1024**3)
        else:
            info["cpu_count"] = "unknown"
            info["memory_total_gb"] = "unknown"

        return info

    def save_benchmark_results(self, suite: MergerBenchmarkSuite):
        """Save benchmark results to JSON"""
        timestamp_str = suite.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"merger_benchmark_results_{timestamp_str}.json"
        filepath = self.output_dir / filename

        # Convert to dict
        data = asdict(suite)
        data["timestamp"] = suite.timestamp.isoformat()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"\n📊 Results saved to: {filepath}")

    def generate_benchmark_report(self, suite: MergerBenchmarkSuite):
        """Generate markdown benchmark report"""
        timestamp_str = suite.timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"merger_benchmark_report_{timestamp_str}.md"

        with open(report_path, "w") as f:
            f.write("# ENAHO Merger Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {suite.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # System info
            f.write("## System Information\n\n")
            for key, value in suite.system_info.items():
                f.write(f"- **{key}:** {value}\n")
            f.write("\n")

            # Summary
            summary = suite.get_summary_stats()
            f.write("## Summary\n\n")
            f.write(f"- **Total Operations:** {summary['total_operations']}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']:.1%}\n")
            f.write(
                f"- **Average Throughput:** {summary['average_throughput_recs_per_sec']:,.0f} records/sec\n"
            )
            f.write(f"- **Average Memory:** {summary['average_memory_peak_mb']:.1f} MB\n")
            f.write(f"- **Average Quality:** {summary['average_merge_quality']:.1f}%\n")
            f.write(f"- **Total Duration:** {summary['total_duration']:.1f}s\n\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            f.write(
                "| Operation | Records | Duration (s) | Throughput (rec/s) | Memory (MB) | Quality | Status |\n"
            )
            f.write(
                "|-----------|---------|--------------|-------------------|-------------|---------|--------|\n"
            )

            for result in suite.results:
                status = "✅" if result.success else "❌"
                f.write(
                    f"| {result.operation} | {result.dataset_size:,} | "
                    f"{result.duration_seconds:.2f} | {result.records_per_second:,.0f} | "
                    f"{result.memory_peak_mb:.1f} | {result.merge_quality_score:.1f}% | {status} |\n"
                )

            # Performance analysis
            f.write("\n## Performance Analysis\n\n")
            f.write(self._generate_performance_analysis(suite))

        self.logger.info(f"📄 Report generated: {report_path}")

    def _generate_performance_analysis(self, suite: MergerBenchmarkSuite) -> str:
        """Generate performance analysis section"""
        analysis = []

        # Group by operation type
        hogar_merges = [r for r in suite.results if "hogar" in r.operation and r.success]
        geo_merges = [r for r in suite.results if "geographic" in r.operation and r.success]

        # Analyze scaling for household merges
        if len(hogar_merges) >= 2:
            analysis.append("### Module Merge (Household) Scaling\n")
            for result in hogar_merges:
                analysis.append(
                    f"- **{result.dataset_size:,} records**: "
                    f"{result.records_per_second:,.0f} rec/s, "
                    f"{result.memory_peak_mb:.1f} MB\n"
                )

            # Calculate scaling factor
            if len(hogar_merges) >= 2:
                small = hogar_merges[0]
                large = hogar_merges[-1]
                size_ratio = large.dataset_size / small.dataset_size
                time_ratio = large.duration_seconds / small.duration_seconds
                memory_ratio = large.memory_peak_mb / small.memory_peak_mb

                analysis.append(
                    f"\n**Scaling Analysis ({small.dataset_size:,} → {large.dataset_size:,} records):**\n"
                )
                analysis.append(f"- Size multiplier: {size_ratio:.0f}x\n")
                analysis.append(
                    f"- Time multiplier: {time_ratio:.1f}x (ideal: {size_ratio:.0f}x)\n"
                )
                analysis.append(f"- Memory multiplier: {memory_ratio:.1f}x\n")

                if time_ratio < size_ratio * 1.2:
                    analysis.append(f"- ✅ Performance scales nearly linearly\n")
                else:
                    analysis.append(
                        f"- ⚠️ Performance degrades with scale (investigate bottlenecks)\n"
                    )

        # Analyze geographic merges
        if len(geo_merges) >= 2:
            analysis.append("\n### Geographic Merge Scaling\n")
            for result in geo_merges:
                analysis.append(
                    f"- **{result.dataset_size:,} records**: "
                    f"{result.records_per_second:,.0f} rec/s\n"
                )

        return "".join(analysis)

    def compare_with_baseline(
        self, baseline_file: Path, current_suite: MergerBenchmarkSuite
    ) -> Dict[str, Any]:
        """
        Compare current results with baseline

        Args:
            baseline_file: Path to baseline JSON file
            current_suite: Current benchmark suite

        Returns:
            Comparison analysis
        """
        try:
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load baseline: {e}")
            return {"error": str(e)}

        comparison = {
            "baseline_date": baseline_data.get("timestamp"),
            "current_date": current_suite.timestamp.isoformat(),
            "improvements": [],
            "regressions": [],
            "overall_improvement": 0.0,
        }

        # Match operations
        baseline_ops = {r["operation"]: r for r in baseline_data["results"]}
        current_ops = {r.operation: r for r in current_suite.results}

        for op_name in set(baseline_ops.keys()) & set(current_ops.keys()):
            baseline = baseline_ops[op_name]
            current = current_ops[op_name]

            if not current.success:
                continue

            # Calculate improvement
            baseline_throughput = baseline.get("records_per_second", 0)
            current_throughput = current.records_per_second

            if baseline_throughput > 0:
                improvement = (current_throughput - baseline_throughput) / baseline_throughput * 100

                if improvement > 5:  # >5% improvement
                    comparison["improvements"].append(
                        {
                            "operation": op_name,
                            "improvement_percent": improvement,
                            "baseline_throughput": baseline_throughput,
                            "current_throughput": current_throughput,
                        }
                    )
                elif improvement < -5:  # >5% regression
                    comparison["regressions"].append(
                        {
                            "operation": op_name,
                            "regression_percent": -improvement,
                            "baseline_throughput": baseline_throughput,
                            "current_throughput": current_throughput,
                        }
                    )

        # Calculate overall improvement
        if comparison["improvements"]:
            avg_improvement = np.mean(
                [i["improvement_percent"] for i in comparison["improvements"]]
            )
            comparison["overall_improvement"] = avg_improvement

        return comparison


def run_merger_benchmark(
    output_dir: Optional[str] = None, log_level: str = "INFO"
) -> MergerBenchmarkSuite:
    """
    Run merger benchmark suite

    Args:
        output_dir: Output directory for results
        log_level: Logging level

    Returns:
        Benchmark suite results
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("merger_benchmark")

    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path("./benchmarks/merger")
    output_path.mkdir(exist_ok=True, parents=True)

    # Run benchmark
    benchmark_suite = MergerBenchmarkSuiteRunner(logger, output_path)
    results = benchmark_suite.run_comprehensive_benchmark()

    return results


if __name__ == "__main__":
    import sys

    # Allow output directory as command-line argument
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None

    results = run_merger_benchmark(output_dir=output_dir, log_level="INFO")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    summary = results.get_summary_stats()
    print(f"\nTotal Operations: {summary['total_operations']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Throughput: {summary['average_throughput_recs_per_sec']:,.0f} records/sec")
    print(f"Average Memory: {summary['average_memory_peak_mb']:.1f} MB")
    print("=" * 70)
