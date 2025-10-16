"""
Benchmarks de Performance - Refactoring
=======================================

Compara la performance antes y después de las optimizaciones:
1. Vectorización en NullAnalyzer
2. Lazy loading en __init__.py
3. Validación centralizada

Ejecutar: python tests/benchmark_refactoring.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enahopy.null_analysis.core.analyzer import NullAnalyzer
from enahopy.validation import (
    validate_column_type,
    validate_columns_exist,
    validate_dataframe_not_empty,
)


class BenchmarkRunner:
    """Ejecuta y reporta benchmarks de performance."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    def benchmark(
        self, name: str, func: Callable, setup: Callable = None, iterations: int = 10
    ) -> Dict[str, float]:
        """
        Ejecuta benchmark de una función.

        Args:
            name: Nombre del benchmark
            func: Función a benchmarkear
            setup: Función de setup (ejecutada antes, no medida)
            iterations: Número de iteraciones

        Returns:
            Dict con métricas de tiempo
        """
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

        # Setup
        if setup:
            data = setup()
        else:
            data = None

        # Warmup
        if data is not None:
            func(data)
        else:
            func()

        # Benchmark
        times = []
        for i in range(iterations):
            if data is not None:
                start = time.perf_counter()
                func(data)
                elapsed = time.perf_counter() - start
            else:
                start = time.perf_counter()
                func()
                elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"  Iteration {i+1:2d}: {elapsed*1000:8.3f}ms")

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        result = {
            "mean_ms": mean_time * 1000,
            "std_ms": std_time * 1000,
            "min_ms": min_time * 1000,
            "max_ms": max_time * 1000,
            "iterations": iterations,
        }

        print(f"\nResults:")
        print(f"  Mean:   {result['mean_ms']:8.3f}ms ± {result['std_ms']:6.3f}ms")
        print(f"  Min:    {result['min_ms']:8.3f}ms")
        print(f"  Max:    {result['max_ms']:8.3f}ms")

        self.results[name] = result
        return result

    def compare(self, baseline_name: str, optimized_name: str) -> None:
        """
        Compara dos benchmarks y calcula speedup.

        Args:
            baseline_name: Nombre del benchmark baseline
            optimized_name: Nombre del benchmark optimizado
        """
        if baseline_name not in self.results or optimized_name not in self.results:
            print(f"\nError: Missing benchmark results")
            return

        baseline = self.results[baseline_name]
        optimized = self.results[optimized_name]

        speedup = baseline["mean_ms"] / optimized["mean_ms"]
        improvement_pct = ((baseline["mean_ms"] - optimized["mean_ms"]) / baseline["mean_ms"]) * 100

        print(f"\n{'='*60}")
        print(f"Comparison: {baseline_name} vs {optimized_name}")
        print(f"{'='*60}")
        print(f"Baseline:   {baseline['mean_ms']:8.3f}ms")
        print(f"Optimized:  {optimized['mean_ms']:8.3f}ms")
        print(f"Speedup:    {speedup:.2f}x faster")
        print(f"Improvement: {improvement_pct:.1f}%")

    def save_results(self, output_path: str) -> None:
        """Guarda resultados en archivo JSON."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def create_test_dataframe(rows: int, cols: int, null_pct: float = 0.2) -> pd.DataFrame:
    """
    Crea DataFrame de prueba con valores nulos.

    Args:
        rows: Número de filas
        cols: Número de columnas
        null_pct: Porcentaje de valores nulos

    Returns:
        DataFrame de prueba
    """
    np.random.seed(42)
    data = {}

    for i in range(cols):
        values = np.random.randn(rows)
        null_mask = np.random.random(rows) < null_pct
        values[null_mask] = np.nan
        data[f"col_{i}"] = values

    return pd.DataFrame(data)


# =========================================================================
# Benchmarks de Vectorización en NullAnalyzer
# =========================================================================


def benchmark_null_analyzer_small():
    """Benchmark NullAnalyzer con DataFrame pequeño (100x10)."""

    def setup():
        return create_test_dataframe(100, 10, null_pct=0.2)

    def run(df):
        analyzer = NullAnalyzer()
        analyzer.analyze(df)

    runner = BenchmarkRunner()
    runner.benchmark("NullAnalyzer - Small (100x10)", run, setup=setup, iterations=100)
    return runner


def benchmark_null_analyzer_medium():
    """Benchmark NullAnalyzer con DataFrame mediano (1000x50)."""

    def setup():
        return create_test_dataframe(1000, 50, null_pct=0.2)

    def run(df):
        analyzer = NullAnalyzer()
        analyzer.analyze(df)

    runner = BenchmarkRunner()
    runner.benchmark("NullAnalyzer - Medium (1000x50)", run, setup=setup, iterations=50)
    return runner


def benchmark_null_analyzer_large():
    """Benchmark NullAnalyzer con DataFrame grande (10000x100)."""

    def setup():
        return create_test_dataframe(10000, 100, null_pct=0.2)

    def run(df):
        analyzer = NullAnalyzer()
        analyzer.analyze(df)

    runner = BenchmarkRunner()
    runner.benchmark("NullAnalyzer - Large (10000x100)", run, setup=setup, iterations=20)
    return runner


def benchmark_null_analyzer_very_wide():
    """Benchmark NullAnalyzer con DataFrame muy ancho (100x500)."""

    def setup():
        return create_test_dataframe(100, 500, null_pct=0.2)

    def run(df):
        analyzer = NullAnalyzer()
        analyzer.analyze(df)

    runner = BenchmarkRunner()
    runner.benchmark("NullAnalyzer - Very Wide (100x500)", run, setup=setup, iterations=20)
    return runner


# =========================================================================
# Benchmarks de Validación Centralizada
# =========================================================================


def benchmark_validation_pipeline():
    """Benchmark pipeline de validación centralizada."""

    def setup():
        return pd.DataFrame(
            {
                "ubigeo": ["010101"] * 1000,
                "departamento": ["LIMA"] * 1000,
                "ingreso": np.random.randn(1000),
                "edad": np.random.randint(18, 80, 1000),
            }
        )

    def run(df):
        validate_dataframe_not_empty(df, "Test")
        validate_columns_exist(df, ["ubigeo", "departamento", "ingreso", "edad"])
        validate_column_type(df, "ubigeo", "object")
        validate_column_type(df, "ingreso", ["float", "int"])
        validate_column_type(df, "edad", "int")

    runner = BenchmarkRunner()
    runner.benchmark("Validation Pipeline (1000 rows)", run, setup=setup, iterations=100)
    return runner


# =========================================================================
# Benchmarks de Lazy Loading
# =========================================================================


def benchmark_import_time():
    """
    Benchmark tiempo de importación del paquete.

    Nota: Este benchmark debe ejecutarse en un proceso separado
    para medir correctamente el tiempo de importación.
    """
    import subprocess

    # Medir tiempo de importación
    cmd = [
        sys.executable,
        "-c",
        "import time; start = time.perf_counter(); import enahopy; print(time.perf_counter() - start)",
    ]

    times = []
    for _ in range(10):
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            elapsed = float(result.stdout.strip())
            times.append(elapsed)
        except ValueError:
            print(f"Error parsing output: {result.stdout}")
            continue

    if times:
        mean_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        print(f"\nImport Time:")
        print(f"  Mean: {mean_time:.3f}ms ± {std_time:.3f}ms")
        print(f"  Min:  {np.min(times)*1000:.3f}ms")
        print(f"  Max:  {np.max(times)*1000:.3f}ms")

        return {
            "mean_ms": mean_time,
            "std_ms": std_time,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
        }
    else:
        print("\nError: Could not measure import time")
        return None


# =========================================================================
# Main Execution
# =========================================================================


def main():
    """Ejecuta todos los benchmarks y genera reporte."""
    print("\n" + "=" * 60)
    print(" ENAHOPY REFACTORING BENCHMARKS")
    print("=" * 60)

    all_results = {}

    # 1. Benchmarks de NullAnalyzer
    print("\n\n1. NULL ANALYZER VECTORIZATION")
    print("-" * 60)

    runner_small = benchmark_null_analyzer_small()
    all_results.update(runner_small.results)

    runner_medium = benchmark_null_analyzer_medium()
    all_results.update(runner_medium.results)

    runner_large = benchmark_null_analyzer_large()
    all_results.update(runner_large.results)

    runner_wide = benchmark_null_analyzer_very_wide()
    all_results.update(runner_wide.results)

    # 2. Benchmarks de Validación
    print("\n\n2. CENTRALIZED VALIDATION")
    print("-" * 60)

    runner_validation = benchmark_validation_pipeline()
    all_results.update(runner_validation.results)

    # 3. Benchmark de Import Time
    print("\n\n3. LAZY LOADING IMPORT TIME")
    print("-" * 60)

    import_result = benchmark_import_time()
    if import_result:
        all_results["Import Time"] = import_result

    # 4. Summary
    print("\n\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    print("\nNull Analyzer Performance:")
    for name, result in all_results.items():
        if "NullAnalyzer" in name:
            print(f"  {name:40s}: {result['mean_ms']:8.3f}ms")

    print("\nValidation Performance:")
    for name, result in all_results.items():
        if "Validation" in name:
            print(f"  {name:40s}: {result['mean_ms']:8.3f}ms")

    # 5. Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    final_runner = BenchmarkRunner()
    final_runner.results = all_results
    final_runner.save_results(str(output_path))

    print("\n" + "=" * 60)
    print(" BENCHMARKS COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
