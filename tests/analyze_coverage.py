"""
Análisis de Cobertura de Código
================================

Script para analizar la cobertura de tests del código refactorizado.

Requiere: pip install pytest pytest-cov

Ejecutar:
    python tests/analyze_coverage.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


class CoverageAnalyzer:
    """Analiza cobertura de código y genera reportes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_dir = project_root / ".coverage_reports"
        self.coverage_dir.mkdir(exist_ok=True)

    def run_coverage(
        self, test_paths: List[str], source_paths: List[str], output_name: str = "coverage"
    ) -> Dict[str, Any]:
        """
        Ejecuta tests con cobertura y genera reporte.

        Args:
            test_paths: Lista de paths de tests
            source_paths: Lista de paths de código fuente
            output_name: Nombre base para archivos de reporte

        Returns:
            Dict con métricas de cobertura
        """
        print(f"\n{'='*60}")
        print(f"Running Coverage Analysis: {output_name}")
        print(f"{'='*60}")

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *test_paths,
            f"--cov={','.join(source_paths)}",
            "--cov-report=html:" + str(self.coverage_dir / f"{output_name}_html"),
            "--cov-report=json:" + str(self.coverage_dir / f"{output_name}.json"),
            "--cov-report=term-missing",
            "-v",
        ]

        print(f"\nCommand: {' '.join(cmd)}\n")

        # Run pytest with coverage
        result = subprocess.run(cmd, cwd=str(self.project_root), capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Parse coverage report
        coverage_file = self.coverage_dir / f"{output_name}.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)

            metrics = self._extract_metrics(coverage_data)
            self._print_summary(metrics, output_name)
            return metrics
        else:
            print(f"\nWarning: Coverage report not found at {coverage_file}")
            return {}

    def _extract_metrics(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae métricas del reporte JSON de coverage."""
        totals = coverage_data.get("totals", {})

        return {
            "total_statements": totals.get("num_statements", 0),
            "covered_statements": totals.get("covered_lines", 0),
            "missing_statements": totals.get("missing_lines", 0),
            "excluded_statements": totals.get("excluded_lines", 0),
            "coverage_percent": totals.get("percent_covered", 0.0),
            "files": self._extract_file_metrics(coverage_data),
        }

    def _extract_file_metrics(self, coverage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae métricas por archivo."""
        files = coverage_data.get("files", {})
        file_metrics = []

        for filepath, file_data in files.items():
            summary = file_data.get("summary", {})
            file_metrics.append(
                {
                    "path": filepath,
                    "statements": summary.get("num_statements", 0),
                    "covered": summary.get("covered_lines", 0),
                    "missing": summary.get("missing_lines", 0),
                    "coverage": summary.get("percent_covered", 0.0),
                    "missing_lines": file_data.get("missing_lines", []),
                }
            )

        # Sort by coverage (ascending)
        file_metrics.sort(key=lambda x: x["coverage"])
        return file_metrics

    def _print_summary(self, metrics: Dict[str, Any], name: str) -> None:
        """Imprime resumen de cobertura."""
        print(f"\n{'='*60}")
        print(f"Coverage Summary: {name}")
        print(f"{'='*60}")

        print(f"\nOverall Coverage: {metrics['coverage_percent']:.1f}%")
        print(f"  Total Statements:   {metrics['total_statements']}")
        print(f"  Covered Statements: {metrics['covered_statements']}")
        print(f"  Missing Statements: {metrics['missing_statements']}")

        # Files with low coverage
        low_coverage_files = [
            f for f in metrics["files"] if f["coverage"] < 80.0 and f["statements"] > 10
        ]

        if low_coverage_files:
            print(f"\nFiles with <80% Coverage ({len(low_coverage_files)}):")
            for file in low_coverage_files[:10]:  # Top 10
                rel_path = Path(file["path"]).relative_to(self.project_root)
                print(
                    f"  {str(rel_path):60s} {file['coverage']:5.1f}%  "
                    f"({file['covered']}/{file['statements']} lines)"
                )

        # Files with excellent coverage
        excellent_coverage_files = [f for f in metrics["files"] if f["coverage"] >= 90.0]

        if excellent_coverage_files:
            print(f"\nFiles with >=90% Coverage ({len(excellent_coverage_files)}):")
            for file in excellent_coverage_files[:10]:
                rel_path = Path(file["path"]).relative_to(self.project_root)
                print(f"  {str(rel_path):60s} {file['coverage']:5.1f}%")

    def analyze_refactored_modules(self) -> Dict[str, Any]:
        """Analiza cobertura de módulos refactorizados."""
        print("\n" + "=" * 60)
        print(" ANALYZING REFACTORED MODULES")
        print("=" * 60)

        # 1. Validation module
        validation_metrics = self.run_coverage(
            test_paths=["tests/test_validation.py"],
            source_paths=["enahopy/validation.py"],
            output_name="validation_coverage",
        )

        # 2. Cache module
        cache_metrics = self.run_coverage(
            test_paths=["tests/test_core_cache.py"],
            source_paths=["enahopy/loader/core/cache.py"],
            output_name="cache_coverage",
        )

        # 3. Null Analyzer module
        analyzer_metrics = self.run_coverage(
            test_paths=["tests/test_null_analyzer_vectorized.py"],
            source_paths=["enahopy/null_analysis/core/analyzer.py"],
            output_name="null_analyzer_coverage",
        )

        # 4. Integration tests
        integration_metrics = self.run_coverage(
            test_paths=["tests/test_integration_validators.py"],
            source_paths=["enahopy/validation.py", "enahopy/null_analysis/core/analyzer.py"],
            output_name="integration_coverage",
        )

        # Aggregate results
        return {
            "validation": validation_metrics,
            "cache": cache_metrics,
            "null_analyzer": analyzer_metrics,
            "integration": integration_metrics,
        }

    def generate_report(self, all_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Genera reporte final consolidado."""
        print("\n" + "=" * 60)
        print(" FINAL COVERAGE REPORT")
        print("=" * 60)

        print("\n" + "-" * 60)
        print(" Module Coverage Summary")
        print("-" * 60)

        for module_name, metrics in all_metrics.items():
            if metrics:
                coverage = metrics.get("coverage_percent", 0.0)
                status = "✓" if coverage >= 80.0 else "✗"
                print(f"{status} {module_name:20s}: {coverage:5.1f}% coverage")

        # Overall recommendations
        print("\n" + "-" * 60)
        print(" Recommendations")
        print("-" * 60)

        for module_name, metrics in all_metrics.items():
            if not metrics:
                continue

            coverage = metrics.get("coverage_percent", 0.0)

            if coverage < 60.0:
                print(f"\n{module_name}:")
                print(f"  - CRITICAL: Coverage is very low ({coverage:.1f}%)")
                print("  - Add comprehensive unit tests")
                print("  - Focus on edge cases and error handling")

            elif coverage < 80.0:
                print(f"\n{module_name}:")
                print(f"  - WARNING: Coverage below recommended ({coverage:.1f}%)")
                print("  - Add tests for uncovered branches")
                print("  - Review missing lines in HTML report")

            else:
                print(f"\n{module_name}:")
                print(f"  - EXCELLENT: Coverage is good ({coverage:.1f}%)")
                print("  - Maintain current test quality")

        # Save consolidated report
        report_path = self.coverage_dir / "consolidated_report.json"
        with open(report_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n\nConsolidated report saved to: {report_path}")
        print(f"HTML reports available in: {self.coverage_dir}")


def check_dependencies() -> bool:
    """Verifica que las dependencias necesarias estén instaladas."""
    try:
        pass

        return True
    except ImportError as e:
        print("\nError: Missing required dependencies")
        print(f"  {e}")
        print("\nInstall with: pip install pytest pytest-cov")
        return False


def main():
    """Ejecuta análisis de cobertura."""
    print("\n" + "=" * 60)
    print(" ENAHOPY CODE COVERAGE ANALYSIS")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Setup
    project_root = Path(__file__).parent.parent
    analyzer = CoverageAnalyzer(project_root)

    # Analyze refactored modules
    all_metrics = analyzer.analyze_refactored_modules()

    # Generate final report
    analyzer.generate_report(all_metrics)

    print("\n" + "=" * 60)
    print(" COVERAGE ANALYSIS COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
