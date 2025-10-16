"""
Test Runner Principal - ENAHOPY
================================
Ejecutor principal de todas las suites de tests con reportes detallados.
"""

import argparse
import importlib.util
import json
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Para reportes más avanzados
try:
    import coverage

    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestResult:
    """Clase para almacenar resultados de tests"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.tests_run = 0
        self.successes = 0
        self.failures = []
        self.errors = []
        self.skipped = []
        self.execution_time = 0.0

    def to_dict(self) -> dict:
        """Convertir a diccionario para serialización"""
        return {
            "module": self.module_name,
            "tests_run": self.tests_run,
            "successes": self.successes,
            "failures": len(self.failures),
            "errors": len(self.errors),
            "skipped": len(self.skipped),
            "execution_time": round(self.execution_time, 2),
            "success_rate": round(
                (self.successes / self.tests_run * 100) if self.tests_run > 0 else 0, 2
            ),
        }


class ENAHOTestRunner:
    """Runner principal para todos los tests de ENAHOPY"""

    def __init__(self, verbose: int = 2, coverage_enabled: bool = False):
        self.verbose = verbose
        self.coverage_enabled = coverage_enabled and COVERAGE_AVAILABLE
        self.results = []
        self.cov = None

        if self.coverage_enabled:
            self.cov = coverage.Coverage(source=["enahopy"])
            self.cov.start()

    def run_module_tests(self, module_name: str, test_file: str) -> TestResult:
        """Ejecutar tests de un módulo específico"""
        print(f"\n{'=' * 70}")
        print(f"Ejecutando tests: {module_name}")
        print(f"{'=' * 70}")

        result = TestResult(module_name)
        start_time = time.time()

        try:
            # Importar módulo de tests dinámicamente
            spec = importlib.util.spec_from_file_location(f"test_{module_name}", test_file)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Cargar y ejecutar tests
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)

            # Ejecutar con unittest
            runner = unittest.TextTestRunner(verbosity=self.verbose)
            test_result = runner.run(suite)

            # Recopilar resultados
            result.tests_run = test_result.testsRun
            result.successes = (
                test_result.testsRun - len(test_result.failures) - len(test_result.errors)
            )
            result.failures = [(str(test), str(trace)) for test, trace in test_result.failures]
            result.errors = [(str(test), str(trace)) for test, trace in test_result.errors]
            result.skipped = [(str(test), reason) for test, reason in test_result.skipped]

        except Exception as e:
            print(f"Error ejecutando tests de {module_name}: {e}")
            result.errors.append(("Module Import", str(e)))

        result.execution_time = time.time() - start_time
        self.results.append(result)

        return result

    def run_all_tests(self) -> Dict:
        """Ejecutar todos los tests de la librería"""
        print("\n" + "=" * 70)
        print("ENAHOPY - Suite Completa de Tests")
        print("=" * 70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        test_modules = [
            ("loader", "tests/test_loader.py"),
            ("merger", "tests/test_merger.py"),
            ("null_analysis", "tests/test_null_analysis.py"),
        ]

        total_start = time.time()

        for module_name, test_file in test_modules:
            if Path(test_file).exists():
                self.run_module_tests(module_name, test_file)
            else:
                print(f"⚠️ Archivo de tests no encontrado: {test_file}")

        total_time = time.time() - total_start

        # Generar resumen
        summary = self.generate_summary(total_time)

        # Detener coverage si está habilitado
        if self.coverage_enabled and self.cov:
            self.cov.stop()
            self.cov.save()

        return summary

    def generate_summary(self, total_time: float) -> Dict:
        """Generar resumen de todos los tests"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": round(total_time, 2),
            "modules_tested": len(self.results),
            "total_tests": sum(r.tests_run for r in self.results),
            "total_successes": sum(r.successes for r in self.results),
            "total_failures": sum(len(r.failures) for r in self.results),
            "total_errors": sum(len(r.errors) for r in self.results),
            "total_skipped": sum(len(r.skipped) for r in self.results),
            "module_results": [r.to_dict() for r in self.results],
        }

        # Calcular tasa de éxito global
        if summary["total_tests"] > 0:
            summary["overall_success_rate"] = round(
                (summary["total_successes"] / summary["total_tests"]) * 100, 2
            )
        else:
            summary["overall_success_rate"] = 0

        return summary

    def print_summary(self, summary: Dict):
        """Imprimir resumen formateado"""
        print("\n" + "=" * 70)
        print("RESUMEN GLOBAL DE TESTS")
        print("=" * 70)

        print(f"\nEstadísticas Generales:")
        print(f"  - Módulos testeados: {summary['modules_tested']}")
        print(f"  - Tests ejecutados: {summary['total_tests']}")
        print(f"  - Éxitos: {summary['total_successes']} ✅")
        print(f"  - Fallos: {summary['total_failures']} ❌")
        print(f"  - Errores: {summary['total_errors']} 💥")
        print(f"  - Omitidos: {summary['total_skipped']} ⏭️")
        print(f"  - Tasa de éxito: {summary['overall_success_rate']}%")
        print(f"  - Tiempo total: {summary['total_execution_time']}s")

        print(f"\nResultados por Módulo:")
        for module in summary["module_results"]:
            status = "✅" if module["failures"] == 0 and module["errors"] == 0 else "❌"
            print(
                f"  {status} {module['module']}: {module['successes']}/{module['tests_run']} "
                + f"({module['success_rate']}%) - {module['execution_time']}s"
            )

        # Mostrar fallos y errores detallados si existen
        if summary["total_failures"] > 0 or summary["total_errors"] > 0:
            print("\n" + "=" * 70)
            print("DETALLES DE FALLOS Y ERRORES")
            print("=" * 70)

            for result in self.results:
                if result.failures or result.errors:
                    print(f"\n{result.module_name}:")

                    if result.failures:
                        print("  Fallos:")
                        for test, trace in result.failures[:3]:  # Mostrar máximo 3
                            print(f"    - {test}")
                            print(f"      {trace.split(chr(10))[0][:100]}...")

                    if result.errors:
                        print("  Errores:")
                        for test, trace in result.errors[:3]:  # Mostrar máximo 3
                            print(f"    - {test}")
                            print(f"      {trace.split(chr(10))[0][:100]}...")

    def save_report(self, summary: Dict, output_file: str = "test_report.json"):
        """Guardar reporte en archivo JSON"""
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Reporte guardado en: {output_file}")

    def generate_coverage_report(self):
        """Generar reporte de cobertura"""
        if self.coverage_enabled and self.cov:
            print("\n" + "=" * 70)
            print("REPORTE DE COBERTURA")
            print("=" * 70)

            # Generar reporte en consola
            self.cov.report()

            # Generar HTML
            self.cov.html_report(directory="htmlcov")
            print("\n📊 Reporte HTML de cobertura generado en: htmlcov/index.html")


def run_with_pytest():
    """Ejecutar tests usando pytest si está disponible"""
    if not PYTEST_AVAILABLE:
        print("⚠️ pytest no está instalado. Usando unittest.")
        return False

    print("\n" + "=" * 70)
    print("Ejecutando tests con pytest")
    print("=" * 70)

    # Configuración de pytest
    pytest_args = [
        "tests/",
        "-v",  # Verbose
        "--tb=short",  # Traceback corto
        "--junit-xml=test_results.xml",  # Generar XML para CI/CD
        "--cov=enahopy",  # Coverage
        "--cov-report=html",  # Reporte HTML
        "--cov-report=term",  # Reporte en terminal
    ]

    # Ejecutar pytest
    exit_code = pytest.main(pytest_args)

    return exit_code == 0


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="ENAHOPY Test Runner")
    parser.add_argument(
        "--module",
        choices=["loader", "merger", "null_analysis", "all"],
        default="all",
        help="Módulo a testear",
    )
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=2, help="Nivel de verbosidad"
    )
    parser.add_argument("--coverage", action="store_true", help="Habilitar análisis de cobertura")
    parser.add_argument("--pytest", action="store_true", help="Usar pytest en lugar de unittest")
    parser.add_argument(
        "--report", type=str, default="test_report.json", help="Archivo para guardar el reporte"
    )

    args = parser.parse_args()

    # Si se solicita pytest
    if args.pytest:
        success = run_with_pytest()
        return 0 if success else 1

    # Usar unittest runner personalizado
    runner = ENAHOTestRunner(verbose=args.verbose, coverage_enabled=args.coverage)

    # Ejecutar tests
    if args.module == "all":
        summary = runner.run_all_tests()
    else:
        test_file = f"tests/test_{args.module}.py"
        result = runner.run_module_tests(args.module, test_file)
        summary = {"timestamp": datetime.now().isoformat(), "module_results": [result.to_dict()]}

    # Mostrar resumen
    runner.print_summary(summary)

    # Guardar reporte
    runner.save_report(summary, args.report)

    # Generar reporte de cobertura
    if args.coverage:
        runner.generate_coverage_report()

    # Retornar código de salida
    total_failures = sum(len(r.failures) + len(r.errors) for r in runner.results)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
