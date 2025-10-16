"""
Tests Unitarios para Null Analyzer Vectorizado
==============================================

Tests de correctitud y performance para el analizador de nulos
refactorizado con operaciones vectorizadas.
"""

import time
import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd

from enahopy.null_analysis.core.analyzer import NullAnalysisConfig, NullAnalyzer


class TestNullAnalyzerCorrectness(unittest.TestCase):
    """Tests de correctitud para el analizador vectorizado."""

    def setUp(self):
        """Set up test environment."""
        self.analyzer = NullAnalyzer()

    def test_analyze_all_complete_dataframe(self):
        """Debe analizar correctamente DataFrame sin nulos."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

        result = self.analyzer.analyze(df)

        self.assertEqual(result["null_values"], 0)
        self.assertEqual(result["null_percentage"], 0.0)
        self.assertEqual(result["severity"], "none")
        self.assertEqual(result["row_analysis"]["rows_with_nulls"], 0)
        self.assertEqual(result["row_analysis"]["complete_rows"], 5)

        for col_analysis in result["columns_analysis"].values():
            self.assertEqual(col_analysis["null_count"], 0)
            self.assertEqual(col_analysis["null_percentage"], 0.0)
            self.assertEqual(col_analysis["severity"], "none")

    def test_analyze_all_null_dataframe(self):
        """Debe analizar correctamente DataFrame completamente nulo."""
        df = pd.DataFrame(
            {
                "col1": [None, None, None],
                "col2": [np.nan, np.nan, np.nan],
                "col3": [pd.NA, pd.NA, pd.NA],
            }
        )

        result = self.analyzer.analyze(df)

        self.assertEqual(result["null_values"], 9)  # 3 rows * 3 cols
        self.assertEqual(result["null_percentage"], 100.0)
        self.assertEqual(result["severity"], "extreme")
        self.assertEqual(result["row_analysis"]["rows_with_nulls"], 3)
        self.assertEqual(result["row_analysis"]["complete_rows"], 0)

        for col_analysis in result["columns_analysis"].values():
            self.assertEqual(col_analysis["null_count"], 3)
            self.assertEqual(col_analysis["null_percentage"], 100.0)
            self.assertEqual(col_analysis["severity"], "extreme")

    def test_analyze_partial_nulls(self):
        """Debe analizar correctamente DataFrame con nulos parciales."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, None, 4, 5],
                "col2": [None, "b", "c", None, "e"],
                "col3": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

        result = self.analyzer.analyze(df)

        # Validaciones globales
        self.assertEqual(result["null_values"], 3)  # 1 in col1, 2 in col2
        self.assertEqual(result["null_percentage"], 20.0)  # 3/15 * 100
        # 20% cae en "high" según thresholds: moderate < 20%, high < 50%
        self.assertEqual(result["severity"], "high")

        # Validaciones por columna
        self.assertEqual(result["columns_analysis"]["col1"]["null_count"], 1)
        self.assertEqual(result["columns_analysis"]["col1"]["null_percentage"], 20.0)

        self.assertEqual(result["columns_analysis"]["col2"]["null_count"], 2)
        self.assertEqual(result["columns_analysis"]["col2"]["null_percentage"], 40.0)

        self.assertEqual(result["columns_analysis"]["col3"]["null_count"], 0)
        self.assertEqual(result["columns_analysis"]["col3"]["null_percentage"], 0.0)

        # Validaciones por fila
        self.assertEqual(result["row_analysis"]["rows_with_nulls"], 3)
        self.assertEqual(result["row_analysis"]["complete_rows"], 2)

    def test_analyze_preserves_dtypes(self):
        """Debe preservar correctamente los tipos de datos."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        result = self.analyzer.analyze(df)

        self.assertIn("int", result["columns_analysis"]["int_col"]["dtype"])
        self.assertIn("float", result["columns_analysis"]["float_col"]["dtype"])
        self.assertEqual(result["columns_analysis"]["str_col"]["dtype"], "object")
        self.assertEqual(result["columns_analysis"]["bool_col"]["dtype"], "bool")

    def test_severity_classification(self):
        """Debe clasificar severidad correctamente."""
        test_cases = [
            (0.0, "none"),
            (3.0, "low"),  # < 5%
            (10.0, "moderate"),  # < 20%
            (35.0, "high"),  # < 50%
            (60.0, "critical"),  # < 70%
            (85.0, "extreme"),  # >= 70%
        ]

        for percentage, expected_severity in test_cases:
            severity = self.analyzer._classify_severity(percentage)
            self.assertEqual(
                severity,
                expected_severity,
                f"Failed for {percentage}%: expected {expected_severity}, got {severity}",
            )

    def test_pattern_detection_complete_columns(self):
        """Debe detectar columnas completas."""
        df = pd.DataFrame(
            {"complete": [1, 2, 3], "partial": [None, 2, 3], "empty": [None, None, None]}
        )

        result = self.analyzer.analyze(df)
        patterns = result["patterns"]

        self.assertTrue(patterns["has_complete_columns"])
        self.assertIn("complete", patterns["complete_columns"])
        self.assertEqual(len(patterns["complete_columns"]), 1)

    def test_pattern_detection_empty_columns(self):
        """Debe detectar columnas vacías."""
        df = pd.DataFrame(
            {
                "complete": [1, 2, 3],
                "empty1": [None, None, None],
                "empty2": [np.nan, np.nan, np.nan],
            }
        )

        result = self.analyzer.analyze(df)
        patterns = result["patterns"]

        self.assertTrue(patterns["has_empty_columns"])
        self.assertEqual(set(patterns["empty_columns"]), {"empty1", "empty2"})

    def test_get_summary_statistics(self):
        """Debe generar estadísticas resumidas correctamente."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, None, 4, 5],
                "col2": ["a", None, None, "d", "e"],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        summary = self.analyzer.get_summary_statistics(df)

        # Verificar estructura
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 3)

        # Verificar columnas
        expected_columns = [
            "column",
            "dtype",
            "null_count",
            "non_null_count",
            "null_percentage",
            "unique_values",
            "has_nulls",
        ]
        for col in expected_columns:
            self.assertIn(col, summary.columns)

        # Verificar ordenamiento (descendente por null_percentage)
        self.assertTrue(
            (summary["null_percentage"].diff().dropna() <= 0).all(),
            "Summary should be sorted by null_percentage descending",
        )

        # Verificar valores específicos
        col2_row = summary[summary["column"] == "col2"].iloc[0]
        self.assertEqual(col2_row["null_count"], 2)
        self.assertEqual(col2_row["null_percentage"], 40.0)
        self.assertTrue(col2_row["has_nulls"])

    def test_identify_columns_for_imputation(self):
        """Debe identificar columnas candidatas para imputación."""
        df = pd.DataFrame(
            {
                "no_nulls": [1, 2, 3, 4, 5],
                "few_nulls": [1, None, 3, 4, 5],  # 20% nulls
                "some_nulls": [1, None, None, 4, 5],  # 40% nulls
                "many_nulls": [1, None, None, None, 5],  # 60% nulls
            }
        )

        # Con max_null_percentage = 50%
        candidates = self.analyzer.identify_columns_for_imputation(df, max_null_percentage=50.0)

        self.assertIn("few_nulls", candidates)
        self.assertIn("some_nulls", candidates)
        self.assertNotIn("no_nulls", candidates)  # No tiene nulos
        self.assertNotIn("many_nulls", candidates)  # Excede 50%

    def test_generate_recommendations(self):
        """Debe generar recomendaciones apropiadas."""
        # Caso 1: Sin nulos
        result_none = {"severity": "none", "null_percentage": 0, "patterns": {}}
        recs = self.analyzer.generate_recommendations(result_none)
        self.assertTrue(any("completos" in rec for rec in recs))

        # Caso 2: Nulos bajos
        result_low = {"severity": "low", "null_percentage": 3, "patterns": {}}
        recs = self.analyzer.generate_recommendations(result_low)
        self.assertTrue(any("simple" in rec for rec in recs))

        # Caso 3: Nulos moderados
        result_mod = {"severity": "moderate", "null_percentage": 15, "patterns": {}}
        recs = self.analyzer.generate_recommendations(result_mod)
        self.assertTrue(any("moderado" in rec for rec in recs))

        # Caso 4: Nulos críticos
        result_crit = {"severity": "critical", "null_percentage": 75, "patterns": {}}
        recs = self.analyzer.generate_recommendations(result_crit)
        self.assertTrue(any("ADVERTENCIA" in rec for rec in recs))

        # Caso 5: Con columnas vacías
        result_empty = {
            "severity": "moderate",
            "null_percentage": 25,
            "patterns": {"has_empty_columns": True, "empty_columns": ["col1", "col2"]},
        }
        recs = self.analyzer.generate_recommendations(result_empty)
        self.assertTrue(any("vacías" in rec for rec in recs))


class TestNullAnalyzerPerformance(unittest.TestCase):
    """Tests de performance para operaciones vectorizadas."""

    def test_performance_large_dataframe(self):
        """Debe procesar DataFrames grandes eficientemente."""
        # Crear DataFrame grande (10,000 rows x 100 cols)
        np.random.seed(42)
        n_rows = 10000
        n_cols = 100

        data = {}
        for i in range(n_cols):
            # 20% de valores nulos aleatorios
            values = np.random.randn(n_rows)
            null_mask = np.random.random(n_rows) < 0.2
            values[null_mask] = np.nan
            data[f"col_{i}"] = values

        df = pd.DataFrame(data)

        analyzer = NullAnalyzer()

        # Medir tiempo de análisis
        start_time = time.perf_counter()
        result = analyzer.analyze(df)
        elapsed_time = time.perf_counter() - start_time

        # Verificar que se completó en menos de 1 segundo
        self.assertLess(
            elapsed_time,
            1.0,
            f"Analysis took {elapsed_time:.3f}s, expected < 1.0s for 10K x 100 DataFrame",
        )

        # Verificar correctitud básica
        self.assertEqual(result["shape"], (n_rows, n_cols))
        self.assertGreater(result["null_percentage"], 15.0)
        self.assertLess(result["null_percentage"], 25.0)  # ~20% esperado

    def test_performance_many_columns(self):
        """Debe manejar DataFrames con muchas columnas eficientemente."""
        # DataFrame con 500 columnas
        n_cols = 500
        df = pd.DataFrame({f"col_{i}": [1, None, 3, 4, 5] for i in range(n_cols)})

        analyzer = NullAnalyzer()

        start_time = time.perf_counter()
        result = analyzer.analyze(df)
        elapsed_time = time.perf_counter() - start_time

        # Debe completar rápidamente incluso con muchas columnas
        self.assertLess(
            elapsed_time, 0.5, f"Analysis took {elapsed_time:.3f}s, expected < 0.5s for 500 columns"
        )

        self.assertEqual(len(result["columns_analysis"]), n_cols)


class TestNullAnalyzerEdgeCases(unittest.TestCase):
    """Tests de casos extremos."""

    def setUp(self):
        """Set up test environment."""
        self.analyzer = NullAnalyzer()

    def test_empty_dataframe(self):
        """Debe manejar DataFrame vacío."""
        df = pd.DataFrame()

        result = self.analyzer.analyze(df)

        self.assertEqual(result["null_values"], 0)
        self.assertEqual(result["null_percentage"], 0.0)
        self.assertEqual(result["row_analysis"]["rows_with_nulls"], 0)
        self.assertEqual(result["row_analysis"]["complete_rows"], 0)

    def test_single_column_dataframe(self):
        """Debe manejar DataFrame de una columna."""
        df = pd.DataFrame({"col1": [1, None, 3, None, 5]})

        result = self.analyzer.analyze(df)

        self.assertEqual(result["null_values"], 2)
        self.assertEqual(result["null_percentage"], 40.0)
        self.assertEqual(len(result["columns_analysis"]), 1)

    def test_single_row_dataframe(self):
        """Debe manejar DataFrame de una fila."""
        df = pd.DataFrame({"col1": [1], "col2": [None], "col3": [3]})

        result = self.analyzer.analyze(df)

        self.assertEqual(result["null_values"], 1)
        self.assertAlmostEqual(result["null_percentage"], 33.33, places=2)
        self.assertEqual(result["row_analysis"]["rows_with_nulls"], 1)

    def test_mixed_null_types(self):
        """Debe manejar diferentes tipos de valores nulos."""
        df = pd.DataFrame(
            {
                "col1": [None, 1, 2],
                "col2": [np.nan, 2, 3],
                "col3": [pd.NA, 3, 4],
                "col4": [float("nan"), 4, 5],
            }
        )

        result = self.analyzer.analyze(df)

        # Todos deben ser detectados como nulos
        for col in ["col1", "col2", "col3", "col4"]:
            col_analysis = result["columns_analysis"][col]
            self.assertEqual(col_analysis["null_count"], 1, f"{col} should have 1 null value")


if __name__ == "__main__":
    unittest.main()
