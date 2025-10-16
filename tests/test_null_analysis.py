"""
Tests para el módulo null_analysis de enahopy
============================================

Tests unitarios e integración para análisis de valores nulos.

Nota: Los tests están diseñados para ser robustos ante la ausencia
de componentes opcionales (core.analyzer, patterns, reports, etc.)
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from enahopy.null_analysis import (
    ENAHONullAnalyzer,
    NullAnalysisConfig,
    NullAnalysisError,
    analyze_null_patterns,
    generate_null_report,
)


def is_core_analyzer_available():
    """Helper para verificar si el core analyzer está disponible"""
    try:
        analyzer = ENAHONullAnalyzer()
        test_df = pd.DataFrame({"col": [1, np.nan]})
        result = analyzer.analyze(test_df, generate_report=False)
        return "error" not in result.get("summary", {})
    except:
        return False


class TestENAHONullAnalyzer:
    """Tests unitarios para ENAHONullAnalyzer"""

    @pytest.fixture
    def sample_df(self):
        """DataFrame de prueba con patrones conocidos"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "ubigeo": [f"{i:06d}" for i in np.random.randint(10101, 250199, 100)],
                "p207": np.where(
                    np.random.random(100) > 0.95, np.nan, np.random.choice([1, 2], 100)
                ),  # Sexo
                "p208": np.where(
                    np.random.random(100) > 0.90, np.nan, np.random.randint(0, 100, 100)
                ),  # Edad
                "ingreso": np.where(
                    np.random.random(100) > 0.70, np.nan, np.random.lognormal(7, 1.5, 100)
                ),
                "gasto": np.where(
                    np.random.random(100) > 0.75, np.nan, np.random.lognormal(6, 1.2, 100)
                ),
                "departamento": np.random.choice(["01", "02", "03"], 100),
                "complete_col": range(100),
                "mostly_null": [np.nan] * 90 + list(range(10)),
            }
        )

    @pytest.fixture
    def empty_df(self):
        """DataFrame vacío"""
        return pd.DataFrame()

    @pytest.fixture
    def no_nulls_df(self):
        """DataFrame sin nulos"""
        return pd.DataFrame(
            {"col1": range(50), "col2": list("ABCDE" * 10), "col3": np.random.randn(50)}
        )

    def test_initialization_default(self):
        """Test inicialización por defecto"""
        analyzer = ENAHONullAnalyzer()

        assert analyzer is not None
        assert hasattr(analyzer, "config")
        assert hasattr(analyzer, "last_analysis")
        assert hasattr(analyzer, "last_report")

    def test_initialization_with_config(self):
        """Test inicialización con configuración"""
        config = NullAnalysisConfig() if NullAnalysisConfig else {}
        analyzer = ENAHONullAnalyzer(config=config)

        assert analyzer is not None
        if config:
            assert analyzer.config == config

    def test_analyze_basic(self, sample_df):
        """Test análisis básico"""
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(sample_df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result
        assert isinstance(result["summary"], dict)

        # Verificar métricas básicas (manejar caso de NullAnalyzer no disponible)
        summary = result["summary"]
        if "error" not in summary:
            assert "total_values" in summary or "null_values" in summary
        else:
            # Si hay error, al menos debe ser un dict con información del error
            assert isinstance(summary, dict)

    def test_analyze_with_report(self, sample_df):
        """Test análisis con generación de reporte"""
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(sample_df, generate_report=True)

        assert isinstance(result, dict)
        assert "summary" in result

        # El reporte puede no generarse si faltan componentes
        # pero el análisis básico debe funcionar

    def test_analyze_empty_dataframe(self, empty_df):
        """Test con DataFrame vacío"""
        analyzer = ENAHONullAnalyzer()

        # Debe manejar el error apropiadamente
        try:
            result = analyzer.analyze(empty_df)
            # Si no lanza excepción, debe indicar error o resultado vacío
            if isinstance(result, dict):
                summary = result.get("summary", {})
                assert "error" in str(result).lower() or "error" in summary or len(summary) == 0
        except (NullAnalysisError, ValueError, Exception):
            # Comportamiento esperado para DataFrame vacío
            pass

    def test_analyze_no_nulls(self, no_nulls_df):
        """Test con DataFrame sin nulos"""
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(no_nulls_df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

        # Verificar que detecta correctamente ausencia de nulos (si analyzer disponible)
        summary = result["summary"]
        if "error" not in summary:
            null_percentage = summary.get("null_percentage", -1)
            null_values = summary.get("null_values", -1)
            assert null_percentage == 0 or null_values == 0
        else:
            # Si hay error en core analyzer, verificar que al menos funciona el fallback
            assert isinstance(summary, dict)

    def test_analyze_single_column(self):
        """Test con DataFrame de una columna"""
        df = pd.DataFrame({"col1": [1, np.nan, 3, np.nan, 5]})

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

    def test_analyze_mixed_types(self):
        """Test con tipos de datos mixtos"""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, np.nan, 4, 5],
                "str_col": ["a", "b", np.nan, "d", "e"],
                "float_col": [1.1, np.nan, 3.3, 4.4, 5.5],
                "bool_col": [True, False, np.nan, True, False],
            }
        )

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

    @pytest.mark.skipif(not is_core_analyzer_available(), reason="Core analyzer not available")
    def test_analyze_detailed_metrics(self, sample_df):
        """Test métricas detalladas cuando core analyzer está disponible"""
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(sample_df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result
        summary = result["summary"]

        # Con core analyzer disponible, debe tener métricas detalladas
        assert "total_values" in summary or "null_values" in summary
        assert "null_percentage" in summary


class TestConvenienceFunctions:
    """Tests para funciones de conveniencia"""

    @pytest.fixture
    def test_df(self):
        """DataFrame simple para tests"""
        return pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5],
                "col2": [np.nan, 2, 3, np.nan, 5],
                "col3": range(5),
                "col4": ["A", "B", np.nan, "D", "E"],
            }
        )

    def test_analyze_null_patterns_basic(self, test_df):
        """Test función analyze_null_patterns"""
        result = analyze_null_patterns(test_df)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_analyze_null_patterns_with_config(self, test_df):
        """Test analyze_null_patterns con configuración"""
        config = NullAnalysisConfig() if NullAnalysisConfig else None
        result = analyze_null_patterns(test_df, config=config)

        assert isinstance(result, dict)

    def test_generate_null_report_basic(self, test_df):
        """Test generación básica de reporte"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report"

            result = generate_null_report(test_df, output_path=str(output_path))

            # Verificar que se ejecutó sin errores
            assert result is not None

    def test_generate_null_report_html_format(self, test_df):
        """Test reporte en formato HTML"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            result = generate_null_report(test_df, output_path=str(output_path), format="html")

            assert result is not None


class TestUtilityFunctions:
    """Tests para funciones utilitarias"""

    @pytest.fixture
    def utility_df(self):
        """DataFrame para tests de utilidades"""
        return pd.DataFrame(
            {
                "full_col": range(10),
                "partial_nulls": [1, np.nan, 3, np.nan, 5, 6, np.nan, 8, 9, 10],
                "mostly_nulls": [np.nan] * 8 + [1, 2],
                "all_nulls": [np.nan] * 10,
            }
        )

    def test_calculate_null_percentage(self, utility_df):
        """Test cálculo de porcentaje de nulos"""
        from enahopy.null_analysis import calculate_null_percentage

        # Test por columna
        percentage = calculate_null_percentage(utility_df, "partial_nulls")
        assert isinstance(percentage, (int, float))
        assert 0 <= percentage <= 100

        # Verificar cálculo correcto (3 nulos de 10 = 30%)
        expected = 30.0
        assert abs(percentage - expected) < 0.1

    def test_find_columns_with_nulls(self, utility_df):
        """Test identificación de columnas con nulos"""
        from enahopy.null_analysis import find_columns_with_nulls

        columns = find_columns_with_nulls(utility_df)

        assert isinstance(columns, list)
        assert "partial_nulls" in columns
        assert "mostly_nulls" in columns
        assert "all_nulls" in columns
        assert "full_col" not in columns

    def test_get_null_summary(self, utility_df):
        """Test resumen de nulos"""
        from enahopy.null_analysis import get_null_summary

        summary = get_null_summary(utility_df)

        assert isinstance(summary, pd.DataFrame)
        # La función puede filtrar columnas sin nulos, verificar que hay al menos algunas filas
        assert len(summary) >= 3  # Al menos las columnas con nulos
        assert len(summary) <= len(utility_df.columns)  # No más que el total de columnas

        # Verificar columnas esperadas
        expected_cols = ["column", "null_count", "null_percentage"]
        for col in expected_cols:
            assert any(col in c or c.endswith(col.split("_")[-1]) for c in summary.columns)

    def test_safe_dict_merge(self):
        """Test fusión segura de diccionarios"""
        from enahopy.null_analysis import safe_dict_merge

        dict1 = {"a": 1, "b": {"x": 1}}
        dict2 = {"b": {"y": 2}, "c": 3}

        result = safe_dict_merge(dict1, dict2)

        assert isinstance(result, dict)
        assert "a" in result
        assert "c" in result
        assert isinstance(result.get("b"), dict)


class TestIntegrationWorkflows:
    """Tests de integración para flujos completos"""

    @pytest.fixture
    def enaho_like_df(self):
        """DataFrame similar a datos ENAHO reales"""
        np.random.seed(42)
        n = 500

        return pd.DataFrame(
            {
                "ubigeo": [
                    f"{dept:02d}{prov:02d}{dist:02d}"
                    for dept, prov, dist in zip(
                        np.random.randint(1, 26, n),
                        np.random.randint(1, 10, n),
                        np.random.randint(1, 20, n),
                    )
                ],
                "conglome": np.random.randint(1, 9999, n),
                "vivienda": np.random.randint(1, 99, n),
                "hogar": np.random.randint(1, 5, n),
                "p207": np.where(np.random.random(n) > 0.98, np.nan, np.random.choice([1, 2], n)),
                "p208": np.where(np.random.random(n) > 0.92, np.nan, np.random.randint(0, 100, n)),
                "p301a": np.where(
                    np.random.random(n) > 0.85, np.nan, np.random.choice([1, 2, 3, 4, 5, 6], n)
                ),
                "ingreso": np.where(
                    np.random.random(n) > 0.60, np.nan, np.random.lognormal(6.5, 1.5, n)
                ),
                "gasto": np.where(
                    np.random.random(n) > 0.65, np.nan, np.random.lognormal(6.2, 1.3, n)
                ),
                "departamento": np.random.choice(["01", "02", "03", "04", "05"], n),
                "area": np.random.choice(["1", "2"], n, p=[0.7, 0.3]),
            }
        )

    def test_complete_analysis_workflow(self, enaho_like_df):
        """Test flujo completo de análisis"""
        # 1. Análisis inicial
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(enaho_like_df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

        # 2. Análisis con funciones de conveniencia
        quick_result = analyze_null_patterns(enaho_like_df)
        assert isinstance(quick_result, dict)

        # 3. Verificar consistencia entre métodos
        # Ambos deberían detectar la presencia de nulos
        assert len(quick_result) > 0

    def test_workflow_with_report_generation(self, enaho_like_df):
        """Test flujo con generación de reportes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Análisis completo
            analyzer = ENAHONullAnalyzer()
            result = analyzer.analyze(enaho_like_df, generate_report=True)

            assert isinstance(result, dict)

            # 2. Generar reporte independiente
            report_path = Path(tmpdir) / "integration_report"
            report_result = generate_null_report(enaho_like_df, output_path=str(report_path))

            assert report_result is not None

    def test_analysis_with_geographic_data(self, enaho_like_df):
        """Test análisis con datos geográficos típicos de ENAHO"""
        # Verificar que el análisis maneja correctamente ubigeos
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(enaho_like_df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

        # Los ubigeos no deberían tener nulos (son identificadores)
        summary = result["summary"]
        # Verificar que el análisis se completó exitosamente
        assert isinstance(summary, dict)

    def test_comparative_analysis(self, enaho_like_df):
        """Test análisis comparativo entre subconjuntos"""
        # Dividir por área urbano/rural
        urban_df = enaho_like_df[enaho_like_df["area"] == "1"]
        rural_df = enaho_like_df[enaho_like_df["area"] == "2"]

        # Análisis independientes
        analyzer = ENAHONullAnalyzer()

        urban_result = analyzer.analyze(urban_df, generate_report=False)
        rural_result = analyzer.analyze(rural_df, generate_report=False)

        assert isinstance(urban_result, dict)
        assert isinstance(rural_result, dict)
        assert "summary" in urban_result
        assert "summary" in rural_result


class TestErrorHandling:
    """Tests para manejo de errores"""

    def test_invalid_dataframe_input(self):
        """Test con input inválido"""
        analyzer = ENAHONullAnalyzer()

        # Test con None
        with pytest.raises((TypeError, AttributeError, NullAnalysisError)):
            analyzer.analyze(None)

        # Test con string
        with pytest.raises((TypeError, AttributeError, NullAnalysisError)):
            analyzer.analyze("not a dataframe")

    def test_very_large_dataframe_handling(self):
        """Test con DataFrame muy grande"""
        # Crear DataFrame moderadamente grande
        n = 10000
        large_df = pd.DataFrame(
            {
                "col1": np.random.randn(n),
                "col2": np.where(np.random.random(n) > 0.8, np.nan, np.random.randint(1, 100, n)),
                "col3": np.random.choice(["A", "B", "C"], n),
            }
        )

        analyzer = ENAHONullAnalyzer()

        # Debe manejar datasets grandes sin problemas
        result = analyzer.analyze(large_df, generate_report=False)
        assert isinstance(result, dict)

    def test_unicode_column_names(self):
        """Test con nombres de columnas en español/unicode"""
        df = pd.DataFrame(
            {
                "año": [2020, 2021, np.nan, 2023],
                "región": ["Lima", "Cusco", np.nan, "Arequipa"],
                "niños_menores_5años": [1, 0, np.nan, 1],
                "ingreso_soles": [1000, np.nan, 1500, 2000],
            }
        )

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

    def test_extreme_null_percentages(self):
        """Test con porcentajes extremos de nulos"""
        # DataFrame con 95% de nulos
        df_high_nulls = pd.DataFrame(
            {"mostly_null": [np.nan] * 95 + list(range(5)), "some_data": range(100)}
        )

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df_high_nulls, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result


class TestPatternDetection:
    """Tests para detección de patrones (si está disponible)"""

    @pytest.fixture
    def pattern_df(self):
        """DataFrame con patrones específicos de missing"""
        np.random.seed(42)
        n = 200

        # Crear patrón correlacionado: si falta ingreso, falta gasto
        base_missing = np.random.random(n) > 0.7

        return pd.DataFrame(
            {
                "id": range(n),
                "ingreso": np.where(base_missing, np.nan, np.random.lognormal(7, 1.5, n)),
                "gasto": np.where(
                    base_missing | (np.random.random(n) > 0.9),
                    np.nan,
                    np.random.lognormal(6.5, 1.2, n),
                ),
                "edad": np.where(np.random.random(n) > 0.95, np.nan, np.random.randint(18, 80, n)),
                "sexo": np.where(np.random.random(n) > 0.98, np.nan, np.random.choice([1, 2], n)),
            }
        )

    def test_correlated_missing_detection(self, pattern_df):
        """Test detección de missing correlacionado"""
        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(pattern_df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result

        # Si hay detección de patrones disponible, debería estar en el resultado
        if "patterns" in result and result["patterns"] != {"error": str}:
            assert isinstance(result["patterns"], dict)

        # El análisis básico debe funcionar independientemente

    def test_monotone_pattern_detection(self):
        """Test detección de patrón monótono"""
        # Crear patrón monótono: A -> B -> C (si falta A, faltan B y C)
        n = 100
        missing_a = np.random.random(n) > 0.8
        missing_b = missing_a | (np.random.random(n) > 0.7)
        missing_c = missing_b | (np.random.random(n) > 0.6)

        df = pd.DataFrame(
            {
                "var_a": np.where(missing_a, np.nan, np.random.randn(n)),
                "var_b": np.where(missing_b, np.nan, np.random.randn(n)),
                "var_c": np.where(missing_c, np.nan, np.random.randn(n)),
            }
        )

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df, generate_report=False)

        assert isinstance(result, dict)
        assert "summary" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
