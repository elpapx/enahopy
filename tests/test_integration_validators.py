"""
Tests de Integración - Validadores y Flujos Completos
=====================================================

Tests que verifican la integración entre validadores,
analizadores y funciones de conveniencia.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from enahopy.exceptions import ENAHOValidationError
from enahopy.null_analysis import ENAHONullAnalyzer, quick_null_analysis
from enahopy.null_analysis.config import AnalysisComplexity, NullAnalysisConfig
from enahopy.validation import (
    require_columns,
    require_dataframe,
    validate_column_type,
    validate_columns_exist,
    validate_dataframe_not_empty,
    validate_dataframe_shape,
)


class TestValidatorIntegration(unittest.TestCase):
    """Tests de integración para validadores."""

    def test_full_validation_pipeline_success(self):
        """Debe ejecutar pipeline completo de validación exitosamente."""
        df = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "010201"],
                "ingreso": [1000.5, 1500.0, 2000.0],
                "edad": [25, 30, 35],
                "departamento": ["AMAZONAS", "AMAZONAS", "AMAZONAS"],
            }
        )

        # Pipeline de validación
        validate_dataframe_not_empty(df, "Datos ENAHO")
        validate_columns_exist(df, ["ubigeo", "ingreso", "edad", "departamento"])
        validate_column_type(df, "ubigeo", "object")
        validate_column_type(df, "ingreso", ["float", "int"])
        validate_column_type(df, "edad", "int")
        validate_dataframe_shape(df, min_rows=1, max_rows=1000, min_cols=3)

        # Si llegó aquí, todas las validaciones pasaron
        self.assertTrue(True)

    def test_validation_pipeline_fails_on_missing_columns(self):
        """Debe fallar cuando faltan columnas requeridas."""
        df = pd.DataFrame({"ubigeo": ["010101"], "ingreso": [1000]})

        with self.assertRaises(ENAHOValidationError) as context:
            validate_columns_exist(df, ["ubigeo", "ingreso", "edad", "departamento"])

        error_dict = context.exception.to_dict()
        missing = set(error_dict["context"]["missing_columns"])
        self.assertEqual(missing, {"edad", "departamento"})

    def test_validation_pipeline_fails_on_wrong_type(self):
        """Debe fallar cuando tipos de datos son incorrectos."""
        df = pd.DataFrame(
            {
                "ubigeo": [1, 2, 3],  # Debería ser string
                "ingreso": ["mil", "dos mil", "tres mil"],  # Debería ser numeric
            }
        )

        with self.assertRaises(ENAHOValidationError):
            validate_column_type(df, "ubigeo", "object")  # int != object en pandas

        with self.assertRaises(ENAHOValidationError):
            validate_column_type(df, "ingreso", ["int", "float"])

    def test_validation_with_decorators(self):
        """Debe validar usando decoradores en funciones."""

        @require_dataframe("data", allow_empty=False)
        @require_columns("ubigeo", "ingreso")
        def analyze_income(data: pd.DataFrame) -> float:
            return data["ingreso"].mean()

        # Caso exitoso
        df = pd.DataFrame({"ubigeo": ["010101", "010102"], "ingreso": [1000, 2000]})

        result = analyze_income(df)
        self.assertEqual(result, 1500.0)

        # Falla con DataFrame vacío
        with self.assertRaises(ENAHOValidationError):
            analyze_income(pd.DataFrame())

        # Falla con columnas faltantes
        df_incomplete = pd.DataFrame({"ubigeo": ["010101"]})
        with self.assertRaises(ENAHOValidationError):
            analyze_income(df_incomplete)


class TestNullAnalyzerIntegration(unittest.TestCase):
    """Tests de integración para analizador de nulos."""

    def test_analyzer_with_validated_input(self):
        """Debe analizar correctamente datos validados."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, None, 4, 5],
                "col2": [None, "b", "c", None, "e"],
                "col3": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

        # Validar primero
        validate_dataframe_not_empty(df, "Test Data")
        validate_dataframe_shape(df, min_rows=1)

        # Luego analizar
        config = NullAnalysisConfig(complexity_level=AnalysisComplexity.STANDARD)
        analyzer = ENAHONullAnalyzer(config=config, verbose=False)

        result = analyzer.analyze_null_patterns(df)

        # Verificar resultado
        self.assertIn("metrics", result)
        self.assertGreater(result["metrics"].missing_percentage, 0)
        self.assertEqual(result["metrics"].total_rows, 5)
        self.assertEqual(result["metrics"].total_columns, 3)

    def test_quick_analysis_convenience_function(self):
        """Debe funcionar función de conveniencia quick_null_analysis."""
        df = pd.DataFrame(
            {
                "departamento": ["LIMA", "LIMA", "CUSCO", "CUSCO"],
                "ingreso": [1000, None, 1500, None],
            }
        )

        result = quick_null_analysis(df, group_by="departamento", complexity="standard")

        self.assertIn("metrics", result)
        self.assertIn("summary", result)
        # 2 nulls out of 8 total cells (4 rows × 2 cols) = 25%
        self.assertEqual(result["metrics"].missing_percentage, 25.0)

    def test_analyzer_with_geographic_grouping(self):
        """Debe agrupar correctamente por variables geográficas."""
        df = pd.DataFrame(
            {
                "departamento": ["LIMA", "LIMA", "CUSCO", "CUSCO", "AREQUIPA"],
                "provincia": ["LIMA", "CALLAO", "CUSCO", "CUSCO", "AREQUIPA"],
                "ingreso": [1000, None, 1500, None, 2000],
                "edad": [25, 30, None, 35, 40],
            }
        )

        config = NullAnalysisConfig(complexity_level=AnalysisComplexity.STANDARD)
        analyzer = ENAHONullAnalyzer(config=config, verbose=False)

        result = analyzer.analyze_null_patterns(df, group_by="departamento")

        # Debe tener análisis agrupado
        self.assertIn("group_analysis", result)
        group_df = result["group_analysis"]

        self.assertIsInstance(group_df, pd.DataFrame)
        self.assertIn("departamento", group_df.columns)

        # LIMA: 1 null in ingreso
        # CUSCO: 1 null in ingreso, 1 null in edad
        # AREQUIPA: 0 nulls


class TestEndToEndWorkflows(unittest.TestCase):
    """Tests de flujos end-to-end completos."""

    def test_complete_enaho_analysis_workflow(self):
        """Debe ejecutar flujo completo de análisis ENAHO."""
        # Simular datos ENAHO
        df = pd.DataFrame(
            {
                "conglome": ["001", "001", "002", "002", "003"],
                "vivienda": ["01", "01", "01", "01", "01"],
                "hogar": ["1", "2", "1", "2", "1"],
                "ubigeo": ["010101", "010102", "020101", "020102", "030101"],
                "departamento": ["AMAZONAS", "AMAZONAS", "ANCASH", "ANCASH", "APURIMAC"],
                "p208a": [1000, None, 1500, None, 2000],  # Ingreso principal
                "p209": [500, 300, None, 400, None],  # Ingreso secundario
                "gashog2d": [800, 700, 900, None, 1000],  # Gasto hogar
            }
        )

        # 1. Validación de estructura
        validate_dataframe_not_empty(df, "Datos ENAHO")
        validate_columns_exist(df, ["conglome", "vivienda", "hogar", "ubigeo"])
        validate_column_type(df, "ubigeo", "object")
        validate_dataframe_shape(df, min_rows=1, min_cols=5)

        # 2. Análisis de valores nulos
        config = NullAnalysisConfig(complexity_level=AnalysisComplexity.STANDARD)
        analyzer = ENAHONullAnalyzer(config=config, verbose=False)

        result = analyzer.analyze_null_patterns(df, group_by="departamento")

        # 3. Verificar resultados
        self.assertIn("metrics", result)
        self.assertGreater(result["metrics"].missing_percentage, 0)

        # 4. Obtener recomendaciones
        recommendations = analyzer.get_imputation_recommendations(result)
        self.assertIsInstance(recommendations, dict)

    def test_workflow_with_report_generation(self):
        """Debe generar reporte completo del análisis."""
        df = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "010201"],
                "departamento": ["AMAZONAS", "AMAZONAS", "AMAZONAS"],
                "ingreso": [1000, None, 2000],
                "gasto": [800, 900, None],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report"

            # Generar reporte
            config = NullAnalysisConfig(
                complexity_level=AnalysisComplexity.ADVANCED,
                export_formats=[],  # No exportar archivos en test
            )
            analyzer = ENAHONullAnalyzer(config=config, verbose=False)

            result = analyzer.generate_comprehensive_report(
                df=df, output_path=str(output_path), group_by="departamento"
            )

            # Verificar estructura del resultado
            self.assertIn("report_metadata", result)
            self.assertIn("analysis_results", result)
            self.assertIn("metrics", result["analysis_results"])

    def test_error_handling_invalid_input(self):
        """Debe manejar correctamente inputs inválidos."""
        # None DataFrame
        with self.assertRaises(ENAHOValidationError) as context:
            validate_dataframe_not_empty(None, "Test")

        self.assertIn("None", str(context.exception))

        # DataFrame vacío
        with self.assertRaises(ENAHOValidationError) as context:
            validate_dataframe_not_empty(pd.DataFrame(), "Test", error_code="CUSTOM_ERROR")

        self.assertEqual(context.exception.error_code, "CUSTOM_ERROR")

        # Columnas faltantes
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with self.assertRaises(ENAHOValidationError) as context:
            validate_columns_exist(df, ["col1", "col2", "col3"])

        error_dict = context.exception.to_dict()
        self.assertEqual(set(error_dict["context"]["missing_columns"]), {"col2", "col3"})

    def test_data_quality_score_integration(self):
        """Debe calcular score de calidad integrado."""
        # DataFrame de alta calidad
        df_high_quality = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.5, 2.5, 3.5, 4.5, 5.5],
            }
        )

        config = NullAnalysisConfig(complexity_level=AnalysisComplexity.STANDARD)
        analyzer = ENAHONullAnalyzer(config=config, verbose=False)

        score_high = analyzer.get_data_quality_score(df_high_quality, detailed=False)
        self.assertEqual(score_high, 100.0)  # Sin nulos = 100%

        # DataFrame de baja calidad
        df_low_quality = pd.DataFrame(
            {
                "col1": [None, None, None, 4, 5],
                "col2": [None, None, "c", None, "e"],
                "col3": [None, 2.5, None, None, 5.5],
            }
        )

        score_low = analyzer.get_data_quality_score(df_low_quality, detailed=False)
        self.assertLess(score_low, 50.0)  # Muchos nulos = score bajo

        # Score detallado
        score_detailed = analyzer.get_data_quality_score(df_low_quality, detailed=True)
        self.assertIsInstance(score_detailed, dict)
        self.assertIn("overall_score", score_detailed)
        self.assertIn("completeness_score", score_detailed)


class TestValidatorWithRealScenarios(unittest.TestCase):
    """Tests con escenarios reales de datos ENAHO."""

    def test_ubigeo_validation(self):
        """Debe validar correctamente códigos UBIGEO."""
        df = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "020101"],
                "departamento": ["AMAZONAS", "AMAZONAS", "ANCASH"],
            }
        )

        # Validar existencia y tipo
        validate_columns_exist(df, ["ubigeo", "departamento"])
        validate_column_type(df, "ubigeo", "object")

        # Validar longitud de UBIGEO (6 dígitos)
        @require_columns("ubigeo")
        def validate_ubigeo_format(df: pd.DataFrame) -> bool:
            return df["ubigeo"].str.len().eq(6).all()

        self.assertTrue(validate_ubigeo_format(df))

    def test_income_validation(self):
        """Debe validar correctamente variables de ingreso."""
        df = pd.DataFrame(
            {
                "p208a": [1000.5, 1500.0, 2000.0],  # Ingreso principal
                "p209": [500.0, 300.0, 400.0],  # Ingreso secundario
            }
        )

        # Validar tipos numéricos
        validate_column_type(df, "p208a", ["int", "float"])
        validate_column_type(df, "p209", ["int", "float"])

        # Validar rango de valores
        @require_dataframe("data")
        def validate_positive_income(data: pd.DataFrame) -> bool:
            return (data["p208a"] >= 0).all() and (data["p209"] >= 0).all()

        self.assertTrue(validate_positive_income(df))

    def test_household_structure_validation(self):
        """Debe validar estructura de hogar correctamente."""
        df = pd.DataFrame(
            {
                "conglome": ["001", "001", "002"],
                "vivienda": ["01", "01", "01"],
                "hogar": ["1", "2", "1"],
                "codperso": ["1", "1", "1"],
            }
        )

        # Validar columnas de estructura
        required_cols = ["conglome", "vivienda", "hogar", "codperso"]
        validate_columns_exist(df, required_cols)

        # Validar tipos
        for col in required_cols:
            validate_column_type(df, col, "object")


if __name__ == "__main__":
    unittest.main()
