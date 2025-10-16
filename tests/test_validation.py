"""
Tests Unitarios para Módulo de Validación
==========================================

Tests comprehensivos para enahopy.validation
"""

import numpy as np
import pandas as pd
import pytest

from enahopy.exceptions import ENAHOValidationError
from enahopy.validation import (
    require_columns,
    require_dataframe,
    validate_column_type,
    validate_columns_exist,
    validate_dataframe_not_empty,
    validate_dataframe_shape,
)

# =====================================================
# Tests para validate_dataframe_not_empty
# =====================================================


class TestValidateDataFrameNotEmpty:
    """Tests para validate_dataframe_not_empty()"""

    def test_valid_dataframe(self):
        """Debe pasar con DataFrame válido"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        # No debe lanzar excepción
        validate_dataframe_not_empty(df)

    def test_none_dataframe(self):
        """Debe fallar con DataFrame None"""
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_not_empty(None)

        assert "es None" in str(exc_info.value)
        assert exc_info.value.error_code == "EMPTY_DATAFRAME"

    def test_empty_dataframe(self):
        """Debe fallar con DataFrame vacío"""
        df = pd.DataFrame()
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_not_empty(df)

        assert "vacío" in str(exc_info.value)
        assert "0 registros" in str(exc_info.value)

    def test_custom_name(self):
        """Debe usar nombre personalizado en mensaje de error"""
        df = pd.DataFrame()
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_not_empty(df, name="Mi Dataset")

        assert "Mi Dataset" in str(exc_info.value)

    def test_custom_error_code(self):
        """Debe usar código de error personalizado"""
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_not_empty(None, error_code="CUSTOM_ERROR")

        assert exc_info.value.error_code == "CUSTOM_ERROR"


# =====================================================
# Tests para validate_columns_exist
# =====================================================


class TestValidateColumnsExist:
    """Tests para validate_columns_exist()"""

    def test_single_column_exists(self):
        """Debe pasar cuando columna existe"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        validate_columns_exist(df, "a")

    def test_multiple_columns_exist(self):
        """Debe pasar cuando todas las columnas existen"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        validate_columns_exist(df, ["a", "b", "c"])

    def test_single_column_missing(self):
        """Debe fallar cuando columna falta"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_columns_exist(df, "missing_column")

        assert "missing_column" in str(exc_info.value)
        assert exc_info.value.error_code == "MISSING_COLUMNS"

    def test_multiple_columns_missing(self):
        """Debe fallar y listar todas las columnas faltantes"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_columns_exist(df, ["a", "b", "c"])

        error_dict = exc_info.value.to_dict()
        assert set(error_dict["context"]["missing_columns"]) == {"b", "c"}

    def test_string_converted_to_list(self):
        """Debe manejar string como lista de un elemento"""
        df = pd.DataFrame({"col1": [1, 2]})
        validate_columns_exist(df, "col1")  # No debe fallar


# =====================================================
# Tests para validate_column_type
# =====================================================


class TestValidateColumnType:
    """Tests para validate_column_type()"""

    def test_numeric_type_valid(self):
        """Debe pasar con tipo numérico correcto"""
        df = pd.DataFrame({"age": [25, 30, 35]})
        validate_column_type(df, "age", "int")

    def test_object_type_valid(self):
        """Debe pasar con tipo object correcto"""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        validate_column_type(df, "name", "object")

    def test_multiple_expected_types(self):
        """Debe pasar si tipo coincide con alguno de la lista"""
        df = pd.DataFrame({"value": [1.5, 2.5, 3.5]})
        validate_column_type(df, "value", [int, float, "float64"])

    def test_type_mismatch(self):
        """Debe fallar con tipo incorrecto"""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_column_type(df, "text", int)

        assert "tiene tipo object" in str(exc_info.value)
        assert "esperado" in str(exc_info.value)

    def test_column_not_exists(self):
        """Debe fallar si columna no existe"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_column_type(df, "missing", int)

        assert exc_info.value.error_code == "COLUMN_NOT_FOUND"


# =====================================================
# Tests para validate_dataframe_shape
# =====================================================


class TestValidateDataFrameShape:
    """Tests para validate_dataframe_shape()"""

    def test_min_rows_valid(self):
        """Debe pasar cuando cumple mínimo de filas"""
        df = pd.DataFrame({"a": range(10)})
        validate_dataframe_shape(df, min_rows=5)

    def test_min_rows_invalid(self):
        """Debe fallar cuando no cumple mínimo de filas"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_shape(df, min_rows=5)

        assert "tiene 2 filas" in str(exc_info.value)
        assert "mínimo requerido: 5" in str(exc_info.value)

    def test_max_rows_valid(self):
        """Debe pasar cuando cumple máximo de filas"""
        df = pd.DataFrame({"a": range(5)})
        validate_dataframe_shape(df, max_rows=10)

    def test_max_rows_invalid(self):
        """Debe fallar cuando excede máximo de filas"""
        df = pd.DataFrame({"a": range(20)})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_shape(df, max_rows=10)

        assert "tiene 20 filas" in str(exc_info.value)
        assert "máximo permitido: 10" in str(exc_info.value)

    def test_min_cols_valid(self):
        """Debe pasar cuando cumple mínimo de columnas"""
        df = pd.DataFrame({f"col{i}": [1, 2] for i in range(5)})
        validate_dataframe_shape(df, min_cols=3)

    def test_min_cols_invalid(self):
        """Debe fallar cuando no cumple mínimo de columnas"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_dataframe_shape(df, min_cols=5)

        assert "tiene 1 columnas" in str(exc_info.value)

    def test_combined_constraints(self):
        """Debe validar múltiples restricciones simultáneamente"""
        df = pd.DataFrame({f"col{i}": range(10) for i in range(3)})
        validate_dataframe_shape(df, min_rows=5, max_rows=15, min_cols=2, max_cols=5)


# =====================================================
# Tests para decorador @require_dataframe
# =====================================================


class TestRequireDataFrameDecorator:
    """Tests para decorador @require_dataframe"""

    def test_valid_dataframe_arg(self):
        """Debe pasar con DataFrame válido"""

        @require_dataframe("data")
        def process(data: pd.DataFrame) -> int:
            return len(data)

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = process(df)
        assert result == 3

    def test_empty_dataframe_not_allowed(self):
        """Debe fallar con DataFrame vacío si allow_empty=False"""

        @require_dataframe("data", allow_empty=False)
        def process(data: pd.DataFrame) -> int:
            return len(data)

        with pytest.raises(ENAHOValidationError):
            process(pd.DataFrame())

    def test_empty_dataframe_allowed(self):
        """Debe pasar con DataFrame vacío si allow_empty=True"""

        @require_dataframe("data", allow_empty=True)
        def process(data: pd.DataFrame) -> int:
            return len(data)

        result = process(pd.DataFrame())
        assert result == 0

    def test_none_dataframe_fails(self):
        """Debe fallar con None incluso si allow_empty=True"""

        @require_dataframe("data", allow_empty=True)
        def process(data: pd.DataFrame) -> int:
            return 0

        with pytest.raises(ENAHOValidationError) as exc_info:
            process(None)

        assert "no puede ser None" in str(exc_info.value)


# =====================================================
# Tests para decorador @require_columns
# =====================================================


class TestRequireColumnsDecorator:
    """Tests para decorador @require_columns"""

    def test_columns_exist(self):
        """Debe pasar cuando columnas existen"""

        @require_columns("ubigeo", "departamento")
        def analyze(df: pd.DataFrame) -> int:
            return df["ubigeo"].nunique()

        df = pd.DataFrame({"ubigeo": [1, 2, 3], "departamento": ["A", "B", "C"]})
        result = analyze(df)
        assert result == 3

    def test_columns_missing(self):
        """Debe fallar cuando falta alguna columna"""

        @require_columns("ubigeo", "provincia")
        def analyze(df: pd.DataFrame) -> dict:
            return {}

        df = pd.DataFrame({"ubigeo": [1, 2, 3]})
        with pytest.raises(ENAHOValidationError):
            analyze(df)

    def test_no_dataframe_arg(self):
        """Debe fallar si no encuentra DataFrame en argumentos"""

        @require_columns("col1")
        def process(x: int, y: str) -> int:
            return x

        with pytest.raises(ValueError) as exc_info:
            process(1, "test")

        assert "no encontró DataFrame" in str(exc_info.value)


# =====================================================
# Tests de integración
# =====================================================


class TestValidationIntegration:
    """Tests de integración combinando validadores"""

    def test_complete_validation_pipeline(self):
        """Debe validar DataFrame completo con múltiples checks"""
        df = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "010201"],
                "ingreso": [1000, 1500, 2000],
                "edad": [25, 30, 35],
            }
        )

        # Validación completa
        validate_dataframe_not_empty(df, "Datos ENAHO")
        validate_columns_exist(df, ["ubigeo", "ingreso", "edad"])
        validate_column_type(df, "ubigeo", "object")
        validate_column_type(df, "ingreso", ["int", "float"])
        validate_dataframe_shape(df, min_rows=1, max_rows=1000, min_cols=3)

    def test_validation_with_error_context(self):
        """Debe proporcionar contexto rico en errores"""
        df = pd.DataFrame({"a": [1, 2]})

        with pytest.raises(ENAHOValidationError) as exc_info:
            validate_columns_exist(df, ["b", "c", "d"])

        error_dict = exc_info.value.to_dict()

        assert error_dict["is_enahopy_exception"] is True
        assert "missing_columns" in error_dict["context"]
        assert "available_columns" in error_dict["context"]
        assert error_dict["operation"] == "validate_columns"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
