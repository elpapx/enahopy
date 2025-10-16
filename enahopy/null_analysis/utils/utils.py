# null_analysis/utils.py

"""

Utilidades y funciones auxiliares para el módulo de análisis de nulos.



Este módulo contiene funciones de apoyo para:

- Fusión segura de estructuras de datos

- Validación de entradas

- Conversión de tipos

- Manejo de casos extremos

"""


import time
import traceback
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from null_analysis.config import AnalysisComplexity, ExportFormat, VisualizationType
from null_analysis.exceptions import NullAnalysisError, ValidationError

# =====================================================

# FUSIÓN SEGURA DE ESTRUCTURAS DE DATOS

# =====================================================


def safe_dict_merge(dict1: Optional[Dict], dict2: Optional[Dict], strategy: str = "update") -> Dict:
    """

    Fusiona diccionarios de forma segura manejando None y estructuras anidadas.



    Args:

        dict1: Primer diccionario (puede ser None)

        dict2: Segundo diccionario (puede ser None)

        strategy: Estrategia de fusión

            - 'update': Actualización simple (dict2 sobrescribe dict1)

            - 'deep_merge': Fusión recursiva de diccionarios anidados

            - 'prefer_non_null': Prefiere valores no nulos

            - 'combine_lists': Combina listas en lugar de reemplazar



    Returns:

        Diccionario fusionado de forma segura



    Examples:

        >>> dict1 = {'a': {'b': 1, 'c': None}, 'd': 2}

        >>> dict2 = {'a': {'b': None, 'c': 3}, 'e': 4}

        >>> result = safe_dict_merge(dict1, dict2, strategy='prefer_non_null')

        >>> # result = {'a': {'b': 1, 'c': 3}, 'd': 2, 'e': 4}

    """

    # Manejar casos None

    if dict1 is None and dict2 is None:
        return {}

    if dict1 is None:
        return dict2.copy() if dict2 else {}

    if dict2 is None:
        return dict1.copy() if dict1 else {}

    result = dict1.copy()

    if strategy == "update":
        result.update(dict2)

    elif strategy == "deep_merge":
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = safe_dict_merge(result[key], value, strategy)

            elif value is not None or key not in result:
                result[key] = value

    elif strategy == "prefer_non_null":
        for key, value in dict2.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = safe_dict_merge(result[key], value, strategy)

                elif value is not None:
                    result[key] = value

            else:
                result[key] = value

    elif strategy == "combine_lists":
        for key, value in dict2.items():
            if key in result:
                if isinstance(result[key], list) and isinstance(value, list):
                    result[key] = result[key] + value

                elif isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = safe_dict_merge(result[key], value, strategy)

                else:
                    result[key] = value

            else:
                result[key] = value

    else:
        raise ValueError(f"Estrategia '{strategy}' no reconocida")

    return result


def merge_analysis_results(
    results: List[Dict[str, Any]], aggregation_strategy: str = "average"
) -> Dict[str, Any]:
    """

    Combina múltiples resultados de análisis en uno consolidado.



    Args:

        results: Lista de resultados de análisis

        aggregation_strategy: 'average', 'max', 'min', 'median'



    Returns:

        Resultado consolidado

    """

    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    # Extraer métricas numéricas para agregación

    numeric_metrics = {}

    for result in results:
        if "metrics" in result and hasattr(result["metrics"], "__dict__"):
            for key, value in result["metrics"].__dict__.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []

                    numeric_metrics[key].append(value)

    # Aplicar estrategia de agregación

    aggregated_metrics = {}

    for key, values in numeric_metrics.items():
        if aggregation_strategy == "average":
            aggregated_metrics[key] = np.mean(values)

        elif aggregation_strategy == "max":
            aggregated_metrics[key] = np.max(values)

        elif aggregation_strategy == "min":
            aggregated_metrics[key] = np.min(values)

        elif aggregation_strategy == "median":
            aggregated_metrics[key] = np.median(values)

    # Combinar otros elementos

    combined_result = {
        "aggregated_metrics": aggregated_metrics,
        "individual_results": results,
        "aggregation_strategy": aggregation_strategy,
        "result_count": len(results),
    }

    return combined_result


# =====================================================

# VALIDACIÓN DE ENTRADAS

# =====================================================


class InputValidator:
    """Validador centralizado para todas las funciones del módulo"""

    @staticmethod
    def validate_dataframe(
        df: Any, min_rows: int = 1, min_cols: int = 1, required_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """

        Valida y convierte entrada a DataFrame.



        Args:

            df: Objeto a validar como DataFrame

            min_rows: Número mínimo de filas requeridas

            min_cols: Número mínimo de columnas requeridas

            required_columns: Lista de columnas que deben estar presentes



        Returns:

            DataFrame validado



        Raises:

            NullAnalysisError: Si la validación falla

        """

        if df is None:
            raise NullAnalysisError("DataFrame no puede ser None")

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)

            except Exception as e:
                raise NullAnalysisError(f"No se pudo convertir a DataFrame: {e}")

        if df.empty:
            raise NullAnalysisError("DataFrame está vacío")

        if len(df) < min_rows:
            raise NullAnalysisError(
                f"DataFrame debe tener al menos {min_rows} filas, tiene {len(df)}"
            )

        if len(df.columns) < min_cols:
            raise NullAnalysisError(
                f"DataFrame debe tener al menos {min_cols} columnas, tiene {len(df.columns)}"
            )

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)

            if missing_cols:
                raise NullAnalysisError(f"Columnas requeridas faltantes: {', '.join(missing_cols)}")

        return df

    @staticmethod
    def validate_group_by(df: pd.DataFrame, group_by: Optional[str]) -> Optional[str]:
        """

        Valida columna de agrupación.



        Args:

            df: DataFrame de referencia

            group_by: Nombre de columna para agrupar



        Returns:

            Nombre de columna validado o None



        Raises:

            ValueError: Si la columna no existe

        """

        if group_by is None:
            return None

        if group_by not in df.columns:
            available_cols = ", ".join(df.columns[:10])

            if len(df.columns) > 10:
                available_cols += f", ... ({len(df.columns)} columnas en total)"

            raise ValueError(
                f"Columna '{group_by}' no existe en el DataFrame. "
                f"Columnas disponibles: {available_cols}"
            )

        return group_by

    @staticmethod
    def validate_geographic_filter(
        filter_dict: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        """

        Valida filtros geográficos.



        Args:

            filter_dict: Diccionario con filtros geográficos



        Returns:

            Filtros validados o None

        """

        if filter_dict is None:
            return None

        valid_keys = {"departamento", "provincia", "distrito", "region", "ubigeo"}

        invalid_keys = set(filter_dict.keys()) - valid_keys

        if invalid_keys:
            warnings.warn(
                f"Claves de filtro geográfico no reconocidas: {invalid_keys}. "
                f"Se utilizarán solo: {valid_keys}"
            )

        # Filtrar solo claves válidas

        validated = {k: v for k, v in filter_dict.items() if k in valid_keys}

        return validated if validated else None


# =====================================================

# CONVERSIÓN SEGURA DE TIPOS

# =====================================================


def safe_enum_conversion(
    value: Union[str, Any], enum_class: type, default: Optional[Any] = None
) -> Any:
    """

    Convierte valores a enum de forma segura.



    Args:

        value: Valor a convertir

        enum_class: Clase de enum objetivo

        default: Valor por defecto si la conversión falla



    Returns:

        Valor de enum o default



    Examples:

        >>> complexity = safe_enum_conversion('advanced', AnalysisComplexity, AnalysisComplexity.STANDARD)

    """

    if value is None:
        return default

    # Si ya es del tipo correcto

    if isinstance(value, enum_class):
        return value

    # Intentar conversión

    try:
        # Intentar por valor

        return enum_class(value)

    except (ValueError, KeyError):
        try:
            # Intentar por nombre (case-insensitive)

            value_upper = str(value).upper()

            for member in enum_class:
                if member.name == value_upper:
                    return member

            # Intentar por valor del enum (case-insensitive)

            for member in enum_class:
                if member.value.upper() == value_upper:
                    return member

        except Exception:
            pass

    #

    if default is not None:
        warnings.warn(
            f"No se pudo convertir '{value}' a {enum_class.__name__}. "
            f"Usando valor por defecto: {default}"
        )

    return default


def validate_column_types(
    df: pd.DataFrame, expected_types: Dict[str, type], coerce: bool = False
) -> Dict[str, Any]:
    """

    Valida y opcionalmente convierte tipos de columnas.



    Args:

        df: DataFrame a validar

        expected_types: Diccionario {columna: tipo_esperado}

        coerce: Si True, intenta convertir los tipos



    Returns:

        Diccionario con resultados de validación

    """

    validation = {
        "valid": True,
        "mismatches": {},
        "conversions_applied": [],
        "conversion_errors": [],
    }

    type_mapping = {
        int: ["int64", "int32", "int16", "int8"],
        float: ["float64", "float32", "float16"],
        str: ["object", "string"],
        bool: ["bool"],
        "numeric": ["int64", "int32", "int16", "int8", "float64", "float32", "float16"],
    }

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            validation["mismatches"][col] = "Columna no existe"

            validation["valid"] = False

            continue

        actual_dtype = str(df[col].dtype)

        # Verificar si el tipo actual es compatible

        if expected_type == "numeric":
            is_valid = actual_dtype in type_mapping["numeric"]

        elif expected_type in type_mapping:
            is_valid = actual_dtype in type_mapping[expected_type]

        else:
            is_valid = False

        if not is_valid:
            validation["valid"] = False

            validation["mismatches"][col] = {"expected": str(expected_type), "actual": actual_dtype}

            if coerce:
                try:
                    if expected_type == int:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

                    elif expected_type == float:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    elif expected_type == str:
                        df[col] = df[col].astype(str)

                    elif expected_type == bool:
                        df[col] = df[col].astype(bool)

                    validation["conversions_applied"].append(col)

                    validation["valid"] = True  # Revalidar después de conversión

                except Exception as e:
                    validation["conversion_errors"].append({"column": col, "error": str(e)})

    return validation


# =====================================================

# CONTEXT MANAGERS Y DECORADORES

# =====================================================


@contextmanager
def null_analysis_context(operation_name: str, logger=None, raise_on_error: bool = True):
    """

    Context manager para operaciones de análisis de nulos.



    Args:

        operation_name: Nombre de la operación

        logger: Logger opcional

        raise_on_error: Si False, captura errores sin relanzar



    Yields:

        Diccionario de contexto con información de la operación

    """

    context = {
        "operation": operation_name,
        "start_time": time.time(),
        "success": False,
        "error": None,
    }

    try:
        if logger:
            logger.info(f"Iniciando: {operation_name}")

        yield context

        context["success"] = True

    except NullAnalysisError:
        context["error"] = traceback.format_exc()

        if raise_on_error:
            raise

    except Exception as e:
        context["error"] = traceback.format_exc()

        error_msg = f"Error en {operation_name}: {str(e)}"

        if logger:
            logger.error(error_msg)

            logger.debug(context["error"])

        if raise_on_error:
            raise NullAnalysisError(error_msg) from e

    finally:
        context["elapsed_time"] = time.time() - context["start_time"]

        if logger:
            status = "exitosamente" if context["success"] else "con errores"

            logger.info(
                f"Completado {status}: {operation_name} " f"({context['elapsed_time']:.2f}s)"
            )


def validate_inputs(func):
    """

    Decorador para validar automáticamente entradas de funciones.



    Valida el primer argumento como DataFrame si es pd.DataFrame en hints.

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Obtener hints de tipo de la función

        import inspect

        sig = inspect.signature(func)

        params = sig.parameters

        # Validar DataFrame si es el primer parámetro

        if args and len(params) > 0:
            first_param = list(params.values())[0]

            if first_param.annotation == pd.DataFrame:
                validator = InputValidator()

                args = list(args)

                args[0] = validator.validate_dataframe(args[0])

                args = tuple(args)

        return func(*args, **kwargs)

    return wrapper


# =====================================================

# UTILIDADES DE CÁLCULO Y ESTADÍSTICA

# =====================================================


def safe_percentage(numerator: float, denominator: float, decimals: int = 2) -> float:
    """

    Calcula porcentaje de forma segura evitando división por cero.



    Args:

        numerator: Numerador

        denominator: Denominador

        decimals: Número de decimales



    Returns:

        Porcentaje calculado o 0.0 si denominador es 0

    """

    if denominator == 0 or pd.isna(denominator):
        return 0.0

    percentage = (numerator / denominator) * 100

    return round(percentage, decimals)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """

    División segura con valor por defecto.



    Args:

        numerator: Numerador

        denominator: Denominador

        default: Valor por defecto si división no es posible



    Returns:

        Resultado de la división o valor por defecto

    """

    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default

    return numerator / denominator


def calculate_missing_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """

    Calcula estadísticas básicas de valores faltantes.



    Args:

        df: DataFrame a analizar



    Returns:

        Diccionario con estadísticas

    """

    total_cells = df.shape[0] * df.shape[1]

    missing_cells = df.isnull().sum().sum()

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": safe_percentage(missing_cells, total_cells),
        "complete_rows": len(df.dropna()),
        "complete_rows_percentage": safe_percentage(len(df.dropna()), len(df)),
        "columns_with_missing": df.isnull().any().sum(),
        "columns_all_missing": df.isnull().all().sum(),
    }


# =====================================================

# UTILIDADES DE FORMATO Y PRESENTACIÓN

# =====================================================


def format_percentage(value: float, decimals: int = 1, include_sign: bool = True) -> str:
    """

    Formatea un valor como porcentaje.



    Args:

        value: Valor a formatear

        decimals: Número de decimales

        include_sign: Si incluir el símbolo %



    Returns:

        String formateado

    """

    if pd.isna(value):
        return "N/A"

    formatted = f"{value:.{decimals}f}"

    if include_sign:
        formatted += "%"

    return formatted


def format_large_number(value: Union[int, float], use_abbreviation: bool = True) -> str:
    """

    Formatea números grandes de forma legible.



    Args:

        value: Número a formatear

        use_abbreviation: Si usar abreviaciones (K, M, B)



    Returns:

        String formateado



    Examples:

        >>> format_large_number(1500000)

        '1.5M'

        >>> format_large_number(1500000, use_abbreviation=False)

        '1,500,000'

    """

    if pd.isna(value):
        return "N/A"

    if use_abbreviation:
        if abs(value) >= 1e9:
            return f"{value / 1e9:.1f}B"

        elif abs(value) >= 1e6:
            return f"{value / 1e6:.1f}M"

        elif abs(value) >= 1e3:
            return f"{value / 1e3:.1f}K"

    return f"{int(value):,}"


# =====================================================

# EXPORTACIÓN DE UTILIDADES

# =====================================================


__all__ = [
    # Fusión de estructuras
    "safe_dict_merge",
    "merge_analysis_results",
    # Validación
    "InputValidator",
    "validate_column_types",
    # Conversión de tipos
    "safe_enum_conversion",
    # Context managers y decoradores
    "null_analysis_context",
    "validate_inputs",
    # Cálculos seguros
    "safe_percentage",
    "safe_division",
    "calculate_missing_stats",
    # Formato
    "format_percentage",
    "format_large_number",
]
