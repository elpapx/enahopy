"""
ENAHOPY - Data Validation Utilities
===================================

Utilidades centralizadas para validación de datos, decoradores y helpers
que previenen código duplicado y aseguran validaciones consistentes.

Author: ENAHOPY Team
Date: 2024-12-09
"""

from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

import pandas as pd

from .exceptions import ENAHOValidationError

# Type variables para decoradores genéricos
F = TypeVar("F", bound=Callable[..., Any])


def validate_dataframe_not_empty(
    df: pd.DataFrame, name: str = "DataFrame", error_code: str = "EMPTY_DATAFRAME"
) -> None:
    """
    Valida que un DataFrame no sea None ni esté vacío.

    Args:
        df: DataFrame a validar.
        name: Nombre descriptivo del DataFrame para mensajes de error.
        error_code: Código de error personalizado.

    Raises:
        ENAHOValidationError: Si el DataFrame es None o está vacío.

    Example:
        >>> import pandas as pd
        >>> from enahopy.validation import validate_dataframe_not_empty
        >>> df = pd.DataFrame()
        >>> validate_dataframe_not_empty(df)
        Traceback (most recent call last):
        ...
        ENAHOValidationError: DataFrame está vacío (0 registros)
    """
    if df is None:
        raise ENAHOValidationError(
            f"{name} es None",
            error_code=error_code,
            operation="validate_dataframe",
            dataframe_name=name,
        )

    if df.empty:
        raise ENAHOValidationError(
            f"{name} está vacío (0 registros)",
            error_code=error_code,
            operation="validate_dataframe",
            dataframe_name=name,
            shape=str(df.shape),
        )


def validate_columns_exist(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    df_name: str = "DataFrame",
    error_code: str = "MISSING_COLUMNS",
) -> None:
    """
    Valida que las columnas especificadas existan en el DataFrame.

    Args:
        df: DataFrame a validar.
        columns: Columna o lista de columnas requeridas.
        df_name: Nombre del DataFrame para mensajes.
        error_code: Código de error personalizado.

    Raises:
        ENAHOValidationError: Si alguna columna no existe.

    Example:
        >>> import pandas as pd
        >>> from enahopy.validation import validate_columns_exist
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> validate_columns_exist(df, ["a", "c"])
        Traceback (most recent call last):
        ...
        ENAHOValidationError: Columnas faltantes en DataFrame: ['c']
    """
    if isinstance(columns, str):
        columns = [columns]

    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        available_cols = list(df.columns)
        raise ENAHOValidationError(
            f"Columnas faltantes en {df_name}: {missing_columns}",
            error_code=error_code,
            operation="validate_columns",
            missing_columns=missing_columns,
            available_columns=available_cols[:10],  # Limitar para no saturar
            dataframe_name=df_name,
        )


def validate_column_type(
    df: pd.DataFrame,
    column: str,
    expected_dtype: Union[type, str, List[Union[type, str]]],
    df_name: str = "DataFrame",
    error_code: str = "INVALID_COLUMN_TYPE",
) -> None:
    """
    Valida que una columna tenga el tipo de dato esperado.

    Args:
        df: DataFrame a validar.
        column: Nombre de la columna.
        expected_dtype: Tipo(s) esperado(s) (ej: 'object', int, [int, float]).
        df_name: Nombre del DataFrame.
        error_code: Código de error.

    Raises:
        ENAHOValidationError: Si el tipo no coincide.

    Example:
        >>> import pandas as pd
        >>> from enahopy.validation import validate_column_type
        >>> df = pd.DataFrame({"age": ["25", "30"]})
        >>> validate_column_type(df, "age", int)
        Traceback (most recent call last):
        ...
        ENAHOValidationError: Columna 'age' en DataFrame tiene tipo object...
    """
    if column not in df.columns:
        raise ENAHOValidationError(
            f"Columna '{column}' no existe en {df_name}",
            error_code="COLUMN_NOT_FOUND",
            operation="validate_column_type",
        )

    actual_dtype = df[column].dtype

    # Normalizar tipos esperados a lista
    if not isinstance(expected_dtype, list):
        expected_dtype = [expected_dtype]

    # Convertir tipos a strings para comparación flexible
    expected_strs = [str(t) if isinstance(t, type) else t for t in expected_dtype]
    actual_str = str(actual_dtype)

    # Verificar coincidencia
    is_valid = any(
        expected in actual_str or actual_str.startswith(expected) for expected in expected_strs
    )

    if not is_valid:
        raise ENAHOValidationError(
            f"Columna '{column}' en {df_name} tiene tipo {actual_dtype}, "
            f"esperado: {', '.join(map(str, expected_dtype))}",
            error_code=error_code,
            operation="validate_column_type",
            column=column,
            actual_type=str(actual_dtype),
            expected_types=expected_strs,
        )


def validate_dataframe_shape(
    df: pd.DataFrame,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    min_cols: Optional[int] = None,
    max_cols: Optional[int] = None,
    df_name: str = "DataFrame",
) -> None:
    """
    Valida que el DataFrame cumpla con restricciones de dimensiones.

    Args:
        df: DataFrame a validar.
        min_rows: Número mínimo de filas requeridas.
        max_rows: Número máximo de filas permitidas.
        min_cols: Número mínimo de columnas requeridas.
        max_cols: Número máximo de columnas permitidas.
        df_name: Nombre del DataFrame.

    Raises:
        ENAHOValidationError: Si no cumple con las dimensiones.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2]})
        >>> validate_dataframe_shape(df, min_rows=5)
        Traceback (most recent call last):
        ...
        ENAHOValidationError: DataFrame tiene 2 filas, mínimo requerido: 5
    """
    rows, cols = df.shape

    if min_rows is not None and rows < min_rows:
        raise ENAHOValidationError(
            f"{df_name} tiene {rows} filas, mínimo requerido: {min_rows}",
            error_code="INSUFFICIENT_ROWS",
            operation="validate_shape",
            actual_rows=rows,
            min_rows=min_rows,
        )

    if max_rows is not None and rows > max_rows:
        raise ENAHOValidationError(
            f"{df_name} tiene {rows} filas, máximo permitido: {max_rows}",
            error_code="EXCESSIVE_ROWS",
            operation="validate_shape",
            actual_rows=rows,
            max_rows=max_rows,
        )

    if min_cols is not None and cols < min_cols:
        raise ENAHOValidationError(
            f"{df_name} tiene {cols} columnas, mínimo requerido: {min_cols}",
            error_code="INSUFFICIENT_COLUMNS",
            operation="validate_shape",
            actual_cols=cols,
            min_cols=min_cols,
        )

    if max_cols is not None and cols > max_cols:
        raise ENAHOValidationError(
            f"{df_name} tiene {cols} columnas, máximo permitido: {max_cols}",
            error_code="EXCESSIVE_COLUMNS",
            operation="validate_shape",
            actual_cols=cols,
            max_cols=max_cols,
        )


# =====================================================
# DECORADORES DE VALIDACIÓN
# =====================================================


def require_dataframe(arg_name: str = "d", allow_empty: bool = False) -> Callable[[F], F]:
    """
    Decorador que valida automáticamente un argumento DataFrame.

    Args:
        arg_name: Nombre del argumento DataFrame a validar.
        allow_empty: Si permitir DataFrames vacíos.

    Returns:
        Función decorada con validación automática.

    Example:
        >>> import pandas as pd
        >>> @require_dataframe("data", allow_empty=False)
        ... def process_data(data: pd.DataFrame) -> int:
        ...     return len(data)
        >>>
        >>> process_data(pd.DataFrame({"a": [1, 2]}))
        2
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener el DataFrame del argumento
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if arg_name not in bound_args.arguments:
                raise ValueError(f"Argumento '{arg_name}' no encontrado en función {func.__name__}")

            df = bound_args.arguments[arg_name]

            # Validar
            if not allow_empty:
                validate_dataframe_not_empty(df, name=arg_name)
            elif df is None:
                raise ENAHOValidationError(
                    f"Argumento '{arg_name}' no puede ser None", error_code="NULL_DATAFRAME"
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_columns(*columns: str) -> Callable[[F], F]:
    """
    Decorador que valida que el DataFrame tenga columnas específicas.

    Args:
        *columns: Nombres de columnas requeridas.

    Returns:
        Función decorada con validación de columnas.

    Example:
        >>> import pandas as pd
        >>> @require_columns("ubigeo", "departamento")
        ... def analyze_geography(df: pd.DataFrame) -> dict:
        ...     return {"departments": df["departamento"].nunique()}
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Asumir primer argumento es DataFrame
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)

            # Buscar primer argumento tipo DataFrame
            df = None
            for param_name, param_value in bound_args.arguments.items():
                if isinstance(param_value, pd.DataFrame):
                    df = param_value
                    break

            if df is None:
                raise ValueError(
                    "Decorador @require_columns no encontró DataFrame "
                    f"en función {func.__name__}"
                )

            # Validar columnas
            validate_columns_exist(df, list(columns), df_name="DataFrame")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Funciones de validación
    "validate_dataframe_not_empty",
    "validate_columns_exist",
    "validate_column_type",
    "validate_dataframe_shape",
    # Decoradores
    "require_dataframe",
    "require_columns",
]
