"""
ENAHO Null Analysis - Utilidades
================================

Funciones auxiliares para análisis de valores nulos.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def safe_dict_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Combina dos diccionarios de forma segura"""
    result = dict1.copy()
    result.update(dict2)
    return result


def calculate_null_percentage(df: pd.DataFrame, column: str = None) -> Union[float, pd.Series]:
    """Calcula porcentaje de valores nulos"""
    if column:
        return (df[column].isnull().sum() / len(df)) * 100
    else:
        return (df.isnull().sum() / len(df)) * 100


def identify_null_patterns(df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, str]:
    """Identifica patrones de valores nulos en columnas"""
    patterns = {}
    null_percentages = calculate_null_percentage(df)

    for col in df.columns:
        null_pct = null_percentages[col]

        if null_pct == 0:
            patterns[col] = "complete"
        elif null_pct < threshold * 100:
            patterns[col] = "low_missing"
        elif null_pct < 30:
            patterns[col] = "moderate_missing"
        elif null_pct < 70:
            patterns[col] = "high_missing"
        else:
            patterns[col] = "mostly_missing"

    return patterns


def get_null_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlación de valores nulos"""
    null_df = df.isnull().astype(int)
    return null_df.corr()


def find_columns_with_nulls(df: pd.DataFrame) -> List[str]:
    """Encuentra columnas que contienen valores nulos"""
    return df.columns[df.isnull().any()].tolist()


def get_null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Genera resumen de valores nulos por columna"""
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "null_count": df.isnull().sum(),
            "null_percentage": (df.isnull().sum() / len(df)) * 100,
            "non_null_count": df.count(),
            "dtype": df.dtypes,
        }
    )

    summary = summary[summary["null_count"] > 0].sort_values("null_percentage", ascending=False)
    return summary


def detect_monotone_pattern(df: pd.DataFrame) -> bool:
    """Detecta si los valores nulos siguen un patrón monótono"""
    null_counts = df.isnull().sum(axis=1)
    diffs = null_counts.diff().dropna()

    if len(diffs) == 0:
        return False

    all_increasing = (diffs >= 0).all()
    all_decreasing = (diffs <= 0).all()

    return all_increasing or all_decreasing


def impute_with_strategy(
    df: pd.DataFrame, strategy: str = "mean", columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Imputa valores nulos con estrategia especificada"""
    df_copy = df.copy()

    if columns is None:
        columns = find_columns_with_nulls(df_copy)

    for col in columns:
        if col not in df_copy.columns:
            continue

        if strategy == "mean" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif strategy == "mode":
            mode_val = df_copy[col].mode()
            if len(mode_val) > 0:
                df_copy[col].fillna(mode_val[0], inplace=True)
        elif strategy == "forward":
            df_copy[col].fillna(method="ffill", inplace=True)
        elif strategy == "backward":
            df_copy[col].fillna(method="bfill", inplace=True)

    return df_copy


__all__ = [
    "safe_dict_merge",
    "calculate_null_percentage",
    "identify_null_patterns",
    "get_null_correlation_matrix",
    "find_columns_with_nulls",
    "get_null_summary",
    "detect_monotone_pattern",
    "impute_with_strategy",
]
