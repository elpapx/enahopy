# null_analysis/utils/validators.py
"""Validadores específicos para análisis de nulos"""

from typing import Any, Dict, List

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Valida que el DataFrame sea válido para análisis"""
    if df.empty:
        return False
    if df.shape[0] == 0 or df.shape[1] == 0:
        return False
    return True


def validate_column_exists(df: pd.DataFrame, column: str) -> bool:
    """Valida que una columna exista en el DataFrame"""
    return column in df.columns


def validate_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de columnas numéricas"""
    return df.select_dtypes(include=["number"]).columns.tolist()
