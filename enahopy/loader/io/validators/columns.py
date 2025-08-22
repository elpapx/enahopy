"""
Column Validation Module
=======================

Validador de columnas con mapeo flexible.
Soporte para validación case-sensitive/insensitive
y mapeo automático de nombres similares.
"""

import logging
from typing import Dict, List, Optional

from .results import ColumnValidationResult


class ColumnValidator:
    """Validador de columnas con mapeo flexible"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate_columns(
        self,
        requested_columns: List[str],
        available_columns: List[str],
        case_sensitive: bool = False,
    ) -> ColumnValidationResult:
        """
        Valida columnas solicitadas contra las disponibles

        Args:
            requested_columns: Columnas que se quieren leer
            available_columns: Columnas disponibles en el archivo
            case_sensitive: Si la comparación es sensible a mayúsculas

        Returns:
            Resultado de validación con mapeos
        """
        found_columns, missing_columns, mapped_columns = [], [], {}

        # Crear mapeo para búsqueda eficiente
        if case_sensitive:
            available_map = {col: col for col in available_columns}
        else:
            available_map = {col.lower(): col for col in available_columns}

        for requested_col in requested_columns:
            found_col = self._find_column(
                requested_col, available_columns, available_map, case_sensitive
            )

            if found_col:
                found_columns.append(found_col)
                mapped_columns[requested_col] = found_col
            else:
                missing_columns.append(requested_col)

        return ColumnValidationResult(found_columns, missing_columns, mapped_columns)

    def _find_column(
        self,
        requested_col: str,
        available_columns: List[str],
        available_map: Dict[str, str],
        case_sensitive: bool,
    ) -> Optional[str]:
        """Busca una columna con diferentes estrategias"""

        # Búsqueda exacta
        if requested_col in available_columns:
            return requested_col

        # Búsqueda case-insensitive
        if not case_sensitive and requested_col.lower() in available_map:
            return available_map[requested_col.lower()]

        return None


__all__ = ["ColumnValidator"]
