"""
Validation Results Module
========================

Dataclasses para resultados de validación.
Contiene estructuras de datos para reportar
resultados de validación de columnas.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ColumnValidationResult:
    """Resultado de la validación de columnas."""

    found_columns: List[str]
    missing_columns: List[str]
    mapped_columns: Dict[str, str]

    def get_summary(self) -> str:
        """Genera resumen textual de la validación"""
        summary = f"Columnas encontradas: {len(self.found_columns)}\n"
        summary += f"Columnas faltantes: {len(self.missing_columns)}\n"

        if self.mapped_columns:
            summary += "Mapeos realizados:\n"
            for original, found in self.mapped_columns.items():
                if original != found:
                    summary += f"  {original} -> {found}\n"

        if self.missing_columns:
            summary += f"Columnas no encontradas: {', '.join(self.missing_columns)}\n"

        return summary


__all__ = ["ColumnValidationResult"]
