"""Tipos y enumeraciones para patrones de valores nulos"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class MissingDataPattern(Enum):
    """Tipos de patrones de datos faltantes"""

    MCAR = "Missing Completely At Random"
    MAR = "Missing At Random"
    MNAR = "Missing Not At Random"
    MONOTONE = "Monotone"
    ARBITRARY = "Arbitrary"
    UNKNOWN = "Unknown"


class PatternType(Enum):
    """Tipos específicos de patrones"""

    COMPLETE = "complete"
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    BLOCK = "block"
    MONOTONE = "monotone"
    CORRELATED = "correlated"


class PatternSeverity(Enum):
    """Severidad del patrón de nulos"""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatternResult:
    """Resultado del análisis de patrones"""

    pattern_type: MissingDataPattern
    severity: PatternSeverity
    affected_columns: List[str]
    percentage_missing: float
    confidence: float
    details: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type.value,
            "severity": self.severity.value,
            "affected_columns": self.affected_columns,
            "percentage_missing": self.percentage_missing,
            "confidence": self.confidence,
            "details": self.details,
            "recommendations": self.recommendations,
        }
