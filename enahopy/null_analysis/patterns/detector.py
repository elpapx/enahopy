"""Detector de patrones de valores nulos"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .types import MissingDataPattern, PatternResult, PatternSeverity


class PatternDetector:
    """Detector principal de patrones de valores nulos"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def detect_pattern(self, df: pd.DataFrame, threshold: float = 0.05) -> PatternResult:
        """Detecta el patrón principal de valores nulos"""
        self.logger.info(f"Analizando patrones en DataFrame de {len(df)} filas")

        null_stats = self._calculate_null_statistics(df)
        pattern_type = self._identify_pattern_type(df, null_stats)
        severity = self._determine_severity(null_stats["overall_percentage"])
        recommendations = self._generate_recommendations(pattern_type, severity)
        affected_cols = [col for col in df.columns if df[col].isnull().any()]

        return PatternResult(
            pattern_type=pattern_type,
            severity=severity,
            affected_columns=affected_cols,
            percentage_missing=null_stats["overall_percentage"],
            confidence=null_stats.get("confidence", 0.8),
            details=null_stats,
            recommendations=recommendations,
        )

    def _calculate_null_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estadísticas de valores nulos"""
        total_values = df.size
        null_values = df.isnull().sum().sum()

        stats = {
            "total_values": total_values,
            "null_values": null_values,
            "overall_percentage": (null_values / total_values) * 100 if total_values > 0 else 0,
            "columns_with_nulls": df.isnull().any().sum(),
            "rows_with_nulls": df.isnull().any(axis=1).sum(),
            "complete_rows": (~df.isnull().any(axis=1)).sum(),
            "complete_columns": (~df.isnull().any()).sum(),
        }

        if len(df.columns) > 1:
            null_corr = df.isnull().astype(int).corr()
            stats["null_correlation_mean"] = null_corr.values[
                np.triu_indices_from(null_corr.values, k=1)
            ].mean()
        else:
            stats["null_correlation_mean"] = 0

        return stats

    def _identify_pattern_type(self, df: pd.DataFrame, stats: Dict[str, Any]) -> MissingDataPattern:
        """Identifica el tipo de patrón de valores nulos"""
        if stats["null_values"] == 0:
            return MissingDataPattern.MCAR

        if self._is_monotone_pattern(df):
            return MissingDataPattern.MONOTONE

        if stats.get("null_correlation_mean", 0) > 0.5:
            return MissingDataPattern.MAR

        if self._test_randomness(df):
            return MissingDataPattern.MCAR

        return MissingDataPattern.MNAR

    def _is_monotone_pattern(self, df: pd.DataFrame) -> bool:
        """Verifica si hay patrón monótono"""
        null_counts = df.isnull().sum(axis=1)
        diffs = null_counts.diff().dropna()

        if len(diffs) == 0:
            return False

        positive_changes = (diffs > 0).sum()
        negative_changes = (diffs < 0).sum()
        total_changes = len(diffs)

        return positive_changes > 0.8 * total_changes or negative_changes > 0.8 * total_changes

    def _test_randomness(self, df: pd.DataFrame, significance: float = 0.05) -> bool:
        """Test de aleatoriedad para valores nulos"""
        null_df = df.isnull().astype(int)

        if len(null_df.columns) < 2:
            return True

        correlations = []
        for col1 in null_df.columns:
            for col2 in null_df.columns:
                if col1 != col2:
                    corr = null_df[col1].corr(null_df[col2])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

        if correlations:
            mean_corr = np.mean(correlations)
            return mean_corr < 0.3

        return True

    def _determine_severity(self, percentage: float) -> PatternSeverity:
        """Determina severidad basada en porcentaje de nulos"""
        if percentage == 0:
            return PatternSeverity.NONE
        elif percentage < 5:
            return PatternSeverity.LOW
        elif percentage < 20:
            return PatternSeverity.MODERATE
        elif percentage < 50:
            return PatternSeverity.HIGH
        else:
            return PatternSeverity.CRITICAL

    def _generate_recommendations(
        self, pattern: MissingDataPattern, severity: PatternSeverity
    ) -> List[str]:
        """Genera recomendaciones basadas en el patrón y severidad"""
        recommendations = []

        if pattern == MissingDataPattern.MCAR:
            recommendations.append("Los datos parecen faltar aleatoriamente")
            recommendations.append("Considere eliminación por lista o imputación simple")
        elif pattern == MissingDataPattern.MAR:
            recommendations.append("Los datos faltan con patrón dependiente de otras variables")
            recommendations.append("Use imputación múltiple o modelos predictivos")
        elif pattern == MissingDataPattern.MNAR:
            recommendations.append("Los datos no faltan aleatoriamente")
            recommendations.append("Investigue las causas de los valores faltantes")
        elif pattern == MissingDataPattern.MONOTONE:
            recommendations.append("Patrón monótono detectado")
            recommendations.append("Considere imputación secuencial")

        if severity == PatternSeverity.CRITICAL:
            recommendations.append("ADVERTENCIA: Alto porcentaje de datos faltantes")
        elif severity == PatternSeverity.HIGH:
            recommendations.append("Porcentaje significativo de datos faltantes")

        return recommendations


class NullPatternAnalyzer:
    """Analizador avanzado de patrones de nulos"""

    def __init__(self, detector: Optional[PatternDetector] = None):
        self.detector = detector or PatternDetector()

    def analyze_patterns(self, df: pd.DataFrame) -> Dict[str, PatternResult]:
        """Analiza patrones por grupos de columnas"""
        results = {}
        results["global"] = self.detector.detect_pattern(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results["numeric"] = self.detector.detect_pattern(df[numeric_cols])

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            results["categorical"] = self.detector.detect_pattern(df[categorical_cols])

        return results
