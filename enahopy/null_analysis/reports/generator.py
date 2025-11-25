"""Generador de reportes de análisis de valores nulos"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ..patterns.types import PatternResult
from ..utils import get_null_summary


class NullAnalysisReport:
    """Reporte de análisis de valores nulos"""

    def __init__(self):
        self.timestamp = datetime.now()
        self.summary = {}
        self.details = {}
        self.patterns = {}
        self.recommendations = []
        self.metadata = {}

    def add_summary(self, key: str, value: Any):
        self.summary[key] = value

    def add_detail(self, key: str, value: Any):
        self.details[key] = value

    def add_pattern(self, name: str, pattern: PatternResult):
        self.patterns[name] = pattern

    def add_recommendation(self, recommendation: str):
        self.recommendations.append(recommendation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "details": self.details,
            "patterns": {
                k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.patterns.items()
            },
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


class ReportGenerator:
    """Generador principal de reportes"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def generate_report(
        self,
        df: pd.DataFrame,
        pattern_results: Optional[Dict[str, PatternResult]] = None,
        include_visualizations: bool = False,
    ) -> NullAnalysisReport:
        """Genera reporte completo de análisis de nulos"""
        self.logger.info("Generando reporte de análisis de valores nulos")

        report = NullAnalysisReport()

        report.metadata["rows"] = len(df)
        report.metadata["columns"] = len(df.columns)
        report.metadata["shape"] = df.shape

        total_values = df.size
        null_values = df.isnull().sum().sum()

        report.add_summary("Total de valores", total_values)
        report.add_summary("Valores nulos", null_values)
        report.add_summary("Porcentaje de nulos", f"{(null_values/total_values)*100:.2f}%")

        null_summary = get_null_summary(df)
        report.add_detail("column_summary", null_summary.to_dict("records"))

        if pattern_results:
            for name, pattern in pattern_results.items():
                report.add_pattern(name, pattern)
                if hasattr(pattern, "recommendations"):
                    for rec in pattern.recommendations:
                        report.add_recommendation(rec)

        return report
