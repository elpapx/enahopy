"""
Estrategia de análisis básico de valores nulos
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import MissingDataMetrics, MissingDataPattern, NullAnalysisConfig


class INullAnalysisStrategy(ABC):
    """Interfaz para estrategias de análisis de nulos"""

    def __init__(self, config: NullAnalysisConfig, logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Ejecuta el análisis específico"""
        pass

    @abstractmethod
    def get_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        pass


class BasicNullAnalysis(INullAnalysisStrategy):
    """Análisis básico de valores nulos - rápido y eficiente"""

    def analyze(self, df: pd.DataFrame, group_by: Optional[str] = None) -> Dict[str, Any]:
        """Análisis básico con estadísticas fundamentales"""

        # Estadísticas básicas
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100

        # DataFrame de resumen
        summary_df = pd.DataFrame(
            {
                "variable": missing_count.index,
                "missing_count": missing_count.values,
                "missing_percentage": missing_percentage.values,
                "complete_count": len(df) - missing_count.values,
                "completeness": 100 - missing_percentage.values,
            }
        ).sort_values("missing_percentage", ascending=False)

        # Análisis por grupos si se especifica
        group_analysis = None
        if group_by and group_by in df.columns:
            group_analysis = self._analyze_by_groups(df, group_by)

        # Métricas básicas
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = missing_count.sum()
        complete_cases = df.dropna().shape[0]

        metrics = MissingDataMetrics(
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_percentage=(missing_cells / total_cells) * 100,
            complete_cases=complete_cases,
            complete_cases_percentage=(complete_cases / len(df)) * 100,
            variables_with_missing=(missing_count > 0).sum(),
            variables_without_missing=(missing_count == 0).sum(),
            missing_pattern_count=0,
            most_common_pattern="Not analyzed",
            missing_data_pattern=MissingDataPattern.UNKNOWN,
            data_quality_score=self._calculate_basic_quality_score(missing_percentage),
        )

        return {
            "analysis_type": "basic",
            "summary": summary_df,
            "group_analysis": group_analysis,
            "metrics": metrics,
            "execution_time": 0,
        }

    def _analyze_by_groups(self, df: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """Análisis de nulos por grupos"""
        group_missing = (
            df.groupby(group_by)
            .apply(
                lambda x: pd.Series(
                    {
                        col: (x[col].isnull().sum() / len(x)) * 100
                        for col in x.columns
                        if col != group_by
                    }
                )
            )
            .reset_index()
        )

        group_stats = (
            df.groupby(group_by)
            .agg({df.columns[0]: "count"})
            .rename(columns={df.columns[0]: "group_size"})
        )

        return group_missing.merge(group_stats, on=group_by, how="left")

    def _calculate_basic_quality_score(self, missing_percentages: pd.Series) -> float:
        """Calcula score básico de calidad de datos"""
        weights = np.exp(-missing_percentages / 20)
        return float(np.average(100 - missing_percentages, weights=weights))

    def get_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Recomendaciones básicas"""
        recommendations = []
        summary = analysis_result["summary"]

        high_missing = summary[summary["missing_percentage"] > 50]
        if not high_missing.empty:
            recommendations.append(
                f"⚠️  {len(high_missing)} variables tienen >50% de valores faltantes. "
                f"Considere eliminarlas: {high_missing['variable'].head(3).tolist()}"
            )

        moderate_missing = summary[
            (summary["missing_percentage"] > 10) & (summary["missing_percentage"] <= 50)
        ]
        if not moderate_missing.empty:
            recommendations.append(
                f"🔧 {len(moderate_missing)} variables tienen 10-50% de faltantes. "
                f"Evalúe técnicas de imputación."
            )

        if analysis_result["metrics"].complete_cases_percentage < 20:
            recommendations.append(
                "❌ Menos del 20% de casos completos. Dataset requiere limpieza intensiva."
            )

        return recommendations
