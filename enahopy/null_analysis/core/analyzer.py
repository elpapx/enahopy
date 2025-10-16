"""
ENAHO Null Analyzer - Core Analyzer
===================================

Analizador principal de valores nulos.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class NullAnalysisConfig:
    """Configuración para análisis de valores nulos"""

    threshold_low: float = 0.05
    threshold_moderate: float = 0.20
    threshold_high: float = 0.50
    threshold_critical: float = 0.70
    generate_report: bool = True
    include_visualizations: bool = False
    verbose: bool = True


class NullAnalyzer:
    """
    Analizador principal de valores nulos para datos ENAHO

    Esta clase proporciona funcionalidad básica de análisis de valores nulos,
    incluyendo estadísticas, detección de patrones y generación de reportes.
    """

    def __init__(
        self, config: Optional[NullAnalysisConfig] = None, logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa el analizador

        Args:
            config: Configuración del análisis
            logger: Logger opcional para mensajes
        """
        self.config = config or NullAnalysisConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza análisis básico de valores nulos con operaciones vectorizadas.

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con estadísticas de valores nulos

        Note:
            Optimizado usando operaciones vectorizadas de pandas para mejor
            performance en DataFrames grandes (especialmente con muchas columnas).
        """
        self.logger.info(f"Analizando DataFrame de forma {df.shape}")

        results = {
            "shape": df.shape,
            "total_values": df.size,
            "null_values": 0,
            "null_percentage": 0.0,
            "columns_analysis": {},
            "row_analysis": {},
            "patterns": {},
            "severity": "none",
        }

        # Cálculos vectorizados para todas las columnas de una vez
        null_counts = df.isnull().sum()  # Serie con conteos por columna
        total_rows = len(df)

        # Calcular porcentaje global usando la serie
        total_nulls = null_counts.sum()
        results["null_values"] = int(total_nulls)

        if results["total_values"] > 0:
            results["null_percentage"] = float(total_nulls / results["total_values"] * 100)

        # Calcular porcentajes vectorizadamente
        null_percentages = (
            (null_counts / total_rows * 100) if total_rows > 0 else pd.Series(0, index=df.columns)
        )
        non_null_counts = total_rows - null_counts

        # Obtener tipos una sola vez
        dtypes = df.dtypes.astype(str)

        # Construir análisis por columnas usando valores pre-calculados
        for col in df.columns:
            null_pct = null_percentages[col]

            results["columns_analysis"][col] = {
                "null_count": int(null_counts[col]),
                "null_percentage": float(null_pct),
                "non_null_count": int(non_null_counts[col]),
                "dtype": dtypes[col],
                "severity": self._classify_severity(null_pct),
            }

        # Análisis por filas (vectorizado)
        null_mask = df.isnull().any(axis=1)
        rows_with_nulls = null_mask.sum()
        complete_rows = (~null_mask).sum()

        results["row_analysis"] = {
            "rows_with_nulls": int(rows_with_nulls),
            "complete_rows": int(complete_rows),
            "percentage_incomplete": (
                float(rows_with_nulls / total_rows * 100) if total_rows > 0 else 0.0
            ),
        }

        # Determinar severidad global
        results["severity"] = self._classify_severity(results["null_percentage"])

        # Detectar patrones básicos
        results["patterns"] = self._detect_basic_patterns(df)

        return results

    def _classify_severity(self, percentage: float) -> str:
        """
        Clasifica la severidad según el porcentaje de nulos

        Args:
            percentage: Porcentaje de valores nulos

        Returns:
            Clasificación de severidad
        """
        if percentage == 0:
            return "none"
        elif percentage < self.config.threshold_low * 100:
            return "low"
        elif percentage < self.config.threshold_moderate * 100:
            return "moderate"
        elif percentage < self.config.threshold_high * 100:
            return "high"
        elif percentage < self.config.threshold_critical * 100:
            return "critical"
        else:
            return "extreme"

    def _detect_basic_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta patrones básicos en valores nulos

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con patrones detectados
        """
        patterns = {
            "has_complete_columns": False,
            "has_empty_columns": False,
            "has_monotone_pattern": False,
            "correlation_detected": False,
            "pattern_type": "unknown",
        }

        # Verificar columnas completas
        complete_cols = df.columns[~df.isnull().any()].tolist()
        patterns["has_complete_columns"] = len(complete_cols) > 0
        patterns["complete_columns"] = complete_cols

        # Verificar columnas vacías
        empty_cols = df.columns[df.isnull().all()].tolist()
        patterns["has_empty_columns"] = len(empty_cols) > 0
        patterns["empty_columns"] = empty_cols

        # Detectar patrón monótono (simplificado)
        null_counts_by_row = df.isnull().sum(axis=1)
        if len(null_counts_by_row) > 1:
            diffs = null_counts_by_row.diff().dropna()
            if len(diffs) > 0:
                all_increasing = (diffs >= 0).all()
                all_decreasing = (diffs <= 0).all()
                patterns["has_monotone_pattern"] = all_increasing or all_decreasing

        # Detectar correlación en nulos (simplificado)
        if len(df.columns) > 1:
            null_matrix = df.isnull().astype(int)
            corr_matrix = null_matrix.corr()

            # Verificar si hay alta correlación
            upper_triangle = np.triu(corr_matrix.values, k=1)
            high_corr = np.any(np.abs(upper_triangle) > 0.7)
            patterns["correlation_detected"] = bool(high_corr)

        # Determinar tipo de patrón
        if patterns["has_empty_columns"]:
            patterns["pattern_type"] = "structural"
        elif patterns["has_monotone_pattern"]:
            patterns["pattern_type"] = "monotone"
        elif patterns["correlation_detected"]:
            patterns["pattern_type"] = "correlated"
        elif not patterns["has_complete_columns"]:
            patterns["pattern_type"] = "random"
        else:
            patterns["pattern_type"] = "mixed"

        return patterns

    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera estadísticas resumidas de valores nulos

        Args:
            df: DataFrame a analizar

        Returns:
            DataFrame con resumen estadístico
        """
        summary_data = []

        for col in df.columns:
            null_count = df[col].isnull().sum()
            total_count = len(df[col])

            summary_data.append(
                {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "null_count": null_count,
                    "non_null_count": total_count - null_count,
                    "null_percentage": (null_count / total_count * 100) if total_count > 0 else 0,
                    "unique_values": df[col].nunique(),
                    "has_nulls": null_count > 0,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("null_percentage", ascending=False)

        return summary_df

    def identify_columns_for_imputation(
        self, df: pd.DataFrame, max_null_percentage: float = 50.0
    ) -> List[str]:
        """
        Identifica columnas candidatas para imputación

        Args:
            df: DataFrame a analizar
            max_null_percentage: Porcentaje máximo de nulos para considerar imputación

        Returns:
            Lista de nombres de columnas candidatas
        """
        candidates = []

        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100

            # Candidata si tiene nulos pero no demasiados
            if 0 < null_pct <= max_null_percentage:
                candidates.append(col)

        return candidates

    def impute_advanced(
        self,
        df: pd.DataFrame,
        strategy: str = "auto",
        categorical_cols: Optional[List[str]] = None,
        compare_strategies: bool = False,
        assess_quality: bool = True,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Advanced imputation with ML strategies.

        Args:
            df: DataFrame with missing values
            strategy: Imputation strategy ('auto', 'mice', 'knn', 'missforest', 'random_forest')
            categorical_cols: List of categorical column names
            compare_strategies: Whether to compare multiple strategies
            assess_quality: Whether to assess imputation quality
            **kwargs: Additional arguments passed to imputation methods

        Returns:
            Tuple of (imputed_df, quality_report)
        """
        try:
            from ..strategies.imputation_quality_assessment import (
                QualityAssessmentConfig,
                assess_imputation_quality,
            )
            from ..strategies.ml_imputation import create_ml_imputation_manager, quick_ml_imputation
        except ImportError:
            self.logger.error("ML imputation modules not available")
            raise ImportError(
                "Advanced ML imputation requires scikit-learn. "
                "Install with: pip install scikit-learn scipy"
            )

        self.logger.info(f"Starting advanced imputation with strategy: {strategy}")

        quality_report = {
            "strategy_used": strategy,
            "comparison_results": None,
            "quality_assessment": None,
        }

        # Compare strategies if requested
        if compare_strategies:
            self.logger.info("Comparing multiple imputation strategies...")
            manager = create_ml_imputation_manager()
            comparison_results = manager.compare_strategies(df, test_size=0.2)
            quality_report["comparison_results"] = comparison_results

            # Select best strategy if auto
            if strategy == "auto":
                best_strategy = manager.get_best_strategy(comparison_results)
                strategy = best_strategy or "knn"
                self.logger.info(f"Auto-selected strategy: {strategy}")
                quality_report["strategy_used"] = strategy

        # Perform imputation
        if strategy == "auto" and not compare_strategies:
            # Quick auto mode without comparison
            imputed_df = quick_ml_imputation(df, strategy="auto")
        else:
            # Use specified strategy
            manager = create_ml_imputation_manager()
            manager.fit_strategy(strategy, df)
            imputed_df = manager.impute(strategy, df)

        # Assess quality if requested
        if assess_quality:
            self.logger.info("Assessing imputation quality...")
            config = QualityAssessmentConfig()
            missing_mask = df.isnull()

            quality_result = assess_imputation_quality(
                original_df=df,
                imputed_df=imputed_df,
                missing_mask=missing_mask,
                categorical_cols=categorical_cols,
                config=config,
            )

            quality_report["quality_assessment"] = {
                "overall_score": quality_result.overall_score,
                "metric_scores": quality_result.metric_scores,
                "recommendations": quality_result.recommendations,
            }

            self.logger.info(f"Imputation quality score: {quality_result.overall_score:.2f}")

        return imputed_df, quality_report

    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones basadas en el análisis

        Args:
            analysis_results: Resultados del análisis

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        severity = analysis_results.get("severity", "unknown")
        null_pct = analysis_results.get("null_percentage", 0)

        # Recomendaciones por severidad
        if severity == "none":
            recommendations.append("✅ No se detectaron valores nulos. Los datos están completos.")
        elif severity == "low":
            recommendations.append(
                "✓ Porcentaje bajo de valores nulos. Considere imputación simple."
            )
            recommendations.append(
                "✓ Los valores faltantes probablemente no afectarán significativamente el análisis."
            )
        elif severity == "moderate":
            recommendations.append("⚠️ Porcentaje moderado de valores nulos detectado.")
            recommendations.append(
                "⚠️ Considere técnicas de imputación múltiple o modelos predictivos."
            )
            recommendations.append("⚠️ Investigue las causas de los valores faltantes.")
        elif severity in ["high", "critical", "extreme"]:
            recommendations.append(f"❌ ADVERTENCIA: {null_pct:.1f}% de valores nulos detectados.")
            recommendations.append("❌ El análisis puede estar severamente comprometido.")
            recommendations.append("❌ Considere excluir variables con exceso de valores faltantes.")
            recommendations.append("❌ Revise la calidad de la fuente de datos.")

        # Recomendaciones por patrones
        patterns = analysis_results.get("patterns", {})

        if patterns.get("has_empty_columns"):
            empty_cols = patterns.get("empty_columns", [])
            recommendations.append(f"⚠️ Columnas completamente vacías detectadas: {empty_cols[:5]}")
            recommendations.append("⚠️ Considere eliminar estas columnas del análisis.")

        if patterns.get("has_monotone_pattern"):
            recommendations.append("📊 Patrón monótono detectado en valores nulos.")
            recommendations.append("📊 Considere imputación secuencial o análisis longitudinal.")

        if patterns.get("correlation_detected"):
            recommendations.append("🔍 Correlación detectada entre patrones de valores nulos.")
            recommendations.append("🔍 Los valores pueden no faltar aleatoriamente (MAR/MNAR).")
            recommendations.append(
                "🔍 Use técnicas avanzadas de imputación que consideren dependencias."
            )

        return recommendations


# Funciones auxiliares
def create_null_analyzer(config: Optional[NullAnalysisConfig] = None) -> NullAnalyzer:
    """
    Función de fábrica para crear un NullAnalyzer

    Args:
        config: Configuración opcional

    Returns:
        Instancia de NullAnalyzer
    """
    return NullAnalyzer(config)


def quick_null_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza un análisis rápido de valores nulos

    Args:
        df: DataFrame a analizar

    Returns:
        Resultados del análisis
    """
    analyzer = NullAnalyzer()
    return analyzer.analyze(df)
