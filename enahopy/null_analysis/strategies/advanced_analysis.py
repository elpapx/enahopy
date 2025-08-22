"""
Estrategia de análisis avanzado de valores nulos
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from ..config import MissingDataMetrics, MissingDataPattern, NullAnalysisConfig
from .basic_analysis import BasicNullAnalysis


class AdvancedNullAnalysis(BasicNullAnalysis):
    """Análisis avanzado con detección de patrones y estadísticas avanzadas"""

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análisis avanzado completo"""

        # Ejecutar análisis básico primero
        basic_analyzer = BasicNullAnalysis(self.config, self.logger)
        basic_result = basic_analyzer.analyze(df, kwargs.get("group_by"))

        # Análisis de patrones de missing data
        patterns_analysis = self._analyze_missing_patterns(df)

        # Análisis de correlaciones entre missing values
        correlation_analysis = self._analyze_missing_correlations(df)

        # Clustering de patrones
        clustering_analysis = self._cluster_missing_patterns(df)

        # Test estadísticos
        statistical_tests = self._perform_statistical_tests(df)

        # Detección de patrones temporales
        temporal_analysis = self._analyze_temporal_patterns(df)

        # Actualizar métricas con información avanzada
        advanced_metrics = self._calculate_advanced_metrics(
            df,
            basic_result["metrics"],
            patterns_analysis,
            correlation_analysis,
            clustering_analysis,
            statistical_tests,
        )

        return {
            "analysis_type": "advanced",
            "basic_analysis": basic_result,
            "patterns": patterns_analysis,
            "correlations": correlation_analysis,
            "clustering": clustering_analysis,
            "statistical_tests": statistical_tests,
            "temporal_analysis": temporal_analysis,
            "metrics": advanced_metrics,
            "execution_time": 0,
        }

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza patrones de datos faltantes"""

        if df.empty:
            return {
                "total_patterns": 0,
                "most_common_patterns": {},
                "is_monotone": False,
                "pattern_diversity": 0,
                "complete_cases_pattern": 0,
                "error": "DataFrame Vacío",
            }

        missing_matrix = df.isnull()

        pattern_strings = missing_matrix.apply(
            lambda row: "".join(["1" if x else "0" for x in row]), axis=1
        )

        pattern_counts = pattern_strings.value_counts()
        is_monotone = self._check_monotone_missing(missing_matrix)
        top_patterns = pattern_counts.head(10)

        # Calclar diversidad con protección contra división por 0

        pattern_diversity = len(pattern_counts) / max(len(df), 1)

        return {
            "total_patterns": len(pattern_counts),
            "most_common_patterns": top_patterns.to_dict(),
            "is_monotone": is_monotone,
            "pattern_diversity": len(pattern_counts) / len(df),
            "complete_cases_pattern": pattern_counts.get("0" * len(df.columns), 0),
        }

    def _analyze_missing_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza correlaciones entre variables con datos faltantes"""
        missing_matrix = df.isnull().astype(int)
        missing_corr = missing_matrix.corr()

        significant_correlations = []
        for i in range(len(missing_corr.columns)):
            for j in range(i + 1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > self.config.correlation_threshold:
                    significant_correlations.append(
                        {
                            "var1": missing_corr.columns[i],
                            "var2": missing_corr.columns[j],
                            "correlation": corr_val,
                        }
                    )

        return {
            "correlation_matrix": missing_corr,
            "significant_correlations": significant_correlations,
            "max_correlation": missing_corr.abs().max().max(),
            "mean_correlation": missing_corr.abs().mean().mean(),
        }

    def _cluster_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Clustering de patrones de missing data"""
        missing_matrix = df.isnull().astype(int)

        if len(df) > 1000:
            sample_size = min(1000, len(df))
            missing_sample = missing_matrix.sample(n=sample_size, random_state=42)
        else:
            missing_sample = missing_matrix

        try:
            kmeans = KMeans(n_clusters=min(5, len(missing_sample)), random_state=42)
            clusters = kmeans.fit_predict(missing_sample)

            cluster_analysis = pd.DataFrame({"cluster": clusters})
            cluster_summary = cluster_analysis["cluster"].value_counts().to_dict()

            return {
                "n_clusters": len(cluster_summary),
                "cluster_distribution": cluster_summary,
                "silhouette_score": None,
                "clustering_successful": True,
            }

        except Exception as e:
            self.logger.warning(f"Error en clustering: {str(e)}")
            return {"clustering_successful": False, "error": str(e)}

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza tests estadísticos para evaluar patrones de missing data"""
        results = {}

        try:
            missing_matrix = df.isnull()
            patterns = missing_matrix.apply(
                lambda row: "".join(["1" if x else "0" for x in row]), axis=1
            )

            pattern_counts = patterns.value_counts()

            if len(pattern_counts) > 1:
                expected_freq = len(df) / len(pattern_counts)
                chi_square = sum(
                    (obs - expected_freq) ** 2 / expected_freq for obs in pattern_counts.values
                )

                df_chi = len(pattern_counts) - 1
                p_value = 1 - stats.chi2.cdf(chi_square, df_chi) if df_chi > 0 else 1.0

                results["simplified_mcar_test"] = {
                    "chi_square": chi_square,
                    "p_value": p_value,
                    "reject_mcar": p_value < 0.05,
                }

        except Exception as e:
            self.logger.warning(f"Error en tests estadísticos: {str(e)}")
            results["error"] = str(e)

        return results

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza patrones temporales en datos faltantes"""
        temporal_analysis = {"temporal_columns_found": False}

        date_columns = []
        for col in df.columns:
            if df[col].dtype in ["datetime64[ns]", "datetime64[D]"] or "fecha" in col.lower():
                date_columns.append(col)

        if date_columns:
            temporal_analysis["temporal_columns_found"] = True
            temporal_analysis["date_columns"] = date_columns

            date_col = date_columns[0]
            if not df[date_col].isnull().all():
                try:
                    df_temp = df.copy()
                    df_temp["period"] = pd.to_datetime(df_temp[date_col]).dt.to_period("M")

                    temporal_missing = (
                        df_temp.groupby("period")
                        .apply(lambda x: x.isnull().sum() / len(x))
                        .mean(axis=1)
                    )

                    temporal_analysis["temporal_patterns"] = temporal_missing.to_dict()
                    temporal_analysis["temporal_trend"] = (
                        "increasing"
                        if temporal_missing.corr(range(len(temporal_missing))) > 0.3
                        else "stable"
                    )

                except Exception as e:
                    self.logger.warning(f"Error en análisis temporal: {str(e)}")

        return temporal_analysis

    def _check_monotone_missing(self, missing_matrix: pd.DataFrame) -> bool:
        """Verifica si el patrón de missing data es monótono"""
        try:
            missing_counts = missing_matrix.sum().sort_values()
            ordered_matrix = missing_matrix[missing_counts.index]

            for i in range(len(ordered_matrix)):
                row = ordered_matrix.iloc[i]
                first_missing = -1

                for j, val in enumerate(row):
                    if val and first_missing == -1:
                        first_missing = j
                    elif not val and first_missing != -1:
                        return False

            return True

        except Exception:
            return False

    def _calculate_advanced_metrics(
        self,
        df: pd.DataFrame,
        basic_metrics: MissingDataMetrics,
        patterns: Dict,
        correlations: Dict,
        clustering: Dict,
        statistical_tests: Dict,
    ) -> MissingDataMetrics:
        """Calcula métricas avanzadas"""

        # Obtener patrón más común de forma segura

        common_patterns = patterns.get("most_common_patterns", {})
        if common_patterns:
            most_common_pattern = str(list(common_patterns.keys())[0])
        else:
            most_common_pattern = "No patterns Found"

        advanced_metrics = MissingDataMetrics(
            total_cells=basic_metrics.total_cells,
            missing_cells=basic_metrics.missing_cells,
            missing_percentage=basic_metrics.missing_percentage,
            complete_cases=basic_metrics.complete_cases,
            complete_cases_percentage=basic_metrics.complete_cases_percentage,
            variables_with_missing=basic_metrics.variables_with_missing,
            variables_without_missing=basic_metrics.variables_without_missing,
            missing_pattern_count=patterns.get("total_patterns", 0),
            most_common_pattern=str(
                list(patterns.get("most_common_patterns", {}).keys())[0]
                if patterns.get("most_common_patterns")
                else "Unknown"
            ),
            missing_data_pattern=self._classify_missing_pattern(statistical_tests, correlations),
            monotone_missing=patterns.get("is_monotone", False),
            little_mcar_test_pvalue=statistical_tests.get("simplified_mcar_test", {}).get(
                "p_value"
            ),
            missing_clustering_score=clustering.get("silhouette_score"),
            temporal_pattern_detected=len(patterns.get("temporal_patterns", {})) > 0,
            data_quality_score=self._calculate_advanced_quality_score(
                basic_metrics, patterns, correlations
            ),
            completeness_score=basic_metrics.complete_cases_percentage,
            consistency_score=self._calculate_consistency_score(correlations, patterns),
        )

        return advanced_metrics

    def _classify_missing_pattern(
        self, statistical_tests: Dict, correlations: Dict
    ) -> MissingDataPattern:
        """Clasifica el patrón de missing data"""
        mcar_test = statistical_tests.get("simplified_mcar_test", {})

        if mcar_test.get("reject_mcar", True):
            max_correlation = correlations.get("max_correlation", 0)

            if max_correlation > 0.7:
                return MissingDataPattern.MNAR
            elif max_correlation > 0.3:
                return MissingDataPattern.MAR
            else:
                return MissingDataPattern.UNKNOWN
        else:
            return MissingDataPattern.MCAR

    def _calculate_advanced_quality_score(
        self, basic_metrics: MissingDataMetrics, patterns: Dict, correlations: Dict
    ) -> float:
        """Calcula score avanzado de calidad"""
        completeness_score = basic_metrics.complete_cases_percentage
        pattern_score = 100 - (patterns.get("pattern_diversity", 1) * 50)
        correlation_score = 100 - (correlations.get("mean_correlation", 0) * 100)

        weights = [0.5, 0.3, 0.2]
        scores = [completeness_score, pattern_score, correlation_score]

        return sum(w * s for w, s in zip(weights, scores))

    def _calculate_consistency_score(self, correlations: Dict, patterns: Dict) -> float:
        """Calcula score de consistencia"""
        base_score = 100

        max_corr = correlations.get("max_correlation", 0)
        correlation_penalty = max_corr * 30

        pattern_diversity = patterns.get("pattern_diversity", 0)
        diversity_penalty = pattern_diversity * 20

        return max(0, base_score - correlation_penalty - diversity_penalty)

    def get_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Recomendaciones avanzadas basadas en análisis completo"""
        recommendations = []

        basic_recs = BasicNullAnalysis(self.config, self.logger).get_recommendations(
            analysis_result["basic_analysis"]
        )
        recommendations.extend(basic_recs)

        metrics = analysis_result["metrics"]
        patterns = analysis_result["patterns"]
        correlations = analysis_result["correlations"]

        if metrics.missing_data_pattern == MissingDataPattern.MCAR:
            recommendations.append(
                "✅ Datos faltantes parecen ser MCAR. Eliminación por lista o imputación simple son apropiadas."
            )
        elif metrics.missing_data_pattern == MissingDataPattern.MAR:
            recommendations.append(
                "🔧 Datos faltantes parecen ser MAR. Use imputación múltiple o métodos basados en modelos."
            )
        elif metrics.missing_data_pattern == MissingDataPattern.MNAR:
            recommendations.append(
                "⚠️  Datos faltantes parecen ser MNAR. Considere modelado específico o consulte experto en el dominio."
            )

        if len(correlations.get("significant_correlations", [])) > 0:
            recommendations.append(
                f"🔗 {len(correlations['significant_correlations'])} correlaciones significativas "
                f"entre patrones de missing. Considere imputación conjunta."
            )

        if patterns.get("is_monotone"):
            recommendations.append(
                "📈 Patrón monótono detectado. Los métodos de imputación secuencial pueden ser efectivos."
            )

        if metrics.data_quality_score < 50:
            recommendations.append(
                "❌ Score de calidad bajo. Considere recolección adicional de datos o revisión del diseño de estudio."
            )

        return recommendations
