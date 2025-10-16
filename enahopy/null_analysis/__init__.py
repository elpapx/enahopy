"""
ENAHO Null Analysis - Análisis de Valores Nulos
===============================================

Módulo completo para análisis de valores nulos en datos ENAHO.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Importar configuración y excepciones
from .config import AnalysisComplexity, NullAnalysisConfig
from .exceptions import NullAnalysisError

# Importar el analizador base desde core
try:
    from .core.analyzer import NullAnalyzer
except ImportError:
    # Si no existe, crear una clase dummy
    class NullAnalyzer:
        def __init__(self, config=None):
            self.config = config or {}

        def analyze(self, df):
            return {"error": "NullAnalyzer not available"}


# Importar detectores de patrones
try:
    from .patterns import (
        MissingDataPattern,
        NullPatternAnalyzer,
        PatternDetector,
        PatternResult,
        PatternSeverity,
        PatternType,
    )

    PATTERNS_AVAILABLE = True
except ImportError:
    PATTERNS_AVAILABLE = False
    PatternDetector = None
    NullPatternAnalyzer = None
    MissingDataPattern = None
    PatternType = None
    PatternSeverity = None
    PatternResult = None

# Importar generadores de reportes
try:
    from .reports import NullAnalysisReport, NullVisualizer, ReportGenerator, VisualizationType

    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False
    ReportGenerator = None
    NullAnalysisReport = None
    NullVisualizer = None
    VisualizationType = None

# Importar utilidades
try:
    from .utils import (
        calculate_null_percentage,
        detect_monotone_pattern,
        find_columns_with_nulls,
        get_null_correlation_matrix,
        get_null_summary,
        identify_null_patterns,
        impute_with_strategy,
        safe_dict_merge,
    )

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Importar funciones de conveniencia
try:
    from .convenience import (
        compare_null_patterns,
        create_null_visualizations,
        detect_missing_patterns_automatically,
    )
    from .convenience import generate_null_report as generate_comprehensive_null_report
    from .convenience import (
        get_data_quality_score,
        quick_null_analysis,
        suggest_imputation_methods,
        validate_data_completeness,
    )

    CONVENIENCE_AVAILABLE = True
except ImportError:
    CONVENIENCE_AVAILABLE = False
    quick_null_analysis = None
    get_data_quality_score = None
    create_null_visualizations = None
    generate_comprehensive_null_report = None
    compare_null_patterns = None
    suggest_imputation_methods = None
    validate_data_completeness = None
    detect_missing_patterns_automatically = None

    # Definir funciones básicas si utils no está disponible
    def calculate_null_percentage(
        df: pd.DataFrame, column: Optional[str] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate null percentage for DataFrame or specific column.

        Args:
            df: DataFrame to analyze.
            column: Optional column name. If None, calculates for all columns.

        Returns:
            Float percentage if column specified, Series of percentages otherwise.

        Example:
            >>> df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
            >>> calculate_null_percentage(df, "a")
            33.33...
        """
        if column:
            return (df[column].isnull().sum() / len(df)) * 100
        return (df.isnull().sum() / len(df)) * 100

    def find_columns_with_nulls(df: pd.DataFrame) -> List[str]:
        """
        Find all columns containing null values.

        Args:
            df: DataFrame to analyze.

        Returns:
            List of column names with at least one null value.

        Example:
            >>> df = pd.DataFrame({"a": [1, None], "b": [1, 2]})
            >>> find_columns_with_nulls(df)
            ['a']
        """
        return df.columns[df.isnull().any()].tolist()

    def get_null_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary DataFrame of null statistics.

        Args:
            df: DataFrame to analyze.

        Returns:
            DataFrame with columns: 'column', 'null_count', 'null_percentage'.

        Example:
            >>> df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
            >>> summary = get_null_summary(df)
            >>> summary.columns.tolist()
            ['column', 'null_count', 'null_percentage']
        """
        return pd.DataFrame(
            {
                "column": df.columns,
                "null_count": df.isnull().sum().values,
                "null_percentage": (df.isnull().sum() / len(df) * 100).values,
            }
        )


# Importar estrategias avanzadas de imputación ML
try:
    from .strategies import (  # Advanced ML imputation; ENAHO pattern-aware imputation; Quality assessment; Availability flags
        ENAHO_PATTERN_AVAILABLE,
        ML_IMPUTATION_AVAILABLE,
        QUALITY_ASSESSMENT_AVAILABLE,
        AutoencoderImputer,
        ENAHOImputationConfig,
        ENAHOMissingPattern,
        ENAHOPatternDetector,
        ENAHOPatternImputer,
        ImputationConfig,
        ImputationQualityAssessor,
        ImputationResult,
        MICEImputer,
        MissForestImputer,
        MLImputationManager,
        QualityAssessmentConfig,
        QualityAssessmentResult,
        QualityMetricType,
        assess_imputation_quality,
        compare_imputation_methods,
        create_advanced_imputer,
        create_enaho_pattern_imputer,
        create_ml_imputation_manager,
    )

    ADVANCED_IMPUTATION_AVAILABLE = True
except ImportError:
    ADVANCED_IMPUTATION_AVAILABLE = False
    ML_IMPUTATION_AVAILABLE = False
    ENAHO_PATTERN_AVAILABLE = False
    QUALITY_ASSESSMENT_AVAILABLE = False


# =====================================================
# CLASE PRINCIPAL ENAHONullAnalyzer
# =====================================================


class ENAHONullAnalyzer:
    """
    Analizador principal de valores nulos para datos ENAHO

    Esta clase orquesta todo el análisis de valores nulos,
    incluyendo detección de patrones y generación de reportes.
    """

    def __init__(self, config: Optional[NullAnalysisConfig] = None, verbose: bool = True):
        """
        Inicializa el analizador

        Args:
            config: Configuración opcional del análisis
            verbose: Si mostrar mensajes de log (default: True)
        """
        self.config = config or NullAnalysisConfig() if NullAnalysisConfig else {}
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Ajustar nivel de log según verbose
        if not verbose:
            self.logger.setLevel(logging.WARNING)

        # Componentes internos
        self.pattern_detector = PatternDetector(self.logger) if PATTERNS_AVAILABLE else None
        self.pattern_analyzer = (
            NullPatternAnalyzer(self.pattern_detector) if PATTERNS_AVAILABLE else None
        )
        self.report_generator = ReportGenerator(self.logger) if REPORTS_AVAILABLE else None
        self.visualizer = NullVisualizer() if REPORTS_AVAILABLE else None

        # Analizador core (usa su propia config simple con thresholds)
        self.core_analyzer = NullAnalyzer()

        # Estado
        self.last_analysis = None
        self.last_report = None

        self.logger.info("ENAHONullAnalyzer inicializado")

    def analyze(
        self, df: pd.DataFrame, generate_report: bool = True, include_visualizations: bool = False
    ) -> Dict[str, Any]:
        """
        Realiza análisis completo de valores nulos

        Args:
            df: DataFrame a analizar
            generate_report: Si generar reporte completo
            include_visualizations: Si incluir visualizaciones

        Returns:
            Diccionario con resultados del análisis
        """
        self.logger.info(f"Iniciando análisis de valores nulos para DataFrame {df.shape}")

        results = {
            "summary": {},
            "patterns": {},
            "recommendations": [],
            "report": None,
            "visualizations": {},
        }

        # 1. Análisis básico usando core_analyzer
        try:
            core_results = self.core_analyzer.analyze(df)
            results["summary"].update(core_results)
        except Exception as e:
            self.logger.warning(f"Error en análisis core: {e}")
            # Fallback a análisis básico
            results["summary"] = {
                "total_values": df.size,
                "null_values": df.isnull().sum().sum(),
                "null_percentage": (df.isnull().sum().sum() / df.size) * 100,
                "columns_with_nulls": find_columns_with_nulls(df) if UTILS_AVAILABLE else [],
            }

        # 2. Detección de patrones si está disponible
        if PATTERNS_AVAILABLE and self.pattern_analyzer:
            try:
                results["patterns"] = self.pattern_analyzer.analyze_patterns(df)
            except Exception as e:
                self.logger.warning(f"Error en detección de patrones: {e}")
                results["patterns"] = {"error": str(e)}

        # 3. Generar reporte si se solicita y está disponible
        if generate_report and REPORTS_AVAILABLE and self.report_generator:
            try:
                report = self.report_generator.generate_report(
                    df, results.get("patterns"), include_visualizations
                )
                results["report"] = report.to_dict()
                self.last_report = report

                if hasattr(report, "recommendations"):
                    results["recommendations"] = report.recommendations
            except Exception as e:
                self.logger.warning(f"Error generando reporte: {e}")

        # 4. Generar visualizaciones si se solicita
        if include_visualizations and REPORTS_AVAILABLE and self.visualizer:
            try:
                results["visualizations"]["matrix"] = self.visualizer.visualize_null_matrix(df)
                results["visualizations"]["bars"] = self.visualizer.visualize_null_bars(df)
                results["visualizations"]["heatmap"] = self.visualizer.visualize_null_heatmap(df)
            except Exception as e:
                self.logger.warning(f"No se pudieron generar visualizaciones: {e}")

        # Guardar último análisis
        self.last_analysis = results

        return results

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene un resumen rápido de valores nulos

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con resumen
        """
        return {
            "total_values": df.size,
            "null_values": df.isnull().sum().sum(),
            "null_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "columns_with_nulls": (
                find_columns_with_nulls(df)
                if UTILS_AVAILABLE
                else df.columns[df.isnull().any()].tolist()
            ),
            "complete_rows": (~df.isnull().any(axis=1)).sum(),
            "rows_with_nulls": df.isnull().any(axis=1).sum(),
        }

    def analyze_null_patterns(
        self,
        df: pd.DataFrame,
        group_by: Optional[str] = None,
        geographic_filter: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Analiza patrones de nulos con agrupación opcional.

        Args:
            df: DataFrame a analizar
            group_by: Columna para agrupar análisis
            geographic_filter: Filtros geográficos opcionales

        Returns:
            Diccionario con métricas y análisis
        """
        # Análisis básico
        result = self.analyze(df, generate_report=False)

        # Agregar métricas en formato esperado (usando SimpleNamespace para compatibilidad)
        from types import SimpleNamespace

        result["metrics"] = SimpleNamespace(
            total_cells=df.size,
            missing_cells=int(result["summary"]["null_values"]),
            missing_percentage=float(result["summary"]["null_percentage"]),
            total_rows=len(df),
            total_columns=len(df.columns),
            variables_with_missing=len(result["summary"].get("columns_with_nulls", [])),
            variables_without_missing=len(df.columns)
            - len(result["summary"].get("columns_with_nulls", [])),
            complete_cases=int(result["summary"].get("complete_rows", 0)),
            complete_cases_percentage=float(
                result["summary"].get("complete_rows", 0) / len(df) * 100 if len(df) > 0 else 0
            ),
            data_quality_score=100.0 - float(result["summary"]["null_percentage"]),
            missing_data_pattern="Unknown",
        )

        # Agregar summary
        result["summary"] = get_null_summary(df) if UTILS_AVAILABLE else pd.DataFrame()

        # Análisis por grupo si se solicita
        if group_by and group_by in df.columns:
            grouped_stats = []
            for group_val, group_df in df.groupby(group_by):
                group_nulls = group_df.isnull().sum().sum()
                group_total = group_df.size
                grouped_stats.append(
                    {
                        group_by: group_val,
                        "null_count": int(group_nulls),
                        "null_percentage": (
                            float(group_nulls / group_total * 100) if group_total > 0 else 0.0
                        ),
                    }
                )
            result["group_analysis"] = pd.DataFrame(grouped_stats)
            result["analysis_type"] = "grouped"
        else:
            result["analysis_type"] = "basic"

        return result

    def get_data_quality_score(
        self, df: pd.DataFrame, detailed: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Calcula score de calidad de datos basado en completitud.

        Args:
            df: DataFrame a analizar
            detailed: Si retornar detalles adicionales

        Returns:
            Score 0-100 o diccionario con detalles
        """
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        score = 100.0 - (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

        if not detailed:
            return score

        return {
            "overall_score": score,
            "completeness_score": score,
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "missing_percentage": (
                float(missing_cells / total_cells * 100) if total_cells > 0 else 0.0
            ),
        }

    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        output_path: str,
        group_by: Optional[str] = None,
        geographic_filter: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Genera reporte comprehensivo de análisis.

        Args:
            df: DataFrame a analizar
            output_path: Ruta para guardar reporte
            group_by: Columna para agrupar
            geographic_filter: Filtros geográficos

        Returns:
            Diccionario con metadatos y resultados
        """
        analysis_results = self.analyze_null_patterns(df, group_by, geographic_filter)

        return {
            "report_metadata": {
                "output_path": output_path,
                "dataframe_shape": df.shape,
                "group_by": group_by,
            },
            "analysis_results": analysis_results,
        }

    def get_imputation_recommendations(
        self, analysis_result: Dict[str, Any], variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sugiere estrategias de imputación basadas en el análisis.

        Args:
            analysis_result: Resultados del análisis
            variable: Variable específica (opcional)

        Returns:
            Diccionario con recomendaciones
        """
        recommendations = {}

        if "metrics" in analysis_result:
            missing_pct = analysis_result["metrics"].missing_percentage

            if missing_pct < 5:
                recommendations["strategy"] = "simple"
                recommendations["methods"] = ["mean", "median", "mode"]
            elif missing_pct < 20:
                recommendations["strategy"] = "moderate"
                recommendations["methods"] = ["knn", "iterative"]
            else:
                recommendations["strategy"] = "advanced"
                recommendations["methods"] = ["mice", "missforest"]

        return recommendations


# =====================================================
# FUNCIONES DE CONVENIENCIA
# =====================================================


def analyze_null_patterns(
    df: pd.DataFrame, config: Optional[NullAnalysisConfig] = None
) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis rápido de patrones de nulos

    Args:
        df: DataFrame a analizar
        config: Configuración opcional

    Returns:
        Resultados del análisis
    """
    analyzer = ENAHONullAnalyzer(config)
    return analyzer.analyze(df, generate_report=False)


def generate_null_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    format: str = "html",
    include_visualizations: bool = True,
) -> Any:
    """
    Genera reporte completo de análisis de nulos.

    Args:
        df: DataFrame a analizar.
        output_path: Ruta para guardar el reporte (opcional).
        format: Formato del reporte ('html', 'json', 'pdf'). Default: 'html'.
        include_visualizations: Si incluir visualizaciones en el reporte.

    Returns:
        Objeto NullAnalysisReport generado, o None si el análisis falló.

    Raises:
        NullAnalysisError: Si el análisis del DataFrame falla críticamente.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, None, 3]})
        >>> report = generate_null_report(df, output_path="report.html")
        >>> print(f"Report generated: {report is not None}")
    """
    logger = logging.getLogger(__name__)

    try:
        analyzer = ENAHONullAnalyzer()
        results = analyzer.analyze(
            df, generate_report=True, include_visualizations=include_visualizations
        )

        # Intentar guardar reporte si se especificó ruta
        if analyzer.last_report and output_path:
            try:
                analyzer.last_report.save(output_path, format)
                logger.info(f"Report saved successfully to: {output_path}")

            except (OSError, IOError, PermissionError) as e:
                # Errores esperados de sistema de archivos - no críticos
                logger.warning(
                    f"Failed to save report to {output_path}: {str(e)}. "
                    f"Report object still returned for programmatic use."
                )

            except AttributeError as e:
                # El metodo .save() puede no existir si REPORTS_AVAILABLE=False
                logger.warning(
                    f"Report saving not available: {str(e)}. "
                    f"Check if reporting module is properly installed."
                )

            except Exception as e:
                # Otros errores inesperados - log completo para debugging
                logger.error(
                    f"Unexpected error saving report: {type(e).__name__}: {str(e)}", exc_info=True
                )

        return analyzer.last_report

    except KeyboardInterrupt:
        # Usuario cancela operación - re-raise para propagar
        logger.info("Report generation cancelled by user")
        raise

    except Exception as e:
        # Error crítico en el análisis - no silenciar
        logger.error(f"Critical error during null analysis: {str(e)}", exc_info=True)
        raise NullAnalysisError(
            f"Failed to generate null report: {str(e)}",
            error_code="ANALYSIS_FAILED",
            operation="generate_null_report",
        ) from e


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Clase principal
    "ENAHONullAnalyzer",
    # Funciones de conveniencia
    "analyze_null_patterns",
    "generate_null_report",
    # Configuración y excepciones
    "NullAnalysisConfig",
    "AnalysisComplexity",
    "NullAnalysisError",
]

# Agregar funciones de conveniencia si están disponibles
if CONVENIENCE_AVAILABLE:
    __all__.extend(
        [
            "quick_null_analysis",
            "get_data_quality_score",
            "create_null_visualizations",
            "generate_comprehensive_null_report",
            "compare_null_patterns",
            "suggest_imputation_methods",
            "validate_data_completeness",
            "detect_missing_patterns_automatically",
        ]
    )

# Agregar exports condicionales
if PATTERNS_AVAILABLE:
    __all__.extend(
        [
            "PatternDetector",
            "NullPatternAnalyzer",
            "MissingDataPattern",
            "PatternType",
            "PatternSeverity",
            "PatternResult",
        ]
    )

if REPORTS_AVAILABLE:
    __all__.extend(
        [
            "ReportGenerator",
            "NullAnalysisReport",
            "NullVisualizer",
            "VisualizationType",
        ]
    )

if UTILS_AVAILABLE:
    __all__.extend(
        [
            "calculate_null_percentage",
            "identify_null_patterns",
            "get_null_correlation_matrix",
            "find_columns_with_nulls",
            "get_null_summary",
            "detect_monotone_pattern",
            "impute_with_strategy",
            "safe_dict_merge",
        ]
    )

# Agregar exports de imputación avanzada
if ADVANCED_IMPUTATION_AVAILABLE:
    __all__.extend(
        [
            # Advanced ML imputation
            "MICEImputer",
            "MissForestImputer",
            "AutoencoderImputer",
            "ImputationConfig",
            "ImputationResult",
            "MLImputationManager",
            "create_advanced_imputer",
            "compare_imputation_methods",
            "create_ml_imputation_manager",
            # ENAHO pattern-aware imputation
            "ENAHOMissingPattern",
            "ENAHOImputationConfig",
            "ENAHOPatternDetector",
            "ENAHOPatternImputer",
            "create_enaho_pattern_imputer",
            # Quality assessment
            "QualityMetricType",
            "QualityAssessmentConfig",
            "QualityAssessmentResult",
            "ImputationQualityAssessor",
            "assess_imputation_quality",
            # Availability flags
            "ML_IMPUTATION_AVAILABLE",
            "ENAHO_PATTERN_AVAILABLE",
            "QUALITY_ASSESSMENT_AVAILABLE",
        ]
    )
