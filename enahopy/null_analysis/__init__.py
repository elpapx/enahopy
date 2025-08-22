"""
ENAHO Null Analysis - Análisis de Valores Nulos
===============================================

Módulo completo para análisis de valores nulos en datos ENAHO.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

# Importar configuración y excepciones
from .config import NullAnalysisConfig
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

    # Definir funciones básicas si utils no está disponible
    def calculate_null_percentage(df, column=None):
        if column:
            return (df[column].isnull().sum() / len(df)) * 100
        return (df.isnull().sum() / len(df)) * 100

    def find_columns_with_nulls(df):
        return df.columns[df.isnull().any()].tolist()

    def get_null_summary(df):
        return pd.DataFrame(
            {
                "column": df.columns,
                "null_count": df.isnull().sum(),
                "null_percentage": (df.isnull().sum() / len(df)) * 100,
            }
        )


# =====================================================
# CLASE PRINCIPAL ENAHONullAnalyzer
# =====================================================


class ENAHONullAnalyzer:
    """
    Analizador principal de valores nulos para datos ENAHO

    Esta clase orquesta todo el análisis de valores nulos,
    incluyendo detección de patrones y generación de reportes.
    """

    def __init__(self, config: Optional[NullAnalysisConfig] = None):
        """
        Inicializa el analizador

        Args:
            config: Configuración opcional del análisis
        """
        self.config = config or NullAnalysisConfig() if NullAnalysisConfig else {}
        self.logger = logging.getLogger(__name__)

        # Componentes internos
        self.pattern_detector = PatternDetector(self.logger) if PATTERNS_AVAILABLE else None
        self.pattern_analyzer = (
            NullPatternAnalyzer(self.pattern_detector) if PATTERNS_AVAILABLE else None
        )
        self.report_generator = ReportGenerator(self.logger) if REPORTS_AVAILABLE else None
        self.visualizer = NullVisualizer() if REPORTS_AVAILABLE else None

        # Analizador core
        self.core_analyzer = NullAnalyzer(self.config)

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
    Genera reporte completo de análisis de nulos

    Args:
        df: DataFrame a analizar
        output_path: Ruta para guardar el reporte
        format: Formato del reporte
        include_visualizations: Si incluir visualizaciones

    Returns:
        Reporte generado
    """
    analyzer = ENAHONullAnalyzer()
    results = analyzer.analyze(
        df, generate_report=True, include_visualizations=include_visualizations
    )

    if analyzer.last_report and output_path:
        try:
            analyzer.last_report.save(output_path, format)
        except:
            pass

    return analyzer.last_report


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
    "NullAnalysisError",
]

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
