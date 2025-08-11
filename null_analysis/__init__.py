"""
ENAHO Null Analysis Module
==========================

Análisis avanzado de valores nulos para microdatos del INEI.
"""

from .config import (
    NullAnalysisConfig,
    MissingDataMetrics,
    MissingDataPattern,
    AnalysisComplexity,
    VisualizationType,
    ExportFormat
)

from .core.analyzer import ENAHONullAnalyzer

from .convenience import (
    quick_null_analysis,
    get_data_quality_score,
    create_null_visualizations,
    generate_null_report,
    compare_null_patterns,
    suggest_imputation_methods,
    validate_data_completeness,
    detect_missing_patterns_automatically,
    # Compatibilidad
    LegacyNullAnalyzer,
    diagnostico_nulos_enaho
)

from .exceptions import (
    NullAnalysisError,
    VisualizationError,
    PatternDetectionError
)

__version__ = '3.0.0'
__author__ = 'ENAHO Analyzer Team'

__all__ = [
    # Clases principales
    'ENAHONullAnalyzer',
    'NullAnalysisConfig',
    'MissingDataMetrics',

    # Enums
    'MissingDataPattern',
    'AnalysisComplexity',
    'VisualizationType',
    'ExportFormat',

    # Excepciones
    'NullAnalysisError',
    'VisualizationError',
    'PatternDetectionError',

    # Funciones de conveniencia
    'quick_null_analysis',
    'get_data_quality_score',
    'create_null_visualizations',
    'generate_null_report',
    'compare_null_patterns',
    'suggest_imputation_methods',
    'validate_data_completeness',
    'detect_missing_patterns_automatically',

    # Compatibilidad (deprecated)
    'LegacyNullAnalyzer',
    'diagnostico_nulos_enaho'
]