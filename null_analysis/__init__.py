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

# Importar utilidades principales
from .utils import (
    safe_dict_merge,
    InputValidator,
    null_analysis_context,
    safe_percentage,
    format_percentage
)

__version__ = '0.0.3'
__author__ = 'ELPAPX'

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

    # Utilidades
    'safe_dict_merge',
    'InputValidator',
    'null_analysis_context',

    # Compatibilidad (deprecated)
    'LegacyNullAnalyzer',
    'diagnostico_nulos_enaho'
]
