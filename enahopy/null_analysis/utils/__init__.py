# null_analysis/utils/__init__.py
"""
Utility functions for null analysis
"""

from .validators import (
    validate_dataframe,
    validate_column_exists,
    validate_numeric_columns
)

from .cache import NullAnalysisCache

# Si utils.py está en la raíz del módulo utils, importar desde ahí
try:
    from .utils import (
        safe_dict_merge,
        InputValidator,
        null_analysis_context,
        safe_percentage,
        format_percentage
    )
except ImportError:
    # Si no está en subdirectorio, importar desde el archivo principal
    pass

# Importar y re-exportar NullAnalysisError para conveniencia
from ..exceptions import NullAnalysisError

__all__ = [
    # Validadores
    'validate_dataframe',
    'validate_column_exists',
    'validate_numeric_columns',
    'NullAnalysisCache',
    
    # Utilidades principales
    'safe_dict_merge',
    'InputValidator',
    'null_analysis_context',
    'safe_percentage',
    'format_percentage',
    
    # Excepciones
    'NullAnalysisError'
]
