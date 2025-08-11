"""
ENAHO Analyzer - Herramienta integral para análisis de microdatos del INEI
=========================================================================

Librería Python para facilitar el análisis de encuestas nacionales peruanas
como ENAHO, ENDES, ENAPRES, etc.
"""

# Importar los submódulos principales
try:
    from . import loader
except ImportError:
    pass

try:
    from . import merger
except ImportError:
    pass

try:
    from . import null_analysis
except ImportError:
    pass

__version__ = '1.0.0'
__author__ = 'el maldito papx'
__description__ = 'Herramienta integral para análisis de microdatos del INEI'

__all__ = [
    'loader',
    'merger',
    'null_analysis'
]