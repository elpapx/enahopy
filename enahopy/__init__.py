"""
enahopy - Análisis de microdatos ENAHO del INEI
================================================

Librería Python para facilitar el trabajo con los microdatos
de la Encuesta Nacional de Hogares (ENAHO) del Perú.

Módulos principales:
- loader: Descarga y carga de datos
- merger: Fusión de módulos
- null_analysis: Análisis de valores nulos
"""

from .version import __version__, __version_info__

# Intentar importar componentes principales con manejo de errores
try:
    from .loader import (
        ENAHODataDownloader,
        ENAHOLocalReader,
        download_enaho_data,
        read_enaho_file,
        ENAHOUtils
    )
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False

try:
    from .merger import (
        ENAHOMerger,
        merge_enaho_modules,
        create_panel_data
    )
    MERGER_AVAILABLE = True
except ImportError:
    MERGER_AVAILABLE = False

try:
    from .null_analysis import (
        ENAHONullAnalyzer,
        analyze_null_patterns,
        generate_null_report
    )
    NULL_ANALYSIS_AVAILABLE = True
except ImportError:
    NULL_ANALYSIS_AVAILABLE = False

# Exportar solo lo que esté disponible
__all__ = ['__version__', '__version_info__']

if LOADER_AVAILABLE:
    __all__.extend([
        'ENAHODataDownloader',
        'ENAHOLocalReader',
        'download_enaho_data',
        'read_enaho_file',
        'ENAHOUtils',
    ])

if MERGER_AVAILABLE:
    __all__.extend([
        'ENAHOMerger',
        'merge_enaho_modules',
        'create_panel_data',
    ])

if NULL_ANALYSIS_AVAILABLE:
    __all__.extend([
        'ENAHONullAnalyzer',
        'analyze_null_patterns',
        'generate_null_report',
    ])

# Metadata
__author__ = 'Paul Camacho Abadie'
__email__ = 'pcamacho447@gmail.com'
__license__ = 'MIT'
