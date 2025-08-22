"""
enahopy - Análisis de microdatos ENAHO del INEI
================================================

Librería Python para facilitar el análisis de microdatos de la
Encuesta Nacional de Hogares (ENAHO) del Instituto Nacional de
Estadística e Informática (INEI) del Perú.
"""

# Version info
__version__ = "0.1.1"
__version_info__ = (0, 1, 0)
__author__ = "Paul Camacho"
__email__ = "pcamacho447@gmail.com"

# Simple import without dynamic loading
try:
    from .loader import ENAHODataDownloader, ENAHOLocalReader, download_enaho_data, read_enaho_file, ENAHOUtils
    _loader_available = True
except ImportError:
    _loader_available = False
    ENAHODataDownloader = None
    ENAHOLocalReader = None
    download_enaho_data = None
    read_enaho_file = None
    ENAHOUtils = None

try:
    from .merger import ENAHOMerger, merge_enaho_modules, create_panel_data
    _merger_available = True
except ImportError:
    _merger_available = False
    ENAHOMerger = None
    merge_enaho_modules = None
    create_panel_data = None

try:
    from .null_analysis import ENAHONullAnalyzer, analyze_null_patterns, generate_null_report
    _null_analysis_available = True
except ImportError:
    _null_analysis_available = False
    ENAHONullAnalyzer = None
    analyze_null_patterns = None
    generate_null_report = None

def show_status(verbose=True):
    """Show the status of all components."""
    print(f"enahopy v{__version__} - Estado de componentes:")
    print("-" * 50)

    components = {
        "Loader": _loader_available,
        "Merger": _merger_available,
        "Null_analysis": _null_analysis_available
    }

    for name, available in components.items():
        status = "Disponible" if available else "No disponible"
        symbol = "✅" if available else "❌"
        print(f"{symbol} {name}: {status}")

def get_available_components():
    """Return status of components."""
    return {
        "loader": _loader_available,
        "merger": _merger_available,
        "null_analysis": _null_analysis_available
    }

# Build __all__ list dynamically
__all__ = ["__version__", "__version_info__", "show_status", "get_available_components"]

if _loader_available:
    __all__.extend(["ENAHODataDownloader", "ENAHOLocalReader", "download_enaho_data", "read_enaho_file", "ENAHOUtils"])

if _merger_available:
    __all__.extend(["ENAHOMerger", "merge_enaho_modules", "create_panel_data"])

if _null_analysis_available:
    __all__.extend(["ENAHONullAnalyzer", "analyze_null_patterns", "generate_null_report"])