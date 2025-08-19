"""
enahopy - Análisis de microdatos ENAHO del INEI
================================================

Librería Python para facilitar el trabajo con los microdatos
de la Encuesta Nacional de Hogares (ENAHO) del Perú.
"""

# Version info
__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Track what's available
_components = {}

# =====================================================
# LOADER MODULE
# =====================================================
try:
    from .loader import (
        ENAHODataDownloader,
        ENAHOLocalReader,
        download_enaho_data,
        read_enaho_file,
        ENAHOUtils
    )
    _components['loader'] = True
except ImportError as e:
    print(f"⚠️ Loader module not available: {e}")
    _components['loader'] = False
    ENAHODataDownloader = None
    ENAHOLocalReader = None
    download_enaho_data = None
    read_enaho_file = None
    ENAHOUtils = None

# =====================================================
# MERGER MODULE
# =====================================================
try:
    from .merger import ENAHOMerger, merge_enaho_modules
    _components['merger'] = True
except ImportError as e:
    print(f"⚠️ Merger module not available: {e}")
    _components['merger'] = False
    ENAHOMerger = None
    merge_enaho_modules = None

try:
    from .merger import create_panel_data
    _components['panel'] = True
except ImportError as e:
    _components['panel'] = False
    create_panel_data = None

# =====================================================
# NULL ANALYSIS MODULE
# =====================================================
try:
    from .null_analysis import ENAHONullAnalyzer
    _components['null_analyzer'] = True
except ImportError as e:
    print(f"⚠️ Null analysis not available: {e}")
    _components['null_analyzer'] = False
    ENAHONullAnalyzer = None

try:
    from .null_analysis import analyze_null_patterns, generate_null_report
    _components['null_functions'] = True
except ImportError:
    _components['null_functions'] = False
    analyze_null_patterns = None
    generate_null_report = None

# =====================================================
# BUILD __all__ DYNAMICALLY
# =====================================================
__all__ = ['__version__', '__version_info__']

if _components.get('loader'):
    __all__.extend([
        'ENAHODataDownloader',
        'ENAHOLocalReader',
        'download_enaho_data',
        'read_enaho_file',
        'ENAHOUtils'
    ])

if _components.get('merger'):
    __all__.extend([
        'ENAHOMerger',
        'merge_enaho_modules'
    ])

if _components.get('panel'):
    __all__.append('create_panel_data')

if _components.get('null_analyzer'):
    __all__.append('ENAHONullAnalyzer')

if _components.get('null_functions'):
    __all__.extend([
        'analyze_null_patterns',
        'generate_null_report'
    ])

# =====================================================
# DIAGNOSTIC FUNCTION
# =====================================================
def check_installation():
    """Verifica el estado de la instalación"""
    print("=" * 50)
    print("ENAHOPY Installation Status")
    print("=" * 50)
    print(f"Version: {__version__}")
    print("")
    print("Components:")

    status_map = {
        'loader': ('Loader (ENAHODataDownloader)', ENAHODataDownloader),
        'merger': ('Merger (ENAHOMerger)', ENAHOMerger),
        'panel': ('Panel (create_panel_data)', create_panel_data),
        'null_analyzer': ('Null Analyzer (ENAHONullAnalyzer)', ENAHONullAnalyzer),
    }

    for key, (name, obj) in status_map.items():
        if obj is not None:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    print("=" * 50)

    # Return status for programmatic use
    return all(obj is not None for _, obj in status_map.values())

__all__.append('check_installation')

# =====================================================
# MODULE INFO
# =====================================================
__author__ = 'ENAHO Development Team'
__license__ = 'MIT'
