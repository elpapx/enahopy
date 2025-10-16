"""
enahopy - Análisis de microdatos ENAHO del INEI
================================================

Librería Python para facilitar el análisis de microdatos de la
Encuesta Nacional de Hogares (ENAHO) del Instituto Nacional de
Estadística e Informática (INEI) del Perú.

Implementa lazy loading para optimizar tiempo de carga inicial.
"""

from typing import Any, Dict

# Version info
__version__ = "0.5.1"
__version_info__ = (0, 5, 1)
__author__ = "Paul Camacho"
__email__ = "pcamacho447@gmail.com"

# =====================================================
# CORE IMPORTS (Siempre cargados - esenciales)
# =====================================================

# Loader (esencial)
try:
    from .loader import (
        ENAHODataDownloader,
        ENAHOLocalReader,
        ENAHOUtils,
        download_enaho_data,
        read_enaho_file,
    )

    _loader_available = True
except ImportError:
    _loader_available = False
    ENAHODataDownloader = None
    ENAHOLocalReader = None
    download_enaho_data = None
    read_enaho_file = None
    ENAHOUtils = None

# Merger (esencial)
try:
    from .merger import ENAHOMerger, create_panel_data, merge_enaho_modules

    _merger_available = True
except ImportError:
    _merger_available = False
    ENAHOMerger = None
    merge_enaho_modules = None
    create_panel_data = None

# Null Analysis (esencial)
try:
    from .null_analysis import ENAHONullAnalyzer, analyze_null_patterns, generate_null_report

    _null_analysis_available = True
except ImportError:
    _null_analysis_available = False
    ENAHONullAnalyzer = None
    analyze_null_patterns = None
    generate_null_report = None

# =====================================================
# LAZY LOADING SETUP (Módulos pesados)
# =====================================================

# Mapeo de atributos a imports perezosos
# Formato: "nombre_atributo": ("modulo.submodulo", "nombre_en_modulo")
_LAZY_IMPORTS = {
    # Statistical Analysis (pesado)
    "PovertyIndicators": ("statistical_analysis", "PovertyIndicators"),
    "InequalityMeasures": ("statistical_analysis", "InequalityMeasures"),
    "WelfareAnalysis": ("statistical_analysis", "WelfareAnalysis"),
    "create_statistical_analyzer": ("statistical_analysis", "create_statistical_analyzer"),
    "quick_poverty_analysis": ("statistical_analysis", "quick_poverty_analysis"),
    # Data Quality (pesado)
    "DataQualityAssessment": ("data_quality", "DataQualityAssessment"),
    "assess_data_quality": ("data_quality", "assess_data_quality"),
    "quick_quality_check": ("data_quality", "quick_quality_check"),
    # Reporting (pesado - visualizaciones)
    "ReportGenerator": ("reporting", "ReportGenerator"),
    "VisualizationEngine": ("reporting", "VisualizationEngine"),
    "generate_enaho_report": ("reporting", "generate_enaho_report"),
    "create_quick_dashboard": ("reporting", "create_quick_dashboard"),
    # ML Imputation (pesado - sklearn, etc)
    "MLImputationManager": ("null_analysis.strategies.ml_imputation", "MLImputationManager"),
    "create_ml_imputation_manager": (
        "null_analysis.strategies.ml_imputation",
        "create_ml_imputation_manager",
    ),
    "quick_ml_imputation": ("null_analysis.strategies.ml_imputation", "quick_ml_imputation"),
    # Performance (pesado - benchmarking, monitoring)
    "MemoryMonitor": ("performance", "MemoryMonitor"),
    "DataFrameOptimizer": ("performance", "DataFrameOptimizer"),
    "memory_optimized_context": ("performance", "memory_optimized_context"),
    "optimize_pandas_settings": ("performance", "optimize_pandas_settings"),
    "create_memory_optimizer": ("performance", "create_memory_optimizer"),
    "StreamingProcessor": ("performance", "StreamingProcessor"),
    "StreamingConfig": ("performance", "StreamingConfig"),
    "create_streaming_processor": ("performance", "create_streaming_processor"),
    "ENAHOBenchmarkSuite": ("performance", "ENAHOBenchmarkSuite"),
    "run_quick_benchmark": ("performance", "run_quick_benchmark"),
    "quick_performance_check": ("performance", "quick_performance_check"),
    "ENAHOAsyncDownloader": ("performance", "ENAHOAsyncDownloader"),
    "create_async_downloader": ("performance", "create_async_downloader"),
    "show_performance_status": ("performance", "show_performance_status"),
    "create_performance_suite": ("performance", "create_performance_suite"),
    "optimize_for_large_datasets": ("performance", "optimize_for_large_datasets"),
}

# Cache de módulos importados
_imported_modules = {}

# Flags de disponibilidad (determinados bajo demanda)
_statistical_analysis_available = None
_data_quality_available = None
_reporting_available = None
_ml_imputation_available = None
_performance_available = None


def __getattr__(name: str) -> Any:
    """
    Lazy import de atributos pesados del módulo.

    Esta función se llama automáticamente cuando se accede a un atributo
    que no existe en el módulo, permitiendo imports diferidos (lazy loading).

    Args:
        name: Nombre del atributo solicitado.

    Returns:
        El objeto importado desde el módulo correspondiente.

    Raises:
        AttributeError: Si el atributo no existe o el módulo no está disponible.

    Example:
        >>> import enahopy
        >>> # Primera llamada carga el módulo performance
        >>> monitor = enahopy.MemoryMonitor()
        >>> # Segunda llamada usa cache (instantáneo)
        >>> optimizer = enahopy.DataFrameOptimizer()
    """
    # Verificar si es un import lazy registrado
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]

        # Importar módulo si no está en cache
        if module_path not in _imported_modules:
            try:
                # Import dinámico del módulo
                parts = module_path.split(".")
                if len(parts) == 1:
                    # Import de primer nivel (ej: "performance")
                    module = __import__(
                        f".{module_path}", fromlist=[attr_name], package=__package__
                    )
                else:
                    # Import anidado (ej: "null_analysis.strategies.ml_imputation")
                    module = __import__(
                        f".{module_path}", fromlist=[attr_name], package=__package__
                    )

                _imported_modules[module_path] = module

                # Actualizar flag de disponibilidad
                _update_availability_flag(module_path, True)

            except ImportError as e:
                # Actualizar flag de disponibilidad
                _update_availability_flag(module_path, False)

                raise AttributeError(
                    f"Módulo '{module_path}' no disponible. "
                    f"Error: {str(e)}. "
                    f"Instalar dependencias: pip install enahopy[all]"
                ) from e

        # Retornar atributo del módulo cacheado
        module = _imported_modules[module_path]
        if hasattr(module, attr_name):
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"'{attr_name}' no encontrado en módulo '{module_path}'")

    # Si no es lazy import, error estándar
    raise AttributeError(f"module 'enahopy' has no attribute '{name}'")


def __dir__():
    """
    Retorna lista de atributos disponibles para autocomplete.

    Returns:
        Lista de nombres de atributos accesibles desde el módulo.
    """
    # Combinar atributos actuales con lazy imports
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())


def _update_availability_flag(module_path: str, available: bool) -> None:
    """
    Actualiza el flag de disponibilidad global para un módulo.

    Args:
        module_path: Ruta del módulo (ej: "performance", "statistical_analysis").
        available: Si el módulo está disponible.
    """
    global _statistical_analysis_available, _data_quality_available
    global _reporting_available, _ml_imputation_available, _performance_available

    # Determinar qué flag actualizar basado en el módulo
    if module_path == "statistical_analysis":
        _statistical_analysis_available = available
    elif module_path == "data_quality":
        _data_quality_available = available
    elif module_path == "reporting":
        _reporting_available = available
    elif module_path.startswith("null_analysis.strategies.ml_imputation"):
        _ml_imputation_available = available
    elif module_path == "performance":
        _performance_available = available


def show_status(verbose: bool = True) -> None:
    """
    Show the status of all components.

    Args:
        verbose: If True, shows detailed information about BUILD and MEASURE phases.
                 Default is True.

    Returns:
        None. Prints status information to stdout.

    Example:
        >>> import enahopy
        >>> enahopy.show_status(verbose=True)
        enahopy v0.1.2 - Estado de componentes:
        --------------------------------------------------
        [OK] Loader: Disponible
        ...
    """
    print(f"enahopy v{__version__} - Estado de componentes:")
    print("-" * 50)

    components = {
        "Loader": _loader_available,
        "Merger": _merger_available,
        "Null_analysis": _null_analysis_available,
        "Statistical_analysis": _statistical_analysis_available,
        "Data_quality": _data_quality_available,
        "Reporting": _reporting_available,
        "ML_imputation": _ml_imputation_available,
        "Performance": _performance_available,
    }

    for name, available in components.items():
        if available is None:
            status = "Lazy (no cargado aún)"
            symbol = "[~]"
        elif available:
            status = "Disponible"
            symbol = "[OK]"
        else:
            status = "No disponible"
            symbol = "[X]"
        print(f"{symbol} {name}: {status}")

    if verbose:
        print("\nBUILD Phase 3 Features:")
        print(
            f"   - Advanced Statistical Analysis: {'[OK]' if _statistical_analysis_available else '[X]'}"
        )
        print(f"   - ML-based Imputation: {'[OK]' if _ml_imputation_available else '[X]'}")
        print(f"   - Data Quality Assessment: {'[OK]' if _data_quality_available else '[X]'}")
        print(f"   - Automated Reporting: {'[OK]' if _reporting_available else '[X]'}")
        print("\nMEASURE Phase Features:")
        print(f"   - Async Downloading: {'[OK]' if _performance_available else '[X]'}")
        print(f"   - Memory Optimization: {'[OK]' if _performance_available else '[X]'}")
        print(f"   - Streaming Processing: {'[OK]' if _performance_available else '[X]'}")
        print(f"   - Performance Benchmarks: {'[OK]' if _performance_available else '[X]'}")


def get_available_components() -> Dict[str, bool]:
    """
    Return status of components.

    Returns:
        Dictionary mapping component names to their availability status.
        Keys are component names (str), values are availability flags (bool).

    Example:
        >>> import enahopy
        >>> components = enahopy.get_available_components()
        >>> if components['loader']:
        ...     print("Loader is available")
    """
    return {
        "loader": _loader_available,
        "merger": _merger_available,
        "null_analysis": _null_analysis_available,
        "statistical_analysis": _statistical_analysis_available,
        "data_quality": _data_quality_available,
        "reporting": _reporting_available,
        "ml_imputation": _ml_imputation_available,
        "performance": _performance_available,
    }


# Build __all__ list dynamically
__all__ = ["__version__", "__version_info__", "show_status", "get_available_components"]

if _loader_available:
    __all__.extend(
        [
            "ENAHODataDownloader",
            "ENAHOLocalReader",
            "download_enaho_data",
            "read_enaho_file",
            "ENAHOUtils",
        ]
    )

if _merger_available:
    __all__.extend(["ENAHOMerger", "merge_enaho_modules", "create_panel_data"])

if _null_analysis_available:
    __all__.extend(["ENAHONullAnalyzer", "analyze_null_patterns", "generate_null_report"])

# Phase 3 Advanced Features
if _statistical_analysis_available:
    __all__.extend(
        [
            "PovertyIndicators",
            "InequalityMeasures",
            "WelfareAnalysis",
            "create_statistical_analyzer",
            "quick_poverty_analysis",
        ]
    )

if _data_quality_available:
    __all__.extend(["DataQualityAssessment", "assess_data_quality", "quick_quality_check"])

if _reporting_available:
    __all__.extend(
        [
            "ReportGenerator",
            "VisualizationEngine",
            "generate_enaho_report",
            "create_quick_dashboard",
        ]
    )

if _ml_imputation_available:
    __all__.extend(["MLImputationManager", "create_ml_imputation_manager", "quick_ml_imputation"])

# MEASURE Phase Performance Features
if _performance_available:
    __all__.extend(
        [
            # Memory optimization
            "MemoryMonitor",
            "DataFrameOptimizer",
            "memory_optimized_context",
            "optimize_pandas_settings",
            "create_memory_optimizer",
            # Streaming processing
            "StreamingProcessor",
            "StreamingConfig",
            "create_streaming_processor",
            # Performance benchmarking
            "ENAHOBenchmarkSuite",
            "run_quick_benchmark",
            "quick_performance_check",
            # Async downloading
            "ENAHOAsyncDownloader",
            "create_async_downloader",
            # Utility functions
            "show_performance_status",
            "create_performance_suite",
            "optimize_for_large_datasets",
        ]
    )
