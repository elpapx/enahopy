"""
ENAHO Merger - API Pública
=========================

Exportaciones principales y funciones de conveniencia para el sistema
de fusión geográfica y merge entre módulos ENAHO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .config import (  # Configuraciones; Enums geográficos; Enums de módulos; Dataclasses de resultados; Constantes
    DEPARTAMENTOS_VALIDOS,
    PATRONES_GEOGRAFICOS,
    GeoMergeConfiguration,
    GeoValidationResult,
    ModuleMergeConfig,
    ModuleMergeLevel,
    ModuleMergeResult,
    ModuleMergeStrategy,
    ModuleType,
    NivelTerritorial,
    TipoManejoDuplicados,
    TipoManejoErrores,
    TipoValidacionUbigeo,
)
from .core import ENAHOGeoMerger
from .exceptions import (  # Excepciones geográficas; Excepciones de módulos; Excepciones de calidad
    ConfigurationError,
    ConflictResolutionError,
    DataQualityError,
    DuplicateHandlingError,
    GeoMergeError,
    IncompatibleModulesError,
    MergeKeyError,
    ModuleMergeError,
    ModuleValidationError,
    TerritorialInconsistencyError,
    UbigeoValidationError,
    ValidationThresholdError,
)
from .geographic.patterns import GeoPatternDetector
from .geographic.strategies import DuplicateStrategyFactory
from .geographic.validators import GeoDataQualityValidator, TerritorialValidator, UbigeoValidator
from .modules.merger import ENAHOModuleMerger
from .modules.validator import ModuleValidator

# Import de panel con manejo de errores
try:
    from .panel.creator import PanelCreator, create_panel_data
except ImportError:
    # Si panel no está disponible, crear función dummy
    PanelCreator = None

    def create_panel_data(*args, **kwargs):
        raise ImportError("Panel module not available")


# =====================================================
# FUNCIONES DE CONVENIENCIA PRINCIPALES
# =====================================================


def merge_with_geography(
    df_principal: "pd.DataFrame",
    df_geografia: "pd.DataFrame",
    columna_union: str = "ubigeo",
    columnas_geograficas: dict = None,
    config: GeoMergeConfiguration = None,
    verbose: bool = True,
) -> tuple:
    """
    Función de conveniencia para fusión geográfica básica.

    Args:
        df_principal: DataFrame principal
        df_geografia: DataFrame geográfico
        columna_union: Columna para la fusión
        columnas_geograficas: Mapeo de columnas geográficas
        config: Configuración personalizada
        verbose: Si mostrar información detallada

    Returns:
        Tupla con (DataFrame fusionado, Resultado de validación)
    """
    geo_config = config or GeoMergeConfiguration(columna_union=columna_union)
    merger = ENAHOGeoMerger(geo_config=geo_config, verbose=verbose)

    return merger.merge_geographic_data(
        df_principal=df_principal,
        df_geografia=df_geografia,
        columnas_geograficas=columnas_geograficas,
        columna_union=columna_union,
    )


def merge_enaho_modules(
    modules_dict: dict,
    base_module: str = "34",
    level: str = "hogar",
    strategy: str = "coalesce",
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Función de conveniencia para merge rápido entre módulos.

    Args:
        modules_dict: Diccionario {codigo_modulo: dataframe}
        base_module: Módulo base
        level: Nivel de merge ("hogar", "persona", "vivienda")
        strategy: Estrategia para conflictos
        verbose: Si mostrar logs

    Returns:
        DataFrame combinado
    """
    module_config = ModuleMergeConfig(
        merge_level=ModuleMergeLevel(level), merge_strategy=ModuleMergeStrategy(strategy)
    )

    merger = ENAHOGeoMerger(module_config=module_config, verbose=verbose)
    result = merger.merge_multiple_modules(modules_dict, base_module, module_config)

    if verbose:
        print(result.get_summary_report())

    return result.merged_df


def merge_modules_with_geography(
    modules_dict: dict,
    df_geografia: "pd.DataFrame",
    base_module: str = "34",
    level: str = "hogar",
    strategy: str = "coalesce",
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Función de conveniencia para merge combinado (módulos + geografía).

    Args:
        modules_dict: Diccionario con módulos ENAHO
        df_geografia: DataFrame con información geográfica
        base_module: Módulo base
        level: Nivel de merge
        strategy: Estrategia para conflictos
        verbose: Si mostrar logs

    Returns:
        DataFrame final con módulos combinados y geografía
    """
    module_config = ModuleMergeConfig(
        merge_level=ModuleMergeLevel(level), merge_strategy=ModuleMergeStrategy(strategy)
    )

    merger = ENAHOGeoMerger(module_config=module_config, verbose=verbose)

    result_df, report = merger.merge_modules_with_geography(
        modules_dict=modules_dict,
        df_geografia=df_geografia,
        base_module=base_module,
        merge_config=module_config,
    )

    if verbose:
        print(
            f"""
🔗🗺️  MERGE COMBINADO COMPLETADO
===============================
Módulos procesados: {report['processing_summary']['modules_processed']}
Secuencia: {report['processing_summary']['merge_sequence']}
Registros finales: {report['processing_summary']['final_records']:,}
Cobertura geográfica: {report['processing_summary']['geographic_coverage']:.1f}%
Calidad general: {report['overall_quality']['quality_grade']}
        """
        )

    return result_df


def validate_ubigeo_data(
    df: "pd.DataFrame",
    columna_ubigeo: str = "ubigeo",
    tipo_validacion: TipoValidacionUbigeo = TipoValidacionUbigeo.STRUCTURAL,
    verbose: bool = True,
) -> GeoValidationResult:
    """
    Función de conveniencia para validar datos UBIGEO.

    Args:
        df: DataFrame a validar
        columna_ubigeo: Columna con códigos UBIGEO
        tipo_validacion: Tipo de validación a realizar
        verbose: Si mostrar información detallada

    Returns:
        Resultado de validación
    """
    config = GeoMergeConfiguration(
        columna_union=columna_ubigeo, tipo_validacion_ubigeo=tipo_validacion
    )

    merger = ENAHOGeoMerger(geo_config=config, verbose=verbose)
    return merger.validate_geographic_data(df, columna_ubigeo)


def detect_geographic_columns(
    df: "pd.DataFrame", confianza_minima: float = 0.8, verbose: bool = True
) -> dict:
    """
    Función de conveniencia para detectar columnas geográficas.

    Args:
        df: DataFrame a analizar
        confianza_minima: Umbral mínimo de confianza
        verbose: Si mostrar información detallada

    Returns:
        Diccionario con columnas geográficas detectadas
    """
    try:
        from ..loader import setup_logging
    except ImportError:
        import logging

        def setup_logging(verbose):
            logger = logging.getLogger("geo_detector")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(levelname)s] %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO if verbose else logging.WARNING)
            return logger

    logger = setup_logging(verbose)
    detector = GeoPatternDetector(logger)
    return detector.detectar_columnas_geograficas(df, confianza_minima)


def extract_ubigeo_components(
    df: "pd.DataFrame", columna_ubigeo: str = "ubigeo", verbose: bool = True
) -> "pd.DataFrame":
    """
    Función de conveniencia para extraer componentes de UBIGEO.

    Args:
        df: DataFrame con códigos UBIGEO
        columna_ubigeo: Columna con UBIGEO
        verbose: Si mostrar información detallada

    Returns:
        DataFrame con componentes territoriales
    """
    merger = ENAHOGeoMerger(verbose=verbose)
    return merger.extract_territorial_components(df, columna_ubigeo)


def validate_module_compatibility(
    modules_dict: dict, level: str = "hogar", verbose: bool = True
) -> dict:
    """
    Función de conveniencia para validar compatibilidad entre módulos.

    Args:
        modules_dict: Diccionario con módulos a validar
        level: Nivel de merge
        verbose: Si mostrar logs

    Returns:
        Reporte de compatibilidad
    """
    merger = ENAHOGeoMerger(verbose=verbose)
    compatibility = merger.validate_module_compatibility(modules_dict, level)

    if verbose:
        status = "✅ COMPATIBLE" if compatibility["overall_compatible"] else "⚠️  CON PROBLEMAS"
        print(
            f"""
📋 REPORTE DE COMPATIBILIDAD
===========================
Estado: {status}
Nivel de merge: {level}
Módulos analizados: {len(compatibility['modules_analyzed'])}

Recomendaciones:
{chr(10).join(['  - ' + rec for rec in compatibility['recommendations']])}
        """
        )

        if compatibility["potential_issues"]:
            print(
                f"""
⚠️  Problemas detectados:
{chr(10).join(['  - ' + issue for issue in compatibility['potential_issues']])}
            """
            )

    return compatibility


def create_merge_report(
    df: "pd.DataFrame",
    include_geographic: bool = True,
    include_quality: bool = True,
    verbose: bool = True,
) -> str:
    """
    Crea reporte integral de análisis de datos.

    Args:
        df: DataFrame a analizar
        include_geographic: Si incluir análisis geográfico
        include_quality: Si incluir métricas de calidad
        verbose: Si mostrar logs

    Returns:
        Reporte formateado como string
    """
    merger = ENAHOGeoMerger(verbose=verbose)
    return merger.create_comprehensive_report(df, include_geographic, include_quality)


# =====================================================
# FUNCIONES UTILITARIAS
# =====================================================


def get_available_duplicate_strategies() -> list:
    """Retorna lista de estrategias disponibles para manejo de duplicados"""
    return DuplicateStrategyFactory.get_available_strategies()


def get_strategy_info(strategy: TipoManejoDuplicados) -> dict:
    """Obtiene información detallada sobre una estrategia de duplicados"""
    return DuplicateStrategyFactory.get_strategy_info(strategy)


def validate_merge_configuration(
    geo_config: GeoMergeConfiguration = None, module_config: ModuleMergeConfig = None
) -> dict:
    """
    Valida configuraciones de merge.

    Args:
        geo_config: Configuración geográfica
        module_config: Configuración de módulos

    Returns:
        Resultado de validación
    """
    validation = {"valid": True, "warnings": [], "errors": []}

    if geo_config:
        # Validar configuración geográfica
        if geo_config.manejo_duplicados == TipoManejoDuplicados.AGGREGATE:
            if not geo_config.funciones_agregacion:
                validation["errors"].append(
                    "funciones_agregacion requerido para estrategia AGGREGATE"
                )
                validation["valid"] = False

        if geo_config.manejo_duplicados == TipoManejoDuplicados.BEST_QUALITY:
            if not geo_config.columna_calidad:
                validation["errors"].append(
                    "columna_calidad requerido para estrategia BEST_QUALITY"
                )
                validation["valid"] = False

        if geo_config.chunk_size <= 0:
            validation["errors"].append("chunk_size debe ser positivo")
            validation["valid"] = False

    if module_config:
        # Validar configuración de módulos
        if not module_config.hogar_keys:
            validation["warnings"].append("hogar_keys vacío - usando valores por defecto")

        if module_config.min_match_rate < 0 or module_config.min_match_rate > 1:
            validation["errors"].append("min_match_rate debe estar entre 0 y 1")
            validation["valid"] = False

        if module_config.max_conflicts_allowed < 0:
            validation["errors"].append("max_conflicts_allowed debe ser positivo")
            validation["valid"] = False

    return validation


def create_optimized_merge_config(
    df_size: int, merge_type: str = "geographic", performance_priority: str = "balanced"
) -> dict:
    """
    Crea configuración optimizada según tamaño de datos.

    Args:
        df_size: Tamaño del DataFrame (número de filas)
        merge_type: Tipo de merge ("geographic" o "module")
        performance_priority: Prioridad ("memory", "speed", "balanced")

    Returns:
        Configuración optimizada
    """
    configs = {}

    if merge_type == "geographic":
        if df_size < 10000:  # Dataset pequeño
            configs["geo_config"] = GeoMergeConfiguration(
                chunk_size=df_size, optimizar_memoria=False, usar_cache=False
            )
        elif df_size < 100000:  # Dataset mediano
            configs["geo_config"] = GeoMergeConfiguration(
                chunk_size=10000,
                optimizar_memoria=performance_priority == "memory",
                usar_cache=True,
            )
        else:  # Dataset grande
            configs["geo_config"] = GeoMergeConfiguration(
                chunk_size=50000 if performance_priority != "memory" else 25000,
                optimizar_memoria=True,
                usar_cache=True,
            )

    elif merge_type == "module":
        if df_size < 50000:  # Dataset pequeño
            configs["module_config"] = ModuleMergeConfig(chunk_processing=False, chunk_size=df_size)
        else:  # Dataset grande
            configs["module_config"] = ModuleMergeConfig(
                chunk_processing=performance_priority == "memory",
                chunk_size=25000 if performance_priority == "memory" else 50000,
            )

    return configs


# =====================================================
# COMPATIBILIDAD CON VERSIÓN ANTERIOR
# =====================================================


def agregar_info_geografica(
    df_principal: "pd.DataFrame",
    df_geografia: "pd.DataFrame",
    columna_union: str = "ubigeo",
    columnas_geograficas: dict = None,
    manejo_duplicados: str = "first",
    manejo_errores: str = "coerce",
    valor_faltante="DESCONOCIDO",
    reporte_duplicados: bool = False,
) -> "pd.DataFrame":
    """
    Función de compatibilidad con la API anterior.

    DEPRECATED: Use merge_with_geography() para nuevas implementaciones.
    """
    import warnings

    warnings.warn(
        "agregar_info_geografica está deprecated. Use merge_with_geography() en su lugar.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convertir parámetros a nueva configuración
    config = GeoMergeConfiguration(
        columna_union=columna_union,
        manejo_duplicados=TipoManejoDuplicados(manejo_duplicados),
        manejo_errores=TipoManejoErrores(manejo_errores),
        valor_faltante=valor_faltante,
        reporte_duplicados=reporte_duplicados,
    )

    result_df, _ = merge_with_geography(
        df_principal=df_principal,
        df_geografia=df_geografia,
        columna_union=columna_union,
        columnas_geograficas=columnas_geograficas,
        config=config,
    )

    return result_df


# =====================================================
# EXPORTACIONES PÚBLICAS
# =====================================================


# Aliases para compatibilidad
ENAHOMerger = ENAHOGeoMerger

__all__ = [
    "create_panel_data",
    # Alias principal
    "ENAHOMerger",
    # Clases principales
    "ENAHOGeoMerger",
    "ENAHOModuleMerger",
    # Configuraciones
    "GeoMergeConfiguration",
    "ModuleMergeConfig",
    # Enums geográficos
    "TipoManejoDuplicados",
    "TipoManejoErrores",
    "NivelTerritorial",
    "TipoValidacionUbigeo",
    # Enums de módulos
    "ModuleMergeLevel",
    "ModuleMergeStrategy",
    "ModuleType",
    # Resultados
    "GeoValidationResult",
    "ModuleMergeResult",
    # Validadores y detectores
    "UbigeoValidator",
    "TerritorialValidator",
    "GeoDataQualityValidator",
    "GeoPatternDetector",
    "ModuleValidator",
    # Factories
    "DuplicateStrategyFactory",
    # Excepciones principales
    "GeoMergeError",
    "ModuleMergeError",
    "UbigeoValidationError",
    "IncompatibleModulesError",
    # Funciones principales
    "merge_with_geography",
    "merge_enaho_modules",
    "merge_modules_with_geography",
    "validate_ubigeo_data",
    "detect_geographic_columns",
    "extract_ubigeo_components",
    "validate_module_compatibility",
    "create_merge_report",
    # Utilidades
    "get_available_duplicate_strategies",
    "get_strategy_info",
    "validate_merge_configuration",
    "create_optimized_merge_config",
    # Constantes
    "DEPARTAMENTOS_VALIDOS",
    "PATRONES_GEOGRAFICOS",
    # Compatibilidad (deprecated)
    "agregar_info_geografica",
]

# Información del módulo
__version__ = "2.0.0"
__author__ = "ENAHO Analyzer Team"
__description__ = "Advanced geographic and module merging system for INEI microdata"
