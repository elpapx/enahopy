"""
ENAHO Merger - Public API
=========================

Main exports and convenience functions for the ENAHO geographic and module
merging system.

The ENAHO Merger module provides comprehensive data integration capabilities
for Peruvian household survey (ENAHO) microdata, enabling researchers to:

1. Enrich survey data with geographic information using UBIGEO codes
2. Merge multiple survey modules at household, person, or dwelling levels
3. Validate data quality with territorial consistency checks
4. Handle duplicates with multiple strategies (first, last, aggregate, best quality)
5. Optimize memory usage for large datasets (>500K records)
6. Generate detailed quality reports and merge statistics

Architecture:
    The module is organized into specialized submodules:

    - **core**: Main ENAHOGeoMerger orchestrating all merge operations
    - **modules**: ENAHO module merging (ENAHOModuleMerger, ModuleValidator)
    - **geographic**: Geographic data integration (validators, patterns, strategies)
    - **config**: Configuration classes and enums
    - **exceptions**: Specialized exceptions for error handling
    - **panel**: Longitudinal panel data creation (optional)

Main Classes:
    - ENAHOGeoMerger: Central integration hub for geographic and module merges
    - ENAHOModuleMerger: Specialized merger for ENAHO survey modules
    - GeoMergeConfiguration: Configuration for geographic merge operations
    - ModuleMergeConfig: Configuration for module-level merges

Main Functions:
    - merge_with_geography(): Quick geographic data enrichment
    - merge_enaho_modules(): Quick multi-module integration
    - merge_modules_with_geography(): Combined module + geographic merge
    - validate_ubigeo_data(): Geographic data validation
    - detect_geographic_columns(): Automatic column detection

Enums and Types:
    - TipoManejoDuplicados: Duplicate handling strategies (FIRST, LAST, AGGREGATE, etc.)
    - ModuleMergeLevel: Merge levels (HOGAR, PERSONA, VIVIENDA)
    - ModuleMergeStrategy: Conflict resolution (COALESCE, KEEP_LEFT, KEEP_RIGHT, etc.)
    - TipoValidacionUbigeo: UBIGEO validation types (STRUCTURAL, SEMANTIC, TERRITORIAL)

Examples:
    Basic geographic enrichment:

    >>> from enahopy.merger import merge_with_geography
    >>> result_df, validation = merge_with_geography(
    ...     df_principal=df_survey_data,
    ...     df_geografia=df_geo_reference,
    ...     columna_union='ubigeo'
    ... )
    >>> print(f"Coverage: {validation.coverage_percentage:.1f}%")

    Multi-module merge with geography:

    >>> from enahopy.merger import merge_modules_with_geography
    >>> modules = {'34': df_sumaria, '01': df_vivienda, '02': df_personas}
    >>> final_df = merge_modules_with_geography(
    ...     modules_dict=modules,
    ...     df_geografia=df_geo,
    ...     base_module='34',
    ...     level='hogar',
    ...     strategy='coalesce'
    ... )

    Advanced usage with custom configuration:

    >>> from enahopy.merger import ENAHOGeoMerger, GeoMergeConfiguration
    >>> from enahopy.merger.config import TipoManejoDuplicados
    >>>
    >>> geo_config = GeoMergeConfiguration(
    ...     manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
    ...     funciones_agregacion={'poblacion': 'sum', 'ingreso': 'mean'},
    ...     validar_consistencia_territorial=True
    ... )
    >>> merger = ENAHOGeoMerger(geo_config=geo_config, verbose=True)
    >>> result, validation = merger.merge_geographic_data(df_data, df_geo)

    Feasibility analysis before merging:

    >>> from enahopy.merger import validate_module_compatibility
    >>> compatibility = validate_module_compatibility(
    ...     modules_dict={'34': df1, '01': df2, '02': df3},
    ...     level='hogar'
    ... )
    >>> if compatibility['overall_compatible']:
    ...     # Proceed with merge
    ...     pass

Performance Considerations:
    - Datasets <100K records: Standard merge (no optimization needed)
    - Datasets 100K-500K records: Enable chunk processing
    - Datasets >500K records: Enable memory optimization and categorical encoding
    - Datasets >1M records: Consider Parquet format and parallel processing

See Also:
    - :class:`~enahopy.merger.ENAHOGeoMerger`: Main merger class
    - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Geographic merge config
    - :class:`~enahopy.merger.config.ModuleMergeConfig`: Module merge config
    - :mod:`~enahopy.loader`: Data loading and caching
    - :mod:`~enahopy.null_analyzer`: Missing data analysis

References:
    INEI ENAHO Technical Documentation: https://www.inei.gob.pe/enaho/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
    columnas_geograficas: Optional[Dict[str, str]] = None,
    config: Optional[GeoMergeConfiguration] = None,
    verbose: bool = True,
) -> Tuple["pd.DataFrame", GeoValidationResult]:
    """Merge ENAHO survey data with geographic reference information using UBIGEO.

    Performs a left join between principal dataset and geographic reference,
    enriching survey records with administrative division information
    (departamento, provincia, distrito) and optionally geographic coordinates.
    Includes automatic UBIGEO validation and quality assessment.

    Args:
        df_principal: Main ENAHO survey DataFrame with UBIGEO codes.
            Must contain the merge column specified in columna_union parameter.
            All records from this DataFrame are preserved in result (left join).
        df_geografia: Geographic reference DataFrame with UBIGEO codes and
            associated geographic metadata. Should have unique UBIGEOs to avoid
            duplicate record creation.
        columna_union: Column name used as merge key. Must exist in both
            DataFrames. Default: "ubigeo". Defaults to "ubigeo".
        columnas_geograficas: Optional mapping of geographic column names to
            include from df_geografia. If None, all columns are included.
            Format: {"source_col": "target_col"}. Defaults to None.
        config: Optional GeoMergeConfiguration for controlling merge behavior,
            duplicate handling strategy, validation settings, and error management.
            If None, uses default configuration. Defaults to None.
        verbose: If True, prints detailed merge progress and validation results.
            Defaults to True.

    Returns:
        Tuple containing:

        - pd.DataFrame: Result DataFrame with geographic columns added from
          df_geografia. All rows from df_principal are preserved (left join).
          Unmatched records have NaN values in geographic columns.
        - GeoValidationResult: Quality metrics including match rate, coverage
          percentage, and any validation issues detected.

    Raises:
        GeoMergeError: If merge operation fails due to missing columns,
            incompatible data types, or pandas merge errors.

    Examples:
        Basic geographic enrichment:

        >>> from enahopy.merger import merge_with_geography
        >>> import pandas as pd
        >>>
        >>> df_survey = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102', '150103'],
        ...     'ingreso': [2000, 1500, 1800],
        ...     'conglome': ['001', '002', '003']
        ... })
        >>> df_geo = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102', '150103'],
        ...     'departamento': ['Lima', 'Lima', 'Lima'],
        ...     'provincia': ['Lima', 'Lima', 'Lima'],
        ...     'distrito': ['Lima', 'San Isidro', 'Miraflores']
        ... })
        >>>
        >>> result_df, validation = merge_with_geography(
        ...     df_principal=df_survey,
        ...     df_geografia=df_geo,
        ...     columna_union='ubigeo'
        ... )
        >>> print(f"Match rate: {validation.match_rate:.1f}%")
        >>> print(f"Columns added: {validation.matched_count}")
        Match rate: 100.0%
        Columns added: 3

        With custom configuration:

        >>> from enahopy.merger import merge_with_geography, GeoMergeConfiguration
        >>> from enahopy.merger.config import TipoManejoDuplicados
        >>>
        >>> config = GeoMergeConfiguration(
        ...     manejo_duplicados=TipoManejoDuplicados.FIRST,
        ...     validar_formato_ubigeo=True
        ... )
        >>> result_df, validation = merge_with_geography(
        ...     df_principal=df_survey,
        ...     df_geografia=df_geo,
        ...     config=config
        ... )

    Note:
        - Performs left join: all df_principal records are preserved
        - Automatically validates UBIGEO format if enabled in config
        - Handles duplicate UBIGEO codes according to configured strategy
        - Memory-optimized for large datasets (>500K records)
        - Results include detailed quality metrics for validation

    See Also:
        - :func:`merge_enaho_modules`: Merge multiple ENAHO modules
        - :func:`merge_modules_with_geography`: Combined module + geographic merge
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Full-featured merger class
        - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Configuration options
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
    modules_dict: Dict[str, "pd.DataFrame"],
    base_module: str = "34",
    level: str = "hogar",
    strategy: str = "coalesce",
    verbose: bool = True,
) -> "pd.DataFrame":
    """Merge multiple ENAHO survey modules with intelligent conflict resolution.

    Combines ENAHO modules (vivienda, personas, sumaria, etc.) into a single
    integrated dataset, handling hierarchical keys (conglome, vivienda, hogar,
    codperso) and resolving column conflicts automatically.

    Args:
        modules_dict: Dictionary mapping module codes to DataFrames.
            Format: {"34": df_sumaria, "01": df_vivienda, "02": df_personas}.
            All modules should contain appropriate household/person keys based
            on the specified merge level.
        base_module: Code of base module to use as anchor for merge.
            Default: "34" (sumaria). This module's records are preserved (left join).
            Defaults to "34".
        level: Merge level determining which keys are used.
            Options: "hogar" (household), "persona" (person), "vivienda" (dwelling).
            Determines which merge keys are required. Defaults to "hogar".
        strategy: Conflict resolution strategy for duplicate columns.
            Options: "coalesce" (fill nulls), "keep_left", "keep_right",
            "average" (for numeric), "concatenate" (for strings).
            Defaults to "coalesce".
        verbose: If True, prints detailed merge progress and statistics.
            Defaults to True.

    Returns:
        pd.DataFrame: Merged DataFrame combining all modules with all rows from
            base module preserved. Unmatched records have NaN in columns from
            other modules.

    Raises:
        IncompatibleModulesError: If modules lack required merge keys or have
            incompatible structures.
        ModuleMergeError: If merge operation fails or critical validation errors occur.

    Examples:
        Basic multi-module merge:

        >>> from enahopy.merger import merge_enaho_modules
        >>> import pandas as pd
        >>>
        >>> modules = {
        ...     '34': df_sumaria,  # Base module
        ...     '01': df_vivienda,
        ...     '02': df_personas
        ... }
        >>>
        >>> merged_df = merge_enaho_modules(
        ...     modules_dict=modules,
        ...     base_module='34',
        ...     level='hogar',
        ...     strategy='coalesce'
        ... )
        >>> print(f"Result shape: {merged_df.shape}")
        >>> print(f"Columns: {len(merged_df.columns)}")

        With person-level merge:

        >>> merged_df = merge_enaho_modules(
        ...     modules_dict=modules,
        ...     base_module='02',  # Personas module
        ...     level='persona'
        ... )

    Note:
        - Uses left join semantics: all base module records preserved
        - Automatically detects and handles merge keys
        - Memory-optimized for large datasets (>500K records)
        - Provides detailed merge statistics when verbose=True

    See Also:
        - :func:`merge_with_geography`: Geographic data enrichment
        - :func:`merge_modules_with_geography`: Combined merge operation
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Advanced merger class
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
    modules_dict: Dict[str, "pd.DataFrame"],
    df_geografia: "pd.DataFrame",
    base_module: str = "34",
    level: str = "hogar",
    strategy: str = "coalesce",
    verbose: bool = True,
) -> "pd.DataFrame":
    """Merge multiple ENAHO modules and enrich with geographic information.

    Combines this operation: module merge + geographic enrichment into a single
    workflow, providing a complete integrated dataset with administrative divisions
    and module-level information.

    Args:
        modules_dict: Dictionary mapping module codes to DataFrames.
            Format: {"34": df_sumaria, "01": df_vivienda, "02": df_personas}.
        df_geografia: Geographic reference DataFrame with UBIGEO codes and
            administrative division information.
        base_module: Code of base module for merge anchor. Default: "34".
            Defaults to "34".
        level: Merge level ("hogar", "persona", or "vivienda").
            Determines which merge keys are used. Defaults to "hogar".
        strategy: Conflict resolution strategy for duplicate columns.
            Options: "coalesce", "keep_left", "keep_right", "average",
            "concatenate". Defaults to "coalesce".
        verbose: If True, prints detailed progress and statistics.
            Defaults to True.

    Returns:
        pd.DataFrame: Final integrated DataFrame with all modules merged
            and geographic information added. Contains all rows from base module
            with matching geographic data attached.

    Raises:
        ModuleMergeError: If module merge operations fail.
        GeoMergeError: If geographic merge operation fails.

    Examples:
        Combined module and geographic merge:

        >>> from enahopy.merger import merge_modules_with_geography
        >>> import pandas as pd
        >>>
        >>> modules = {
        ...     '34': df_sumaria,
        ...     '01': df_vivienda,
        ...     '02': df_personas
        ... }
        >>> final_df = merge_modules_with_geography(
        ...     modules_dict=modules,
        ...     df_geografia=df_geo,
        ...     base_module='34',
        ...     level='hogar'
        ... )
        >>> print(f"Final shape: {final_df.shape}")
        >>> print("Columns:", final_df.columns.tolist())

    Note:
        - Performs modules merge first, then geographic enrichment
        - Uses left join semantics throughout
        - Memory-optimized for large datasets
        - Provides integrated quality metrics

    See Also:
        - :func:`merge_with_geography`: Geographic merge only
        - :func:`merge_enaho_modules`: Module merge only
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Advanced operations
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
    """Validate UBIGEO codes in ENAHO data for format and territorial consistency.

    Performs comprehensive validation of UBIGEO geographic codes including
    format checking, structural validation, and optional territorial hierarchy
    consistency verification.

    Args:
        df: DataFrame containing UBIGEO codes to validate.
        columna_ubigeo: Column name containing UBIGEO codes.
            Default: "ubigeo". Defaults to "ubigeo".
        tipo_validacion: Type of validation to perform.
            Options: STRUCTURAL (format only), SEMANTIC (with codes),
            TERRITORIAL (includes hierarchy checks).
            Default: STRUCTURAL. Defaults to TipoValidacionUbigeo.STRUCTURAL.
        verbose: If True, prints validation results and issues.
            Defaults to True.

    Returns:
        GeoValidationResult: Validation report with pass/fail status,
            error details, and quality metrics.

    Raises:
        GeoMergeError: If DataFrame validation fails.

    Examples:
        Basic UBIGEO validation:

        >>> from enahopy.merger import validate_ubigeo_data
        >>> from enahopy.merger.config import TipoValidacionUbigeo
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102', '150131', 'INVALID']
        ... })
        >>> result = validate_ubigeo_data(
        ...     df,
        ...     columna_ubigeo='ubigeo',
        ...     tipo_validacion=TipoValidacionUbigeo.STRUCTURAL
        ... )
        >>> print(f"Valid: {result.is_valid}")
        >>> print(f"Invalid records: {result.invalid_count}")

    Note:
        - STRUCTURAL: Checks format (6 digits)
        - SEMANTIC: Format + code existence
        - TERRITORIAL: Format + code + hierarchy consistency
        - Results cached for performance

    See Also:
        - :func:`detect_geographic_columns`: Auto-detect geographic columns
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Full merger with validation
    """
    config = GeoMergeConfiguration(
        columna_union=columna_ubigeo, tipo_validacion_ubigeo=tipo_validacion
    )

    merger = ENAHOGeoMerger(geo_config=config, verbose=verbose)
    return merger.validate_geographic_data(df, columna_ubigeo)


def detect_geographic_columns(
    df: "pd.DataFrame", confianza_minima: float = 0.8, verbose: bool = True
) -> Dict[str, Any]:
    """Automatically detect geographic columns in DataFrame.

    Uses pattern matching to identify UBIGEO codes and geographic columns
    (departamento, provincia, distrito) in ENAHO data, useful for automatic
    configuration of merge operations.

    Args:
        df: DataFrame to analyze for geographic columns.
        confianza_minima: Minimum confidence threshold (0-1) for column
            identification. Default: 0.8. Defaults to 0.8.
        verbose: If True, prints detection results.
            Defaults to True.

    Returns:
        Dict[str, Any]: Detection results with identified columns, confidence
            scores, and recommendations.

    Examples:
        Detect geographic columns:

        >>> from enahopy.merger import detect_geographic_columns
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102'],
        ...     'departamento': ['Lima', 'Lima'],
        ...     'provincia': ['Lima', 'Lima'],
        ...     'ingreso': [2000, 1500]
        ... })
        >>> results = detect_geographic_columns(df)
        >>> print(f"Detected: {results.get('detected_columns', {})}")

    See Also:
        - :func:`extract_ubigeo_components`: Extract territorial components
        - :func:`validate_ubigeo_data`: Validate geographic data
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
    """Extract departamento, provincia, and distrito codes from UBIGEO.

    Breaks down 6-digit UBIGEO codes into their territorial components:
    - Digits 1-2: Departamento code
    - Digits 3-4: Provincia code
    - Digits 5-6: Distrito code

    Args:
        df: DataFrame containing UBIGEO codes.
        columna_ubigeo: Column name with UBIGEO codes.
            Default: "ubigeo". Defaults to "ubigeo".
        verbose: If True, prints extraction details.
            Defaults to True.

    Returns:
        pd.DataFrame: Original DataFrame with added columns:
            - cod_departamento: 2-digit code
            - cod_provincia: 2-digit code
            - cod_distrito: 2-digit code

    Examples:
        Extract territorial components:

        >>> from enahopy.merger import extract_ubigeo_components
        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'ubigeo': ['150101', '080101']
        ... })
        >>> result = extract_ubigeo_components(df)
        >>> print(result[['ubigeo', 'cod_departamento', 'cod_provincia']])

    See Also:
        - :func:`validate_ubigeo_data`: Validate UBIGEO codes
        - :func:`merge_with_geography`: Geographic merge with validation
    """
    merger = ENAHOGeoMerger(verbose=verbose)
    return merger.extract_territorial_components(df, columna_ubigeo)


def validate_module_compatibility(
    modules_dict: Dict[str, "pd.DataFrame"], level: str = "hogar", verbose: bool = True
) -> Dict[str, Any]:
    """Validate compatibility between multiple ENAHO modules before merging.

    Pre-merge feasibility analysis checking module structure, required keys,
    and potential issues that could affect merge quality.

    Args:
        modules_dict: Dictionary mapping module codes to DataFrames to validate.
            Format: {"34": df_sumaria, "01": df_vivienda, "02": df_personas}.
        level: Merge level to validate compatibility for.
            Options: "hogar" (household), "persona" (person), "vivienda" (dwelling).
            Defaults to "hogar".
        verbose: If True, prints validation results and recommendations.
            Defaults to True.

    Returns:
        Dict[str, Any]: Compatibility report with:
            - overall_compatible: Bool indicating if modules can merge
            - modules_analyzed: List of modules checked
            - potential_issues: List of detected problems
            - recommendations: List of suggested fixes

    Examples:
        Validate module compatibility:

        >>> from enahopy.merger import validate_module_compatibility
        >>>
        >>> modules = {
        ...     '34': df_sumaria,
        ...     '01': df_vivienda,
        ...     '02': df_personas
        ... }
        >>> compatibility = validate_module_compatibility(
        ...     modules,
        ...     level='hogar'
        ... )
        >>> if compatibility['overall_compatible']:
        ...     print("Modules can be merged")
        ... else:
        ...     print("Issues detected:")
        ...     for issue in compatibility['potential_issues']:
        ...         print(f"  - {issue}")

    See Also:
        - :func:`merge_enaho_modules`: Merge modules after validation
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Advanced merger with validation
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
    """Create comprehensive merge analysis report for merged ENAHO data.

    Generates detailed report of merge quality, geographic coverage,
    and data completeness metrics.

    Args:
        df: Merged DataFrame to analyze.
        include_geographic: If True, includes geographic analysis metrics.
            Defaults to True.
        include_quality: If True, includes data quality metrics.
            Defaults to True.
        verbose: If True, prints report to console.
            Defaults to True.

    Returns:
        str: Formatted report as multi-line string.

    Examples:
        Generate merge report:

        >>> from enahopy.merger import create_merge_report
        >>>
        >>> report = create_merge_report(
        ...     df=merged_df,
        ...     include_geographic=True,
        ...     include_quality=True,
        ...     verbose=True
        ... )
        >>> with open('merge_report.txt', 'w') as f:
        ...     f.write(report)

    See Also:
        - :func:`validate_module_compatibility`: Pre-merge validation
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Advanced merger
    """
    merger = ENAHOGeoMerger(verbose=verbose)
    return merger.create_comprehensive_report(df, include_geographic, include_quality)


# =====================================================
# FUNCIONES UTILITARIAS
# =====================================================


def get_available_duplicate_strategies() -> List[str]:
    """Get list of available duplicate handling strategies.

    Returns:
        List[str]: Strategy names available for use in merge operations.

    Examples:
        >>> from enahopy.merger import get_available_duplicate_strategies
        >>> strategies = get_available_duplicate_strategies()
        >>> print(f"Available: {strategies}")
        Available: ['FIRST', 'LAST', 'AGGREGATE', ...]
    """
    return DuplicateStrategyFactory.get_available_strategies()


def get_strategy_info(strategy: TipoManejoDuplicados) -> Dict[str, Any]:
    """Get detailed information about a duplicate handling strategy.

    Args:
        strategy: Strategy enum value to get information for.

    Returns:
        Dict[str, Any]: Strategy information including description,
            use cases, and parameters.

    Examples:
        >>> from enahopy.merger import get_strategy_info
        >>> from enahopy.merger.config import TipoManejoDuplicados
        >>> info = get_strategy_info(TipoManejoDuplicados.AGGREGATE)
        >>> print(info['description'])
    """
    return DuplicateStrategyFactory.get_strategy_info(strategy)


def validate_merge_configuration(
    geo_config: Optional[GeoMergeConfiguration] = None,
    module_config: Optional[ModuleMergeConfig] = None,
) -> Dict[str, Any]:
    """Validate merge configurations for errors and inconsistencies.

    Performs comprehensive validation of GeoMergeConfiguration and
    ModuleMergeConfig, checking for required parameters, valid values,
    and potential issues.

    Args:
        geo_config: Optional geographic merge configuration to validate.
            Defaults to None.
        module_config: Optional module merge configuration to validate.
            Defaults to None.

    Returns:
        Dict[str, Any]: Validation result with:
            - valid (bool): Configuration is valid
            - warnings (List[str]): Non-critical issues
            - errors (List[str]): Critical issues preventing merge

    Examples:
        Validate configurations:

        >>> from enahopy.merger import (
        ...     validate_merge_configuration,
        ...     GeoMergeConfiguration,
        ...     ModuleMergeConfig
        ... )
        >>> from enahopy.merger.config import TipoManejoDuplicados
        >>>
        >>> geo_config = GeoMergeConfiguration(
        ...     manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
        ...     funciones_agregacion={'ingreso': 'mean'}
        ... )
        >>> result = validate_merge_configuration(geo_config=geo_config)
        >>> if result['valid']:
        ...     print("Configuration is valid")
        ... else:
        ...     for error in result['errors']:
        ...         print(f"ERROR: {error}")

    See Also:
        - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Geo config
        - :class:`~enahopy.merger.config.ModuleMergeConfig`: Module config
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
) -> Dict[str, Union[GeoMergeConfiguration, ModuleMergeConfig]]:
    """Create optimized merge configuration based on dataset size.

    Automatically generates appropriate configuration tuning chunk size,
    memory optimization, and caching based on dataset size and performance
    priorities.

    Args:
        df_size: Number of rows in DataFrame(s) to merge.
        merge_type: Type of merge operation.
            Options: "geographic" (UBIGEO merge), "module" (ENAHO modules).
            Defaults to "geographic".
        performance_priority: Optimization priority.
            Options: "memory" (minimize RAM), "speed" (maximize performance),
            "balanced" (tradeoff). Defaults to "balanced".

    Returns:
        Dict[str, Union[GeoMergeConfiguration, ModuleMergeConfig]]: Optimized
            configuration dictionary containing either 'geo_config' or
            'module_config' key with tuned settings.

    Examples:
        Create optimized config for large geographic merge:

        >>> from enahopy.merger import create_optimized_merge_config
        >>>
        >>> config_dict = create_optimized_merge_config(
        ...     df_size=500000,
        ...     merge_type='geographic',
        ...     performance_priority='speed'
        ... )
        >>> geo_config = config_dict['geo_config']
        >>> print(f"Chunk size: {geo_config.chunk_size}")

        For module merge with memory optimization:

        >>> config_dict = create_optimized_merge_config(
        ...     df_size=1000000,
        ...     merge_type='module',
        ...     performance_priority='memory'
        ... )

    Note:
        - <10K rows: Standard processing, no optimization
        - 10K-100K rows: Balanced chunk processing
        - 100K-500K rows: Chunk processing with caching
        - >500K rows: Aggressive optimization for memory/speed

    See Also:
        - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Geo config
        - :class:`~enahopy.merger.config.ModuleMergeConfig`: Module config
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
