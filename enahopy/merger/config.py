"""

ENAHO Merger - Configuraciones y Enums

=====================================



Configuraciones, enums y dataclasses para el sistema de fusión geográfica

y merge entre módulos ENAHO.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

# =====================================================

# ENUMS GEOGRÁFICOS

# =====================================================


class TipoManejoDuplicados(Enum):
    """Strategies for handling duplicate geographic records in UBIGEO-based merges.

    Defines different approaches to resolve situations where multiple records share
    the same UBIGEO code during geographic data fusion. The choice of strategy
    significantly impacts merge results and should align with data quality goals.

    Attributes:
        FIRST: Keep the first occurrence encountered and discard subsequent duplicates.
            Fast and deterministic but may not preserve the best quality record.
        LAST: Keep the last occurrence encountered and discard previous duplicates.
            Useful when later records represent updated or corrected information.
        ERROR: Raise GeoMergeError when duplicates are detected. Use this for strict
            validation when duplicates are unexpected and indicate data quality issues.
        KEEP_ALL: Retain all duplicate records without deduplication. Results in
            a one-to-many relationship that may inflate record counts.
        AGGREGATE: Combine duplicate records using specified aggregation functions.
            Requires funciones_agregacion configuration. Ideal for numerical data.
        MOST_RECENT: Select the most recent record based on a date/timestamp column.
            Requires columna_orden_duplicados configuration pointing to date field.
        BEST_QUALITY: Select the record with the best quality score. Requires
            columna_calidad configuration with numeric quality indicator.

    Examples:
        Using FIRST strategy (default):

        >>> from enahopy.merger.config import (
        ...     GeoMergeConfiguration,
        ...     TipoManejoDuplicados
        ... )
        >>> config = GeoMergeConfiguration(
        ...     manejo_duplicados=TipoManejoDuplicados.FIRST
        ... )

        Using AGGREGATE with custom functions:

        >>> config = GeoMergeConfiguration(
        ...     manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
        ...     funciones_agregacion={
        ...         'poblacion': 'sum',
        ...         'ingreso': 'mean',
        ...         'area': 'first'
        ...     }
        ... )

        Using BEST_QUALITY with quality column:

        >>> config = GeoMergeConfiguration(
        ...     manejo_duplicados=TipoManejoDuplicados.BEST_QUALITY,
        ...     columna_calidad='quality_score'
        ... )

    See Also:
        - :class:`GeoMergeConfiguration`: Configuration containing duplicate strategy
        - :class:`~enahopy.merger.geographic.strategies.DuplicateStrategyFactory`: Strategy implementation
    """

    FIRST = "first"

    LAST = "last"

    ERROR = "error"

    KEEP_ALL = "keep_all"

    AGGREGATE = "aggregate"

    MOST_RECENT = "most_recent"

    BEST_QUALITY = "best_quality"


class TipoManejoErrores(Enum):
    """Estrategias para manejar errores de fusión"""

    COERCE = "coerce"

    RAISE = "raise"

    IGNORE = "ignore"

    LOG_WARNING = "log_warning"


class NivelTerritorial(Enum):
    """Niveles territoriales del Perú según INEI"""

    DEPARTAMENTO = "departamento"

    PROVINCIA = "provincia"

    DISTRITO = "distrito"

    CENTRO_POBLADO = "centro_poblado"

    CONGLOMERADO = "conglomerado"


class TipoValidacionUbigeo(Enum):
    """Tipos de validación de UBIGEO"""

    BASIC = "basic"  # Solo formato

    STRUCTURAL = "structural"  # Validar estructura jerárquica

    EXISTENCE = "existence"  # Validar existencia real

    TEMPORAL = "temporal"  # Validar vigencia temporal


# =====================================================

# ENUMS DE MÓDULOS

# =====================================================


class ModuleMergeLevel(Enum):
    """Hierarchical levels for merging ENAHO survey modules.

    Defines the granularity level at which ENAHO modules should be merged,
    determining which identification keys are required and how records are matched.
    The choice of merge level depends on the analytical objective and the structure
    of the modules being combined.

    Attributes:
        HOGAR: Household level merge using keys ['conglome', 'vivienda', 'hogar'].
            Use for combining household-level modules like sumaria (34), housing
            characteristics (01), or household expenditures (07). Results in one
            record per household.
        PERSONA: Person level merge using keys ['conglome', 'vivienda', 'hogar', 'codperso'].
            Use for combining person-level modules like demographics (02), education
            (03), health (04), or employment (05). Results in one record per person
            within each household.
        VIVIENDA: Dwelling level merge using keys ['conglome', 'vivienda'].
            Use for combining dwelling-level data that may span multiple households
            within the same physical structure. Less commonly used than HOGAR or PERSONA.

    Examples:
        Household-level merge:

        >>> from enahopy.merger import ENAHOGeoMerger
        >>> from enahopy.merger.config import (
        ...     ModuleMergeConfig,
        ...     ModuleMergeLevel
        ... )
        >>> config = ModuleMergeConfig(
        ...     merge_level=ModuleMergeLevel.HOGAR
        ... )
        >>> merger = ENAHOGeoMerger(module_config=config)
        >>> # Merge sumaria + housing modules at household level

        Person-level merge:

        >>> config = ModuleMergeConfig(
        ...     merge_level=ModuleMergeLevel.PERSONA
        ... )
        >>> merger = ENAHOGeoMerger(module_config=config)
        >>> # Merge demographics + education modules at person level

    Note:
        - Household-level modules cannot be directly merged at person level without
          first aggregating or broadcasting the household data
        - Person-level modules can be aggregated to household level using groupby
        - The merge level must be compatible with all modules being merged

    See Also:
        - :class:`ModuleMergeConfig`: Configuration including merge level
        - :class:`ModuleType`: Module structural types (HOGAR_LEVEL vs PERSONA_LEVEL)
        - :meth:`~enahopy.merger.ENAHOGeoMerger.validate_module_compatibility`: Check level compatibility
    """

    HOGAR = "hogar"  # Merge a nivel hogar

    PERSONA = "persona"  # Merge a nivel persona

    VIVIENDA = "vivienda"  # Merge a nivel vivienda


class ModuleMergeStrategy(Enum):
    """Strategies for resolving column name conflicts during module merges.

    When merging ENAHO modules, column name conflicts occur if both modules contain
    columns with the same name (other than merge keys). This enum defines strategies
    to automatically resolve these conflicts, determining which values to keep in
    the final merged DataFrame.

    Attributes:
        KEEP_LEFT: Preserve values from the left (base) DataFrame and discard right values.
            Use when the left module has priority or more authoritative data.
        KEEP_RIGHT: Preserve values from the right DataFrame and discard left values.
            Use when the right module contains updated or more reliable information.
        COALESCE: Combine values by preferring non-null values: use left value if present,
            otherwise use right value. Ideal for filling gaps where modules have
            complementary coverage (left module: 70% complete, right module: fills remaining 30%).
        AVERAGE: Calculate the arithmetic mean of numeric values from both columns.
            Non-numeric columns fall back to COALESCE. Use when both modules provide
            independent measurements of the same variable.
        CONCATENATE: Concatenate string values with " | " separator when values differ.
            Use for textual fields where combining information is valuable (e.g., comments).
            Numeric columns fall back to COALESCE.
        ERROR: Raise ConflictResolutionError if any conflicts are detected. Use for
            strict validation when conflicts indicate data quality issues or require
            manual resolution.

    Examples:
        Using COALESCE (default) to fill gaps:

        >>> from enahopy.merger.config import (
        ...     ModuleMergeConfig,
        ...     ModuleMergeStrategy
        ... )
        >>> config = ModuleMergeConfig(
        ...     merge_strategy=ModuleMergeStrategy.COALESCE
        ... )
        >>> # Left module has ingreso=2000, area=NaN
        >>> # Right module has ingreso=NaN, area=50
        >>> # Result: ingreso=2000, area=50

        Using AVERAGE for numerical reconciliation:

        >>> config = ModuleMergeConfig(
        ...     merge_strategy=ModuleMergeStrategy.AVERAGE
        ... )
        >>> # Left module has gasto=1000
        >>> # Right module has gasto=1200
        >>> # Result: gasto=1100

        Strict validation with ERROR:

        >>> config = ModuleMergeConfig(
        ...     merge_strategy=ModuleMergeStrategy.ERROR
        ... )
        >>> # Any conflict raises ConflictResolutionError

    Note:
        - Conflict resolution only applies to overlapping column names
        - Merge keys are never considered conflicts
        - Strategy applies globally to all conflicting columns
        - AVERAGE and CONCATENATE have type-specific fallback behavior

    See Also:
        - :class:`ModuleMergeConfig`: Configuration including conflict strategy
        - :exc:`~enahopy.merger.exceptions.ConflictResolutionError`: Raised by ERROR strategy
        - :meth:`~enahopy.merger.modules.merger.ENAHOModuleMerger.merge_modules`: Uses strategy
    """

    KEEP_LEFT = "keep_left"  # Mantener valores del DataFrame izquierdo

    KEEP_RIGHT = "keep_right"  # Mantener valores del DataFrame derecho

    COALESCE = "coalesce"  # Combinar (usar derecho si izquierdo es nulo)

    AVERAGE = "average"  # Promediar valores numéricos

    CONCATENATE = "concatenate"  # Concatenar strings

    ERROR = "error"  # Error si hay conflictos


class ModuleType(Enum):
    """Tipos de módulos ENAHO según su estructura"""

    HOGAR_LEVEL = "hogar_level"  # Módulos a nivel hogar (01, 07, 08, 34)

    PERSONA_LEVEL = "persona_level"  # Módulos a nivel persona (02, 03, 04, 05)

    MIXED_LEVEL = "mixed_level"  # Módulos con ambos niveles

    SPECIAL = "special"  # Módulos especiales (37)


# =====================================================

# CONFIGURACIONES

# =====================================================


@dataclass
class GeoMergeConfiguration:
    """Complete configuration for geographic data fusion operations.

    Comprehensive configuration dataclass controlling all aspects of geographic
    merging including UBIGEO validation, duplicate handling, territorial
    consistency checks, and performance optimization. Provides fine-grained
    control over merge behavior and data quality requirements.

    Attributes:
        columna_union: Name of the column containing UBIGEO codes for joining
            geographic and survey data. Must exist in both DataFrames. Defaults
            to "ubigeo".
        manejo_duplicados: Strategy for handling duplicate UBIGEOs in geographic
            data. See TipoManejoDuplicados for options. Defaults to FIRST (keep
            first occurrence).
        manejo_errores: Error handling strategy when validation fails. COERCE
            attempts to fix issues, RAISE stops execution, IGNORE continues with
            warnings. Defaults to COERCE.
        valor_faltante: Value to use for missing geographic information after merge.
            Can be string or numeric. Records without geographic match will have
            this value. Defaults to "DESCONOCIDO".
        validar_formato_ubigeo: If True, validates UBIGEO format (2/4/6 digits) and
            structure before merge. Recommended for data quality assurance. Defaults
            to True.
        tipo_validacion_ubigeo: Level of UBIGEO validation. BASIC checks format only,
            STRUCTURAL validates hierarchical consistency, EXISTENCE checks against
            official catalog. Defaults to STRUCTURAL.
        validar_consistencia_territorial: If True, validates that province codes
            belong to departments and districts belong to provinces. Computationally
            intensive for large datasets. Defaults to True.
        validar_coordenadas: If True, validates geographic coordinates (lat, lon)
            for valid ranges and missing values. Only applies if coordinate columns
            are present. Defaults to False.
        generar_reporte_calidad: If True, generates detailed quality report including
            coverage metrics, validation results, and recommendations. Defaults to True.
        reporte_duplicados: If True, includes duplicate analysis in quality report.
            Defaults to True.
        mostrar_estadisticas: If True, displays merge statistics including record
            counts, coverage percentage, and quality scores. Defaults to True.
        funciones_agregacion: Dictionary mapping column names to aggregation functions
            for AGGREGATE duplicate strategy. Example: {'poblacion': 'sum', 'ingreso': 'mean'}.
            Required when manejo_duplicados is AGGREGATE. Defaults to None.
        sufijo_duplicados: Suffix to append to duplicate column names. Defaults to "_dup".
        columna_orden_duplicados: Column name for ordering duplicates when using
            MOST_RECENT strategy. Must contain sortable values (dates, timestamps).
            Defaults to None.
        columna_calidad: Column name containing quality scores for BEST_QUALITY
            strategy. Must contain numeric values where higher means better quality.
            Defaults to None.
        usar_cache: If True, caches validation results and geographic lookups for
            repeated operations. Improves performance for iterative workflows.
            Defaults to True.
        optimizar_memoria: If True, enables memory optimization for large datasets
            including chunked processing and dtype optimization. Defaults to True.
        chunk_size: Number of records per chunk when processing large DataFrames.
            Only used if optimizar_memoria is True. Defaults to 50000.
        nivel_territorial_objetivo: Target territorial level for merge (DEPARTAMENTO,
            PROVINCIA, DISTRITO). Determines UBIGEO validation requirements. Defaults
            to DISTRITO (most granular).
        incluir_niveles_superiores: If True, includes higher territorial levels
            (e.g., when merging at distrito level, include provincia and departamento
            columns). Defaults to True.
        prefijo_columnas: Prefix to add to geographic column names in result
            (e.g., "geo_" produces "geo_departamento"). Defaults to "" (no prefix).
        sufijo_columnas: Suffix to add to geographic column names in result
            (e.g., "_ref" produces "departamento_ref"). Defaults to "" (no suffix).

    Examples:
        Basic configuration with defaults:

        >>> from enahopy.merger.config import GeoMergeConfiguration
        >>> config = GeoMergeConfiguration()
        >>> # Uses FIRST for duplicates, COERCE for errors, validates UBIGEO

        Custom configuration for aggregation:

        >>> config = GeoMergeConfiguration(
        ...     columna_union='cod_ubigeo',
        ...     manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
        ...     funciones_agregacion={
        ...         'poblacion_total': 'sum',
        ...         'ingreso_promedio': 'mean',
        ...         'area_urbana': 'first'
        ...     },
        ...     validar_formato_ubigeo=True,
        ...     generar_reporte_calidad=True
        ... )

        Memory-optimized configuration for large datasets:

        >>> config = GeoMergeConfiguration(
        ...     optimizar_memoria=True,
        ...     chunk_size=100000,
        ...     validar_consistencia_territorial=False,  # Skip expensive validation
        ...     usar_cache=True
        ... )

        Strict validation configuration:

        >>> config = GeoMergeConfiguration(
        ...     manejo_duplicados=TipoManejoDuplicados.ERROR,
        ...     manejo_errores=TipoManejoErrores.RAISE,
        ...     validar_formato_ubigeo=True,
        ...     validar_consistencia_territorial=True,
        ...     validar_coordenadas=True
        ... )

    Note:
        - Configuration objects are immutable after creation (frozen dataclass)
        - Some strategies require additional configuration (e.g., AGGREGATE needs funciones_agregacion)
        - Validation can be computationally expensive for datasets >100K records
        - Memory optimization recommended for datasets >500MB

    See Also:
        - :class:`TipoManejoDuplicados`: Duplicate handling strategies
        - :class:`TipoManejoErrores`: Error handling modes
        - :class:`NivelTerritorial`: Territorial levels
        - :class:`~enahopy.merger.ENAHOGeoMerger`: Main merger using this configuration
    """

    # Configuración básica

    columna_union: str = "ubigeo"

    manejo_duplicados: TipoManejoDuplicados = TipoManejoDuplicados.FIRST

    manejo_errores: TipoManejoErrores = TipoManejoErrores.COERCE

    valor_faltante: Union[str, float] = "DESCONOCIDO"

    # Validaciones

    validar_formato_ubigeo: bool = True

    tipo_validacion_ubigeo: TipoValidacionUbigeo = TipoValidacionUbigeo.STRUCTURAL

    validar_consistencia_territorial: bool = True

    validar_coordenadas: bool = False

    # Reportes y logging

    generar_reporte_calidad: bool = True

    reporte_duplicados: bool = True

    mostrar_estadisticas: bool = True

    # Manejo de duplicados avanzado

    funciones_agregacion: Optional[Dict[str, str]] = None

    sufijo_duplicados: str = "_dup"

    columna_orden_duplicados: Optional[str] = None

    columna_calidad: Optional[str] = None  # Para BEST_QUALITY

    # Performance

    usar_cache: bool = True

    optimizar_memoria: bool = True

    chunk_size: int = 50000

    # Configuración territorial específica

    nivel_territorial_objetivo: NivelTerritorial = NivelTerritorial.DISTRITO

    incluir_niveles_superiores: bool = True

    prefijo_columnas: str = ""

    sufijo_columnas: str = ""


@dataclass
class ModuleMergeConfig:
    """Configuración para merge entre módulos ENAHO"""

    merge_level: ModuleMergeLevel = ModuleMergeLevel.HOGAR

    merge_strategy: ModuleMergeStrategy = ModuleMergeStrategy.COALESCE

    validate_keys: bool = True

    allow_partial_matches: bool = False

    suffix_conflicts: Tuple[str, str] = ("_x", "_y")

    # ====== OPTIMIZACIÓN: Merge type parametrizable ======
    merge_type: str = "left"  # "left", "outer", "inner", "right"

    # ====== OPTIMIZACIÓN: Validación post-merge ======
    validate_cardinality: bool = True  # Validar que merge no infle filas inesperadamente

    # ====== OPTIMIZACIÓN: Performance ======
    use_validation_cache: bool = True  # Cachear validaciones costosas

    # Columnas de identificación por nivel

    hogar_keys: List[str] = field(default_factory=lambda: ["conglome", "vivienda", "hogar"])

    persona_keys: List[str] = field(
        default_factory=lambda: ["conglome", "vivienda", "hogar", "codperso"]
    )

    vivienda_keys: List[str] = field(default_factory=lambda: ["conglome", "vivienda"])

    # Configuración específica por módulo

    module_validations: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "01": {
                "level": ModuleType.HOGAR_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar"],
            },
            "02": {
                "level": ModuleType.PERSONA_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar", "codperso"],
            },
            "03": {
                "level": ModuleType.PERSONA_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar", "codperso"],
            },
            "04": {
                "level": ModuleType.PERSONA_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar", "codperso"],
            },
            "05": {
                "level": ModuleType.PERSONA_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar", "codperso"],
            },
            "07": {
                "level": ModuleType.HOGAR_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar"],
            },
            "08": {
                "level": ModuleType.HOGAR_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar"],
            },
            "09": {
                "level": ModuleType.HOGAR_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar"],
            },
            "34": {
                "level": ModuleType.HOGAR_LEVEL,
                "required_keys": ["conglome", "vivienda", "hogar"],
            },
            "37": {"level": ModuleType.SPECIAL, "required_keys": ["conglome", "vivienda", "hogar"]},
        }
    )

    # Configuración de calidad

    min_match_rate: float = 0.7  # Tasa mínima de match para considerar exitoso

    max_conflicts_allowed: int = 1000  # Máximo número de conflictos permitidos

    # Configuración de performance

    chunk_processing: bool = False

    chunk_size: int = 10000

    continue_on_error: bool = False


# =====================================================

# DATACLASSES DE RESULTADOS

# =====================================================


@dataclass
class GeoValidationResult:
    """Resultado de validación geográfica"""

    is_valid: bool

    total_records: int

    valid_ubigeos: int

    invalid_ubigeos: int

    duplicate_ubigeos: int

    missing_coordinates: int

    territorial_inconsistencies: int

    coverage_percentage: float

    errors: List[str]

    warnings: List[str]

    quality_metrics: Dict[str, float]

    def get_summary_report(self) -> str:
        """Genera reporte resumido de validación"""

        status = "✅ VÁLIDO" if self.is_valid else "❌ INVÁLIDO"

        report = [
            "=== REPORTE DE VALIDACIÓN GEOGRÁFICA ===",
            f"Estado: {status}",
            f"Registros totales: {self.total_records:,}",
            f"UBIGEOs válidos: {self.valid_ubigeos:,} ({self.coverage_percentage:.1f}%)",
            f"UBIGEOs inválidos: {self.invalid_ubigeos:,}",
            f"Duplicados: {self.duplicate_ubigeos:,}",
        ]

        if self.missing_coordinates > 0:
            report.append(f"Sin coordenadas: {self.missing_coordinates:,}")

        if self.territorial_inconsistencies > 0:
            report.append(f"Inconsistencias territoriales: {self.territorial_inconsistencies:,}")

        if self.warnings:
            report.append("\n⚠️  Advertencias:")

            for warning in self.warnings[:5]:  # Máximo 5
                report.append(f"  - {warning}")

        if self.errors:
            report.append("\n❌ Errores:")

            for error in self.errors[:5]:  # Máximo 5
                report.append(f"  - {error}")

        return "\n".join(report)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""

        return {
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "valid_ubigeos": self.valid_ubigeos,
            "invalid_ubigeos": self.invalid_ubigeos,
            "duplicate_ubigeos": self.duplicate_ubigeos,
            "missing_coordinates": self.missing_coordinates,
            "territorial_inconsistencies": self.territorial_inconsistencies,
            "coverage_percentage": self.coverage_percentage,
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_metrics": self.quality_metrics,
        }


@dataclass
class ModuleMergeResult:
    """Resultado del merge entre módulos"""

    merged_df: "pd.DataFrame"

    merge_report: Dict[str, Any]

    conflicts_resolved: int

    unmatched_left: int

    unmatched_right: int

    validation_warnings: List[str]

    quality_score: float

    # ====== OPTIMIZACIÓN: Métricas de merge ======
    merge_metrics: Optional[Dict[str, Any]] = None  # Métricas detalladas de performance y calidad

    def get_summary_report(self) -> str:
        """Genera reporte resumido del merge de módulos"""

        status = (
            "✅ EXITOSO"
            if self.quality_score >= 70
            else "⚠️ CON ADVERTENCIAS"
            if self.quality_score >= 50
            else "❌ PROBLEMÁTICO"
        )

        report = [
            "=== REPORTE DE MERGE ENTRE MÓDULOS ===",
            f"Estado: {status}",
            f"Registros finales: {len(self.merged_df):,}",
            f"Score de calidad: {self.quality_score:.1f}%",
            f"Conflictos resueltos: {self.conflicts_resolved:,}",
            f"No coincidentes izq: {self.unmatched_left:,}",
            f"No coincidentes der: {self.unmatched_right:,}",
        ]

        # ====== OPTIMIZACIÓN: Mostrar métricas si existen ======
        if self.merge_metrics:
            report.append("\n📊 Métricas de Performance:")
            if "time_elapsed" in self.merge_metrics:
                report.append(f"  Tiempo: {self.merge_metrics['time_elapsed']:.2f}s")
            if "memory_peak_mb" in self.merge_metrics:
                report.append(f"  Memoria pico: {self.merge_metrics['memory_peak_mb']:.1f} MB")
            if "match_rate" in self.merge_metrics:
                report.append(f"  Tasa de match: {self.merge_metrics['match_rate']:.1%}")
            if "cardinality_change" in self.merge_metrics:
                change = self.merge_metrics["cardinality_change"]
                symbol = "✅" if abs(change - 1.0) < 0.01 else "⚠️"
                report.append(f"  Factor de cardinalidad: {change:.2f}x {symbol}")

        if self.validation_warnings:
            report.append(f"\n⚠️  Advertencias ({len(self.validation_warnings)}):")

            for warning in self.validation_warnings[:3]:
                report.append(f"  - {warning}")

            if len(self.validation_warnings) > 3:
                report.append(f"  ... y {len(self.validation_warnings) - 3} más")

        return "\n".join(report)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""

        return {
            "shape": self.merged_df.shape,
            "merge_report": self.merge_report,
            "conflicts_resolved": self.conflicts_resolved,
            "unmatched_left": self.unmatched_left,
            "unmatched_right": self.unmatched_right,
            "validation_warnings": self.validation_warnings,
            "quality_score": self.quality_score,
        }


# =====================================================

# CONSTANTES Y PATRONES

# =====================================================


# Rangos válidos para departamentos según INEI

DEPARTAMENTOS_VALIDOS = {
    "01": "AMAZONAS",
    "02": "ÁNCASH",
    "03": "APURÍMAC",
    "04": "AREQUIPA",
    "05": "AYACUCHO",
    "06": "CAJAMARCA",
    "07": "CALLAO",
    "08": "CUSCO",
    "09": "HUANCAVELICA",
    "10": "HUÁNUCO",
    "11": "ICA",
    "12": "JUNÍN",
    "13": "LA LIBERTAD",
    "14": "LAMBAYEQUE",
    "15": "LIMA",
    "16": "LORETO",
    "17": "MADRE DE DIOS",
    "18": "MOQUEGUA",
    "19": "PASCO",
    "20": "PIURA",
    "21": "PUNO",
    "22": "SAN MARTÍN",
    "23": "TACNA",
    "24": "TUMBES",
    "25": "UCAYALI",
}


# Patrones extendidos de columnas geográficas

PATRONES_GEOGRAFICOS = {
    "departamento": [
        "departamento",
        "dep",
        "dpto",
        "depto",
        "department",
        "region",
        "cod_dep",
        "codigo_departamento",
        "dpto_id",
        "dep_code",
    ],
    "provincia": [
        "provincia",
        "prov",
        "province",
        "cod_prov",
        "codigo_provincia",
        "prov_id",
        "prov_code",
        "provincia_cod",
    ],
    "distrito": [
        "distrito",
        "dist",
        "district",
        "cod_dist",
        "codigo_distrito",
        "dist_id",
        "dist_code",
        "distrito_cod",
    ],
    "ubigeo": [
        "ubigeo",
        "cod_ubigeo",
        "codigo_ubigeo",
        "ubigeo_code",
        "geo_code",
        "ubigeo_inei",
        "codigo_geografico",
        "cod_geo",
    ],
    "centro_poblado": [
        "centro_poblado",
        "ccpp",
        "poblado",
        "centropoblado",
        "cod_ccpp",
        "codigo_ccpp",
        "ccpp_code",
        "centro_pob",
    ],
    "conglomerado": ["conglome", "conglomerado", "conglo", "cod_conglome", "cluster"],
    "coordenada_x": ["longitud", "longitude", "lon", "coord_x", "x", "este", "east"],
    "coordenada_y": ["latitud", "latitude", "lat", "coord_y", "y", "norte", "north"],
}


# Alias para compatibilidad

MergerConfig = ModuleMergeConfig
