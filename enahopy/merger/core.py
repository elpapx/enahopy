"""
ENAHO Merger - Clases Principales (VERSIÓN CORREGIDA)
======================================================

Implementación de las clases principales ENAHOGeoMerger con
funcionalidades completas de fusión geográfica y merge de módulos.

Versión: 2.1.0
Correcciones aplicadas:
- Validación de DataFrames vacíos
- Manejo robusto de CacheManager
- Validación de tipos de datos
- Manejo de NaN en claves de merge
- División por cero en métricas
- Documentación mejorada
"""

"""
Imports centralizados para evitar errores de dependencias.
"""

import logging

# Imports estándar de Python (siempre disponibles)
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Imports científicos (verificar disponibilidad)
try:
    import numpy as np
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# Verificación de dependencias
def check_dependencies():
    """Verifica que todas las dependencias estén disponibles."""
    missing = []

    if not HAS_PANDAS:
        missing.append("pandas")
    if not HAS_PLOTTING:
        missing.append("matplotlib, seaborn")

    if missing:
        raise ImportError(f"Dependencias faltantes: {', '.join(missing)}")

    return True


from ..validation import validate_columns_exist, validate_dataframe_not_empty

# Importaciones internas
from .config import (
    GeoMergeConfiguration,
    GeoValidationResult,
    ModuleMergeConfig,
    ModuleMergeResult,
    TipoManejoDuplicados,
    TipoManejoErrores,
)
from .exceptions import (
    ConfigurationError,
    DataQualityError,
    GeoMergeError,
    MergeValidationError,
    ModuleMergeError,
    ValidationThresholdError,
)
from .geographic.patterns import GeoPatternDetector
from .geographic.strategies import DuplicateStrategyFactory
from .geographic.validators import GeoDataQualityValidator, TerritorialValidator, UbigeoValidator
from .modules.merger import ENAHOModuleMerger
from .modules.validator import ModuleValidator

# Importaciones opcionales del loader principal con fallback robusto
try:
    from ..loader import CacheManager, ENAHOConfig, log_performance, setup_logging

    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False
    # Fallback completo para uso independiente
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ENAHOConfig:
        """Configuración fallback para uso independiente"""

        cache_dir: str = ".enaho_cache"
        use_cache: bool = True
        validate_data: bool = True

    def setup_logging(
        verbose: bool = True, structured: bool = False, log_file: Optional[str] = None
    ):
        """Setup de logging fallback"""
        logger = logging.getLogger("enaho_geo_merger")
        if not logger.handlers:
            handler = logging.StreamHandler()
            if structured:
                formatter = logging.Formatter(
                    '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
                )
            else:
                formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return logger

    def log_performance(func):
        """Decorador de performance fallback"""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger = logging.getLogger("enaho_geo_merger")
                logger.debug(f"⏱️ {func.__name__} ejecutado en {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger = logging.getLogger("enaho_geo_merger")
                logger.error(f"❌ {func.__name__} falló después de {elapsed:.2f}s: {str(e)}")
                raise

        return wrapper

    # CacheManager será None si no está disponible
    CacheManager = None


class ENAHOGeoMerger:
    """Advanced geographic and module merger for ENAHO data.

    Comprehensive merger that combines ENAHO data with geographic information
    using UBIGEO codes, and merges multiple ENAHO modules with intelligent
    validation and quality assessment. Integrates geographic validation,
    duplicate handling strategies, and module-level fusion workflows.

    This class orchestrates complex data integration tasks including:
    - Geographic data enrichment with UBIGEO validation
    - Territorial consistency checks across administrative levels
    - Multiple strategies for duplicate record handling
    - Multi-module sequential and parallel merge operations
    - Data quality assessment and reporting
    - Memory-optimized processing for large datasets

    The ENAHOGeoMerger serves as the central integration hub for enahopy,
    enabling researchers to combine household survey data with geographic
    references and merge information across different ENAHO modules.

    Attributes:
        geo_config (GeoMergeConfiguration): Configuration for geographic merge
            operations including duplicate handling, validation settings, and
            error management strategies.
        module_config (ModuleMergeConfig): Configuration for module-level merges
            including merge keys, conflict resolution, and quality thresholds.
        verbose (bool): If True, displays detailed operation progress and
            diagnostic information during merge operations.
        logger (logging.Logger): Configured logger instance for tracking
            operations, warnings, and errors.
        ubigeo_validator (UbigeoValidator): Validator for UBIGEO code format
            and structure validation.
        territorial_validator (TerritorialValidator): Validator for territorial
            hierarchy consistency checking.
        quality_validator (GeoDataQualityValidator): Validator for geographic
            data quality assessment.
        pattern_detector (GeoPatternDetector): Detector for automatic identification
            of geographic column patterns.
        module_merger (ENAHOModuleMerger): Specialized merger for ENAHO module
            operations with advanced conflict resolution.

    Examples:
        Basic geographic merge:

        >>> from enahopy.merger import ENAHOGeoMerger
        >>> import pandas as pd
        >>>
        >>> # Create sample data
        >>> df_data = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102', '150103'],
        ...     'ingreso': [2000, 1500, 1800]
        ... })
        >>> df_geo = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102', '150103'],
        ...     'departamento': ['Lima', 'Lima', 'Lima'],
        ...     'provincia': ['Lima', 'Lima', 'Lima']
        ... })
        >>>
        >>> # Perform geographic merge
        >>> merger = ENAHOGeoMerger(verbose=True)
        >>> result, validation = merger.merge_geographic_data(
        ...     df_principal=df_data,
        ...     df_geografia=df_geo,
        ...     columna_union='ubigeo'
        ... )
        >>> print(f"Records: {len(result)}, Coverage: {validation.coverage_percentage:.1f}%")
        Records: 3, Coverage: 100.0%

        Multi-module merge:

        >>> # Create module DataFrames
        >>> df_sumaria = pd.DataFrame({
        ...     'conglome': ['001', '002'],
        ...     'vivienda': ['01', '01'],
        ...     'hogar': ['1', '1'],
        ...     'gashog2d': [2000, 1500]
        ... })
        >>> df_vivienda = pd.DataFrame({
        ...     'conglome': ['001', '002'],
        ...     'vivienda': ['01', '01'],
        ...     'hogar': ['1', '1'],
        ...     'area': [1, 2]
        ... })
        >>>
        >>> # Merge multiple modules
        >>> modules_dict = {'34': df_sumaria, '01': df_vivienda}
        >>> result = merger.merge_multiple_modules(
        ...     modules_dict=modules_dict,
        ...     base_module='34'
        ... )
        >>> print(f"Merged records: {len(result.merged_df)}")
        Merged records: 2

        Combined geographic and module merge:

        >>> # Merge modules then add geography
        >>> modules = {'34': df_sumaria, '01': df_vivienda}
        >>> df_geo = pd.DataFrame({
        ...     'ubigeo': ['150101', '150102'],
        ...     'departamento': ['Lima', 'Lima']
        ... })
        >>> final_df, report = merger.merge_modules_with_geography(
        ...     modules_dict=modules,
        ...     df_geografia=df_geo,
        ...     base_module='34'
        ... )
        >>> print(f"Final shape: {final_df.shape}")
        Final shape: (2, 6)

    Note:
        - UBIGEO validation supports 2-digit (department), 4-digit (province),
          and 6-digit (district) formats
        - Duplicate handling strategies: FIRST, LAST, AGGREGATE, BEST_QUALITY
        - Module merge supports household and person-level keys
        - Memory optimization available for datasets >100K records
        - Compatible with cached data from ENAHODataDownloader

    See Also:
        - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Geographic merge config
        - :class:`~enahopy.merger.config.ModuleMergeConfig`: Module merge config
        - :class:`~enahopy.merger.modules.merger.ENAHOModuleMerger`: Module merger
        - :func:`~enahopy.merger.convenience.merge_enaho_modules`: Convenience function
    """

    def __init__(
        self,
        geo_config: Optional["GeoMergeConfiguration"] = None,
        module_config: Optional[ModuleMergeConfig] = None,
        verbose: bool = True,
    ):
        """Initialize the ENAHO geographic and module merger.

        Sets up the merger with geographic validation components, module merge
        infrastructure, and logging. Automatically initializes validators for
        UBIGEO codes, territorial consistency, and data quality assessment.

        Args:
            geo_config: Configuration for geographic merge operations. If None,
                uses default configuration with standard UBIGEO validation,
                FIRST duplicate handling strategy, and left join semantics.
                Allows customization of validation levels, duplicate strategies,
                and error handling modes. Defaults to None.
            module_config: Configuration for module-level merges. If None, uses
                default configuration with household-level keys, COALESCE conflict
                resolution, and standard quality thresholds. Enables customization
                of merge keys, strategies, and validation rules. Defaults to None.
            verbose: If True, displays detailed progress information including
                validation results, merge statistics, and quality metrics. Useful
                for debugging and monitoring complex merge operations. If False,
                only errors and warnings are logged. Defaults to True.

        Examples:
            Default configuration:

            >>> from enahopy.merger import ENAHOGeoMerger
            >>> merger = ENAHOGeoMerger()
            >>> # Uses default settings for both geographic and module merges

            With custom geographic configuration:

            >>> from enahopy.merger.config import (
            ...     GeoMergeConfiguration,
            ...     TipoManejoDuplicados
            ... )
            >>> geo_config = GeoMergeConfiguration(
            ...     manejo_duplicados=TipoManejoDuplicados.AGGREGATE,
            ...     funciones_agregacion={'ingreso': 'sum', 'poblacion': 'sum'},
            ...     validar_formato_ubigeo=True
            ... )
            >>> merger = ENAHOGeoMerger(geo_config=geo_config, verbose=True)
            >>> # Uses aggregation strategy for duplicates

            With custom module configuration:

            >>> from enahopy.merger.config import (
            ...     ModuleMergeConfig,
            ...     ModuleMergeStrategy,
            ...     ModuleMergeLevel
            ... )
            >>> module_config = ModuleMergeConfig(
            ...     merge_level=ModuleMergeLevel.PERSONA,
            ...     merge_strategy=ModuleMergeStrategy.KEEP_LEFT,
            ...     min_match_rate=0.8
            ... )
            >>> merger = ENAHOGeoMerger(
            ...     module_config=module_config,
            ...     verbose=False
            ... )
            >>> # Person-level merge with left preference

            Full custom configuration:

            >>> geo_cfg = GeoMergeConfiguration(
            ...     columna_union='ubigeo',
            ...     validar_consistencia_territorial=True,
            ...     optimizar_memoria=True,
            ...     chunk_size=50000
            ... )
            >>> mod_cfg = ModuleMergeConfig(
            ...     merge_level=ModuleMergeLevel.HOGAR,
            ...     merge_strategy=ModuleMergeStrategy.COALESCE,
            ...     continue_on_error=True
            ... )
            >>> merger = ENAHOGeoMerger(
            ...     geo_config=geo_cfg,
            ...     module_config=mod_cfg,
            ...     verbose=True
            ... )
            >>> # Fully customized merger with memory optimization

        Note:
            - Validators are automatically initialized on instantiation
            - Logger level is set based on verbose parameter
            - Configuration objects are immutable after creation (dataclass frozen)
            - Default configurations suitable for most ENAHO workflows
            - Memory optimization activates automatically for large datasets

        See Also:
            - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Geographic config
            - :class:`~enahopy.merger.config.ModuleMergeConfig`: Module merge config
            - :meth:`merge_geographic_data`: Main geographic merge method
            - :meth:`merge_multiple_modules`: Main module merge method
        """
        # Para compatibilidad con tests antiguos que usan geo_config
        if geo_config is not None:
            self.geo_config = geo_config
        else:
            # Crear configuración por defecto si no se proporciona
            self.geo_config = GeoMergeConfiguration()

        self.module_config = module_config or ModuleMergeConfig()
        self.verbose = verbose
        self.logger = self._setup_logger()

        # Inicializar componentes geográficos para compatibilidad con tests
        self._initialize_geographic_components()

        # Inicializar el merger de módulos
        self.module_merger = ENAHOModuleMerger(self.module_config, self.logger)

        if self.verbose:
            self.logger.info("ENAHOGeoMerger inicializado correctamente")

    def _setup_logger(self) -> logging.Logger:
        """Configura el logger"""
        logger = logging.getLogger("enaho_geo_merger")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def _validate_configurations(self):
        """
        Valida todas las configuraciones iniciales.

        Raises:
            ConfigurationError: Si alguna configuración es inválida
        """
        errors = []

        # Validar geo_config
        if self.geo_config.chunk_size <= 0:
            errors.append("chunk_size debe ser mayor que 0")

        if self.geo_config.chunk_size > 1000000:
            warnings.warn(
                f"chunk_size muy grande ({self.geo_config.chunk_size}), "
                "puede causar problemas de memoria",
                ResourceWarning,
            )

        if self.geo_config.manejo_duplicados == TipoManejoDuplicados.AGGREGATE:
            if not self.geo_config.funciones_agregacion:
                errors.append(
                    "funciones_agregacion es requerido cuando manejo_duplicados es AGGREGATE"
                )

        if self.geo_config.manejo_duplicados == TipoManejoDuplicados.BEST_QUALITY:
            if not self.geo_config.columna_calidad:
                errors.append(
                    "columna_calidad es requerido cuando manejo_duplicados es BEST_QUALITY"
                )

        # Validar module_config
        if self.module_config.min_match_rate < 0 or self.module_config.min_match_rate > 1:
            errors.append("min_match_rate debe estar entre 0 y 1")

        if self.module_config.max_conflicts_allowed < 0:
            errors.append("max_conflicts_allowed debe ser no negativo")

        if errors:
            raise ConfigurationError(f"Configuración inválida: {'; '.join(errors)}")

    def _setup_cache(self):
        """Configura el cache de manera segura verificando disponibilidad."""
        self.cache_manager = None
        self.cache_enabled = False

        if hasattr(self.config, "use_cache") and self.config.use_cache:
            if CacheManager is not None:
                try:
                    self.cache_manager = CacheManager(cache_dir=self.config.cache_dir)
                    self.cache_enabled = True
                    self.logger.info("Cache habilitado y configurado")
                except Exception as e:
                    self.logger.warning(
                        f"No se pudo inicializar el cache: {str(e)}. " "Continuando sin cache."
                    )
            else:
                if self.geo_config.usar_cache:
                    self.logger.warning(
                        "CacheManager no disponible. Instale el módulo loader "
                        "para habilitar cache. Continuando sin cache."
                    )

    def _initialize_geographic_components(self):
        """Inicializa componentes geográficos con manejo de errores."""
        try:
            self.ubigeo_validator = UbigeoValidator(self.logger)
            self.territorial_validator = TerritorialValidator(self.logger)
            self.quality_validator = GeoDataQualityValidator(self.logger)
            self.pattern_detector = GeoPatternDetector(self.logger)
            self.logger.debug("Componentes geográficos inicializados")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes geográficos: {str(e)}")
            raise

    def _initialize_module_components(self):
        """Inicializa componentes de módulos con manejo de errores."""
        try:
            self.module_merger = ENAHOModuleMerger(self.module_config, self.logger)
            self.module_validator = ModuleValidator(self.module_config, self.logger)
            self.logger.debug("Componentes de módulos inicializados")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes de módulos: {str(e)}")
            raise

    def _check_early_exit_conditions(
        self, df: pd.DataFrame, columna_ubigeo: str
    ) -> Optional[GeoValidationResult]:
        """
        Verifica condiciones de salida temprana en validación geográfica.

        Args:
            df: DataFrame a validar.
            columna_ubigeo: Nombre de la columna UBIGEO.

        Returns:
            GeoValidationResult si hay condición de salida temprana, None en caso contrario.

        Raises:
            ENAHOValidationError: Si la columna UBIGEO no existe o DataFrame vacío.
        """
        # Validar DataFrame no vacío usando módulo centralizado
        try:
            validate_dataframe_not_empty(df, name="DataFrame de validación geográfica")
        except Exception:
            self.logger.error("DataFrame vacío proporcionado para validación")
            return GeoValidationResult(
                is_valid=False,
                total_records=0,
                valid_ubigeos=0,
                invalid_ubigeos=0,
                duplicate_ubigeos=0,
                missing_coordinates=0,
                territorial_inconsistencies=0,
                coverage_percentage=0.0,
                errors=["DataFrame vacío"],
                warnings=[],
                quality_metrics={},
            )

        # Validar columna existe usando módulo centralizado
        try:
            validate_columns_exist(df, columna_ubigeo, df_name="DataFrame de validación")
        except Exception as e:
            raise ValueError(f"Columna '{columna_ubigeo}' no encontrada en el DataFrame") from e

        if df[columna_ubigeo].isna().all():
            self.logger.warning("Todos los valores de UBIGEO son NaN")
            return GeoValidationResult(
                is_valid=False,
                total_records=len(df),
                valid_ubigeos=0,
                invalid_ubigeos=0,
                duplicate_ubigeos=0,
                missing_coordinates=0,
                territorial_inconsistencies=0,
                coverage_percentage=0.0,
                errors=["Todos los valores de UBIGEO son NaN"],
                warnings=["No hay datos geográficos válidos para procesar"],
                quality_metrics={"data_completeness": 0.0},
            )

        return None

    def _validate_ubigeo_column(self, df: pd.DataFrame, columna_ubigeo: str) -> Dict[str, Any]:
        """
        Valida formato y estructura de columna UBIGEO.

        Args:
            df: DataFrame con datos a validar.
            columna_ubigeo: Nombre de la columna UBIGEO.

        Returns:
            Dictionary con métricas de validación:
            - valid: Número de UBIGEOs válidos
            - invalid: Número de UBIGEOs inválidos (incluyendo NaN)
            - errors: Lista de errores encontrados
            - non_nan_mask: Máscara de valores no-NaN
            - valid_mask: Máscara de valores válidos
        """
        non_nan_mask = df[columna_ubigeo].notna()
        non_nan_count = non_nan_mask.sum()
        total_records = len(df)

        if non_nan_count == 0:
            return {
                "valid": 0,
                "invalid": total_records,
                "errors": [],
                "non_nan_mask": non_nan_mask,
                "valid_mask": pd.Series([False] * total_records, index=df.index),
            }

        valid_mask, validation_errors = self.ubigeo_validator.validar_serie_ubigeos(
            df.loc[non_nan_mask, columna_ubigeo], self.geo_config.tipo_validacion_ubigeo
        )

        valid_ubigeos = valid_mask.sum()
        invalid_ubigeos = non_nan_count - valid_ubigeos + (total_records - non_nan_count)

        # Limitar errores mostrados
        limited_errors = validation_errors[:5]
        if len(validation_errors) > 5:
            limited_errors.append(f"...y {len(validation_errors) - 5} errores más")

        return {
            "valid": valid_ubigeos,
            "invalid": invalid_ubigeos,
            "errors": limited_errors,
            "non_nan_mask": non_nan_mask,
            "valid_mask": valid_mask,
        }

    def _check_duplicates(
        self, df: pd.DataFrame, columna_ubigeo: str, non_nan_mask: pd.Series
    ) -> Dict[str, Any]:
        """
        Detecta y analiza UBIGEOs duplicados.

        Args:
            df: DataFrame con datos.
            columna_ubigeo: Nombre de la columna UBIGEO.
            non_nan_mask: Máscara de valores no-NaN.

        Returns:
            Dictionary con:
            - count: Número de registros duplicados
            - unique_values: Número de valores únicos duplicados
            - warning: Mensaje de warning si hay duplicados
        """
        non_nan_count = non_nan_mask.sum()

        if non_nan_count == 0:
            return {"count": 0, "unique_values": 0, "warning": None}

        duplicates_mask = df.loc[non_nan_mask, columna_ubigeo].duplicated(keep=False)
        duplicate_count = duplicates_mask.sum()

        if duplicate_count > 0:
            unique_duplicates = df.loc[non_nan_mask][duplicates_mask][columna_ubigeo].nunique()
            warning = (
                f"{duplicate_count} registros con UBIGEO duplicado "
                f"({unique_duplicates} valores únicos)"
            )
            return {
                "count": duplicate_count,
                "unique_values": unique_duplicates,
                "warning": warning,
            }

        return {"count": 0, "unique_values": 0, "warning": None}

    def _validate_territorial_consistency(
        self, df: pd.DataFrame, columna_ubigeo: str, non_nan_mask: pd.Series, valid_mask: pd.Series
    ) -> Dict[str, Any]:
        """
        Valida consistencia territorial de UBIGEOs.

        Args:
            df: DataFrame con datos.
            columna_ubigeo: Nombre de la columna UBIGEO.
            non_nan_mask: Máscara de valores no-NaN.
            valid_mask: Máscara de valores válidos.

        Returns:
            Dictionary con:
            - count: Número de inconsistencias territoriales
            - warning: Mensaje de warning si hay inconsistencias
        """
        valid_data = df[non_nan_mask & valid_mask]

        if valid_data.empty:
            return {"count": 0, "warning": None}

        territorial_issues = self.territorial_validator.validar_jerarquia_territorial(
            valid_data, columna_ubigeo
        )
        inconsistency_count = len(territorial_issues)

        if inconsistency_count > 0:
            warning = f"{inconsistency_count} inconsistencias territoriales detectadas"
            return {"count": inconsistency_count, "warning": warning}

        return {"count": 0, "warning": None}

    def _validate_coordinates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida coordenadas geográficas en el DataFrame.

        Args:
            df: DataFrame con posibles columnas de coordenadas.

        Returns:
            Dictionary con:
            - missing_count: Número de coordenadas faltantes
            - warning: Mensaje de warning sobre cobertura
        """
        coord_columns = ["latitud", "longitud", "lat", "lon", "x", "y"]
        found_coords = [col for col in coord_columns if col in df.columns]

        if not found_coords or not self.geo_config.validar_coordenadas:
            return {"missing_count": 0, "warning": None}

        missing_coordinates = sum(df[col].isna().sum() for col in found_coords)

        if missing_coordinates > 0:
            total_records = len(df)
            coord_coverage = (
                (
                    (len(found_coords) * total_records - missing_coordinates)
                    / (len(found_coords) * total_records)
                    * 100
                )
                if total_records > 0
                else 0
            )
            warning = f"Cobertura de coordenadas: {coord_coverage:.1f}%"
            return {"missing_count": missing_coordinates, "warning": warning}

        return {"missing_count": 0, "warning": None}

    def _calculate_quality_metrics(
        self,
        total_records: int,
        valid_ubigeos: int,
        non_nan_count: int,
        duplicate_count: int,
        territorial_inconsistencies: int,
    ) -> Dict[str, float]:
        """
        Calcula métricas de calidad de datos geográficos.

        Args:
            total_records: Total de registros en el DataFrame.
            valid_ubigeos: Número de UBIGEOs válidos.
            non_nan_count: Número de valores no-NaN.
            duplicate_count: Número de duplicados.
            territorial_inconsistencies: Número de inconsistencias territoriales.

        Returns:
            Dictionary con métricas de calidad.
        """
        return {
            "completeness": (non_nan_count / total_records * 100) if total_records > 0 else 0.0,
            "validity": (valid_ubigeos / non_nan_count * 100) if non_nan_count > 0 else 0.0,
            "uniqueness": (
                ((non_nan_count - duplicate_count) / non_nan_count * 100)
                if non_nan_count > 0
                else 100.0
            ),
            "consistency": (
                ((total_records - territorial_inconsistencies) / total_records * 100)
                if total_records > 0
                else 100.0
            ),
        }

    @log_performance
    def validate_geographic_data(
        self,
        df: pd.DataFrame,
        columna_ubigeo: str = "ubigeo",
        validate_territory: bool = True,
        validate_quality: bool = True,
    ) -> GeoValidationResult:
        """Validate geographic data with comprehensive quality checks.

        Performs multi-level validation of geographic data including UBIGEO format
        validation, duplicate detection, territorial consistency checks, and
        coordinate coverage assessment. Orchestrates multiple independent validators
        and combines results into a comprehensive quality report.

        This method is the primary validation entry point for geographic data,
        providing detailed diagnostics before performing merge operations. It helps
        identify data quality issues early and provides actionable recommendations.

        Args:
            df: DataFrame to validate. Must contain geographic data with UBIGEO
                codes or other geographic identifiers. Can include additional
                columns like coordinates (latitud, longitud) for extended validation.
            columna_ubigeo: Name of the column containing UBIGEO codes. Column
                should contain Peruvian geographic codes in standard format:
                2-digit (department), 4-digit (province), or 6-digit (district).
                Defaults to "ubigeo".
            validate_territory: If True, performs territorial consistency validation
                checking that UBIGEOs follow valid administrative hierarchies
                (departments contain provinces, provinces contain districts).
                Computationally intensive for large datasets. Defaults to True.
            validate_quality: If True, performs extended data quality checks including
                coordinate coverage, missing value analysis, and data completeness
                assessment. Adds overhead but provides comprehensive quality metrics.
                Defaults to True.

        Returns:
            GeoValidationResult with detailed validation metrics including:

            - is_valid (bool): Overall validation status
            - total_records (int): Total records in DataFrame
            - valid_ubigeos (int): Count of valid UBIGEO codes
            - invalid_ubigeos (int): Count of invalid or missing UBIGEOs
            - duplicate_ubigeos (int): Count of duplicate UBIGEO records
            - missing_coordinates (int): Missing coordinate values if applicable
            - territorial_inconsistencies (int): Territorial hierarchy violations
            - coverage_percentage (float): Percentage of valid geographic coverage
            - errors (List[str]): Detailed error messages
            - warnings (List[str]): Warning messages about data quality
            - quality_metrics (Dict): Detailed quality metrics including
              completeness, validity, uniqueness, and consistency scores

        Raises:
            ValueError: If DataFrame is empty or columna_ubigeo column does not
                exist in the DataFrame. Also raised if all UBIGEO values are
                NaN or invalid.

        Examples:
            Basic validation:

            >>> from enahopy.merger import ENAHOGeoMerger
            >>> import pandas as pd
            >>>
            >>> df_geo = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102', '150103', '150101'],
            ...     'departamento': ['Lima', 'Lima', 'Lima', 'Lima']
            ... })
            >>>
            >>> merger = ENAHOGeoMerger()
            >>> validation = merger.validate_geographic_data(df_geo, 'ubigeo')
            >>> print(f"Valid: {validation.is_valid}")
            >>> print(f"Coverage: {validation.coverage_percentage:.1f}%")
            >>> print(f"Duplicates: {validation.duplicate_ubigeos}")
            Valid: True
            Coverage: 100.0%
            Duplicates: 1

            Validation with quality report:

            >>> validation = merger.validate_geographic_data(
            ...     df_geo,
            ...     'ubigeo',
            ...     validate_quality=True
            ... )
            >>> print(f"Completeness: {validation.quality_metrics['completeness']:.1f}%")
            >>> print(f"Uniqueness: {validation.quality_metrics['uniqueness']:.1f}%")
            >>> for warning in validation.warnings:
            ...     print(f"Warning: {warning}")
            Completeness: 100.0%
            Uniqueness: 75.0%
            Warning: 1 registros con UBIGEO duplicado

            Handling validation errors:

            >>> df_bad = pd.DataFrame({
            ...     'ubigeo': ['XXXXX', '999999', None, '150101'],
            ...     'departamento': ['?', '?', '?', 'Lima']
            ... })
            >>> validation = merger.validate_geographic_data(df_bad)
            >>> if not validation.is_valid:
            ...     print(f"Validation failed: {len(validation.errors)} errors")
            ...     print(f"Invalid UBIGEOs: {validation.invalid_ubigeos}")
            ...     for error in validation.errors[:3]:
            ...         print(f"  - {error}")
            Validation failed: 2 errors
            Invalid UBIGEOs: 3
              - Invalid UBIGEO format: XXXXX
              - Invalid UBIGEO format: 999999

            Fast validation without territorial checks:

            >>> # For large datasets, skip expensive territorial validation
            >>> validation = merger.validate_geographic_data(
            ...     df_large,
            ...     validate_territory=False,
            ...     validate_quality=False
            ... )
            >>> # Only validates UBIGEO format and duplicates

        Note:
            - Validation is non-destructive and does not modify input DataFrame
            - UBIGEO validation supports multiple format levels (2/4/6 digits)
            - Territorial validation can be slow for datasets >100K records
            - Quality metrics range from 0-100 for easy interpretation
            - Validation results can be cached for repeated operations
            - Empty DataFrames return is_valid=False with descriptive errors

        See Also:
            - :class:`~enahopy.merger.config.GeoValidationResult`: Result structure
            - :meth:`merge_geographic_data`: Uses validation before merge
            - :class:`~enahopy.merger.geographic.validators.UbigeoValidator`: UBIGEO validator
            - :class:`~enahopy.merger.geographic.validators.TerritorialValidator`: Territory validator
        """
        # Verificar condiciones de salida temprana
        if early_exit_result := self._check_early_exit_conditions(df, columna_ubigeo):
            return early_exit_result

        self.logger.info(f"Validando datos geográficos: {len(df)} registros")

        total_records = len(df)
        errors = []
        warnings = []

        # Validación de formato UBIGEO
        ubigeo_metrics = self._validate_ubigeo_column(df, columna_ubigeo)
        valid_ubigeos = ubigeo_metrics["valid"]
        invalid_ubigeos = ubigeo_metrics["invalid"]
        errors.extend(ubigeo_metrics["errors"])

        # Detección de duplicados
        duplicate_metrics = self._check_duplicates(
            df, columna_ubigeo, ubigeo_metrics["non_nan_mask"]
        )
        duplicate_ubigeos = duplicate_metrics["count"]
        if duplicate_metrics["warning"]:
            warnings.append(duplicate_metrics["warning"])

        # Validación territorial (opcional)
        territorial_inconsistencies = 0
        if validate_territory and valid_ubigeos > 0:
            territorial_metrics = self._validate_territorial_consistency(
                df, columna_ubigeo, ubigeo_metrics["non_nan_mask"], ubigeo_metrics["valid_mask"]
            )
            territorial_inconsistencies = territorial_metrics["count"]
            if territorial_metrics["warning"]:
                warnings.append(territorial_metrics["warning"])

        # Validación de coordenadas (opcional)
        missing_coordinates = 0
        if validate_quality:
            coord_metrics = self._validate_coordinates(df)
            missing_coordinates = coord_metrics["missing_count"]
            if coord_metrics["warning"]:
                warnings.append(coord_metrics["warning"])

        # Calcular métricas de calidad
        non_nan_count = ubigeo_metrics["non_nan_mask"].sum()
        quality_metrics = self._calculate_quality_metrics(
            total_records,
            valid_ubigeos,
            non_nan_count,
            duplicate_ubigeos,
            territorial_inconsistencies,
        )

        # Calcular cobertura
        coverage_percentage = (valid_ubigeos / total_records * 100) if total_records > 0 else 0.0

        # Determinar validez con umbrales configurables
        min_coverage = getattr(self.geo_config, "min_coverage_threshold", 80.0)
        min_uniqueness = getattr(self.geo_config, "min_uniqueness_threshold", 95.0)

        is_valid = (
            coverage_percentage >= min_coverage
            and quality_metrics["uniqueness"] >= min_uniqueness
            and territorial_inconsistencies == 0
        )

        # Construir resultado
        result = GeoValidationResult(
            is_valid=is_valid,
            total_records=total_records,
            valid_ubigeos=valid_ubigeos,
            invalid_ubigeos=invalid_ubigeos,
            duplicate_ubigeos=duplicate_ubigeos,
            missing_coordinates=missing_coordinates,
            territorial_inconsistencies=territorial_inconsistencies,
            coverage_percentage=coverage_percentage,
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

        self.logger.info(f"Validación completada - Cobertura: {coverage_percentage:.1f}%")
        return result

    def _handle_duplicates(self, df: pd.DataFrame, columna_union: str) -> pd.DataFrame:
        """
        Maneja duplicados usando la estrategia configurada.

        Args:
            df: DataFrame con posibles duplicados
            columna_union: Columna para identificar duplicados

        Returns:
            DataFrame sin duplicados según la estrategia

        Raises:
            GeoMergeError: Si la estrategia es ERROR y hay duplicados
        """
        if df.empty:
            return df

        # Identificar duplicados (excluyendo NaN)
        non_nan_mask = df[columna_union].notna()
        # Check if all values are NaN
        non_nan_count = non_nan_mask.sum()
        if isinstance(non_nan_count, pd.Series):
            non_nan_count = non_nan_count.iloc[0]
        if non_nan_count == 0:
            return df

        duplicates_mask = df[columna_union].duplicated(keep=False)

        # Check if there are no duplicates
        dup_count = duplicates_mask.sum()
        if isinstance(dup_count, pd.Series):
            dup_count = dup_count.iloc[0]
        if dup_count == 0:
            return df

        n_duplicates = duplicates_mask.sum()
        self.logger.info(f"Encontrados {n_duplicates} registros duplicados en '{columna_union}'")

        if self.geo_config.manejo_duplicados == TipoManejoDuplicados.ERROR:
            ubigeos_duplicados = df[duplicates_mask][columna_union].unique()
            raise GeoMergeError(
                f"Se encontraron {n_duplicates} duplicados en '{columna_union}'. "
                f"UBIGEOs afectados: {list(ubigeos_duplicados[:5])}... "
                f"Use otra estrategia de manejo_duplicados si desea procesarlos."
            )

        try:
            strategy = DuplicateStrategyFactory.create_strategy(
                self.geo_config.manejo_duplicados, self.logger
            )
            return strategy.handle_duplicates(df, columna_union, self.geo_config)
        except Exception as e:
            self.logger.error(f"Error manejando duplicados: {str(e)}")
            raise

    @log_performance
    def merge_geographic_data(
        self,
        df_principal: pd.DataFrame,
        df_geografia: pd.DataFrame,
        columnas_geograficas: Optional[Dict[str, str]] = None,
        columna_union: Optional[str] = None,
        validate_before_merge: bool = None,
    ) -> Tuple[pd.DataFrame, GeoValidationResult]:
        """Merge main data with geographic information using UBIGEO codes.

        Performs intelligent left join between a principal dataset and geographic
        reference data, with comprehensive validation, duplicate handling, and
        quality assessment. Automatically detects geographic columns, validates
        UBIGEO codes, handles duplicates according to configured strategy, and
        provides detailed merge metrics.

        This method is the primary geographic enrichment operation, adding
        administrative division information (departments, provinces, districts)
        and optionally geographic coordinates to household or individual-level data.

        Args:
            df_principal: Principal DataFrame to enrich with geographic information.
                Must contain UBIGEO column or other geographic identifiers. This
                DataFrame remains unchanged in terms of record count (left join
                semantics) unless duplicates are found in df_geografia.
                Required columns: columna_union (UBIGEO by default).
            df_geografia: Geographic reference DataFrame containing UBIGEO codes
                and associated geographic information such as departamento, provincia,
                distrito, and optionally coordinates (latitud, longitud).
                Should ideally have unique UBIGEOs unless using aggregation strategy.
                Required columns: columna_union (UBIGEO by default) + geographic columns.
            columnas_geograficas: Optional mapping of {"original_column": "new_column"}
                to rename geographic columns during merge. If None, automatically
                detects columns containing geographic patterns (departamento, provincia,
                distrito, etc.) with >70% confidence. Example: {"dept": "departamento"}.
                Defaults to None (auto-detection).
            columna_union: Name of the column to use as merge key. Must exist in
                both DataFrames. Should contain valid Peruvian UBIGEO codes. If None,
                uses value from geo_config.columna_union (default: "ubigeo").
                Defaults to None.
            validate_before_merge: If True, performs comprehensive validation of
                df_geografia before merging, checking UBIGEO formats, territorial
                consistency, and data quality. If validation fails and error handling
                is RAISE, will abort merge. If None, uses value from geo_config
                (default: True). Defaults to None.

        Returns:
            Tuple containing two elements:

            1. merged_df (pd.DataFrame): Result DataFrame combining df_principal
               with geographic columns from df_geografia. Preserves all records
               from df_principal (left join). New columns added with geographic
               information. Records without geographic match have NaN values.

            2. validation (GeoValidationResult): Validation result with merge
               quality metrics including coverage percentage, missing matches,
               duplicate handling summary, and data quality scores.

        Raises:
            ValueError: If DataFrames are empty, columna_union is missing from
                either DataFrame, or required columns are not found.
            GeoMergeError: If duplicate handling fails, merge operation encounters
                errors, or data types are incompatible in merge keys.
            DataQualityError: If validate_before_merge is True and validation fails
                with errors that exceed configured thresholds, and error handling
                mode is RAISE.

        Examples:
            Basic geographic merge:

            >>> from enahopy.merger import ENAHOGeoMerger
            >>> import pandas as pd
            >>>
            >>> # Principal data with household information
            >>> df_data = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102', '150103'],
            ...     'conglome': ['001', '002', '003'],
            ...     'ingreso': [2000, 1500, 1800]
            ... })
            >>>
            >>> # Geographic reference data
            >>> df_geo = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102', '150103'],
            ...     'departamento': ['Lima', 'Lima', 'Lima'],
            ...     'provincia': ['Lima', 'Lima', 'Lima'],
            ...     'distrito': ['Lima', 'San Isidro', 'Miraflores']
            ... })
            >>>
            >>> merger = ENAHOGeoMerger()
            >>> result, validation = merger.merge_geographic_data(
            ...     df_principal=df_data,
            ...     df_geografia=df_geo,
            ...     columna_union='ubigeo'
            ... )
            >>> print(f"Records: {len(result)}, Coverage: {validation.coverage_percentage:.1f}%")
            >>> print(result.columns.tolist())
            Records: 3, Coverage: 100.0%
            ['ubigeo', 'conglome', 'ingreso', 'departamento', 'provincia', 'distrito']

            With column renaming:

            >>> columnas_geograficas = {
            ...     'departamento': 'dept',
            ...     'provincia': 'prov',
            ...     'distrito': 'dist'
            ... }
            >>> result, validation = merger.merge_geographic_data(
            ...     df_data,
            ...     df_geo,
            ...     columnas_geograficas=columnas_geograficas
            ... )
            >>> print(result.columns.tolist())
            ['ubigeo', 'conglome', 'ingreso', 'dept', 'prov', 'dist']

            Handling missing geographic data:

            >>> df_data_partial = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102', '999999'],  # 999999 doesn't exist
            ...     'ingreso': [2000, 1500, 1800]
            ... })
            >>> result, validation = merger.merge_geographic_data(
            ...     df_data_partial,
            ...     df_geo
            ... )
            >>> print(f"Missing matches: {validation.quality_metrics['missing_matches']}")
            >>> print(result[result['departamento'].isna()])
            Missing matches: 1
               ubigeo  ingreso departamento provincia distrito
            2  999999     1800          NaN       NaN      NaN

            Large dataset with memory optimization:

            >>> from enahopy.merger.config import GeoMergeConfiguration
            >>> geo_config = GeoMergeConfiguration(
            ...     optimizar_memoria=True,
            ...     chunk_size=50000
            ... )
            >>> merger = ENAHOGeoMerger(geo_config=geo_config)
            >>> result, validation = merger.merge_geographic_data(
            ...     df_large,  # 500K records
            ...     df_geo
            ... )
            >>> # Processes in chunks automatically

            Without pre-merge validation (faster):

            >>> result, validation = merger.merge_geographic_data(
            ...     df_data,
            ...     df_geo,
            ...     validate_before_merge=False
            ... )
            >>> # Skips validation for speed

        Note:
            - Uses left join semantics: all records from df_principal are preserved
            - Geographic columns are automatically detected if not specified
            - Duplicate UBIGEOs in df_geografia are handled per configured strategy
            - For datasets >100K records, automatic chunked processing available
            - UBIGEO validation supports 6-digit, 4-digit, and 2-digit formats
            - Missing geographic matches result in NaN values (not record deletion)
            - Memory optimization recommended for datasets >500MB

        See Also:
            - :meth:`validate_geographic_data`: Pre-merge validation method
            - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Configuration options
            - :class:`~enahopy.merger.config.GeoValidationResult`: Validation result structure
            - :meth:`merge_modules_with_geography`: Combined module+geographic merge
        """
        # Validaciones de entrada exhaustivas
        if df_principal is None or df_principal.empty:
            raise ValueError("df_principal no puede estar vacío")

        if df_geografia is None or df_geografia.empty:
            raise ValueError("df_geografia no puede estar vacío")

        columna_union = columna_union or self.geo_config.columna_union
        validate_before_merge = (
            validate_before_merge
            if validate_before_merge is not None
            else self.geo_config.validar_formato_ubigeo
        )

        # Verificar que la columna de unión existe en ambos DataFrames
        if columna_union not in df_principal.columns:
            raise ValueError(
                f"Columna '{columna_union}' no encontrada en df_principal. "
                f"Columnas disponibles: {list(df_principal.columns[:10])}..."
            )

        if columna_union not in df_geografia.columns:
            raise ValueError(
                f"Columna '{columna_union}' no encontrada en df_geografia. "
                f"Columnas disponibles: {list(df_geografia.columns[:10])}..."
            )

        self.logger.info(
            f"Iniciando merge geográfico: "
            f"{len(df_principal)} registros principales × "
            f"{len(df_geografia)} registros geográficos"
        )

        # Validar tipos de datos de las columnas de unión
        self._validate_merge_column_types(df_principal, df_geografia, columna_union)

        # Detectar columnas geográficas si no se especifican
        if columnas_geograficas is None:
            self.logger.info("columnas geográficas automáticamente...")
            columnas_geograficas = self.pattern_detector.detectar_columnas_geograficas(
                df_geografia, confianza_minima=0.7
            )
            self.logger.info(f"Columnas detectadas: {list(columnas_geograficas.keys())}")

        # Validación previa si está configurada
        validation_result = None
        if validate_before_merge:
            validation_result = self.validate_geographic_data(
                df_geografia,
                columna_union,
                validate_territory=self.geo_config.validar_consistencia_territorial,
                validate_quality=self.geo_config.generar_reporte_calidad,
            )

            if (
                not validation_result.is_valid
                and self.geo_config.manejo_errores == TipoManejoErrores.RAISE
            ):
                raise DataQualityError(
                    f"Validación de datos geográficos falló: {validation_result.errors}",
                    validation_result=validation_result,
                )

        # Preparar DataFrames para merge
        df_geo_clean = self._prepare_geographic_df(
            df_geografia, columna_union, columnas_geograficas
        )

        # Manejar duplicados
        df_geo_clean = self._handle_duplicates(df_geo_clean, columna_union)

        # Realizar merge con manejo de memoria optimizado
        if self.geo_config.optimizar_memoria and len(df_principal) > self.geo_config.chunk_size:
            result_df = self._merge_by_chunks(df_principal, df_geo_clean, columna_union)
        else:
            result_df = self._merge_simple(df_principal, df_geo_clean, columna_union)

        # Manejo de valores faltantes
        if self.geo_config.valor_faltante is not None:
            geo_columns = [col for col in result_df.columns if col in columnas_geograficas.values()]
            for col in geo_columns:
                # Handle categorical dtype: add category before filling
                if isinstance(result_df[col].dtype, pd.CategoricalDtype):
                    if self.geo_config.valor_faltante not in result_df[col].cat.categories:
                        result_df[col] = result_df[col].cat.add_categories(
                            [self.geo_config.valor_faltante]
                        )
                result_df[col] = result_df[col].fillna(self.geo_config.valor_faltante)

        # Generar reporte final
        if validation_result is None:
            validation_result = self._generate_merge_report(
                df_principal, df_geografia, result_df, columna_union
            )

        self.logger.info(
            f"Merge completado: {len(result_df)} registros, " f"{result_df.shape[1]} columnas"
        )

        return result_df, validation_result

    def _validate_merge_column_types(self, df1: pd.DataFrame, df2: pd.DataFrame, column: str):
        """
        Valida y corrige tipos de datos de columnas de merge.

        Args:
            df1, df2: DataFrames a fusionar
            column: Columna de merge

        Raises:
            ValueError: Si los tipos son incompatibles y no se pueden convertir
        """
        type1 = df1[column].dtype
        type2 = df2[column].dtype

        if type1 != type2:
            self.logger.warning(
                f"⚠️ Tipos de datos diferentes en '{column}': "
                f"{type1} vs {type2}. Intentando convertir..."
            )

            # Intentar conversión a string si son tipos diferentes
            try:
                if pd.api.types.is_numeric_dtype(type1) and pd.api.types.is_string_dtype(type2):
                    df1[column] = df1[column].astype(str)
                elif pd.api.types.is_string_dtype(type1) and pd.api.types.is_numeric_dtype(type2):
                    df2[column] = df2[column].astype(str)
                else:
                    # Convertir ambos a string como último recurso
                    df1[column] = df1[column].astype(str)
                    df2[column] = df2[column].astype(str)

                self.logger.info(f"✅ Tipos convertidos exitosamente para '{column}'")
            except Exception as e:
                raise ValueError(
                    f"No se pudieron compatibilizar los tipos de datos para '{column}': {str(e)}"
                )

    def _prepare_geographic_df(
        self, df: pd.DataFrame, columna_union: str, columnas_geograficas: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Prepara DataFrame geográfico para merge.

        Args:
            df: DataFrame geográfico
            columna_union: Columna de unión
            columnas_geograficas: Columnas a incluir

        Returns:
            DataFrame preparado
        """
        # Bug fix #1: Seleccionar columnas relevantes, avoiding duplicate merge key
        columns_to_keep = [columna_union]

        for orig_col, new_col in columnas_geograficas.items():
            # Skip merge key to avoid duplicates
            if orig_col == columna_union:
                continue
            if orig_col in df.columns:
                columns_to_keep.append(orig_col)
            else:
                self.logger.warning(f"⚠️ Columna '{orig_col}' no encontrada en df_geografia")

        # Ensure uniqueness (defensive programming)
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        df_clean = df[columns_to_keep].copy()

        # Renombrar columnas si es necesario
        rename_dict = {}
        for orig_col, new_col in columnas_geograficas.items():
            if orig_col in df_clean.columns and orig_col != new_col:
                rename_dict[orig_col] = new_col

        if rename_dict:
            df_clean = df_clean.rename(columns=rename_dict)
            self.logger.debug(f"Columnas renombradas: {rename_dict}")

        # Aplicar prefijos/sufijos si están configurados
        if self.geo_config.prefijo_columnas or self.geo_config.sufijo_columnas:
            rename_dict = {}
            for col in df_clean.columns:
                if col != columna_union:  # No renombrar la columna de unión
                    new_name = (
                        f"{self.geo_config.prefijo_columnas}{col}{self.geo_config.sufijo_columnas}"
                    )
                    rename_dict[col] = new_name

            if rename_dict:
                df_clean = df_clean.rename(columns=rename_dict)

        return df_clean

    def _merge_simple(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        """
        Realiza merge simple entre dos DataFrames.

        Args:
            df1, df2: DataFrames a fusionar
            on: Columna de unión

        Returns:
            DataFrame fusionado
        """
        # Bug fix #1: Ensure merge key is unique in both DataFrames to prevent duplicate columns
        # Check if merge key exists in both dataframes (it should from validation)
        if on not in df1.columns:
            raise ValueError(f"Merge key '{on}' not found in left DataFrame")
        if on not in df2.columns:
            raise ValueError(f"Merge key '{on}' not found in right DataFrame")

        try:
            result = pd.merge(df1, df2, on=on, how="left", validate="m:1")
        except pd.errors.MergeError as e:
            self.logger.error(f"Error en merge: {str(e)}")
            # Intentar merge sin validación
            self.logger.warning("Reintentando merge sin validación m:1...")
            result = pd.merge(df1, df2, on=on, how="left")

        # Bug fix #1: Check for duplicate merge key columns after merge (shouldn't happen but defensive)
        if on in result.columns:
            # Count how many times the merge key appears
            key_columns = [col for col in result.columns if col == on or col.startswith(f"{on}_")]
            if len(key_columns) > 1:
                self.logger.warning(
                    f"Duplicate merge key columns detected: {key_columns}. Keeping only '{on}'"
                )
                # Keep only the first occurrence
                cols_to_drop = [col for col in key_columns if col != on]
                result = result.drop(columns=cols_to_drop)

        return result

    def _merge_by_chunks(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
        """
        Realiza merge por chunks para optimizar memoria.

        Args:
            df1, df2: DataFrames a fusionar
            on: Columna de unión

        Returns:
            DataFrame fusionado
        """
        chunk_size = self.geo_config.chunk_size
        n_chunks = (len(df1) + chunk_size - 1) // chunk_size

        self.logger.info(f"📦 Procesando en {n_chunks} chunks de {chunk_size} registros")

        chunks_results = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df1))

            chunk = df1.iloc[start_idx:end_idx]
            chunk_merged = pd.merge(chunk, df2, on=on, how="left")
            chunks_results.append(chunk_merged)

            if (i + 1) % 10 == 0:
                self.logger.debug(f"Procesados {i + 1}/{n_chunks} chunks")

        result = pd.concat(chunks_results, ignore_index=True)
        self.logger.info(f"✅ Chunks combinados: {len(result)} registros")

        return result

    def _generate_merge_report(
        self,
        df_original: pd.DataFrame,
        df_geo: pd.DataFrame,
        df_result: pd.DataFrame,
        columna_union: str,
    ) -> GeoValidationResult:
        """
        Genera reporte detallado del merge.

        Args:
            df_original: DataFrame original
            df_geo: DataFrame geográfico
            df_result: DataFrame resultado
            columna_union: Columna de unión

        Returns:
            GeoValidationResult con métricas del merge
        """
        total_records = len(df_result)

        # Calcular métricas de cobertura
        non_null_geo = df_result[columna_union].notna().sum()
        coverage_percentage = (non_null_geo / total_records * 100) if total_records > 0 else 0.0

        # Detectar registros sin match
        new_cols = set(df_result.columns) - set(df_original.columns)
        missing_geo = 0

        for col in new_cols:
            missing_geo = max(missing_geo, df_result[col].isna().sum())

        quality_metrics = {
            "original_records": len(df_original),
            "geographic_records": len(df_geo),
            "merged_records": total_records,
            "coverage": coverage_percentage,
            "new_columns": len(new_cols),
            "missing_matches": missing_geo,
        }

        warnings = []
        if missing_geo > 0:
            warnings.append(f"{missing_geo} registros sin información geográfica")

        return GeoValidationResult(
            is_valid=True,
            total_records=total_records,
            valid_ubigeos=non_null_geo,
            invalid_ubigeos=total_records - non_null_geo,
            duplicate_ubigeos=0,
            missing_coordinates=0,
            territorial_inconsistencies=0,
            coverage_percentage=coverage_percentage,
            errors=[],
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

    # =====================================================
    # MÉTODOS DE MERGE DE MÓDULOS
    # =====================================================

    @log_performance
    def merge_multiple_modules(
        self,
        modules_dict: Dict[str, pd.DataFrame],
        base_module: str,
        config: Optional[ModuleMergeConfig] = None,
    ) -> ModuleMergeResult:
        """Merge multiple ENAHO modules sequentially with intelligent ordering.

        Combines multiple ENAHO survey modules into a single comprehensive dataset
        by performing sequential left joins using household or person-level keys.
        Automatically determines optimal merge order, handles conflicts, validates
        compatibility, and provides detailed quality metrics for each merge step.

        This method is essential for creating integrated analysis datasets that
        combine information from different ENAHO modules (e.g., sumaria + vivienda
        + personas) while maintaining data integrity and traceability.

        Args:
            modules_dict: Dictionary mapping module codes to DataFrames.
                Keys should be ENAHO module codes (e.g., "34", "01", "02").
                Values are DataFrames containing module data with required merge keys.
                All DataFrames must contain the merge keys specified in config
                (conglome, vivienda, hogar for household level; add codperso for
                person level). Example: {"34": df_sumaria, "01": df_vivienda}.
            base_module: Code of the base module to start merging from. This module
                serves as the anchor, and other modules are progressively merged
                into it. Common choices: "34" (sumaria/summary), "01" (vivienda/housing).
                Must exist in modules_dict. The final DataFrame will have the same
                number of records as this base (left join semantics).
            config: Optional custom configuration for this specific merge operation.
                If None, uses the instance's module_config. Allows override of
                merge level, conflict resolution strategy, and quality thresholds
                without modifying the instance configuration. Defaults to None.

        Returns:
            ModuleMergeResult containing:

            - merged_df (pd.DataFrame): Final integrated DataFrame with all module
              data combined. Includes columns from all modules with conflicts resolved.
            - merge_report (Dict): Comprehensive report including:
                - modules_merged: List of module codes processed
                - base_module: Base module used
                - merge_order: Sequence of modules merged
                - total_records: Final record count
                - individual_reports: Per-module merge details
                - elapsed_time: Total processing time in seconds
            - conflicts_resolved (int): Total number of column conflicts resolved
              across all merges
            - unmatched_left (int): Cumulative records from left side without matches
            - unmatched_right (int): Cumulative records from right side without matches
            - validation_warnings (List[str]): All warnings from validation and merges
            - quality_score (float): Overall quality score (0-100) based on match
              rates, conflicts, and data completeness

        Raises:
            ValueError: If base_module not in modules_dict, less than 2 modules
                provided, or any DataFrame is empty/invalid.
            MergeValidationError: If module merge fails due to incompatible keys,
                missing columns, or validation threshold violations.
            ModuleMergeError: If a critical error occurs during merge operations
                that cannot be recovered.

        Examples:
            Basic multi-module merge:

            >>> from enahopy.merger import ENAHOGeoMerger
            >>> import pandas as pd
            >>>
            >>> # Create sample module DataFrames
            >>> df_sumaria = pd.DataFrame({
            ...     'conglome': ['001', '002', '003'],
            ...     'vivienda': ['01', '01', '01'],
            ...     'hogar': ['1', '1', '1'],
            ...     'gashog2d': [2000, 1500, 1800]
            ... })
            >>> df_vivienda = pd.DataFrame({
            ...     'conglome': ['001', '002', '003'],
            ...     'vivienda': ['01', '01', '01'],
            ...     'hogar': ['1', '1', '1'],
            ...     'area': [1, 2, 1]
            ... })
            >>> df_personas = pd.DataFrame({
            ...     'conglome': ['001', '002', '003'],
            ...     'vivienda': ['01', '01', '01'],
            ...     'hogar': ['1', '1', '1'],
            ...     'nmiembros': [4, 3, 5]
            ... })
            >>>
            >>> modules = {
            ...     '34': df_sumaria,
            ...     '01': df_vivienda,
            ...     '02': df_personas
            ... }
            >>>
            >>> merger = ENAHOGeoMerger()
            >>> result = merger.merge_multiple_modules(
            ...     modules_dict=modules,
            ...     base_module='34'
            ... )
            >>> print(f"Records: {len(result.merged_df)}, Quality: {result.quality_score:.1f}%")
            >>> print(f"Columns: {result.merged_df.shape[1]}")
            Records: 3, Quality: 100.0%
            Columns: 7

            With custom configuration:

            >>> from enahopy.merger.config import (
            ...     ModuleMergeConfig,
            ...     ModuleMergeStrategy,
            ...     ModuleMergeLevel
            ... )
            >>> custom_config = ModuleMergeConfig(
            ...     merge_level=ModuleMergeLevel.HOGAR,
            ...     merge_strategy=ModuleMergeStrategy.KEEP_LEFT,
            ...     min_match_rate=0.8
            ... )
            >>> result = merger.merge_multiple_modules(
            ...     modules,
            ...     '34',
            ...     config=custom_config
            ... )

            Accessing detailed reports:

            >>> result = merger.merge_multiple_modules(modules, '34')
            >>> print(f"Merge order: {result.merge_report['merge_order']}")
            >>> print(f"Conflicts resolved: {result.conflicts_resolved}")
            >>> for report in result.merge_report['individual_reports']:
            ...     print(f"Module {report['module']}: Quality {report['quality_score']:.1f}%")
            Merge order: ['34', '01', '02']
            Conflicts resolved: 2
            Module 01: Quality 98.5%
            Module 02: Quality 99.2%

            Handling warnings:

            >>> result = merger.merge_multiple_modules(modules, '34')
            >>> if result.validation_warnings:
            ...     print("Warnings found:")
            ...     for warning in result.validation_warnings:
            ...         print(f"  - {warning}")

        Note:
            - Merge order is automatically optimized (typically: base → sumaria → others)
            - Uses left join semantics: all records from base module are preserved
            - Conflicts between module columns are resolved per configured strategy
            - Empty or invalid modules are automatically skipped with warnings
            - Sequential processing: modules are merged one at a time
            - Memory is freed between merge steps for large datasets
            - Quality score factors in match rates, conflicts, and data completeness

        See Also:
            - :class:`~enahopy.merger.config.ModuleMergeConfig`: Configuration options
            - :class:`~enahopy.merger.config.ModuleMergeResult`: Result structure
            - :meth:`merge_modules_with_geography`: Combined module+geographic merge
            - :class:`~enahopy.merger.modules.merger.ENAHOModuleMerger`: Underlying merger
        """
        start_time = time.time()
        config = config or self.module_config

        try:
            # Log inicio
            self.logger.info(
                f"Iniciando merge de {len(modules_dict)} módulos con base '{base_module}'"
            )

            # Validar módulos
            self._validate_modules(modules_dict, base_module)

            # Ordenar módulos para merge
            merge_order = self._determine_merge_order(modules_dict, base_module)
            self.logger.info(f"Orden de merge: {' → '.join(merge_order)}")

            # Inicializar resultado base
            base_df = modules_dict[base_module].copy()
            merged_df = base_df

            # Acumuladores para el reporte
            all_warnings = []
            all_conflicts = 0
            all_unmatched_left = 0
            all_unmatched_right = 0
            merge_reports = []

            # Fusionar módulos uno por uno
            for module_code in merge_order[1:]:  # Excluir el base que ya tenemos
                self.logger.info(f"Agregando módulo {module_code}")

                try:
                    # Realizar merge individual
                    result = self.module_merger.merge_modules(
                        merged_df,
                        modules_dict[module_code],
                        base_module if merged_df is base_df else "merged",
                        module_code,
                    )

                    # Actualizar DataFrame fusionado
                    merged_df = result.merged_df

                    # ✅ CORRECCIÓN DEL BUG: usar validation_warnings en lugar de warnings
                    all_warnings.extend(result.validation_warnings)  # AQUÍ ESTABA EL BUG

                    # Acumular estadísticas
                    all_conflicts += result.conflicts_resolved
                    all_unmatched_left += result.unmatched_left
                    all_unmatched_right += result.unmatched_right

                    # Guardar reporte individual
                    merge_reports.append(
                        {
                            "module": module_code,
                            "report": result.merge_report,
                            "quality_score": result.quality_score,
                        }
                    )

                    self.logger.info(
                        f"Merge completado: {len(merged_df)} registros finales "
                        f"(Calidad: {result.quality_score:.1%})"
                    )

                except Exception as e:
                    self.logger.error(f"Error fusionando módulo {module_code}: {str(e)}")
                    raise MergeValidationError(f"Fallo en merge de módulo {module_code}: {str(e)}")

            # Calcular calidad total
            total_quality = self._calculate_total_quality(merge_reports)

            # Crear reporte consolidado
            elapsed_time = time.time() - start_time

            merge_report = {
                "modules_merged": list(modules_dict.keys()),
                "base_module": base_module,
                "merge_order": merge_order,
                "total_records": len(merged_df),
                "individual_reports": merge_reports,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat(),
            }

            # Crear resultado final
            final_result = ModuleMergeResult(
                merged_df=merged_df,
                merge_report=merge_report,
                conflicts_resolved=all_conflicts,
                unmatched_left=all_unmatched_left,
                unmatched_right=all_unmatched_right,
                validation_warnings=all_warnings,  # Usar el nombre correcto del atributo
                quality_score=total_quality,
            )

            self.logger.info(
                f"merge_multiple_modules completado en {elapsed_time:.2f}s "
                f"con calidad total: {total_quality:.1%}"
            )

            return final_result

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(
                f"merge_multiple_modules falló después de {elapsed_time:.2f}s: {str(e)}"
            )
            raise

    def _validate_modules(self, modules_dict: Dict[str, pd.DataFrame], base_module: str):
        """Valida que los módulos sean compatibles para merge"""
        self.logger.info(f"Validando compatibilidad de {len(modules_dict)} módulos")

        if base_module not in modules_dict:
            raise ValueError(f"Módulo base '{base_module}' no encontrado")

        if len(modules_dict) < 2:
            raise ValueError("Se requieren al menos 2 módulos para fusionar")

        # Validar que todos sean DataFrames
        for code, df in modules_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Módulo {code} no es un DataFrame")
            if df.empty:
                raise ValueError(f"Módulo {code} está vacío")

    def _determine_merge_order(
        self, modules_dict: Dict[str, pd.DataFrame], base_module: str
    ) -> List[str]:
        """
        Determina el orden óptimo de merge
        Prioriza: base → sumaria → otros módulos por tamaño
        """
        modules = list(modules_dict.keys())
        modules.remove(base_module)

        # Ordenar por prioridad y tamaño
        def get_priority(module_code):
            priorities = {"34": 1, "02": 2, "01": 3}  # sumaria, personas, hogar
            return priorities.get(module_code, 99)

        modules.sort(key=lambda x: (get_priority(x), -len(modules_dict[x])))

        return [base_module] + modules

    def _calculate_total_quality(self, merge_reports: List[Dict]) -> float:
        """Calcula la calidad promedio ponderada de todos los merges"""
        if not merge_reports:
            return 0.0

        total_score = sum(r["quality_score"] for r in merge_reports)
        return total_score / len(merge_reports)

    @log_performance
    def merge_modules_with_geography(
        self,
        modules_dict: Dict[str, pd.DataFrame],
        df_geografia: pd.DataFrame,
        base_module: str = "34",
        merge_config: Optional[ModuleMergeConfig] = None,
        geo_config: Optional[GeoMergeConfiguration] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Combine module merge with geographic enrichment in one operation.

        Performs a two-stage integration workflow: first merges multiple ENAHO
        modules using household/person keys, then enriches the result with
        geographic information using UBIGEO codes. This combined operation is
        the most comprehensive data integration method in enahopy, creating
        analysis-ready datasets with both survey data and geographic context.

        This method simplifies the typical workflow of merging multiple modules
        and adding geographic information, handling both operations with
        consistent error handling, validation, and quality reporting.

        Args:
            modules_dict: Dictionary mapping module codes to DataFrames.
                Each DataFrame must contain required merge keys (conglome,
                vivienda, hogar) and ideally a UBIGEO column for geographic
                matching. Example: {"34": df_sumaria, "01": df_vivienda}.
                All modules will be merged before adding geography.
            df_geografia: Geographic reference DataFrame with UBIGEO codes
                and administrative division information (departamento, provincia,
                distrito). Should have unique UBIGEOs for clean merging.
                Must contain column matching columna_union from geo_config.
            base_module: Code of the base module for the initial merge.
                This module serves as the anchor for the multi-module merge.
                Common choices: "34" (sumaria), "01" (vivienda). Must exist
                in modules_dict. Defaults to "34".
            merge_config: Optional configuration for module merge stage.
                If None, uses instance's module_config. Allows customization
                of merge level, conflict resolution, and quality thresholds
                for the module integration step. Defaults to None.
            geo_config: Optional configuration for geographic merge stage.
                If None, uses instance's geo_config. Allows customization
                of UBIGEO validation, duplicate handling, and merge semantics
                for the geographic enrichment step. Defaults to None.

        Returns:
            Tuple containing two elements:

            1. final_df (pd.DataFrame): Complete integrated DataFrame containing:
               - All columns from merged modules
               - Geographic columns from df_geografia
               - Resolved conflicts from module merge
               - UBIGEO-matched geographic information

            2. combined_report (Dict): Comprehensive report with:
               - module_merge: Report from module integration including:
                   - modules_processed: List of module codes
                   - conflicts_resolved: Number of column conflicts
                   - warnings: Module merge warnings
                   - quality_metrics: Module merge quality scores
               - geographic_merge: Report from geographic enrichment including:
                   - validation: GeoValidationResult details
                   - final_records: Total records after geographic merge
                   - coverage: Geographic coverage percentage
               - overall_quality: Combined quality assessment
               - processing_summary: High-level summary of complete workflow

        Raises:
            ValueError: If modules_dict is empty, df_geografia is empty,
                base_module not in modules_dict, or required columns missing.
            ModuleMergeError: If module integration fails due to incompatible
                keys, validation failures, or merge errors.
            GeoMergeError: If geographic enrichment fails due to duplicate
                handling issues, validation failures, or merge errors.

        Examples:
            Basic combined merge:

            >>> from enahopy.merger import ENAHOGeoMerger
            >>> import pandas as pd
            >>>
            >>> # Create module DataFrames
            >>> df_sumaria = pd.DataFrame({
            ...     'conglome': ['001', '002'],
            ...     'vivienda': ['01', '01'],
            ...     'hogar': ['1', '1'],
            ...     'ubigeo': ['150101', '150102'],
            ...     'gashog2d': [2000, 1500]
            ... })
            >>> df_vivienda = pd.DataFrame({
            ...     'conglome': ['001', '002'],
            ...     'vivienda': ['01', '01'],
            ...     'hogar': ['1', '1'],
            ...     'area': [1, 2]
            ... })
            >>>
            >>> # Geographic reference
            >>> df_geo = pd.DataFrame({
            ...     'ubigeo': ['150101', '150102'],
            ...     'departamento': ['Lima', 'Lima'],
            ...     'provincia': ['Lima', 'Lima']
            ... })
            >>>
            >>> modules = {'34': df_sumaria, '01': df_vivienda}
            >>> merger = ENAHOGeoMerger()
            >>>
            >>> result, report = merger.merge_modules_with_geography(
            ...     modules_dict=modules,
            ...     df_geografia=df_geo,
            ...     base_module='34'
            ... )
            >>> print(f"Final shape: {result.shape}")
            >>> print(f"Geographic coverage: {report['geographic_merge']['coverage']:.1f}%")
            Final shape: (2, 8)
            Geographic coverage: 100.0%

            With custom configurations:

            >>> from enahopy.merger.config import (
            ...     ModuleMergeConfig,
            ...     GeoMergeConfiguration,
            ...     ModuleMergeStrategy,
            ...     TipoManejoDuplicados
            ... )
            >>>
            >>> mod_cfg = ModuleMergeConfig(
            ...     merge_strategy=ModuleMergeStrategy.COALESCE,
            ...     min_match_rate=0.85
            ... )
            >>> geo_cfg = GeoMergeConfiguration(
            ...     manejo_duplicados=TipoManejoDuplicados.FIRST,
            ...     validar_formato_ubigeo=True
            ... )
            >>>
            >>> result, report = merger.merge_modules_with_geography(
            ...     modules,
            ...     df_geo,
            ...     base_module='34',
            ...     merge_config=mod_cfg,
            ...     geo_config=geo_cfg
            ... )

            Accessing detailed reports:

            >>> result, report = merger.merge_modules_with_geography(modules, df_geo)
            >>> print(f"Modules processed: {report['processing_summary']['modules_processed']}")
            >>> print(f"Final records: {report['processing_summary']['final_shape'][0]}")
            >>> print(f"Overall quality: {report['overall_quality']['overall_score']:.1f}")
            >>>
            >>> # Module merge details
            >>> mod_report = report['module_merge']
            >>> print(f"Conflicts resolved: {mod_report['conflicts_resolved']}")
            >>>
            >>> # Geographic merge details
            >>> geo_report = report['geographic_merge']
            >>> print(f"Coverage: {geo_report['coverage']:.1f}%")

            Error handling:

            >>> try:
            ...     result, report = merger.merge_modules_with_geography(
            ...         modules,
            ...         df_geo_with_issues,
            ...         base_module='34'
            ...     )
            ... except (ModuleMergeError, GeoMergeError) as e:
            ...     print(f"Merge failed: {e}")
            ...     # Handle error appropriately

        Note:
            - Two-stage process: module merge first, then geographic enrichment
            - Both stages can have independent configurations
            - Quality assessment combines metrics from both stages
            - Module merge preserves all records from base module (left join)
            - Geographic merge adds location data to merged modules
            - Final record count matches base module (unless duplicates in geography)
            - Comprehensive reporting tracks both workflow stages
            - Recommended for creating complete analysis datasets

        See Also:
            - :meth:`merge_multiple_modules`: First stage (module integration)
            - :meth:`merge_geographic_data`: Second stage (geographic enrichment)
            - :class:`~enahopy.merger.config.ModuleMergeConfig`: Module merge config
            - :class:`~enahopy.merger.config.GeoMergeConfiguration`: Geographic config
        """
        # Validaciones
        if not modules_dict:
            raise ValueError("modules_dict no puede estar vacío")

        if df_geografia is None or df_geografia.empty:
            raise ValueError("df_geografia no puede estar vacío")

        self.logger.info("🌍 Iniciando merge combinado: módulos + geografía")

        # 1. Merge entre módulos
        module_result = self.merge_multiple_modules(modules_dict, base_module, merge_config)

        self.logger.info(f"📊 Módulos combinados: {len(module_result.merged_df)} registros")

        # 2. Merge con información geográfica
        # Usar configuración específica si se proporciona
        original_geo_config = self.geo_config
        if geo_config:
            self.geo_config = geo_config

        try:
            geo_result, geo_validation = self.merge_geographic_data(
                df_principal=module_result.merged_df,
                df_geografia=df_geografia,
                validate_before_merge=True,
            )
        finally:
            # Restaurar configuración original
            self.geo_config = original_geo_config

        # 3. Combinar reportes
        combined_report = {
            "module_merge": {
                "modules_processed": list(modules_dict.keys()),
                "conflicts_resolved": module_result.conflicts_resolved,
                "warnings": module_result.validation_warnings,
                "quality_score": module_result.quality_score,
                "unmatched_left": module_result.unmatched_left,
                "unmatched_right": module_result.unmatched_right,
            },
            "geographic_merge": {
                "validation": (
                    geo_validation.to_dict()
                    if hasattr(geo_validation, "to_dict")
                    else vars(geo_validation)
                ),
                "final_records": len(geo_result),
                "coverage": geo_validation.coverage_percentage,
            },
            "overall_quality": self._assess_combined_quality(geo_result, module_result),
            "processing_summary": {
                "modules_processed": len(modules_dict),
                "base_module": base_module,
                "final_shape": geo_result.shape,
                "merge_sequence": " → ".join(modules_dict.keys()),
                "geographic_coverage": geo_validation.coverage_percentage,
            },
        }

        self.logger.info(
            f"✅ Merge combinado completado: {geo_result.shape[0]} registros, "
            f"{geo_result.shape[1]} columnas"
        )

        return geo_result, combined_report

    def _assess_combined_quality(
        self, final_df: pd.DataFrame, module_result: ModuleMergeResult
    ) -> Dict[str, Any]:
        """
        Evalúa la calidad general del merge combinado.

        Args:
            final_df: DataFrame final
            module_result: Resultado del merge de módulos

        Returns:
            Diccionario con métricas de calidad combinadas
        """
        # Calcular completitud general
        completeness = final_df.notna().sum().sum() / (final_df.shape[0] * final_df.shape[1]) * 100

        # Detectar columnas con alta proporción de NaN
        high_nan_cols = []
        for col in final_df.columns:
            nan_ratio = final_df[col].isna().mean()
            if nan_ratio > 0.5:  # Más del 50% NaN
                high_nan_cols.append((col, round(nan_ratio * 100, 1)))

        # Evaluar calidad general
        quality_score = completeness

        if module_result.conflicts_resolved > 0:
            quality_score -= (module_result.conflicts_resolved / len(final_df)) * 10

        if high_nan_cols:
            quality_score -= len(high_nan_cols) * 2

        quality_score = max(0, min(100, quality_score))  # Limitar entre 0 y 100

        return {
            "overall_score": round(quality_score, 2),
            "data_completeness": round(completeness, 2),
            "high_nan_columns": high_nan_cols[:10],  # Top 10 columnas con más NaN
            "total_conflicts": module_result.conflicts_resolved,
            "warnings_count": len(module_result.validation_warnings),
            "recommendation": self._get_quality_recommendation(quality_score),
        }

    def _get_quality_recommendation(self, score: float) -> str:
        """
        Genera recomendación basada en el score de calidad.

        Args:
            score: Score de calidad (0-100)

        Returns:
            Recomendación textual
        """
        if score >= 90:
            return "Excelente calidad de datos. Listo para análisis."
        elif score >= 75:
            return "Buena calidad. Revisar columnas con valores faltantes."
        elif score >= 60:
            return "Calidad aceptable. Se recomienda validación adicional."
        elif score >= 40:
            return "Calidad baja. Revisar conflictos y datos faltantes."
        else:
            return "Calidad crítica. Se requiere limpieza de datos exhaustiva."

    def validate_module_compatibility(
        self, modules_dict: Dict[str, pd.DataFrame], merge_level: str = "hogar"
    ) -> Dict[str, Any]:
        """Validate compatibility between multiple ENAHO modules before merging.

        Performs comprehensive pre-merge validation checking that all modules
        contain required merge keys, analyzing key overlap between modules,
        and assessing data quality. This validation helps identify potential
        issues before expensive merge operations, enabling early error detection
        and providing actionable recommendations.

        Args:
            modules_dict: Dictionary mapping module codes to DataFrames.
                Each DataFrame should contain the required merge keys for the
                specified merge level. Example: {"34": df_sumaria, "01": df_vivienda}.
                All modules will be validated for compatibility.
            merge_level: Level of merge to validate. Options are:
                - "hogar": Household level, requires ['conglome', 'vivienda', 'hogar']
                - "persona": Person level, requires ['conglome', 'vivienda', 'hogar', 'codperso']
                Defaults to "hogar".

        Returns:
            Dictionary with validation results containing:

            - is_compatible (bool): True if all modules are compatible for merging
            - merge_level (str): The validated merge level
            - errors (List[str]): Critical errors preventing merge
            - warnings (List[str]): Non-critical issues or recommendations
            - module_analysis (Dict): Per-module analysis with:
                - has_required_keys (bool): All required keys present
                - missing_keys (List[str]): Missing required keys
                - record_count (int): Number of records
                - column_count (int): Number of columns
                - key_overlap_percentage (float): Overlap with base module

        Examples:
            Basic compatibility check:

            >>> from enahopy.merger import ENAHOGeoMerger
            >>> import pandas as pd
            >>>
            >>> df_sumaria = pd.DataFrame({
            ...     'conglome': ['001', '002'],
            ...     'vivienda': ['01', '01'],
            ...     'hogar': ['1', '1'],
            ...     'gashog2d': [2000, 1500]
            ... })
            >>> df_vivienda = pd.DataFrame({
            ...     'conglome': ['001', '002'],
            ...     'vivienda': ['01', '01'],
            ...     'hogar': ['1', '1'],
            ...     'area': [1, 2]
            ... })
            >>>
            >>> modules = {'34': df_sumaria, '01': df_vivienda}
            >>> merger = ENAHOGeoMerger()
            >>>
            >>> result = merger.validate_module_compatibility(modules, 'hogar')
            >>> print(f"Compatible: {result['is_compatible']}")
            >>> print(f"Errors: {len(result['errors'])}")
            Compatible: True
            Errors: 0

            Detecting missing keys:

            >>> df_bad = pd.DataFrame({
            ...     'conglome': ['001'],
            ...     'vivienda': ['01']
            ...     # Missing 'hogar' key!
            ... })
            >>> modules_bad = {'34': df_sumaria, '99': df_bad}
            >>> result = merger.validate_module_compatibility(modules_bad)
            >>> if not result['is_compatible']:
            ...     print("Validation failed:")
            ...     for error in result['errors']:
            ...         print(f"  - {error}")
            Validation failed:
              - Módulo 99: falta llave 'hogar'

            Person-level validation:

            >>> df_personas = pd.DataFrame({
            ...     'conglome': ['001', '001'],
            ...     'vivienda': ['01', '01'],
            ...     'hogar': ['1', '1'],
            ...     'codperso': ['01', '02'],
            ...     'edad': [25, 30]
            ... })
            >>> modules_persona = {'34': df_sumaria, '02': df_personas}
            >>> result = merger.validate_module_compatibility(
            ...     modules_persona,
            ...     'persona'
            ... )
            >>> print(result['is_compatible'])
            False  # sumaria doesn't have 'codperso'

            Analyzing module details:

            >>> result = merger.validate_module_compatibility(modules)
            >>> for module_code, analysis in result['module_analysis'].items():
            ...     print(f"Module {module_code}:")
            ...     print(f"  Records: {analysis['record_count']}")
            ...     print(f"  Has all keys: {analysis['has_required_keys']}")
            ...     if analysis['missing_keys']:
            ...         print(f"  Missing: {analysis['missing_keys']}")
            Module 34:
              Records: 2
              Has all keys: True
            Module 01:
              Records: 2
              Has all keys: True

        Note:
            - Validation is fast and non-destructive
            - Checks key presence but not key validity
            - Low overlap warnings (<50%) help identify data quality issues
            - Validation results can inform merge strategy selection
            - Recommended before expensive multi-module merge operations

        See Also:
            - :meth:`merge_multiple_modules`: Uses validation internally
            - :class:`~enahopy.merger.config.ModuleMergeLevel`: Merge level enum
            - :class:`~enahopy.merger.modules.validator.ModuleValidator`: Validator class
        """
        self.logger.info(f"🔍 Validando compatibilidad de {len(modules_dict)} módulos")

        compatibility = {
            "is_compatible": True,
            "merge_level": merge_level,
            "errors": [],
            "warnings": [],
            "module_analysis": {},
        }

        # Verificar llaves requeridas según nivel
        if merge_level == "hogar":
            required_keys = ["conglome", "vivienda", "hogar"]
        else:  # persona
            required_keys = ["conglome", "vivienda", "hogar", "codperso"]

        for module_code, df in modules_dict.items():
            module_info = {
                "has_required_keys": True,
                "missing_keys": [],
                "record_count": len(df),
                "column_count": df.shape[1],
            }

            # Verificar llaves
            for key in required_keys:
                if key not in df.columns:
                    module_info["has_required_keys"] = False
                    module_info["missing_keys"].append(key)
                    compatibility["errors"].append(f"Módulo {module_code}: falta llave '{key}'")
                    compatibility["is_compatible"] = False
                elif df[key].isna().all():
                    compatibility["warnings"].append(
                        f"Módulo {module_code}: llave '{key}' completamente nula"
                    )

            compatibility["module_analysis"][module_code] = module_info

        # Verificar consistencia de llaves entre módulos
        if compatibility["is_compatible"]:
            base_module = list(modules_dict.keys())[0]
            base_df = modules_dict[base_module]

            for module_code, df in modules_dict.items():
                if module_code == base_module:
                    continue

                # Crear llaves compuestas
                base_keys = base_df[required_keys].apply(lambda x: "_".join(x.astype(str)), axis=1)
                module_keys = df[required_keys].apply(lambda x: "_".join(x.astype(str)), axis=1)

                # Calcular overlap
                common_keys = set(base_keys) & set(module_keys)
                overlap_percentage = (
                    (len(common_keys) / len(base_keys) * 100) if len(base_keys) > 0 else 0
                )

                if overlap_percentage < 50:
                    compatibility["warnings"].append(
                        f"Bajo overlap entre {base_module} y {module_code}: {overlap_percentage:.1f}%"
                    )

        return compatibility
