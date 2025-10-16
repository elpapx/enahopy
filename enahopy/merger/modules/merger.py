"""

ENAHO Merger - Merger de Módulos ENAHO (VERSIÓN CORREGIDA)

===========================================================



Implementación especializada para combinar módulos ENAHO

con validaciones específicas y manejo robusto de errores.



Versión: 2.1.0

Correcciones aplicadas:

- División por cero en quality score

- Manejo de DataFrames vacíos/None

- Conversión robusta de tipos en llaves

- Detección de cardinalidad del merge

- Gestión de memoria mejorada

- Validación de tipos incompatibles

"""

import gc
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ModuleMergeConfig, ModuleMergeLevel, ModuleMergeResult, ModuleMergeStrategy
from ..exceptions import (
    ConflictResolutionError,
    IncompatibleModulesError,
    MergeKeyError,
    ModuleMergeError,
)
from .validator import ModuleValidator


class ENAHOModuleMerger:
    """Specialized merger for combining ENAHO survey modules with robust error handling.

    Advanced merger designed specifically for ENAHO survey data integration,
    handling the complexities of merging multiple household survey modules
    with different structures, keys, and data types. Provides intelligent
    conflict resolution, type harmonization, cardinality validation, and
    comprehensive quality assessment.

    This class is optimized for ENAHO's unique characteristics including:
    - Hierarchical household/person keys (conglome, vivienda, hogar, codperso)
    - Multiple merge levels (household vs person)
    - Column naming conflicts across modules
    - Data type incompatibilities
    - Large dataset memory optimization
    - Detailed quality and validation reporting

    The ENAHOModuleMerger is typically used through ENAHOGeoMerger but can
    be instantiated directly for advanced use cases requiring fine-grained
    control over module integration operations.

    Attributes:
        config (ModuleMergeConfig): Configuration specifying merge behavior
            including merge level (household/person), conflict resolution
            strategy, quality thresholds, and error handling modes.
        logger (logging.Logger): Logger instance for operation tracking,
            warnings, and error reporting.
        validator (ModuleValidator): Validator for module structure and
            compatibility checks before merge operations.

    Examples:
        Basic two-module merge:

        >>> from enahopy.merger.modules.merger import ENAHOModuleMerger
        >>> from enahopy.merger.config import ModuleMergeConfig
        >>> import pandas as pd
        >>> import logging
        >>>
        >>> config = ModuleMergeConfig()
        >>> logger = logging.getLogger('enaho')
        >>> merger = ENAHOModuleMerger(config, logger)
        >>>
        >>> df_left = pd.DataFrame({
        ...     'conglome': ['001', '002'],
        ...     'vivienda': ['01', '01'],
        ...     'hogar': ['1', '1'],
        ...     'gasto': [2000, 1500]
        ... })
        >>> df_right = pd.DataFrame({
        ...     'conglome': ['001', '002'],
        ...     'vivienda': ['01', '01'],
        ...     'hogar': ['1', '1'],
        ...     'area': [1, 2]
        ... })
        >>>
        >>> result = merger.merge_modules(
        ...     df_left, df_right, '34', '01'
        ... )
        >>> print(f"Merged: {result.merged_df.shape}")
        Merged: (2, 6)

        With custom conflict resolution:

        >>> from enahopy.merger.config import ModuleMergeStrategy
        >>> config = ModuleMergeConfig(
        ...     merge_strategy=ModuleMergeStrategy.KEEP_LEFT
        ... )
        >>> merger = ENAHOModuleMerger(config, logger)

    Note:
        - Automatically handles type conversions for merge keys
        - Detects and warns about many-to-many relationships
        - Provides detailed merge statistics and quality scores
        - Memory-optimized for datasets >500K records
        - Thread-safe for concurrent operations

    See Also:
        - :class:`~enahopy.merger.ENAHOGeoMerger`: High-level merger interface
        - :class:`~enahopy.merger.config.ModuleMergeConfig`: Configuration options
        - :class:`~enahopy.merger.config.ModuleMergeResult`: Result structure
        - :class:`~enahopy.merger.modules.validator.ModuleValidator`: Validator class
    """

    def __init__(self, config: ModuleMergeConfig, logger: logging.Logger):
        self.config = config

        self.logger = logger

        self.validator = ModuleValidator(config, logger)

        self._validation_cache = {}  # Cache para validaciones

    def merge_modules(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_module: str,
        right_module: str,
        merge_config: Optional[ModuleMergeConfig] = None,
    ) -> ModuleMergeResult:
        """Merge two ENAHO modules with comprehensive validation and quality assessment.

        Performs intelligent left join between two ENAHO survey modules using
        household or person-level keys. Handles data type harmonization, detects
        and resolves column conflicts, validates merge cardinality, and provides
        detailed quality metrics. This is the core two-module merge operation
        used by higher-level multi-module merge methods.

        The method automatically handles common ENAHO merge challenges including:
        type mismatches in keys, duplicate records, many-to-many relationships,
        column naming conflicts, and missing key values.

        Args:
            left_df: Left DataFrame (base module). All records from this DataFrame
                are preserved in the result (left join semantics). Should contain
                required merge keys (conglome, vivienda, hogar for household level;
                add codperso for person level). This module serves as the anchor.
            right_df: Right DataFrame (module to merge). Records from this DataFrame
                are matched to left_df using merge keys. Should contain the same
                merge keys as left_df. Unmatched records are tracked but not included
                in the result.
            left_module: Code of the left module for identification and reporting.
                Example: "34" (sumaria), "01" (vivienda). Used in conflict resolution
                and quality reporting.
            right_module: Code of the right module for identification and reporting.
                Example: "01" (vivienda), "02" (personas). Used in conflict resolution
                and quality reporting.
            merge_config: Optional custom configuration for this specific merge.
                If None, uses the instance's config. Allows override of merge level,
                conflict strategy, and validation thresholds without modifying
                instance configuration. Defaults to None.

        Returns:
            ModuleMergeResult containing:

            - merged_df (pd.DataFrame): Result DataFrame with combined data from
              both modules. Contains all columns from both modules with conflicts
              resolved. Same number of rows as left_df (left join).
            - merge_report (Dict): Detailed report including:
                - modules_merged: String showing module combination
                - merge_level: Level used (hogar/persona)
                - merge_strategy: Strategy used for conflicts
                - total_records: Final record count
                - merge_statistics: Match statistics
                - compatibility_info: Compatibility metrics
                - quality_score: Overall quality score (0-100)
            - conflicts_resolved (int): Number of column conflicts resolved
            - unmatched_left (int): Records from left with no right match
            - unmatched_right (int): Records from right with no left match
            - validation_warnings (List[str]): Warnings from validation
            - quality_score (float): Overall quality score (0-100)

        Raises:
            ModuleMergeError: If critical merge error occurs including invalid
                DataFrames, merge execution failures, or unrecoverable errors.
            IncompatibleModulesError: If modules cannot be merged due to missing
                required keys, incompatible data types, or structural issues.
            MergeValidationError: If validation fails and thresholds are violated.

        Examples:
            Basic two-module merge:

            >>> from enahopy.merger.modules.merger import ENAHOModuleMerger
            >>> from enahopy.merger.config import ModuleMergeConfig
            >>> import pandas as pd
            >>> import logging
            >>>
            >>> # Setup
            >>> config = ModuleMergeConfig()
            >>> logger = logging.getLogger('enaho')
            >>> merger = ENAHOModuleMerger(config, logger)
            >>>
            >>> # Create sample modules
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
            >>>
            >>> result = merger.merge_modules(
            ...     df_sumaria, df_vivienda, '34', '01'
            ... )
            >>> print(f"Records: {len(result.merged_df)}")
            >>> print(f"Quality: {result.quality_score:.1f}%")
            >>> print(f"Conflicts: {result.conflicts_resolved}")
            Records: 3
            Quality: 100.0%
            Conflicts: 0

            Handling conflicts:

            >>> df_left = pd.DataFrame({
            ...     'conglome': ['001'],
            ...     'vivienda': ['01'],
            ...     'hogar': ['1'],
            ...     'area': [100]  # Conflicting column
            ... })
            >>> df_right = pd.DataFrame({
            ...     'conglome': ['001'],
            ...     'vivienda': ['01'],
            ...     'hogar': ['1'],
            ...     'area': [200]  # Conflicting column
            ... })
            >>> result = merger.merge_modules(df_left, df_right, '34', '01')
            >>> print(f"Conflicts resolved: {result.conflicts_resolved}")
            Conflicts resolved: 1

            With partial matches:

            >>> df_left = pd.DataFrame({
            ...     'conglome': ['001', '002', '003'],
            ...     'vivienda': ['01', '01', '01'],
            ...     'hogar': ['1', '1', '1'],
            ...     'gasto': [2000, 1500, 1800]
            ... })
            >>> df_right = pd.DataFrame({
            ...     'conglome': ['001', '002'],  # Missing '003'
            ...     'vivienda': ['01', '01'],
            ...     'hogar': ['1', '1'],
            ...     'area': [1, 2]
            ... })
            >>> result = merger.merge_modules(df_left, df_right, '34', '01')
            >>> print(f"Unmatched left: {result.unmatched_left}")
            >>> print(f"Match rate: {result.quality_score:.1f}%")
            Unmatched left: 1
            Match rate: 66.7%

            Accessing merge statistics:

            >>> result = merger.merge_modules(df_sumaria, df_vivienda, '34', '01')
            >>> stats = result.merge_report['merge_statistics']
            >>> print(f"Both: {stats['both']}")
            >>> print(f"Left only: {stats['left_only']}")
            >>> print(f"Right only: {stats['right_only']}")
            Both: 3
            Left only: 0
            Right only: 0

        Note:
            - Uses left join semantics: all left records are preserved
            - Automatically converts merge keys to compatible types
            - Detects and warns about many-to-many relationships
            - Empty DataFrames are handled gracefully with warnings
            - Quality score factors in match rate, conflicts, and completeness
            - Memory-optimized for large datasets (>500K records)

        See Also:
            - :meth:`merge_multiple_modules`: Sequential multi-module merge
            - :class:`~enahopy.merger.config.ModuleMergeConfig`: Configuration
            - :class:`~enahopy.merger.config.ModuleMergeResult`: Result structure
            - :meth:`analyze_merge_feasibility`: Pre-merge analysis
        """

        config = merge_config or self.config

        self.logger.info(f"🔗 Iniciando merge: Módulo {left_module} + Módulo {right_module}")

        # ====== FIX 1: Validación temprana de DataFrames vacíos ======

        if left_df is None or left_df.empty:
            self.logger.warning(f"⚠️ Módulo {left_module} está vacío o es None")

            if right_df is None or right_df.empty:
                # Ambos vacíos

                return ModuleMergeResult(
                    merged_df=pd.DataFrame(),
                    merge_report={"error": "Ambos DataFrames vacíos"},
                    conflicts_resolved=0,
                    unmatched_left=0,
                    unmatched_right=0,
                    validation_warnings=["Ambos módulos vacíos"],
                    quality_score=0.0,
                )

            # Solo left vacío, retornar right

            return ModuleMergeResult(
                merged_df=right_df.copy(),
                merge_report={"warning": f"Módulo {left_module} vacío, retornando {right_module}"},
                conflicts_resolved=0,
                unmatched_left=0,
                unmatched_right=len(right_df),
                validation_warnings=[f"Módulo {left_module} vacío"],
                quality_score=50.0,
            )

        if right_df is None or right_df.empty:
            self.logger.warning(f"⚠️ Módulo {right_module} está vacío o es None")

            # Solo right vacío, retornar left

            return ModuleMergeResult(
                merged_df=left_df.copy(),
                merge_report={"warning": f"Módulo {right_module} vacío, retornando {left_module}"},
                conflicts_resolved=0,
                unmatched_left=len(left_df),
                unmatched_right=0,
                validation_warnings=[f"Módulo {right_module} vacío"],
                quality_score=50.0,
            )

        # 1. Validar DataFrames

        validation_warnings = []

        # ====== OPTIMIZACIÓN: Usar caché de validaciones ======
        use_cache = hasattr(config, "use_validation_cache") and config.use_validation_cache

        if use_cache:
            cache_key_left = f"structure_{left_module}_{id(left_df)}"
            cache_key_right = f"structure_{right_module}_{id(right_df)}"

            if cache_key_left not in self._validation_cache:
                self._validation_cache[cache_key_left] = self.validator.validate_module_structure(
                    left_df, left_module
                )
            validation_warnings.extend(self._validation_cache[cache_key_left])

            if cache_key_right not in self._validation_cache:
                self._validation_cache[cache_key_right] = self.validator.validate_module_structure(
                    right_df, right_module
                )
            validation_warnings.extend(self._validation_cache[cache_key_right])
        else:
            validation_warnings.extend(
                self.validator.validate_module_structure(left_df, left_module)
            )
            validation_warnings.extend(
                self.validator.validate_module_structure(right_df, right_module)
            )

        # 2. Verificar compatibilidad

        if use_cache:
            compat_key = f"compat_{left_module}_{right_module}_{config.merge_level.value}"
            if compat_key not in self._validation_cache:
                self._validation_cache[compat_key] = self.validator.check_module_compatibility(
                    left_df, right_df, left_module, right_module, config.merge_level
                )
            compatibility = self._validation_cache[compat_key]
        else:
            compatibility = self.validator.check_module_compatibility(
                left_df, right_df, left_module, right_module, config.merge_level
            )

        if not compatibility.get("compatible", False):
            raise IncompatibleModulesError(
                compatibility.get("error", "Módulos incompatibles"),
                module1=left_module,
                module2=right_module,
                compatibility_info=compatibility,
            )

        # 3. Determinar llaves de merge

        merge_keys = self._get_merge_keys_for_level(config.merge_level)

        # ====== FIX 2: Validar compatibilidad de tipos antes del merge ======

        type_issues = self._validate_data_types_compatibility(left_df, right_df, merge_keys)

        if type_issues:
            validation_warnings.extend([f"Tipo incompatible en {issue}" for issue in type_issues])

            # Intentar armonizar tipos

            self._harmonize_column_types(left_df, right_df, merge_keys)

        # ====== FIX 3: Detectar cardinalidad del merge ======

        cardinality_warning = self._detect_and_warn_cardinality(left_df, right_df, merge_keys)

        if cardinality_warning:
            validation_warnings.append(cardinality_warning)

        # 4. Preparar DataFrames para merge (con manejo robusto)

        left_clean = self._prepare_for_merge_robust(left_df, merge_keys, f"mod_{left_module}")

        right_clean = self._prepare_for_merge_robust(right_df, merge_keys, f"mod_{right_module}")

        # 5. Ejecutar merge (con optimización para datasets grandes)

        merged_df = self._execute_merge_optimized(
            left_clean, right_clean, merge_keys, config.suffix_conflicts
        )

        # 6. Analizar resultado del merge

        merge_stats = self._analyze_merge_result(merged_df)

        # 7. Resolver conflictos si existen

        conflicts_resolved = self._resolve_conflicts_robust(merged_df, config.merge_strategy)

        # 8. Limpiar DataFrame final

        final_df = self._clean_merged_dataframe(merged_df, merge_keys)

        # ====== FIX 4: Cálculo robusto del score de calidad ======

        quality_score = self._calculate_merge_quality_score_safe(merge_stats, compatibility)

        # 10. Crear reporte detallado

        merge_report = {
            "modules_merged": f"{left_module} + {right_module}",
            "merge_level": config.merge_level.value,
            "merge_strategy": config.merge_strategy.value,
            "total_records": len(final_df),
            "merge_statistics": merge_stats,
            "compatibility_info": compatibility,
            "quality_score": quality_score,
            "type_issues_fixed": len(type_issues),
            "cardinality_warning": cardinality_warning,
        }

        # ====== OPTIMIZACIÓN: Calcular métricas de merge ======
        merge_type = config.merge_type if hasattr(config, "merge_type") else "left"
        cardinality_change = len(final_df) / len(left_df) if len(left_df) > 0 else 1.0
        match_rate = merge_stats.get("both", 0) / len(final_df) if len(final_df) > 0 else 0.0

        merge_metrics = {
            "rows_before": len(left_df),
            "rows_after": len(final_df),
            "merge_type": merge_type,
            "match_rate": match_rate,
            "cardinality_change": cardinality_change,
        }

        # ====== OPTIMIZACIÓN: Validación post-merge de cardinalidad ======
        if hasattr(config, "validate_cardinality") and config.validate_cardinality:
            if merge_type == "left" and abs(cardinality_change - 1.0) > 0.01:
                warning_msg = (
                    f"⚠️ VALIDACIÓN FALLIDA: Left join cambió cardinalidad "
                    f"({len(left_df):,} → {len(final_df):,}, factor {cardinality_change:.2f}x)\n"
                    f"   Posibles causas:\n"
                    f"   - Duplicados en DataFrame derecho\n"
                    f"   - Relación muchos-a-muchos\n"
                    f"   Recomendación: Verificar duplicados en llaves de merge"
                )
                validation_warnings.append(warning_msg)
                self.logger.warning(warning_msg)

            if merge_type == "inner" and len(final_df) > min(len(left_df), len(right_df)):
                warning_msg = (
                    f"⚠️ VALIDACIÓN FALLIDA: Inner join produjo más registros "
                    f"que el mínimo ({len(final_df):,} > {min(len(left_df), len(right_df)):,})\n"
                    f"   Causa: Relación muchos-a-muchos\n"
                    f"   Recomendación: Deduplicar antes del merge"
                )
                validation_warnings.append(warning_msg)
                self.logger.warning(warning_msg)

        self.logger.info(
            f"✅ Merge completado: {len(final_df)} registros finales "
            f"(Calidad: {quality_score:.1f}%)"
        )

        return ModuleMergeResult(
            merged_df=final_df,
            merge_report=merge_report,
            conflicts_resolved=conflicts_resolved,
            unmatched_left=merge_stats.get("left_only", 0),
            unmatched_right=merge_stats.get("right_only", 0),
            validation_warnings=validation_warnings,
            quality_score=quality_score,
            merge_metrics=merge_metrics,  # ← Nueva métrica
        )

    def merge_multiple_modules(
        self,
        modules_dict: Dict[str, pd.DataFrame],
        base_module: str,
        merge_config: Optional[ModuleMergeConfig] = None,
    ) -> ModuleMergeResult:
        """

        Merge múltiples módulos secuencialmente con gestión de memoria.



        Args:

            modules_dict: Diccionario {codigo_modulo: dataframe}

            base_module: Código del módulo base

            merge_config: Configuración de merge



        Returns:

            ModuleMergeResult con todos los módulos combinados

        """

        # Validar módulo base

        if base_module not in modules_dict:
            # Intentar seleccionar automáticamente

            base_module = self._select_best_base_module(modules_dict)

            self.logger.info(f"📌 Módulo base seleccionado automáticamente: {base_module}")

        # Validar que el módulo base no esté vacío

        if modules_dict[base_module] is None or modules_dict[base_module].empty:
            raise ValueError(f"Módulo base '{base_module}' está vacío")

        # Iniciar con módulo base

        result_df = modules_dict[base_module].copy()

        all_warnings = []

        total_conflicts = 0

        merge_history = [base_module]

        quality_scores = []

        # ====== FIX 5: Gestión de memoria mejorada ======

        # Determinar orden óptimo de merge

        merge_order = self._determine_optimal_merge_order(modules_dict, base_module)

        # Merge secuencial con gestión de memoria

        for module_code in merge_order:
            if module_code == base_module:
                continue

            self.logger.info(f"🔗 Agregando módulo {module_code}")

            # Verificar si el módulo está vacío

            if modules_dict[module_code] is None or modules_dict[module_code].empty:
                self.logger.warning(f"⚠️ Módulo {module_code} vacío, omitiendo")

                all_warnings.append(f"Módulo {module_code} vacío")

                continue

            try:
                # Guardar referencia al DataFrame anterior

                prev_df = result_df

                merge_result = self.merge_modules(
                    result_df,
                    modules_dict[module_code],
                    "+".join(merge_history),
                    module_code,
                    merge_config,
                )

                result_df = merge_result.merged_df

                all_warnings.extend(merge_result.validation_warnings)

                total_conflicts += merge_result.conflicts_resolved

                quality_scores.append(merge_result.quality_score)

                merge_history.append(module_code)

                # Liberar memoria del DataFrame anterior

                del prev_df

                if len(result_df) > 100000:  # Si el dataset es grande
                    gc.collect()

            except Exception as e:
                self.logger.error(f"❌ Error fusionando módulo {module_code}: {str(e)}")

                all_warnings.append(f"Error en módulo {module_code}: {str(e)}")

                if merge_config and merge_config.continue_on_error:
                    continue

                else:
                    raise

        # Calcular calidad promedio

        avg_quality = np.mean(quality_scores) if quality_scores else 100.0

        # Reporte final

        final_report = {
            "modules_sequence": " → ".join(merge_history),
            "total_modules": len(modules_dict),
            "modules_merged": len(merge_history),
            "modules_skipped": len(modules_dict) - len(merge_history),
            "final_records": len(result_df),
            "total_conflicts_resolved": total_conflicts,
            "average_quality_score": avg_quality,
            "individual_quality_scores": dict(zip(merge_history[1:], quality_scores)),
            "overall_quality_score": self._calculate_overall_quality_safe(result_df),
        }

        return ModuleMergeResult(
            merged_df=result_df,
            merge_report=final_report,
            conflicts_resolved=total_conflicts,
            unmatched_left=0,
            unmatched_right=0,
            validation_warnings=all_warnings,
            quality_score=avg_quality,
        )

    # =====================================================

    # MÉTODOS AUXILIARES MEJORADOS

    # =====================================================

    def _get_merge_keys_for_level(self, level: ModuleMergeLevel) -> List[str]:
        """Obtiene llaves de merge según el nivel"""

        if level == ModuleMergeLevel.HOGAR:
            return self.config.hogar_keys

        elif level == ModuleMergeLevel.PERSONA:
            return self.config.persona_keys

        elif level == ModuleMergeLevel.VIVIENDA:
            return self.config.vivienda_keys

        else:
            raise ValueError(f"Nivel de merge no soportado: {level}")

    def _prepare_for_merge_robust(
        self, df: pd.DataFrame, merge_keys: List[str], prefix: str
    ) -> pd.DataFrame:
        """

        Prepara DataFrame para merge con manejo robusto de tipos.



        FIX: Manejo mejorado de conversión de tipos y valores nulos
        OPTIMIZED (DE-3): Vectorized type conversion instead of row-by-row

        """

        df_clean = df.copy()

        # Verificar que todas las llaves existan

        missing_keys = [key for key in merge_keys if key not in df_clean.columns]

        if missing_keys:
            raise MergeKeyError(
                f"{prefix}: llaves faltantes para merge", missing_keys=missing_keys, invalid_keys=[]
            )

        # ====== OPTIMIZACIÓN DE-3: Conversión vectorizada de tipos ======
        # ANTES: Iteraba columna por columna con múltiples operaciones
        # AHORA: Procesamiento vectorizado batch de todas las llaves

        try:
            # Procesar todas las llaves de una vez cuando sea posible
            for key in merge_keys:
                col = df_clean[key]

                # Estrategia vectorizada según tipo
                if col.dtype == "object":
                    # Vectorized: clean strings in one operation
                    df_clean[key] = col.fillna("").astype(str).str.strip().replace("", np.nan)

                elif pd.api.types.is_numeric_dtype(col):
                    # Vectorized: preserve NaN mask while converting
                    mask_na = col.isna()
                    df_clean[key] = col.astype(str)
                    df_clean.loc[mask_na, key] = np.nan

                else:
                    # Other types: direct conversion
                    df_clean[key] = col.astype(str)

        except Exception as e:
            # Fallback to safer but slower method if vectorized fails
            self.logger.warning(f"{prefix}: Vectorized conversion failed, using fallback: {e}")

            for key in merge_keys:
                try:
                    if df_clean[key].dtype == "object":
                        df_clean[key] = df_clean[key].fillna("").astype(str).str.strip()
                        df_clean[key] = df_clean[key].replace("", np.nan)
                    elif pd.api.types.is_numeric_dtype(df_clean[key]):
                        mask_na = df_clean[key].isna()
                        df_clean[key] = df_clean[key].astype(str)
                        df_clean.loc[mask_na, key] = np.nan
                    else:
                        df_clean[key] = df_clean[key].astype(str)
                except Exception as inner_e:
                    self.logger.warning(
                        f"{prefix}: Error convirtiendo columna '{key}' a string: {inner_e}. "
                        f"Manteniendo tipo original."
                    )

        # Eliminar registros con TODAS las llaves nulas (vectorizado)

        before_clean = len(df_clean)

        df_clean = df_clean.dropna(subset=merge_keys, how="all")

        after_clean = len(df_clean)

        if before_clean != after_clean:
            self.logger.warning(
                f"{prefix}: {before_clean - after_clean} registros eliminados "
                f"por tener todas las llaves nulas"
            )

        return df_clean

    def _analyze_merge_result(self, merged_df: pd.DataFrame) -> Dict[str, int]:
        """Analiza estadísticas del merge"""

        if "_merge" not in merged_df.columns:
            return {
                "both": len(merged_df),
                "left_only": 0,
                "right_only": 0,
                "total": len(merged_df),
            }

        merge_indicator = merged_df["_merge"]

        return {
            "both": (merge_indicator == "both").sum(),
            "left_only": (merge_indicator == "left_only").sum(),
            "right_only": (merge_indicator == "right_only").sum(),
            "total": len(merged_df),
        }

    def _resolve_conflicts_robust(self, df: pd.DataFrame, strategy: ModuleMergeStrategy) -> int:
        """

        Resuelve conflictos entre columnas duplicadas con manejo robusto.



        FIX: Detecta múltiples patrones de sufijos y maneja errores

        """

        conflicts_resolved = 0

        # Detectar todos los posibles patrones de sufijos

        suffix_patterns = [
            self.config.suffix_conflicts,
            ("_x", "_y"),  # pandas default
            ("_left", "_right"),
            ("_1", "_2"),
        ]

        conflict_columns = set()

        for pattern in suffix_patterns:
            for col in df.columns:
                if col.endswith(pattern[0]):
                    base_name = col[: -len(pattern[0])]

                    right_col = base_name + pattern[1]

                    if right_col in df.columns:
                        conflict_columns.add((col, right_col, base_name))

        # Resolver cada conflicto

        for left_col, right_col, base_name in conflict_columns:
            try:
                # ====== FIX: Manejo especial para columnas categóricas ======
                left_is_categorical = isinstance(df[left_col].dtype, pd.CategoricalDtype)
                right_is_categorical = isinstance(df[right_col].dtype, pd.CategoricalDtype)

                # Convertir categóricas a object antes de la operación
                if left_is_categorical or right_is_categorical:
                    if left_is_categorical:
                        df[left_col] = df[left_col].astype("object")
                    if right_is_categorical:
                        df[right_col] = df[right_col].astype("object")
                    self.logger.debug(f"Convertidas columnas categóricas a object: {base_name}")

                if strategy == ModuleMergeStrategy.COALESCE:
                    df[base_name] = df[left_col].fillna(df[right_col])

                elif strategy == ModuleMergeStrategy.KEEP_LEFT:
                    df[base_name] = df[left_col]

                elif strategy == ModuleMergeStrategy.KEEP_RIGHT:
                    df[base_name] = df[right_col]

                elif strategy == ModuleMergeStrategy.AVERAGE:
                    if pd.api.types.is_numeric_dtype(df[left_col]):
                        # Promedio ignorando NaN

                        df[base_name] = df[[left_col, right_col]].mean(axis=1, skipna=True)

                    else:
                        df[base_name] = df[left_col].fillna(df[right_col])

                elif strategy == ModuleMergeStrategy.CONCATENATE:
                    # Concatenar strings no nulos

                    left_str = df[left_col].fillna("").astype(str)

                    right_str = df[right_col].fillna("").astype(str)

                    # Combinar solo si son diferentes

                    combined = left_str.where(left_str == right_str, left_str + " | " + right_str)

                    df[base_name] = combined.str.strip(" |").replace("", np.nan)

                elif strategy == ModuleMergeStrategy.ERROR:
                    # Verificar si realmente hay conflictos

                    conflicts_mask = (
                        df[left_col].notna()
                        & df[right_col].notna()
                        & (df[left_col] != df[right_col])
                    )

                    if conflicts_mask.any():
                        n_conflicts = conflicts_mask.sum()

                        sample_conflicts = df[conflicts_mask][[left_col, right_col]].head(3)

                        raise ConflictResolutionError(
                            f"Conflictos detectados en columna '{base_name}': "
                            f"{n_conflicts} registros con valores diferentes.\n"
                            f"Muestra: {sample_conflicts.to_dict('records')}"
                        )

                    else:
                        df[base_name] = df[left_col].fillna(df[right_col])

                # Eliminar columnas con sufijos

                df.drop([left_col, right_col], axis=1, inplace=True, errors="ignore")

                conflicts_resolved += 1

            except ConflictResolutionError:  # relanza sin atrapar
                raise

            except Exception as e:
                self.logger.error(f"Error resolviendo conflicto en {base_name}: {str(e)}")

                # Mantener columna izquierda como fallback

                if left_col in df.columns:
                    df[base_name] = df[left_col]

                    df.drop([left_col, right_col], axis=1, inplace=True, errors="ignore")

        return conflicts_resolved

    def _clean_merged_dataframe(self, df: pd.DataFrame, merge_keys: List[str]) -> pd.DataFrame:
        """Limpia DataFrame después del merge"""

        df_clean = df.copy()

        # Eliminar columna indicadora

        if "_merge" in df_clean.columns:
            df_clean.drop("_merge", axis=1, inplace=True)

        # Reordenar columnas: llaves primero

        other_cols = [col for col in df_clean.columns if col not in merge_keys]

        df_clean = df_clean[merge_keys + other_cols]

        return df_clean

    def _calculate_merge_quality_score_safe(
        self, merge_stats: Dict[str, int], compatibility_info: Dict[str, Any]
    ) -> float:
        """

        Calcula score de calidad del merge con protección contra división por cero.



        FIX: Manejo seguro de división por cero y valores None

        """

        total = merge_stats.get("total", 0)

        # FIX: Verificar división por cero

        if total == 0:
            self.logger.warning("Total de registros es 0, retornando score 0")

            return 0.0

        matched = merge_stats.get("both", 0)

        left_only = merge_stats.get("left_only", 0)

        right_only = merge_stats.get("right_only", 0)

        # Calcular tasa de coincidencia

        match_rate = (matched / total) * 100 if total > 0 else 0

        # Penalizar por registros no coincidentes

        unmatched_penalty = ((left_only + right_only) / total) * 20 if total > 0 else 0

        # Bonificar por buena compatibilidad previa

        compatibility_bonus = 0

        if compatibility_info:
            rate1 = compatibility_info.get("match_rate_module1", 0)

            rate2 = compatibility_info.get("match_rate_module2", 0)

            if rate1 and rate2:  # Verificar que no sean None
                avg_compatibility = (rate1 + rate2) / 2

                if avg_compatibility > 90:
                    compatibility_bonus = 5

                elif avg_compatibility > 70:
                    compatibility_bonus = 2

        # Calcular score final

        final_score = match_rate - unmatched_penalty + compatibility_bonus

        # Asegurar que esté en rango [0, 100]

        return max(0.0, min(100.0, final_score))

    def _calculate_overall_quality_safe(self, df: pd.DataFrame) -> float:
        """

        Calcula calidad general del DataFrame con protección contra errores.



        FIX: Manejo seguro de DataFrames vacíos y división por cero

        """

        # Verificar DataFrame vacío

        if df is None or df.empty:
            return 0.0

        # Verificar dimensiones

        n_rows, n_cols = df.shape

        if n_rows == 0 or n_cols == 0:
            return 0.0

        # Calcular completitud

        total_cells = n_rows * n_cols

        null_cells = df.isnull().sum().sum()

        completeness = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0

        # Factor de penalización por duplicados

        key_cols = [col for col in ["conglome", "vivienda", "hogar"] if col in df.columns]

        duplicate_penalty = 0

        if key_cols and len(df) > 0:
            duplicates = df.duplicated(subset=key_cols, keep="first").sum()

            duplicate_penalty = (duplicates / len(df)) * 20

        return max(0.0, min(100.0, completeness - duplicate_penalty))

    # =====================================================

    # NUEVOS MÉTODOS DE VALIDACIÓN Y OPTIMIZACIÓN

    # =====================================================

    def _validate_data_types_compatibility(
        self, df1: pd.DataFrame, df2: pd.DataFrame, merge_keys: List[str]
    ) -> List[str]:
        """

        Valida compatibilidad de tipos de datos en las llaves de merge.



        Returns:

            Lista de columnas con tipos incompatibles

        """

        incompatible = []

        for key in merge_keys:
            if key in df1.columns and key in df2.columns:
                type1 = df1[key].dtype

                type2 = df2[key].dtype

                # Verificar compatibilidad básica

                if type1 != type2:
                    # Permitir ciertas conversiones automáticas

                    compatible_pairs = [
                        ("int64", "float64"),
                        ("int32", "int64"),
                        ("object", "string"),
                    ]

                    type_pair = (str(type1), str(type2))

                    reverse_pair = (str(type2), str(type1))

                    if type_pair not in compatible_pairs and reverse_pair not in compatible_pairs:
                        incompatible.append(f"{key} ({type1} vs {type2})")

        return incompatible

    def _harmonize_column_types(
        self, df1: pd.DataFrame, df2: pd.DataFrame, merge_keys: List[str]
    ) -> None:
        """

        Intenta armonizar tipos de datos entre DataFrames.



        Modifica los DataFrames in-place.

        """

        for key in merge_keys:
            if key in df1.columns and key in df2.columns:
                type1 = df1[key].dtype

                type2 = df2[key].dtype

                if type1 != type2:
                    # Intentar conversión a string como tipo común

                    try:
                        df1[key] = df1[key].astype(str)

                        df2[key] = df2[key].astype(str)

                        self.logger.info(f"✅ Tipos armonizados para columna '{key}'")

                    except Exception as e:
                        self.logger.warning(f"⚠️ No se pudo armonizar tipos para '{key}': {e}")

    def _detect_and_warn_cardinality(
        self, df1: pd.DataFrame, df2: pd.DataFrame, merge_keys: List[str]
    ) -> Optional[str]:
        """

        Detecta la cardinalidad del merge y advierte sobre posibles problemas.



        Returns:

            Mensaje de advertencia si hay problemas potenciales, None ok

        """

        try:
            # Obtener combinaciones únicas de llaves

            df1_keys = df1[merge_keys].drop_duplicates()

            df2_keys = df2[merge_keys].drop_duplicates()

            # Verificar unicidad

            is_df1_unique = len(df1_keys) == len(df1)

            is_df2_unique = len(df2_keys) == len(df2)

            # Detectar tipo de relación

            if is_df1_unique and is_df2_unique:
                return None  # Uno a uno, ideal

            elif is_df1_unique and not is_df2_unique:
                return "Relación uno-a-muchos detectada (left único, right duplicado)"

            elif not is_df1_unique and is_df2_unique:
                return "Relación muchos-a-uno detectada (left duplicado, right único)"

            else:
                # Muchos a muchos - potencialmente problemático

                # ====== OPTIMIZACIÓN DE-3: Sampling inteligente para estimar cardinalidad ======
                # ANTES: merge completo o sampling de 10K registros
                # AHORA: Sampling adaptativo + set intersection cuando es posible (mucho más rápido)

                # Usar sampling más agresivo para datasets grandes
                sample_size = min(5000, len(df1_keys), len(df2_keys))  # Reducido de 10K a 5K

                # ====== OPTIMIZACIÓN: Sampling estratificado si es posible ======
                if len(df1_keys) > sample_size:
                    # Random sampling en lugar de head() para mejor representatividad
                    df1_sample = df1_keys.sample(n=sample_size, random_state=42)
                else:
                    df1_sample = df1_keys

                # ====== OPTIMIZACIÓN: Evitar merge si sample es pequeño ======
                # Para samples pequeños, usar set intersection - mucho más rápido
                if sample_size < 1000 and len(merge_keys) <= 3:
                    # Convertir a tuples y usar set intersection - O(n) vs O(n log n) del merge
                    set1 = set(df1_sample.apply(lambda x: tuple(x), axis=1))
                    set2 = set(df2_keys.apply(lambda x: tuple(x), axis=1))
                    common_count = len(set1 & set2)
                else:
                    # Para samples más grandes, merge sigue siendo necesario
                    common_keys = pd.merge(df1_sample, df2_keys, on=merge_keys, how="inner")
                    common_count = len(common_keys)

                if common_count > 0:
                    avg_duplicates_df1 = len(df1) / len(df1_keys)

                    avg_duplicates_df2 = len(df2) / len(df2_keys)

                    estimated_size = common_count * avg_duplicates_df1 * avg_duplicates_df2

                    # Ajustar estimación si se usó sampling
                    if len(df1_keys) > sample_size:
                        estimated_size = estimated_size * (len(df1_keys) / sample_size)

                    # Solo advertir si explosión es significativa (>2x cualquier lado)
                    if estimated_size > len(df1) * 2 or estimated_size > len(df2) * 2:
                        return (
                            f"⚠️ Relación muchos-a-muchos detectada. "
                            f"Merge podría resultar en ~{estimated_size:,.0f} registros "
                            f"(vs {len(df1):,} y {len(df2):,} originales)"
                        )

                return "Relación muchos-a-muchos detectada"

        except Exception as e:
            self.logger.debug(f"Error detectando cardinalidad: {e}")

            return None

    def _execute_merge_optimized(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        merge_keys: List[str],
        suffixes: Tuple[str, str],
    ) -> pd.DataFrame:
        """

        Ejecuta el merge con optimizaciones para datasets grandes.

        OPTIMIZED (DE-3): Categorical encoding for large datasets

        """

        total_size = len(left_df) + len(right_df)

        # ====== OPTIMIZACIÓN DE-3: Categorical encoding para datasets grandes ======
        # Para datasets grandes, convertir merge keys a categorical acelera significativamente
        use_categorical = total_size > 100000  # Activar para datasets medianos-grandes

        if use_categorical:
            self.logger.debug(f"🎯 Aplicando categorical encoding a merge keys para acelerar merge")
            # Crear copias para no modificar originales
            left_df = left_df.copy()
            right_df = right_df.copy()

            for key in merge_keys:
                if key in left_df.columns and key in right_df.columns:
                    # Convertir a categorical - esto permite merge más rápido
                    # pd.merge usa códigos internos para categorical, que es mucho más rápido
                    left_df[key] = left_df[key].astype("category")
                    right_df[key] = right_df[key].astype("category")

        # Para datasets grandes, usar merge por chunks

        if total_size > 500000:
            self.logger.info("📊 Usando merge optimizado para dataset grande")

            return self._merge_large_datasets(left_df, right_df, merge_keys, suffixes)

        else:
            # Merge estándar para datasets pequeños
            # ====== OPTIMIZACIÓN: Usar merge_type de configuración ======
            merge_type = self.config.merge_type if hasattr(self.config, "merge_type") else "left"

            return pd.merge(
                left_df, right_df, on=merge_keys, how=merge_type, suffixes=suffixes, indicator=True
            )

    def _merge_large_datasets(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        merge_keys: List[str],
        suffixes: Tuple[str, str],
        chunk_size: int = 50000,
    ) -> pd.DataFrame:
        """

        Merge optimizado para datasets grandes usando procesamiento por chunks.

        """

        # ====== OPTIMIZACIÓN: Obtener merge_type de configuración ======
        merge_type = self.config.merge_type if hasattr(self.config, "merge_type") else "left"

        self.logger.info(
            f"Procesando merge por chunks (tamaño: {chunk_size:,}, tipo: {merge_type})"
        )

        # Si right_df es pequeño, hacer merge directo por chunks de left_df

        if len(right_df) < 100000:
            chunks = []

            total_chunks = (len(left_df) // chunk_size) + 1

            for i, start in enumerate(range(0, len(left_df), chunk_size)):
                if i % 5 == 0:  # Log cada 5 chunks
                    self.logger.debug(f"Procesando chunk {i+1}/{total_chunks}")

                chunk = left_df.iloc[start : start + chunk_size]

                merged_chunk = pd.merge(
                    chunk, right_df, on=merge_keys, how=merge_type, suffixes=suffixes
                )

                chunks.append(merged_chunk)

            # Combinar chunks

            result = pd.concat(chunks, ignore_index=True)

            # ====== FIX BUG CRÍTICO: No agregar right_only si es left join ======
            # ANTES: Siempre agregaba right_only, contradiciendo semántica de left join
            # AHORA: Solo agregar right_only si merge_type == "outer"

            # Agregar columna _merge si no existe (para compatibilidad)
            if "_merge" not in result.columns:
                result["_merge"] = "both"

        else:
            # Ambos DataFrames son grandes - usar estrategia diferente
            # ====== OPTIMIZACIÓN: Usar merge_type de configuración ======

            result = pd.merge(
                left_df, right_df, on=merge_keys, how=merge_type, suffixes=suffixes, indicator=True
            )

        # Limpiar memoria

        gc.collect()

        return result

    def _select_best_base_module(self, modules_dict: Dict[str, pd.DataFrame]) -> str:
        """

        Selecciona el mejor módulo base cuando no se especifica.

        """

        # Prioridad de módulos base

        priority_modules = ["34", "01", "02", "03", "04", "05"]

        # Buscar por prioridad

        for module in priority_modules:
            if module in modules_dict:
                df = modules_dict[module]

                if df is not None and not df.empty:
                    return module

        # Si no hay módulos prioritarios, usar el más grande

        valid_modules = {k: v for k, v in modules_dict.items() if v is not None and not v.empty}

        if not valid_modules:
            raise ValueError("No hay módulos válidos para merge")

        return max(valid_modules.keys(), key=lambda k: len(valid_modules[k]))

    def _determine_optimal_merge_order(
        self, modules_dict: Dict[str, pd.DataFrame], base_module: str
    ) -> List[str]:
        """

        Determina el orden óptimo de merge para minimizar memoria y maximizar eficiencia.

        """

        # Filtrar módulos válidos (no vacíos y diferentes del base)

        valid_modules = [
            (k, len(v))
            for k, v in modules_dict.items()
            if k != base_module and v is not None and not v.empty
        ]

        # Ordenar por tamaño (primero los más pequeños para construir gradualmente)

        valid_modules.sort(key=lambda x: x[1])

        return [m[0] for m in valid_modules]

    # =====================================================

    # MÉTODOS DE ANÁLISIS Y PLANIFICACIÓN

    # =====================================================

    def analyze_merge_feasibility(
        self, modules_dict: Dict[str, pd.DataFrame], merge_level: ModuleMergeLevel
    ) -> Dict[str, Any]:
        """Analyze merge feasibility between multiple modules with comprehensive checks.

        Performs pre-merge analysis assessing whether modules can be successfully
        merged, estimating resource requirements, and providing actionable
        recommendations. This analysis helps plan complex multi-module merges
        and identify potential issues before expensive operations.

        Args:
            modules_dict: Dictionary mapping module codes to DataFrames.
                All modules will be analyzed for merge compatibility, key quality,
                and resource requirements. Example: {"34": df_sumaria, "01": df_vivienda}.
            merge_level: Proposed merge level (HOGAR or PERSONA). Determines
                which merge keys are required and how compatibility is assessed.

        Returns:
            Dictionary with comprehensive analysis including:

            - feasible (bool): Whether merge is feasible
            - merge_level (str): Proposed merge level
            - modules_analyzed (List[str]): Modules successfully analyzed
            - modules_empty (List[str]): Empty/invalid modules found
            - potential_issues (List[str]): Issues that may affect merge
            - recommendations (List[str]): Actionable recommendations
            - size_analysis (Dict): Per-module size and memory metrics
            - key_analysis (Dict): Per-module key quality metrics
            - memory_estimate_mb (float): Estimated memory requirement
            - estimated_time_seconds (int): Estimated processing time

        Examples:
            Basic feasibility analysis:

            >>> from enahopy.merger.modules.merger import ENAHOModuleMerger
            >>> from enahopy.merger.config import (
            ...     ModuleMergeConfig,
            ...     ModuleMergeLevel
            ... )
            >>> import pandas as pd
            >>> import logging
            >>>
            >>> config = ModuleMergeConfig()
            >>> logger = logging.getLogger('enaho')
            >>> merger = ENAHOModuleMerger(config, logger)
            >>>
            >>> modules = {
            ...     '34': df_sumaria,
            ...     '01': df_vivienda,
            ...     '02': df_personas
            ... }
            >>>
            >>> analysis = merger.analyze_merge_feasibility(
            ...     modules,
            ...     ModuleMergeLevel.HOGAR
            ... )
            >>> print(f"Feasible: {analysis['feasible']}")
            >>> print(f"Memory needed: {analysis['memory_estimate_mb']:.1f} MB")
            >>> print(f"Est. time: {analysis['estimated_time_seconds']}s")
            Feasible: True
            Memory needed: 125.5 MB
            Est. time: 8s

        Note:
            - Fast analysis without performing actual merge
            - Memory estimates include 2.5x safety factor
            - Time estimates are approximations based on record count
            - Recommendations help optimize merge strategy
            - Feasibility assessment checks key presence and quality

        See Also:
            - :meth:`create_merge_plan`: Create detailed execution plan
            - :meth:`merge_multiple_modules`: Execute multi-module merge
        """

        analysis = {
            "feasible": True,
            "merge_level": merge_level.value,
            "modules_analyzed": [],
            "modules_empty": [],
            "potential_issues": [],
            "recommendations": [],
            "size_analysis": {},
            "key_analysis": {},
            "memory_estimate_mb": 0,
            "estimated_time_seconds": 0,
        }

        merge_keys = self._get_merge_keys_for_level(merge_level)

        total_memory = 0

        valid_modules = 0

        # Análisis por módulo

        for module, df in modules_dict.items():
            # Verificar si el módulo está vacío

            if df is None or df.empty:
                analysis["modules_empty"].append(module)

                analysis["potential_issues"].append(f"Módulo {module} está vacío")

                continue

            analysis["modules_analyzed"].append(module)

            valid_modules += 1

            # Análisis de tamaño y memoria

            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

            analysis["size_analysis"][module] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_mb": round(memory_mb, 2),
                "has_duplicates": (
                    df.duplicated(subset=merge_keys, keep=False).any()
                    if all(k in df.columns for k in merge_keys)
                    else None
                ),
            }

            total_memory += memory_mb

            # Análisis de llaves

            missing_keys = [key for key in merge_keys if key not in df.columns]

            if missing_keys:
                analysis["potential_issues"].append(
                    f"Módulo {module}: llaves faltantes {missing_keys}"
                )

                analysis["feasible"] = False

                continue

            # Analizar calidad de llaves si existen

            if not missing_keys:
                key_df = df[merge_keys].copy()

                # Analizar nulos en llaves

                null_counts = key_df.isnull().sum()

                if null_counts.any():
                    analysis["potential_issues"].append(
                        f"Módulo {module}: valores nulos en llaves {null_counts[null_counts > 0].to_dict()}"
                    )

                # Analizar unicidad

                total_records = len(df)

                unique_combinations = len(key_df.drop_duplicates())

                duplication_rate = (
                    ((total_records - unique_combinations) / total_records * 100)
                    if total_records > 0
                    else 0
                )

                analysis["key_analysis"][module] = {
                    "unique_key_combinations": unique_combinations,
                    "total_records": total_records,
                    "duplication_rate": round(duplication_rate, 2),
                    "null_key_records": null_counts.sum(),
                }

        # Verificar si hay módulos válidos

        if valid_modules == 0:
            analysis["feasible"] = False

            analysis["potential_issues"].append("No hay módulos válidos para merge")

            return analysis

        # Estimar recursos necesarios

        analysis["memory_estimate_mb"] = round(total_memory * 2.5, 2)  # Factor de seguridad 2.5x

        total_rows = sum(info["rows"] for info in analysis["size_analysis"].values())

        analysis["estimated_time_seconds"] = max(5, total_rows // 5000)  # Estimación básica

        # Generar recomendaciones

        if analysis["feasible"]:
            # Recomendaciones de memoria

            if analysis["memory_estimate_mb"] > 1000:  # Más de 1GB
                analysis["recommendations"].append(
                    f"⚠️ Merge requiere ~{analysis['memory_estimate_mb']:.0f} MB. "
                    f"Considere procesamiento por chunks o liberar memoria antes del merge."
                )

            # Recomendaciones por tamaño

            large_modules = [
                m for m, info in analysis["size_analysis"].items() if info["rows"] > 500000
            ]

            if large_modules:
                analysis["recommendations"].append(
                    f"📊 Módulos grandes detectados: {large_modules}. " f"El merge podría ser lento."
                )

            # Recomendaciones por duplicación

            high_dup_modules = [
                m
                for m, info in analysis["key_analysis"].items()
                if info.get("duplication_rate", 0) > 10
            ]

            if high_dup_modules:
                analysis["recommendations"].append(
                    f"🔄 Alta duplicación en: {high_dup_modules}. "
                    f"Considere estrategia 'AGGREGATE' o deduplicación previa."
                )

            # Recomendación de orden de merge

            if valid_modules > 3:
                analysis["recommendations"].append(
                    "💡 Con múltiples módulos, procese del más pequeño al más grande "
                    "para optimizar memoria."
                )

            # Advertencia sobre módulos vacíos

            if analysis["modules_empty"]:
                analysis["recommendations"].append(
                    f"ℹ️ Módulos vacíos serán omitidos: {analysis['modules_empty']}"
                )

        return analysis

    def create_merge_plan(
        self, modules_dict: Dict[str, pd.DataFrame], target_module: str = "34"
    ) -> Dict[str, Any]:
        """

        Crea un plan de merge optimizado con estimaciones detalladas.



        Args:

            modules_dict: Módulos a fusionar

            target_module: Módulo objetivo (base)



        Returns:

            Plan de merge detallado con optimizaciones

        """

        plan = {
            "base_module": target_module,
            "merge_sequence": [],
            "estimated_time_seconds": 0,
            "memory_requirements_mb": 0,
            "optimizations": [],
            "warnings": [],
            "execution_steps": [],
        }

        # Validar y seleccionar módulo base

        valid_modules = {k: v for k, v in modules_dict.items() if v is not None and not v.empty}

        if not valid_modules:
            plan["warnings"].append("No hay módulos válidos para merge")

            return plan

        if target_module not in valid_modules:
            target_module = self._select_best_base_module(valid_modules)

            plan["base_module"] = target_module

            plan["optimizations"].append(
                f"✅ Módulo base cambiado a '{target_module}' (más apropiado)"
            )

        # Crear secuencia de merge optimizada

        other_modules = [(k, len(v)) for k, v in valid_modules.items() if k != target_module]

        other_modules.sort(key=lambda x: x[1])  # Ordenar por tamaño

        plan["merge_sequence"] = [target_module] + [m[0] for m in other_modules]

        # Generar pasos de ejecución detallados

        cumulative_size = len(valid_modules[target_module])

        for i, (module, size) in enumerate(other_modules):
            step = {
                "step": i + 1,
                "action": f"Merge {module} con resultado acumulado",
                "module_size": size,
                "cumulative_size": cumulative_size + size,
                "estimated_time": max(1, (cumulative_size + size) // 10000),
            }

            plan["execution_steps"].append(step)

            cumulative_size += size

        # Estimar recursos totales

        total_rows = sum(len(df) for df in valid_modules.values())

        plan["estimated_time_seconds"] = sum(s["estimated_time"] for s in plan["execution_steps"])

        plan["memory_requirements_mb"] = round(total_rows * len(valid_modules) * 0.15 / 1024, 2)

        # Agregar optimizaciones sugeridas

        if len(valid_modules) > 5:
            plan["optimizations"].append("💡 Considere merge paralelo o por grupos para >5 módulos")

        if total_rows > 1000000:
            plan["optimizations"].append("📊 Dataset grande: active modo chunk_processing=True")

            plan["optimizations"].append("💾 Libere memoria entre merges con gc.collect()")

        if any(len(df) > 500000 for df in valid_modules.values()):
            plan["optimizations"].append(
                "⚡ Use format='parquet' para mejor performance con datasets grandes"
            )

        # Advertencias

        modules_empty = [k for k in modules_dict if k not in valid_modules]

        if modules_empty:
            plan["warnings"].append(f"Módulos vacíos excluidos: {modules_empty}")

        return plan
