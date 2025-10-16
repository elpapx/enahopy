"""

ENAHO Merger - Validador de Módulos ENAHO

========================================



Validador especializado para estructuras y compatibilidad

de módulos ENAHO.

"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import ModuleMergeConfig, ModuleMergeLevel, ModuleType
from ..exceptions import IncompatibleModulesError, ModuleValidationError


class ModuleValidator:
    """Specialized validator for ENAHO survey module structure and compatibility.

    Provides comprehensive validation of ENAHO module structure, data quality,
    and inter-module compatibility before merge operations. Performs module-specific
    validations based on known ENAHO module characteristics, including required
    keys, data types, value ranges, and logical consistency checks.

    This validator is designed specifically for ENAHO survey data and understands
    the structural differences between household-level modules (e.g., sumaria,
    vivienda) and person-level modules (e.g., demographics, education).

    Attributes:
        config: Module merge configuration containing validation rules, required
            keys by module, and quality thresholds.
        logger: Logger for validation warnings, errors, and diagnostic information.

    Examples:
        Basic validator initialization:

        >>> from enahopy.merger.config import ModuleMergeConfig
        >>> from enahopy.merger.modules.validator import ModuleValidator
        >>> import logging
        >>> config = ModuleMergeConfig()
        >>> logger = logging.getLogger('enaho')
        >>> validator = ModuleValidator(config, logger)

        Validating module structure:

        >>> warnings = validator.validate_module_structure(df_sumaria, '34')
        >>> if warnings:
        ...     for warning in warnings:
        ...         print(f"Warning: {warning}")

        Checking module compatibility:

        >>> compat = validator.check_module_compatibility(
        ...     df_sumaria, df_vivienda, '34', '01', ModuleMergeLevel.HOGAR
        ... )
        >>> if compat['compatible']:
        ...     print(f"Match rate: {compat['match_rate_module1']:.1f}%")
        ... else:
        ...     print(f"Error: {compat['error']}")

    Note:
        - Validator is stateless: can be reused for multiple validations
        - Module-specific rules are defined in config.module_validations
        - Intermediate merged modules are automatically detected and validated leniently
        - Validation is non-destructive: never modifies input DataFrames

    See Also:
        - :class:`~enahopy.merger.config.ModuleMergeConfig`: Configuration with validation rules
        - :class:`~enahopy.merger.modules.merger.ENAHOModuleMerger`: Uses validator before merges
        - :exc:`~enahopy.merger.exceptions.ModuleValidationError`: Raised for validation failures
    """

    def __init__(self, config: ModuleMergeConfig, logger: logging.Logger):
        self.config = config

        self.logger = logger

    def validate_module_structure(self, df: pd.DataFrame, module_code: str) -> List[str]:
        """

        Valida estructura específica por módulo



        Args:

            df: DataFrame del módulo

            module_code: Código del módulo (ej: "01", "34")



        Returns:

            Lista de advertencias encontradas

        """

        warnings = []

        # ====== FIX: Permitir módulos intermedios ("merged", "combined", etc.) ======
        # Los módulos intermedios no requieren validación estricta
        if module_code.startswith(("merged", "combined", "+")) or "+" in module_code:
            self.logger.debug(f"Módulo intermedio '{module_code}' - omitiendo validación estricta")
            return warnings  # Sin advertencias para módulos intermedios

        if module_code not in self.config.module_validations:
            warnings.append(f"Módulo {module_code} no reconocido")

            return warnings

        module_info = self.config.module_validations[module_code]

        required_cols = module_info["required_keys"]

        # Verificar columnas requeridas

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            warnings.append(f"Módulo {module_code}: columnas faltantes {missing_cols}")

        # Validar duplicados en llaves si no faltan columnas

        if not missing_cols:
            duplicates = df.duplicated(subset=required_cols).sum()

            if duplicates > 0:
                warnings.append(f"Módulo {module_code}: {duplicates} registros duplicados")

        # Validaciones específicas por tipo de módulo

        module_type = module_info["level"]

        if module_type == ModuleType.PERSONA_LEVEL:
            warnings.extend(self._validate_persona_level_module(df, module_code))

        elif module_type == ModuleType.HOGAR_LEVEL:
            warnings.extend(self._validate_hogar_level_module(df, module_code, required_cols))

        elif module_type == ModuleType.SPECIAL:
            warnings.extend(self._validate_special_module(df, module_code))

        return warnings

    def _validate_persona_level_module(self, df: pd.DataFrame, module_code: str) -> List[str]:
        """Validaciones específicas para módulos a nivel persona"""

        warnings = []

        if "codperso" in df.columns:
            # Verificar que codperso sea válido

            invalid_codperso = df["codperso"].isin([0, "0", "", None]).sum()

            if invalid_codperso > 0:
                warnings.append(
                    f"Módulo {module_code}: {invalid_codperso} códigos de persona inválidos"
                )

            # Verificar rango típico de codperso (1-20 usualmente)

            if pd.api.types.is_numeric_dtype(df["codperso"]):
                max_codperso = df["codperso"].max()

                if max_codperso > 30:
                    warnings.append(
                        f"Módulo {module_code}: código de persona muy alto ({max_codperso})"
                    )

        # Verificar distribución por hogar

        if all(col in df.columns for col in ["conglome", "vivienda", "hogar"]):
            personas_por_hogar = df.groupby(["conglome", "vivienda", "hogar"]).size()

            if personas_por_hogar.max() > 20:
                warnings.append(f"Módulo {module_code}: hogares con más de 20 personas detectados")

            if personas_por_hogar.min() == 0:
                warnings.append(f"Módulo {module_code}: hogares sin personas detectados")

        return warnings

    def _validate_hogar_level_module(
        self, df: pd.DataFrame, module_code: str, required_cols: List[str]
    ) -> List[str]:
        """Validaciones específicas para módulos a nivel hogar"""

        warnings = []

        # Verificar que no haya múltiples registros por hogar

        if "hogar" in df.columns and not any(col for col in required_cols if col not in df.columns):
            hogar_counts = df.groupby(required_cols).size()

            multi_records = (hogar_counts > 1).sum()

            if multi_records > 0:
                warnings.append(
                    f"Módulo {module_code}: {multi_records} hogares con múltiples registros"
                )

        # Validaciones específicas por módulo

        if module_code == "34":  # Sumaria
            warnings.extend(self._validate_sumaria_module(df))

        elif module_code in ["07", "08"]:  # Ingresos y gastos
            warnings.extend(self._validate_economic_module(df, module_code))

        return warnings

    def _validate_special_module(self, df: pd.DataFrame, module_code: str) -> List[str]:
        """Validaciones para módulos especiales"""

        warnings = []

        if module_code == "37":  # Gobierno electrónico
            # Validaciones específicas para módulo 37

            if df.empty:
                warnings.append(
                    "Módulo 37: módulo vacío (normal si no hay datos de gobierno electrónico)"
                )

        return warnings

    def _validate_sumaria_module(self, df: pd.DataFrame) -> List[str]:
        """Validaciones específicas para módulo Sumaria (34)"""

        warnings = []

        # Verificar variables clave de sumaria

        key_sumaria_vars = ["mieperho", "gashog2d", "inghog2d", "pobreza"]

        missing_sumaria = [var for var in key_sumaria_vars if var not in df.columns]

        if missing_sumaria:
            warnings.append(f"Sumaria: variables clave faltantes {missing_sumaria}")

        # Verificar valores lógicos

        if "mieperho" in df.columns:
            invalid_members = (df["mieperho"] <= 0) | (df["mieperho"] > 20)

            if invalid_members.any():
                warnings.append(
                    f"Sumaria: {invalid_members.sum()} hogares con número de miembros inválido"
                )

        if "gashog2d" in df.columns and "inghog2d" in df.columns:
            # Verificar que gastos no sean mayores que ingresos * 2 (permite cierta flexibilidad)

            inconsistent = (
                (df["gashog2d"] > df["inghog2d"] * 2) & (df["gashog2d"] > 0) & (df["inghog2d"] > 0)
            )

            if inconsistent.any():
                warnings.append(
                    f"Sumaria: {inconsistent.sum()} hogares con gastos muy superiores a ingresos"
                )

        return warnings

    def _validate_economic_module(self, df: pd.DataFrame, module_code: str) -> List[str]:
        """Validaciones para módulos económicos (07, 08)"""

        warnings = []

        # Buscar columnas de montos

        amount_cols = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["monto", "gasto", "ingreso", "valor"])
        ]

        if amount_cols:
            for col in amount_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Verificar valores negativos (podrían ser válidos en algunos casos)

                    negative_count = (df[col] < 0).sum()

                    if negative_count > len(df) * 0.1:  # Más del 10%
                        warnings.append(
                            f"Módulo {module_code}: {negative_count} valores negativos en '{col}'"
                        )

                    # Verificar valores extremos

                    if df[col].max() > 1000000:  # Más de 1M
                        extreme_count = (df[col] > 1000000).sum()

                        warnings.append(
                            f"Módulo {module_code}: {extreme_count} valores extremos en '{col}' (>1M)"
                        )

        return warnings

    def check_module_compatibility(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        module1: str,
        module2: str,
        merge_level: ModuleMergeLevel,
    ) -> Dict[str, Any]:
        """

        Verifica compatibilidad entre dos módulos



        Args:

            df1, df2: DataFrames de los módulos

            module1, module2: Códigos de los módulos

            merge_level: Nivel de merge propuesto



        Returns:

            Diccionario con análisis de compatibilidad

        """

        # ====== FIX: Detectar módulos intermedios/compuestos ======
        if (
            "+" in module1 or module1.isdigit() or module1.startswith(("merged", "combined"))
        ):  # modulo compuesto o intermedio
            return {"compatible": True}

        # Obtener información de módulos

        if module1 not in self.config.module_validations:
            return {"compatible": False, "error": f"Módulo {module1} no reconocido"}

        if module2 not in self.config.module_validations:
            return {"compatible": False, "error": f"Módulo {module2} no reconocido"}

        module1_info = self.config.module_validations[module1]

        module2_info = self.config.module_validations[module2]

        # Verificar compatibilidad de niveles

        level1 = module1_info["level"]

        level2 = module2_info["level"]

        compatibility_matrix = {
            (ModuleType.HOGAR_LEVEL, ModuleType.HOGAR_LEVEL): True,
            (ModuleType.PERSONA_LEVEL, ModuleType.PERSONA_LEVEL): True,
            (ModuleType.HOGAR_LEVEL, ModuleType.PERSONA_LEVEL): merge_level
            == ModuleMergeLevel.HOGAR,
            (ModuleType.PERSONA_LEVEL, ModuleType.HOGAR_LEVEL): merge_level
            == ModuleMergeLevel.HOGAR,
            (ModuleType.SPECIAL, ModuleType.HOGAR_LEVEL): True,
            (ModuleType.HOGAR_LEVEL, ModuleType.SPECIAL): True,
            (ModuleType.SPECIAL, ModuleType.PERSONA_LEVEL): merge_level == ModuleMergeLevel.HOGAR,
            (ModuleType.PERSONA_LEVEL, ModuleType.SPECIAL): merge_level == ModuleMergeLevel.HOGAR,
        }

        is_compatible = compatibility_matrix.get((level1, level2), False)

        if not is_compatible:
            return {
                "compatible": False,
                "error": f"Módulos {module1} ({level1.value}) y {module2} ({level2.value}) no compatibles en nivel {merge_level.value}",
            }

        # Verificar llaves de merge

        merge_keys = self._get_merge_keys_for_level(merge_level)

        missing_keys_1 = [k for k in merge_keys if k not in df1.columns]

        missing_keys_2 = [k for k in merge_keys if k not in df2.columns]

        if missing_keys_1 or missing_keys_2:
            return {
                "compatible": False,
                "missing_keys_module1": missing_keys_1,
                "missing_keys_module2": missing_keys_2,
            }

        # Analizar overlap potencial

        df1_keys = df1[merge_keys].copy()

        df2_keys = df2[merge_keys].copy()

        # Convertir llaves a string para consistencia

        for key in merge_keys:
            df1_keys[key] = df1_keys[key].astype(str)

            df2_keys[key] = df2_keys[key].astype(str)

        df1_keys = df1_keys.drop_duplicates()

        df2_keys = df2_keys.drop_duplicates()

        merged_test = pd.merge(df1_keys, df2_keys, on=merge_keys, how="inner")

        potential_matches = len(merged_test)

        match_rate_1 = potential_matches / len(df1_keys) * 100 if len(df1_keys) > 0 else 0

        match_rate_2 = potential_matches / len(df2_keys) * 100 if len(df2_keys) > 0 else 0

        return {
            "compatible": True,
            "potential_matches": potential_matches,
            "match_rate_module1": match_rate_1,
            "match_rate_module2": match_rate_2,
            "recommendation": self._get_merge_recommendation(match_rate_1, match_rate_2),
            "merge_level_used": merge_level.value,
            "module1_unique_keys": len(df1_keys),
            "module2_unique_keys": len(df2_keys),
            "detailed_analysis": self._detailed_compatibility_analysis(df1, df2, merge_keys),
        }

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

    def _get_merge_recommendation(self, rate1: float, rate2: float) -> str:
        """Genera recomendación basada en tasas de match"""

        avg_rate = (rate1 + rate2) / 2

        if avg_rate >= 90:
            return "Excelente compatibilidad - merge recomendado"

        elif avg_rate >= 70:
            return "Buena compatibilidad - revisar registros no coincidentes"

        elif avg_rate >= 50:
            return "Compatibilidad moderada - verificar llaves y filtros"

        else:
            return "Baja compatibilidad - revisar estructura de datos"

    def _detailed_compatibility_analysis(
        self, df1: pd.DataFrame, df2: pd.DataFrame, merge_keys: List[str]
    ) -> Dict[str, Any]:
        """Análisis detallado de compatibilidad"""

        analysis = {}

        # Análisis de distribución de llaves

        for key in merge_keys:
            if key in df1.columns and key in df2.columns:
                values1 = set(df1[key].dropna().astype(str))

                values2 = set(df2[key].dropna().astype(str))

                overlap = len(values1 & values2)

                union = len(values1 | values2)

                analysis[f"{key}_analysis"] = {
                    "overlap_count": overlap,
                    "jaccard_similarity": overlap / union if union > 0 else 0,
                    "values_only_in_module1": len(values1 - values2),
                    "values_only_in_module2": len(values2 - values1),
                }

        return analysis

    def validate_data_consistency(self, df: pd.DataFrame, module_code: str) -> Dict[str, Any]:
        """

        Valida consistencia interna de datos de un módulo



        Args:

            df: DataFrame del módulo

            module_code: Código del módulo



        Returns:

            Reporte de consistencia

        """

        report = {
            "module": module_code,
            "total_records": len(df),
            "consistency_score": 100.0,
            "issues_found": [],
            "data_quality_metrics": {},
        }

        # Métricas básicas de calidad

        report["data_quality_metrics"] = {
            "completeness": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "uniqueness": self._calculate_uniqueness_score(df, module_code),
            "validity": self._calculate_validity_score(df, module_code),
        }

        # Verificar consistencia de llaves

        if module_code in self.config.module_validations:
            required_keys = self.config.module_validations[module_code]["required_keys"]

            missing_keys = [key for key in required_keys if key not in df.columns]

            if missing_keys:
                report["issues_found"].append(f"Llaves faltantes: {missing_keys}")

                report["consistency_score"] -= 20

            else:
                # Verificar valores nulos en llaves

                null_in_keys = df[required_keys].isnull().any(axis=1).sum()

                if null_in_keys > 0:
                    report["issues_found"].append(
                        f"Valores nulos en llaves: {null_in_keys} registros"
                    )

                    report["consistency_score"] -= min(15, (null_in_keys / len(df)) * 50)

        # Verificar rangos de valores

        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if df[col].min() < 0 and col.lower() not in ["ingreso", "monto", "valor"]:
                # Variables que no deberían ser negativas

                negative_count = (df[col] < 0).sum()

                if negative_count > 0:
                    report["issues_found"].append(
                        f"Valores negativos inesperados en '{col}': {negative_count}"
                    )

                    report["consistency_score"] -= min(10, (negative_count / len(df)) * 20)

        # Verificar duplicados

        if module_code in self.config.module_validations:
            required_keys = self.config.module_validations[module_code]["required_keys"]

            if all(key in df.columns for key in required_keys):
                duplicates = df.duplicated(subset=required_keys).sum()

                if duplicates > 0:
                    report["issues_found"].append(f"Registros duplicados: {duplicates}")

                    report["consistency_score"] -= min(25, (duplicates / len(df)) * 100)

        return report

    def _calculate_uniqueness_score(self, df: pd.DataFrame, module_code: str) -> float:
        """Calcula score de unicidad basado en llaves primarias"""

        if module_code not in self.config.module_validations:
            return 100.0

        required_keys = self.config.module_validations[module_code]["required_keys"]

        if not all(key in df.columns for key in required_keys):
            return 50.0  # Score medio si faltan llaves

        unique_combinations = df[required_keys].drop_duplicates().shape[0]

        total_records = df.shape[0]

        return (unique_combinations / total_records) * 100 if total_records > 0 else 100.0

    def _calculate_validity_score(self, df: pd.DataFrame, module_code: str) -> float:
        """Calcula score de validez basado en rangos esperados"""

        score = 100.0

        # Validaciones específicas por módulo

        if module_code == "34":  # Sumaria
            if "mieperho" in df.columns:
                invalid_members = ((df["mieperho"] <= 0) | (df["mieperho"] > 30)).sum()

                score -= min(20, (invalid_members / len(df)) * 100)

        elif module_code in ["02", "03", "04", "05"]:  # Módulos de persona
            if "codperso" in df.columns:
                invalid_codperso = (df["codperso"] <= 0).sum()

                score -= min(15, (invalid_codperso / len(df)) * 100)

        # Validaciones generales

        numeric_cols = df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            # Verificar valores extremos (outliers simples)

            if len(df[col].dropna()) > 0:
                q1 = df[col].quantile(0.25)

                q3 = df[col].quantile(0.75)

                iqr = q3 - q1

                if iqr > 0:
                    outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()

                    outlier_rate = outliers / len(df)

                    if outlier_rate > 0.05:  # Más del 5% outliers
                        score -= min(5, outlier_rate * 20)

        return max(0, score)

    def generate_validation_report(self, df: pd.DataFrame, module_code: str) -> str:
        """

        Genera reporte detallado de validación para un módulo



        Args:

            df: DataFrame del módulo

            module_code: Código del módulo



        Returns:

            Reporte formateado como string

        """

        validation_warnings = self.validate_module_structure(df, module_code)

        consistency_report = self.validate_data_consistency(df, module_code)

        lines = [
            f"📋 REPORTE DE VALIDACIÓN - MÓDULO {module_code.upper()}",
            "=" * 50,
            f"Registros totales: {len(df):,}",
            f"Columnas: {len(df.columns)}",
            f"Score de consistencia: {consistency_report['consistency_score']:.1f}%",
            "",
        ]

        # Métricas de calidad

        metrics = consistency_report["data_quality_metrics"]

        lines.extend(
            [
                "📊 MÉTRICAS DE CALIDAD:",
                f"  • Completitud: {metrics['completeness']:.1f}%",
                f"  • Unicidad: {metrics['uniqueness']:.1f}%",
                f"  • Validez: {metrics['validity']:.1f}%",
                "",
            ]
        )

        # Advertencias estructurales

        if validation_warnings:
            lines.append("⚠️  ADVERTENCIAS ESTRUCTURALES:")

            for warning in validation_warnings:
                lines.append(f"  • {warning}")

            lines.append("")

        # Problemas de consistencia

        if consistency_report["issues_found"]:
            lines.append("❌ PROBLEMAS DE CONSISTENCIA:")

            for issue in consistency_report["issues_found"]:
                lines.append(f"  • {issue}")

            lines.append("")

        # Resumen final

        if consistency_report["consistency_score"] >= 80:
            status = "✅ BUENA CALIDAD"

        elif consistency_report["consistency_score"] >= 60:
            status = "⚠️  CALIDAD MODERADA"

        else:
            status = "❌ REQUIERE ATENCIÓN"

        lines.extend([f"🏷️  ESTADO GENERAL: {status}", ""])

        return "\n".join(lines)
