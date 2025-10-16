"""

ENAHO Merger - Funciones de Conveniencia Adicionales (VERSIÓN CORREGIDA)

=========================================================================



Funciones utilitarias y de conveniencia para casos de uso específicos

del merger geográfico y de módulos ENAHO.



Versión: 2.1.0

Correcciones aplicadas:

- Eliminación de imports circulares potenciales

- Validación de DataFrames vacíos y None

- Manejo robusto de tipos de datos

- Documentación completa con ejemplos

- Validación de configuraciones

- Manejo de edge cases

"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .config import (
    GeoMergeConfiguration,
    ModuleMergeConfig,
    ModuleMergeLevel,
    ModuleMergeStrategy,
    NivelTerritorial,
    TipoManejoDuplicados,
    TipoManejoErrores,
    TipoValidacionUbigeo,
)
from .exceptions import ConfigurationError, DataQualityError, ValidationThresholdError

# Import seguro para evitar circularidad


# =====================================================

# FUNCIONES DE VALIDACIÓN Y UTILIDAD

# =====================================================


def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0,
) -> None:
    """

    Valida que un DataFrame cumpla con requisitos básicos.



    Args:

        df: DataFrame a validar

        name: Nombre del DataFrame para mensajes de error

        required_columns: Lista de columnas requeridas

        min_rows: Número mínimo de filas requerido



    Raises:

        ValueError: Si el DataFrame no cumple los requisitos



    Example:

        >>> validate_dataframe(df, "df_principal", ["ubigeo", "valor"])

    """

    if df is None:
        raise ValueError(f"{name} no puede ser None")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} debe ser un pandas DataFrame, recibido: {type(df)}")

    if df.empty and min_rows > 0:
        raise ValueError(f"{name} no puede estar vacío (requiere al menos {min_rows} filas)")

    if len(df) < min_rows:
        raise ValueError(f"{name} debe tener al menos {min_rows} filas, tiene {len(df)}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)

        if missing_cols:
            raise ValueError(
                f"{name} no tiene las columnas requeridas: {missing_cols}. "
                f"Columnas disponibles: {list(df.columns[:10])}..."
            )


def ensure_compatible_types(
    df1: pd.DataFrame, df2: pd.DataFrame, merge_column: str, logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Asegura que las columnas de merge tengan tipos compatibles.



    Args:

        df1, df2: DataFrames a compatibilizar

        merge_column: Columna de merge

        logger: Logger opcional



    Returns:

        Tupla de DataFrames con tipos compatibles



    Example:

        >>> df1_fix, df2_fix = ensure_compatible_types(df1, df2, "ubigeo")

    """

    if merge_column not in df1.columns or merge_column not in df2.columns:
        return df1, df2

    df1_copy = df1.copy()

    df2_copy = df2.copy()

    type1 = df1_copy[merge_column].dtype

    type2 = df2_copy[merge_column].dtype

    if type1 != type2:
        if logger:
            logger.warning(
                f"Tipos incompatibles en '{merge_column}': {type1} vs {type2}. Convirtiendo..."
            )

        # Estrategia: convertir ambos a string para máxima compatibilidad

        try:
            # Manejar NaN antes de convertir

            df1_copy[merge_column] = df1_copy[merge_column].fillna("").astype(str)

            df2_copy[merge_column] = df2_copy[merge_column].fillna("").astype(str)

            # Reemplazar string vacío con NaN

            df1_copy[merge_column] = df1_copy[merge_column].replace("", np.nan)

            df2_copy[merge_column] = df2_copy[merge_column].replace("", np.nan)

        except Exception as e:
            if logger:
                logger.error(f"Error convirtiendo tipos: {str(e)}")

            raise ValueError(f"No se pudieron compatibilizar tipos para '{merge_column}': {str(e)}")

    return df1_copy, df2_copy


# =====================================================

# FUNCIONES DE MERGE RÁPIDO

# =====================================================


def quick_geographic_merge(
    df_principal: pd.DataFrame,
    df_geografia: pd.DataFrame,
    ubigeo_column: str = "ubigeo",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """

    Merge geográfico rápido con configuración automática y validaciones.



    Args:

        df_principal: DataFrame principal con datos

        df_geografia: DataFrame con información geográfica

        ubigeo_column: Nombre de la columna UBIGEO

        verbose: Si mostrar información de progreso

        **kwargs: Argumentos adicionales para personalización



    Returns:

        DataFrame fusionado con información geográfica



    Raises:

        ValueError: Si los DataFrames son inválidos



    Example:

        >>> df_result = quick_geographic_merge(

        ...     df_data,

        ...     df_geo,

        ...     ubigeo_column='ubigeo_6'

        ... )

        >>> print(f"Registros: {len(df_result)}")

    """

    # Validaciones exhaustivas

    validate_dataframe(df_principal, "df_principal", required_columns=[ubigeo_column])

    validate_dataframe(df_geografia, "df_geografia", required_columns=[ubigeo_column])

    # Compatibilizar tipos

    logger = logging.getLogger("quick_merge") if verbose else None

    df_principal, df_geografia = ensure_compatible_types(
        df_principal, df_geografia, ubigeo_column, logger
    )

    # Detectar tamaño y optimizar configuración

    size = len(df_principal)

    if size < 10000:
        config = GeoMergeConfiguration(
            columna_union=ubigeo_column,
            optimizar_memoria=False,
            chunk_size=size,
            mostrar_estadisticas=verbose,
            validar_formato_ubigeo=kwargs.get("validate", True),
            manejo_errores=TipoManejoErrores.COERCE,
        )

    elif size < 100000:
        config = GeoMergeConfiguration(
            columna_union=ubigeo_column,
            optimizar_memoria=True,
            chunk_size=min(50000, size // 2),
            mostrar_estadisticas=verbose,
            validar_formato_ubigeo=kwargs.get("validate", True),
        )

    else:
        config = GeoMergeConfiguration(
            columna_union=ubigeo_column,
            optimizar_memoria=True,
            chunk_size=50000,
            mostrar_estadisticas=verbose,
            validar_formato_ubigeo=False,  # Desactivar para datasets grandes
            usar_cache=True,
        )

    # Aplicar configuraciones personalizadas

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Realizar merge usando la función principal (importación diferida)

    from .core import ENAHOGeoMerger

    merger = ENAHOGeoMerger(geo_config=config, verbose=verbose)

    try:
        result_df, validation = merger.merge_geographic_data(
            df_principal, df_geografia, columna_union=ubigeo_column
        )

        if verbose and validation:
            print(f"✅ Merge completado: {validation.coverage_percentage:.1f}% cobertura")

        return result_df

    except Exception as e:
        if verbose:
            print(f"❌ Error en merge: {str(e)}")

        raise


def smart_module_merge(
    modules_dict: Dict[str, pd.DataFrame],
    auto_detect_strategy: bool = True,
    base_module: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """

    Merge inteligente de módulos con detección automática de estrategia.



    Args:

        modules_dict: Diccionario {código_módulo: DataFrame}

        auto_detect_strategy: Si detectar automáticamente la mejor estrategia

        base_module: Módulo base (si None, se detecta automáticamente)

        verbose: Si mostrar información de progreso

        **kwargs: Configuraciones adicionales



    Returns:

        DataFrame con todos los módulos fusionados



    Raises:

        ValueError: Si modules_dict está vacío o contiene DataFrames inválidos



    Example:

        >>> modules = {"34": df_sumaria, "01": df_vivienda}

        >>> df_merged = smart_module_merge(modules)

        >>> print(f"Forma final: {df_merged.shape}")

    """

    # Validaciones

    if not modules_dict:
        raise ValueError("modules_dict no puede estar vacío")

    # Validar cada módulo

    for module_code, df in modules_dict.items():
        validate_dataframe(
            df,
            f"Módulo {module_code}",
            required_columns=["conglome", "vivienda", "hogar"],
            min_rows=1,
        )

    # Detectar módulo base si no se especifica

    if base_module is None:
        base_module = _detect_optimal_base_module(modules_dict)

        if verbose:
            print(f"📊 Módulo base detectado: {base_module}")

    # Validar que el módulo base existe

    if base_module not in modules_dict:
        raise ValueError(
            f"Módulo base '{base_module}' no encontrado. "
            f"Módulos disponibles: {list(modules_dict.keys())}"
        )

    # Detectar estrategia y nivel óptimos

    if auto_detect_strategy:
        strategy = _detect_optimal_merge_strategy(modules_dict)

        level = _detect_optimal_merge_level(modules_dict)

        if verbose:
            print(f"🎯 Estrategia detectada: {strategy}")

            print(f"📍 Nivel detectado: {level}")

    else:
        strategy = kwargs.get("strategy", "coalesce")

        level = kwargs.get("level", "hogar")

    # Crear configuración

    config = ModuleMergeConfig(
        merge_level=ModuleMergeLevel(level),
        merge_strategy=ModuleMergeStrategy(strategy),
        min_match_rate=kwargs.get("min_match_rate", 0.8),
        continue_on_error=kwargs.get("continue_on_error", True),
        generate_report=kwargs.get("generate_report", True),
    )

    # Realizar merge (importación diferida)

    from .core import ENAHOGeoMerger

    merger = ENAHOGeoMerger(module_config=config, verbose=verbose)

    try:
        result = merger.merge_multiple_modules(
            modules_dict, base_module=base_module, merge_config=config
        )

        if verbose:
            print(f"✅ Módulos fusionados: {', '.join(result.modules_merged)}")

            print(f"📊 Registros finales: {len(result.merged_df)}")

        return result.merged_df

    except Exception as e:
        if verbose:
            print(f"❌ Error en merge de módulos: {str(e)}")

        raise


# =====================================================

# FUNCIONES DE DETECCIÓN AUTOMÁTICA

# =====================================================


def detect_merge_type(
    df1: pd.DataFrame, df2: pd.DataFrame, confidence_threshold: float = 0.7
) -> str:
    """

    Detecta el tipo de merge más apropiado entre dos DataFrames.



    Args:

        df1, df2: DataFrames a analizar

        confidence_threshold: Umbral de confianza para detección



    Returns:

        Tipo de merge recomendado ('geographic', 'module', 'general')



    Example:

        >>> merge_type = detect_merge_type(df1, df2)

        >>> print(f"Tipo recomendado: {merge_type}")

    """

    # Validar inputs

    if df1 is None or df1.empty or df2 is None or df2.empty:
        return "general"

    # Detectar si son módulos ENAHO

    enaho_keys = ["conglome", "vivienda", "hogar"]

    has_enaho_keys = all(key in df1.columns and key in df2.columns for key in enaho_keys)

    if has_enaho_keys:
        # Verificar si tienen datos válidos en las llaves

        keys_valid = all(
            not df1[key].isna().all() and not df2[key].isna().all() for key in enaho_keys
        )

        if keys_valid:
            return "module"

    # Detectar si hay información geográfica (importación diferida)

    try:
        from .geographic.patterns import GeoPatternDetector

        logger = logging.getLogger("detector")

        detector = GeoPatternDetector(logger)

        geo_cols1 = detector.detectar_columnas_geograficas(df1, confidence_threshold)

        geo_cols2 = detector.detectar_columnas_geograficas(df2, confidence_threshold)

        if geo_cols1 and geo_cols2:
            common_geo = set(geo_cols1.keys()) & set(geo_cols2.keys())

            if common_geo:
                return "geographic"

    except ImportError:
        pass

    return "general"


def _detect_optimal_base_module(modules_dict: Dict[str, pd.DataFrame]) -> str:
    """

    Detecta el módulo base óptimo para merge.



    Args:

        modules_dict: Diccionario de módulos



    Returns:

        Código del módulo base recomendado

    """

    # Prioridad de módulos base (orden preferido)

    priority_order = ["34", "01", "02", "03", "04", "05", "07", "08", "37"]

    # Buscar por prioridad

    for module_code in priority_order:
        if module_code in modules_dict:
            # Verificar que tiene datos válidos

            df = modules_dict[module_code]

            if not df.empty and len(df) > 0:
                return module_code

    # Si no hay módulos prioritarios, usar el más grande

    largest_module = max(modules_dict.items(), key=lambda x: len(x[1]) if x[1] is not None else 0)

    return largest_module[0]


def _detect_optimal_merge_strategy(modules_dict: Dict[str, pd.DataFrame]) -> str:
    """

    Detecta la estrategia óptima de merge para módulos.



    Args:

        modules_dict: Diccionario de módulos



    Returns:

        Estrategia recomendada ('coalesce', 'keep_left', 'average', etc.)

    """

    if len(modules_dict) < 2:
        return "coalesce"

    # Analizar conflictos potenciales

    total_conflicts = 0

    total_comparisons = 0

    numeric_conflicts = 0

    modules = list(modules_dict.items())

    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            name1, df1 = modules[i]

            name2, df2 = modules[j]

            if df1 is None or df2 is None or df1.empty or df2.empty:
                continue

            # Columnas comunes (excluyendo llaves)

            keys = {"conglome", "vivienda", "hogar", "codperso"}

            common_cols = (set(df1.columns) & set(df2.columns)) - keys

            for col in common_cols:
                total_comparisons += 1

                # Verificar tipo de dato

                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(
                    df2[col]
                ):
                    numeric_conflicts += 1

                # Estimar conflictos basado en valores únicos

                try:
                    unique1 = df1[col].nunique()

                    unique2 = df2[col].nunique()

                    if unique1 > 1 or unique2 > 1:
                        total_conflicts += 0.5

                except:
                    pass

    if total_comparisons == 0:
        return "coalesce"

    conflict_rate = total_conflicts / total_comparisons

    numeric_rate = numeric_conflicts / total_comparisons if total_comparisons > 0 else 0

    # Decidir estrategia basada en análisis

    if numeric_rate > 0.5:
        return "average"  # Muchas columnas numéricas

    elif conflict_rate > 0.7:
        return "keep_left"  # Muchos conflictos, priorizar primer módulo

    elif conflict_rate > 0.3:
        return "coalesce"  # Conflictos moderados

    else:
        return "coalesce"  # Pocos conflictos, usar primer no-nulo


def _detect_optimal_merge_level(modules_dict: Dict[str, pd.DataFrame]) -> str:
    """

    Detecta el nivel óptimo de merge (hogar vs persona).



    Args:

        modules_dict: Diccionario de módulos



    Returns:

        Nivel recomendado ('hogar' o 'persona')

    """

    # Verificar si todos los módulos tienen codperso con datos válidos

    all_have_codperso = True

    for module_code, df in modules_dict.items():
        if df is None or df.empty:
            all_have_codperso = False

            break

        if "codperso" not in df.columns:
            all_have_codperso = False

            break

        # Verificar que codperso tiene datos válidos

        if df["codperso"].isna().all():
            all_have_codperso = False

            break

        # Verificar que hay variación en codperso (no todos son iguales)

        if df["codperso"].nunique() < 2:
            all_have_codperso = False

            break

    return "persona" if all_have_codperso else "hogar"


# =====================================================

# FUNCIONES DE OPTIMIZACIÓN Y SUGERENCIAS

# =====================================================


def suggest_merge_optimization(
    df_size: int,
    num_modules: int = 1,
    available_memory_gb: float = 4.0,
    target_performance: str = "balanced",
) -> Dict[str, Any]:
    """

    Sugiere optimizaciones para merge basado en recursos disponibles.



    Args:

        df_size: Número de registros en el DataFrame principal

        num_modules: Número de módulos a fusionar

        available_memory_gb: Memoria RAM disponible en GB

        target_performance: Objetivo ('speed', 'memory', 'balanced')



    Returns:

        Diccionario con sugerencias de optimización



    Example:

        >>> suggestions = suggest_merge_optimization(

        ...     df_size=1000000,

        ...     num_modules=5,

        ...     available_memory_gb=8.0

        ... )

        >>> print(suggestions['recommended_config'])

    """

    if df_size <= 0:
        raise ValueError("df_size debe ser positivo")

    # Estimar uso de memoria (aproximado)

    estimated_memory_mb = (df_size * num_modules * 200) / (1024 * 1024)  # ~200 bytes por celda

    memory_ratio = estimated_memory_mb / (available_memory_gb * 1024)

    suggestions = {
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "memory_usage_ratio": round(memory_ratio, 2),
        "warnings": [],
        "recommended_config": {},
        "performance_tips": [],
    }

    # Advertencias de memoria

    if memory_ratio > 0.8:
        suggestions["warnings"].append(
            f"⚠️ Alto uso de memoria esperado ({memory_ratio:.0%}). "
            "Considere procesar por chunks."
        )

    # Configuración recomendada según objetivo

    if target_performance == "speed":
        suggestions["recommended_config"] = {
            "chunk_size": min(100000, df_size),
            "optimizar_memoria": False,
            "usar_cache": True,
            "validar_formato_ubigeo": False,
            "parallel_processing": True if df_size > 500000 else False,
        }

        suggestions["performance_tips"].append("Desactivar validaciones no críticas")

        suggestions["performance_tips"].append("Usar cache para operaciones repetidas")

    elif target_performance == "memory":
        chunk_size = min(10000, df_size // 10)

        suggestions["recommended_config"] = {
            "chunk_size": chunk_size,
            "optimizar_memoria": True,
            "usar_cache": False,
            "drop_intermediate": True,
            "inplace_operations": True,
        }

        suggestions["performance_tips"].append(f"Procesar en chunks de {chunk_size:,} registros")

        suggestions["performance_tips"].append("Liberar DataFrames intermedios")

    else:  # balanced
        chunk_size = min(50000, df_size // 4)

        suggestions["recommended_config"] = {
            "chunk_size": chunk_size,
            "optimizar_memoria": memory_ratio > 0.5,
            "usar_cache": memory_ratio < 0.7,
            "validar_formato_ubigeo": df_size < 100000,
            "parallel_processing": False,
        }

        suggestions["performance_tips"].append("Configuración balanceada para estabilidad")

    # Sugerencias adicionales basadas en tamaño

    if df_size > 1000000:
        suggestions["performance_tips"].append(
            "Considere guardar resultados intermedios en archivos"
        )

    if num_modules > 5:
        suggestions["performance_tips"].append("Fusione módulos en grupos pequeños primero")

    return suggestions


# =====================================================

# FUNCIONES DE REPORTE Y ANÁLISIS

# =====================================================


def analyze_merge_quality(
    df_result: pd.DataFrame,
    df_original: pd.DataFrame,
    merge_columns: List[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    """

    Analiza la calidad del resultado de un merge.



    Args:

        df_result: DataFrame resultado del merge

        df_original: DataFrame original antes del merge

        merge_columns: Columnas usadas para el merge

        verbose: Si imprimir resumen



    Returns:

        Diccionario con métricas de calidad



    Example:

        >>> quality = analyze_merge_quality(

        ...     df_merged,

        ...     df_original,

        ...     ['ubigeo']

        ... )

        >>> print(f"Score de calidad: {quality['quality_score']}")

    """

    # Validaciones

    validate_dataframe(df_result, "df_result")

    validate_dataframe(df_original, "df_original")

    analysis = {
        "original_shape": df_original.shape,
        "result_shape": df_result.shape,
        "rows_change": len(df_result) - len(df_original),
        "columns_added": df_result.shape[1] - df_original.shape[1],
        "merge_columns": merge_columns,
        "quality_metrics": {},
        "warnings": [],
        "quality_score": 100.0,
    }

    # Verificar cambio en número de filas

    if analysis["rows_change"] > 0:
        pct_increase = (analysis["rows_change"] / len(df_original)) * 100

        analysis["warnings"].append(
            f"Aumento de {analysis['rows_change']} filas ({pct_increase:.1f}%). "
            "Posibles duplicados en el merge."
        )

        analysis["quality_score"] -= min(20, pct_increase)

    elif analysis["rows_change"] < 0:
        analysis["warnings"].append(
            f"Pérdida de {abs(analysis['rows_change'])} filas. " "Revisar tipo de join."
        )

        analysis["quality_score"] -= 30

    # Analizar completitud de nuevas columnas

    new_columns = set(df_result.columns) - set(df_original.columns)

    if new_columns:
        completeness_scores = []

        for col in new_columns:
            completeness = df_result[col].notna().mean()

            completeness_scores.append(completeness)

            if completeness < 0.5:
                analysis["warnings"].append(
                    f"Columna '{col}' tiene {(1 - completeness) * 100:.1f}% valores faltantes"
                )

        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0

        analysis["quality_metrics"]["new_columns_completeness"] = round(avg_completeness, 3)

        analysis["quality_score"] -= (1 - avg_completeness) * 20

    # Verificar integridad de merge keys

    for col in merge_columns:
        if col in df_result.columns:
            null_ratio = df_result[col].isna().mean()

            if null_ratio > 0:
                analysis["warnings"].append(
                    f"Columna de merge '{col}' tiene {null_ratio * 100:.1f}% NaN"
                )

                analysis["quality_score"] -= null_ratio * 10

    # Score final

    analysis["quality_score"] = max(0, min(100, analysis["quality_score"]))

    analysis["quality_grade"] = _get_quality_grade(analysis["quality_score"])

    if verbose:
        print(f"\n📊 ANÁLISIS DE CALIDAD DEL MERGE")

        print(f"{'=' * 40}")

        print(f"Shape original: {analysis['original_shape']}")

        print(f"Shape resultado: {analysis['result_shape']}")

        print(f"Columnas agregadas: {analysis['columns_added']}")

        print(
            f"Score de calidad: {analysis['quality_score']:.1f}/100 ({analysis['quality_grade']})"
        )

        if analysis["warnings"]:
            print(f"\n⚠️ Advertencias:")

            for warning in analysis["warnings"][:5]:
                print(f"  • {warning}")

    return analysis


def _get_quality_grade(score: float) -> str:
    """Convierte score numérico en calificación textual."""

    if score >= 90:
        return "Excelente"

    elif score >= 75:
        return "Bueno"

    elif score >= 60:
        return "Aceptable"

    elif score >= 40:
        return "Deficiente"

    else:
        return "Crítico"


# =====================================================

# FUNCIONES DE MIGRACIÓN Y COMPATIBILIDAD

# =====================================================


def migrate_legacy_config(old_config: Dict[str, Any], version: str = "1.0") -> Dict[str, Any]:
    """

    Migra configuración de versiones anteriores al formato actual.



    Args:

        old_config: Configuración en formato antiguo

        version: Versión de la configuración antigua



    Returns:

        Configuración migrada al formato actual



    Example:

        >>> old_cfg = {'columna_ubigeo': 'ubigeo', 'manejo_duplicados': 'first'}

        >>> new_cfg = migrate_legacy_config(old_cfg, "1.0")

    """

    warnings.warn(
        f"Migrando configuración desde versión {version}. "
        "Revise la nueva configuración para aprovechar nuevas funcionalidades.",
        DeprecationWarning,
        stacklevel=2,
    )

    new_config = {}

    # Mapear configuraciones geográficas

    if "columna_ubigeo" in old_config or "columna_union" in old_config:
        geo_params = {}

        # Mapeo de nombres antiguos a nuevos

        field_mapping = {
            "columna_ubigeo": "columna_union",
            "manejo_duplicados": "manejo_duplicados",
            "valor_faltante": "valor_faltante",
            "reporte_duplicados": "reporte_duplicados",
            "validar_ubigeo": "validar_formato_ubigeo",
        }

        for old_field, new_field in field_mapping.items():
            if old_field in old_config:
                value = old_config[old_field]

                # Convertir valores si es necesario

                if new_field == "manejo_duplicados":
                    try:
                        value = TipoManejoDuplicados(value)

                    except:
                        value = TipoManejoDuplicados.FIRST

                geo_params[new_field] = value

        new_config["geo_config"] = GeoMergeConfiguration(**geo_params)

    # Mapear configuraciones de módulos si existen

    if "merge_level" in old_config or "merge_strategy" in old_config:
        module_params = {}

        if "merge_level" in old_config:
            try:
                module_params["merge_level"] = ModuleMergeLevel(old_config["merge_level"])

            except:
                module_params["merge_level"] = ModuleMergeLevel.HOGAR

        if "merge_strategy" in old_config:
            try:
                module_params["merge_strategy"] = ModuleMergeStrategy(old_config["merge_strategy"])

            except:
                module_params["merge_strategy"] = ModuleMergeStrategy.COALESCE

        new_config["module_config"] = ModuleMergeConfig(**module_params)

    # Mapear otras configuraciones

    if "cache_dir" in old_config:
        new_config["cache_dir"] = old_config["cache_dir"]

    if "verbose" in old_config:
        new_config["verbose"] = old_config["verbose"]

    return new_config


def validate_legacy_data(
    df: pd.DataFrame, expected_structure: str = "enaho", fix_issues: bool = False
) -> Dict[str, Any]:
    """

    Valida y opcionalmente corrige datos con estructuras de versiones anteriores.



    Args:

        df: DataFrame a validar

        expected_structure: Estructura esperada ('enaho', 'geographic', 'custom')

        fix_issues: Si intentar corregir problemas encontrados



    Returns:

        Diccionario con resultado de validación y datos corregidos si aplica



    Example:

        >>> validation = validate_legacy_data(df_old, 'enaho', fix_issues=True)

        >>> if validation['is_valid']:

        ...     df_fixed = validation['fixed_data']

    """

    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "fixes_applied": [],
        "original_shape": df.shape if df is not None else (0, 0),
        "fixed_data": None,
    }

    # Validación básica

    if df is None or df.empty:
        validation["is_valid"] = False

        validation["issues"].append("DataFrame vacío o None")

        return validation

    df_work = df.copy() if fix_issues else df

    if expected_structure == "enaho":
        # Validar estructura ENAHO

        required_columns = ["conglome", "vivienda", "hogar"]

        missing_cols = set(required_columns) - set(df.columns)

        if missing_cols:
            validation["issues"].append(f"Columnas ENAHO faltantes: {missing_cols}")

            if fix_issues:
                # Intentar detectar columnas con nombres similares

                for col in missing_cols:
                    alternatives = [
                        c
                        for c in df.columns
                        if col.lower() in c.lower() or c.lower() in col.lower()
                    ]

                    if alternatives:
                        df_work[col] = df_work[alternatives[0]]

                        validation["fixes_applied"].append(
                            f"Columna '{col}' creada desde '{alternatives[0]}'"
                        )

                    else:
                        validation["is_valid"] = False

        # Validar tipos de datos

        for col in required_columns:
            if col in df_work.columns:
                if not pd.api.types.is_string_dtype(df_work[col]):
                    validation["warnings"].append(
                        f"Columna '{col}' no es string, puede causar problemas"
                    )

                    if fix_issues:
                        try:
                            df_work[col] = df_work[col].astype(str)

                            validation["fixes_applied"].append(
                                f"Columna '{col}' convertida a string"
                            )

                        except:
                            validation["issues"].append(f"No se pudo convertir '{col}' a string")

    elif expected_structure == "geographic":
        # Validar estructura geográfica

        ubigeo_columns = ["ubigeo", "ccpp", "ccdd", "ubigeo_distrito"]

        found_ubigeo = False

        for col in ubigeo_columns:
            if col in df.columns:
                found_ubigeo = True

                # Validar formato UBIGEO

                if col == "ubigeo" or col == "ubigeo_distrito":
                    sample = df_work[col].dropna().head(10)

                    # Verificar longitud (6 dígitos típicamente)

                    invalid_length = sample.astype(str).str.len() != 6

                    if invalid_length.any():
                        validation["warnings"].append(
                            f"Columna '{col}' tiene valores con longitud incorrecta"
                        )

                        if fix_issues:
                            # Intentar rellenar con ceros

                            df_work[col] = df_work[col].astype(str).str.zfill(6)

                            validation["fixes_applied"].append(
                                f"Columna '{col}' ajustada a 6 dígitos"
                            )

                break

        if not found_ubigeo:
            validation["is_valid"] = False

            validation["issues"].append("No se encontró columna UBIGEO válida")

    # Verificar duplicados en índice

    if df_work.index.duplicated().any():
        validation["warnings"].append("Índice con valores duplicados")

        if fix_issues:
            df_work = df_work.reset_index(drop=True)

            validation["fixes_applied"].append("Índice reiniciado")

    # Verificar columnas completamente nulas

    null_columns = df_work.columns[df_work.isna().all()].tolist()

    if null_columns:
        validation["warnings"].append(f"Columnas completamente nulas: {null_columns[:5]}")

        if fix_issues:
            df_work = df_work.drop(columns=null_columns)

            validation["fixes_applied"].append(f"Eliminadas {len(null_columns)} columnas nulas")

    # Resultado final

    if fix_issues:
        validation["fixed_data"] = df_work

        validation["final_shape"] = df_work.shape

    return validation


# =====================================================

# FUNCIONES DE TESTING Y DEBUGGING

# =====================================================


def create_test_data(
    data_type: str = "enaho",
    size: int = 1000,
    include_missing: bool = True,
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """

    Crea datos de prueba para testing del merger.



    Args:

        data_type: Tipo de datos ('enaho', 'geographic', 'module')

        size: Número de registros

        include_missing: Si incluir valores faltantes

        seed: Semilla para reproducibilidad

        **kwargs: Parámetros adicionales específicos del tipo



    Returns:

        DataFrame de prueba



    Example:

        >>> df_test = create_test_data('enaho', size=100)

        >>> print(f"Test data shape: {df_test.shape}")

    """

    np.random.seed(seed)

    if data_type == "enaho":
        # Datos tipo ENAHO

        data = {
            "conglome": np.random.randint(100000, 999999, size).astype(str),
            "vivienda": np.random.randint(1, 99, size).astype(str).str.zfill(2),
            "hogar": np.random.randint(1, 9, size).astype(str),
            "codperso": np.random.randint(1, 20, size).astype(str).str.zfill(2),
            "factor07": np.random.uniform(50, 500, size),
            "p203": np.random.choice([1, 2], size),  # Sexo
            "p208a": np.random.randint(0, 100, size),  # Edad
        }

        # Agregar módulo específico si se solicita

        module = kwargs.get("module", "34")

        if module == "34":
            data["gashog2d"] = np.random.uniform(100, 10000, size)

            data["inghog2d"] = np.random.uniform(100, 15000, size)

        elif module == "01":
            data["nbi1"] = np.random.choice([0, 1], size)

            data["nbi2"] = np.random.choice([0, 1], size)

    elif data_type == "geographic":
        # Datos geográficos

        # Departamentos reales del Perú

        departamentos = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
        ]

        deps = np.random.choice(departamentos, size)

        provs = np.random.randint(1, 20, size)

        dists = np.random.randint(1, 30, size)

        data = {
            "ubigeo": [f"{d}{p:02d}{dist:02d}" for d, p, dist in zip(deps, provs, dists)],
            "departamento": deps,
            "provincia": [f"Provincia_{p}" for p in provs],
            "distrito": [f"Distrito_{d}" for d in dists],
            "area": np.random.choice(["urbano", "rural"], size),
            "latitud": np.random.uniform(-18, 0, size),
            "longitud": np.random.uniform(-81, -68, size),
        }

    elif data_type == "module":
        # Datos de módulo específico

        module_code = kwargs.get("module_code", "05")

        data = {
            "conglome": np.random.randint(100000, 999999, size).astype(str),
            "vivienda": np.random.randint(1, 99, size).astype(str).str.zfill(2),
            "hogar": np.random.randint(1, 9, size).astype(str),
            "codperso": np.random.randint(1, 20, size).astype(str).str.zfill(2),
        }

        # Agregar columnas específicas del módulo

        if module_code == "05":  # Empleo
            data["p501"] = np.random.choice([1, 2, 3, 4, 5, 6], size)

            data["p502"] = np.random.choice([1, 2], size)

            data["i524a1"] = np.random.uniform(0, 5000, size)

        elif module_code == "03":  # Educación
            data["p301a"] = np.random.choice([1, 2, 3, 4, 5, 6], size)

            data["p306"] = np.random.choice([1, 2], size)

    else:
        raise ValueError(f"Tipo de datos '{data_type}' no reconocido")

    df = pd.DataFrame(data)

    # Agregar valores faltantes si se solicita

    if include_missing:
        missing_ratio = kwargs.get("missing_ratio", 0.1)

        for col in df.columns:
            if col not in ["conglome", "vivienda", "hogar"]:  # No en llaves principales
                mask = np.random.random(size) < missing_ratio

                df.loc[mask, col] = np.nan

    # Agregar duplicados si se solicita

    if kwargs.get("include_duplicates", False):
        dup_ratio = kwargs.get("duplicate_ratio", 0.05)

        n_duplicates = int(size * dup_ratio)

        if n_duplicates > 0:
            dup_indices = np.random.choice(df.index, n_duplicates)

            df_duplicates = df.loc[dup_indices].copy()

            df = pd.concat([df, df_duplicates], ignore_index=True)

    return df


def debug_merge_issues(
    df1: pd.DataFrame, df2: pd.DataFrame, merge_column: str, sample_size: int = 10
) -> Dict[str, Any]:
    """

    Diagnostica problemas potenciales en un merge.



    Args:

        df1, df2: DataFrames a fusionar

        merge_column: Columna de merge

        sample_size: Tamaño de muestra para análisis



    Returns:

        Diccionario con diagnóstico detallado



    Example:

        >>> issues = debug_merge_issues(df1, df2, 'ubigeo')

        >>> for issue in issues['problems']:

        ...     print(f"⚠️ {issue}")

    """

    diagnosis = {"problems": [], "recommendations": [], "merge_preview": None, "statistics": {}}

    # Verificar existencia de columnas

    if merge_column not in df1.columns:
        diagnosis["problems"].append(f"Columna '{merge_column}' no existe en df1")

        return diagnosis

    if merge_column not in df2.columns:
        diagnosis["problems"].append(f"Columna '{merge_column}' no existe en df2")

        return diagnosis

    # Analizar tipos de datos

    type1 = df1[merge_column].dtype

    type2 = df2[merge_column].dtype

    if type1 != type2:
        diagnosis["problems"].append(f"Tipos incompatibles: {type1} vs {type2}")

        diagnosis["recommendations"].append(
            f"Convertir ambas columnas al mismo tipo con ensure_compatible_types()"
        )

    # Analizar valores únicos y overlap

    unique1 = set(df1[merge_column].dropna().unique())

    unique2 = set(df2[merge_column].dropna().unique())

    common = unique1 & unique2

    only_df1 = unique1 - unique2

    only_df2 = unique2 - unique1

    diagnosis["statistics"] = {
        "df1_unique_count": len(unique1),
        "df2_unique_count": len(unique2),
        "common_values": len(common),
        "only_in_df1": len(only_df1),
        "only_in_df2": len(only_df2),
        "overlap_percentage": (len(common) / len(unique1) * 100) if unique1 else 0,
    }

    # Detectar problemas de overlap

    if len(common) == 0:
        diagnosis["problems"].append("No hay valores comunes entre las columnas de merge")

        diagnosis["recommendations"].append(
            "Verificar que los valores sean comparables (ej: formato, padding)"
        )

    elif diagnosis["statistics"]["overlap_percentage"] < 50:
        diagnosis["problems"].append(
            f"Bajo overlap: solo {diagnosis['statistics']['overlap_percentage']:.1f}% "
            f"de valores coinciden"
        )

    # Analizar NaN

    nan1 = df1[merge_column].isna().sum()

    nan2 = df2[merge_column].isna().sum()

    if nan1 > 0:
        diagnosis["problems"].append(
            f"df1 tiene {nan1} ({nan1 / len(df1) * 100:.1f}%) valores NaN en '{merge_column}'"
        )

    if nan2 > 0:
        diagnosis["problems"].append(
            f"df2 tiene {nan2} ({nan2 / len(df2) * 100:.1f}%) valores NaN en '{merge_column}'"
        )

    # Detectar duplicados

    dup1 = df1[merge_column].duplicated().sum()

    dup2 = df2[merge_column].duplicated().sum()

    if dup1 > 0:
        diagnosis["problems"].append(f"df1 tiene {dup1} valores duplicados en '{merge_column}'")

        diagnosis["recommendations"].append("Considerar estrategia de manejo de duplicados")

    if dup2 > 0:
        diagnosis["problems"].append(f"df2 tiene {dup2} valores duplicados en '{merge_column}'")

    # Preview del merge

    if len(common) > 0 and sample_size > 0:
        sample_values = list(common)[:sample_size]

        sample1 = df1[df1[merge_column].isin(sample_values)].head(sample_size)

        sample2 = df2[df2[merge_column].isin(sample_values)].head(sample_size)

        try:
            preview = pd.merge(sample1, sample2, on=merge_column, how="inner")

            diagnosis["merge_preview"] = {
                "successful": True,
                "sample_shape": preview.shape,
                "sample_columns": list(preview.columns),
            }

        except Exception as e:
            diagnosis["problems"].append(f"Error en merge de prueba: {str(e)}")

            diagnosis["merge_preview"] = {"successful": False, "error": str(e)}

    # Generar recomendaciones finales

    if not diagnosis["problems"]:
        diagnosis["recommendations"].append(
            "✅ No se detectaron problemas. Merge debería funcionar correctamente."
        )

    else:
        diagnosis["recommendations"].append(
            f"Se detectaron {len(diagnosis['problems'])} problemas potenciales"
        )

    return diagnosis


# =====================================================

# FUNCIONES DE EXPORTACIÓN Y REPORTE

# =====================================================


def export_merge_report(
    merge_result: Union[pd.DataFrame, Dict[str, Any]],
    output_path: str,
    format: str = "excel",
    include_diagnostics: bool = True,
) -> None:
    """

    Exporta reporte detallado del merge.



    Args:

        merge_result: DataFrame resultado o diccionario de reporte

        output_path: Ruta de salida

        format: Formato de exportación ('excel', 'html', 'json')

        include_diagnostics: Si incluir diagnósticos detallados



    Example:

        >>> export_merge_report(

        ...     merge_result,

        ...     'merge_report.xlsx',

        ...     format='excel'

        ... )

    """

    output_path = Path(output_path)

    if format == "excel":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Si es DataFrame, exportar directamente

            if isinstance(merge_result, pd.DataFrame):
                merge_result.to_excel(writer, sheet_name="Datos_Fusionados", index=False)

                if include_diagnostics:
                    # Crear hoja de resumen

                    summary = pd.DataFrame(
                        {
                            "Métrica": ["Total Registros", "Total Columnas", "Memoria (MB)"],
                            "Valor": [
                                len(merge_result),
                                merge_result.shape[1],
                                merge_result.memory_usage(deep=True).sum() / 1024**2,
                            ],
                        }
                    )

                    summary.to_excel(writer, sheet_name="Resumen", index=False)

            # Si es diccionario de reporte

            elif isinstance(merge_result, dict):
                for key, value in merge_result.items():
                    if isinstance(value, pd.DataFrame):
                        sheet_name = key[:31]  # Excel limite de 31 caracteres

                        value.to_excel(writer, sheet_name=sheet_name, index=False)

                    elif isinstance(value, dict):
                        df_temp = pd.DataFrame.from_dict(value, orient="index", columns=["Valor"])

                        sheet_name = key[:31]

                        df_temp.to_excel(writer, sheet_name=sheet_name)

    elif format == "html":
        # Implementar exportación HTML

        html_content = "<html><head><title>Merge Report</title></head><body>"

        if isinstance(merge_result, pd.DataFrame):
            html_content += merge_result.to_html()

        elif isinstance(merge_result, dict):
            import json

            html_content += f"<pre>{json.dumps(merge_result, indent=2, default=str)}</pre>"

        html_content += "</body></html>"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    elif format == "json":
        import json

        if isinstance(merge_result, pd.DataFrame):
            data = merge_result.to_dict(orient="records")

        else:
            data = merge_result

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    else:
        raise ValueError(f"Formato '{format}' no soportado. Use 'excel', 'html' o 'json'")

    print(f"✅ Reporte exportado a: {output_path}")
