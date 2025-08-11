"""
ENAHO Merger - Funciones de Conveniencia Adicionales
===================================================

Funciones utilitarias y de conveniencia para casos de uso específicos
del merger geográfico y de módulos ENAHO.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

from .core import ENAHOGeoMerger
from .config import (
    GeoMergeConfiguration, ModuleMergeConfig, ModuleMergeLevel,
    ModuleMergeStrategy, TipoValidacionUbigeo, TipoManejoDuplicados
)


def quick_geographic_merge(df_principal: pd.DataFrame,
                           df_geografia: pd.DataFrame,
                           **kwargs) -> pd.DataFrame:
    """
    Merge geográfico rápido con configuración automática.

    Args:
        df_principal: DataFrame principal
        df_geografia: DataFrame geográfico
        **kwargs: Argumentos opcionales para personalizar

    Returns:
        DataFrame fusionado
    """
    # Detectar automáticamente configuración óptima
    size = len(df_principal)

    if size < 10000:
        config = GeoMergeConfiguration(
            optimizar_memoria=False,
            mostrar_estadisticas=kwargs.get('verbose', True)
        )
    else:
        config = GeoMergeConfiguration(
            optimizar_memoria=True,
            chunk_size=min(50000, size // 4),
            mostrar_estadisticas=kwargs.get('verbose', True)
        )

    merger = ENAHOGeoMerger(
        geo_config=config,
        verbose=kwargs.get('verbose', True)
    )

    result_df, _ = merger.merge_geographic_data(df_principal, df_geografia)
    return result_df


def smart_module_merge(modules_dict: Dict[str, pd.DataFrame],
                       auto_detect_strategy: bool = True,
                       **kwargs) -> pd.DataFrame:
    """
    Merge inteligente de módulos con detección automática de estrategia.

    Args:
        modules_dict: Diccionario de módulos
        auto_detect_strategy: Si detectar automáticamente la mejor estrategia
        **kwargs: Argumentos adicionales

    Returns:
        DataFrame combinado
    """
    # Detectar mejor estrategia si se solicita
    if auto_detect_strategy:
        strategy = _detect_optimal_merge_strategy(modules_dict)
    else:
        strategy = kwargs.get('strategy', 'coalesce')

    # Detectar mejor nivel de merge
    level = _detect_optimal_merge_level(modules_dict)

    config = ModuleMergeConfig(
        merge_level=ModuleMergeLevel(kwargs.get('level', level)),
        merge_strategy=ModuleMergeStrategy(strategy)
    )

    merger = ENAHOGeoMerger(
        module_config=config,
        verbose=kwargs.get('verbose', True)
    )

    result = merger.merge_multiple_modules(
        modules_dict,
        kwargs.get('base_module', '34'),
        config
    )

    return result.merged_df


def validate_and_merge_geography(df_principal: pd.DataFrame,
                                 df_geografia: pd.DataFrame,
                                 strict_validation: bool = True,
                                 **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge geográfico con validación estricta y reporte detallado.

    Args:
        df_principal: DataFrame principal
        df_geografia: DataFrame geográfico
        strict_validation: Si usar validación estricta
        **kwargs: Argumentos adicionales

    Returns:
        Tupla con DataFrame y reporte de validación
    """
    config = GeoMergeConfiguration(
        tipo_validacion_ubigeo=TipoValidacionUbigeo.STRUCTURAL if strict_validation else TipoValidacionUbigeo.BASIC,
        validar_consistencia_territorial=strict_validation,
        generar_reporte_calidad=True
    )

    merger = ENAHOGeoMerger(geo_config=config, verbose=kwargs.get('verbose', True))

    # Validación previa
    validation_result = merger.validate_geographic_data(df_geografia)

    if not validation_result.is_valid and strict_validation:
        raise ValueError(f"Validación geográfica falló:\n{validation_result.get_summary_report()}")

    # Merge
    result_df, _ = merger.merge_geographic_data(df_principal, df_geografia)

    # Reporte completo
    report = {
        'validation_result': validation_result.to_dict(),
        'merge_summary': {
            'original_records': len(df_principal),
            'final_records': len(result_df),
            'geographic_coverage': (result_df.iloc[:, -1].notna().sum() / len(result_df)) * 100
        },
        'quality_assessment': merger.create_comprehensive_report(result_df)
    }

    return result_df, report


def clean_and_merge_modules(modules_dict: Dict[str, pd.DataFrame],
                            remove_duplicates: bool = True,
                            validate_structures: bool = True,
                            **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Limpia y merge módulos con validaciones automáticas.

    Args:
        modules_dict: Diccionario de módulos
        remove_duplicates: Si remover duplicados antes del merge
        validate_structures: Si validar estructuras de módulos
        **kwargs: Argumentos adicionales

    Returns:
        Tupla con DataFrame y reporte de limpieza
    """
    cleaning_report = {
        'modules_processed': len(modules_dict),
        'cleaning_actions': [],
        'validation_warnings': []
    }

    cleaned_modules = {}

    for module_code, df in modules_dict.items():
        cleaned_df = df.copy()

        # Remover duplicados si se solicita
        if remove_duplicates:
            initial_size = len(cleaned_df)
            # Detectar llaves según el módulo
            if module_code in ['02', '03', '04', '05']:  # Módulos de persona
                keys = ['conglome', 'vivienda', 'hogar', 'codperso']
            else:  # Módulos de hogar
                keys = ['conglome', 'vivienda', 'hogar']

            available_keys = [key for key in keys if key in cleaned_df.columns]
            if available_keys:
                cleaned_df = cleaned_df.drop_duplicates(subset=available_keys)
                removed = initial_size - len(cleaned_df)

                if removed > 0:
                    cleaning_report['cleaning_actions'].append(
                        f"Módulo {module_code}: {removed} duplicados removidos"
                    )

        # Validar estructura si se solicita
        if validate_structures:
            merger = ENAHOGeoMerger(verbose=False)
            warnings = merger.module_merger.validator.validate_module_structure(cleaned_df, module_code)

            if warnings:
                cleaning_report['validation_warnings'].extend([
                    f"Módulo {module_code}: {warning}" for warning in warnings
                ])

        cleaned_modules[module_code] = cleaned_df

    # Realizar merge
    result = smart_module_merge(cleaned_modules, **kwargs)

    return result, cleaning_report


def batch_geographic_merge(dataframes: Dict[str, pd.DataFrame],
                           df_geografia: pd.DataFrame,
                           common_key: str = 'ubigeo',
                           **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Merge geográfico en lote para múltiples DataFrames.

    Args:
        dataframes: Diccionario de DataFrames a procesar
        df_geografia: DataFrame geográfico común
        common_key: Columna común para merge
        **kwargs: Argumentos adicionales

    Returns:
        Diccionario con DataFrames procesados
    """
    config = GeoMergeConfiguration(
        columna_union=common_key,
        optimizar_memoria=kwargs.get('optimize_memory', True)
    )

    merger = ENAHOGeoMerger(geo_config=config, verbose=kwargs.get('verbose', True))

    results = {}

    for name, df in dataframes.items():
        try:
            result_df, _ = merger.merge_geographic_data(df, df_geografia)
            results[name] = result_df

            if kwargs.get('verbose', True):
                print(f"✅ Procesado {name}: {len(df)} → {len(result_df)} registros")

        except Exception as e:
            if kwargs.get('verbose', True):
                print(f"❌ Error procesando {name}: {str(e)}")

            if kwargs.get('skip_errors', True):
                continue
            else:
                raise

    return results


def create_merge_comparison(df1: pd.DataFrame, df2: pd.DataFrame,
                            merge_configs: List[Dict],
                            comparison_metric: str = 'completeness') -> pd.DataFrame:
    """
    Compara diferentes configuraciones de merge.

    Args:
        df1, df2: DataFrames a fusionar
        merge_configs: Lista de configuraciones a probar
        comparison_metric: Métrica para comparar ('completeness', 'quality', 'speed')

    Returns:
        DataFrame con resultados de comparación
    """
    results = []

    for i, config_dict in enumerate(merge_configs):
        try:
            # Crear configuración
            if 'geo_config' in config_dict:
                merger = ENAHOGeoMerger(geo_config=config_dict['geo_config'], verbose=False)
                start_time = pd.Timestamp.now()
                result_df, validation = merger.merge_geographic_data(df1, df2)
                end_time = pd.Timestamp.now()

                # Calcular métricas
                completeness = (1 - result_df.isnull().sum().sum() / (result_df.shape[0] * result_df.shape[1])) * 100
                quality = validation.quality_metrics.get('completeness', 0) if validation else 0

            else:  # Module merge
                merger = ENAHOGeoMerger(module_config=config_dict.get('module_config'), verbose=False)
                start_time = pd.Timestamp.now()
                result = merger.merge_modules(df1, df2, "mod1", "mod2")
                end_time = pd.Timestamp.now()
                result_df = result.merged_df

                completeness = (1 - result_df.isnull().sum().sum() / (result_df.shape[0] * result_df.shape[1])) * 100
                quality = result.quality_score

            execution_time = (end_time - start_time).total_seconds()

            results.append({
                'config_id': f"Config_{i + 1}",
                'completeness': completeness,
                'quality_score': quality,
                'execution_time_seconds': execution_time,
                'final_records': len(result_df),
                'memory_usage_mb': result_df.memory_usage(deep=True).sum() / 1024 / 1024
            })

        except Exception as e:
            results.append({
                'config_id': f"Config_{i + 1}",
                'completeness': 0,
                'quality_score': 0,
                'execution_time_seconds': float('inf'),
                'final_records': 0,
                'memory_usage_mb': 0,
                'error': str(e)
            })

    comparison_df = pd.DataFrame(results)

    # Ordenar por métrica especificada
    if comparison_metric in comparison_df.columns:
        ascending = comparison_metric == 'execution_time_seconds'  # Menor tiempo es mejor
        comparison_df = comparison_df.sort_values(comparison_metric, ascending=ascending)

    return comparison_df


def auto_detect_merge_type(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    """
    Detecta automáticamente el tipo de merge más apropiado.

    Args:
        df1, df2: DataFrames a analizar

    Returns:
        Tipo de merge recomendado ('geographic', 'module', 'general')
    """
    # Detectar si son módulos ENAHO
    enaho_keys = ['conglome', 'vivienda', 'hogar']
    has_enaho_keys = all(key in df1.columns and key in df2.columns for key in enaho_keys)

    if has_enaho_keys:
        return 'module'

    # Detectar si hay información geográfica
    merger = ENAHOGeoMerger(verbose=False)
    geo_cols1 = merger.pattern_detector.detectar_columnas_geograficas(df1)
    geo_cols2 = merger.pattern_detector.detectar_columnas_geograficas(df2)

    if geo_cols1 and geo_cols2:
        common_geo = set(geo_cols1.keys()) & set(geo_cols2.keys())
        if common_geo:
            return 'geographic'

    return 'general'


def _detect_optimal_merge_strategy(modules_dict: Dict[str, pd.DataFrame]) -> str:
    """Detecta la estrategia óptima de merge para módulos"""

    # Analizar conflictos potenciales
    total_conflicts = 0
    total_comparisons = 0

    modules = list(modules_dict.items())

    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            name1, df1 = modules[i]
            name2, df2 = modules[j]

            # Columnas comunes (excluyendo llaves)
            keys = ['conglome', 'vivienda', 'hogar', 'codperso']
            common_cols = set(df1.columns) & set(df2.columns) - set(keys)

            for col in common_cols:
                if col in df1.columns and col in df2.columns:
                    # Verificar si hay valores diferentes para las mismas llaves
                    # (análisis simplificado)
                    total_comparisons += 1

                    # Si hay muchas columnas numéricas, usar promedio
                    if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                        total_conflicts += 0.5  # Peso menor para numéricos

    if total_comparisons == 0:
        return 'coalesce'

    conflict_rate = total_conflicts / total_comparisons

    if conflict_rate > 0.7:
        return 'average'  # Muchos conflictos numéricos
    elif conflict_rate > 0.3:
        return 'coalesce'  # Conflictos moderados
    else:
        return 'keep_left'  # Pocos conflictos


def _detect_optimal_merge_level(modules_dict: Dict[str, pd.DataFrame]) -> str:
    """Detecta el nivel óptimo de merge"""

    # Verificar si todos los módulos tienen codperso
    all_have_codperso = all(
        'codperso' in df.columns and df['codperso'].notna().any()
        for df in modules_dict.values()
    )

    if all_have_codperso:
        return 'persona'
    else:
        return 'hogar'


def suggest_merge_optimization(df_size: int,
                               num_modules: int = 1,
                               available_memory_gb: float = 4.0) -> Dict[str, Any]:
    """
    Sugiere optimizaciones para merge basado en recursos disponibles.

    Args:
        df_size: Tamaño del DataFrame principal
        num_modules: Número de módulos a fusionar
        available_memory_gb: Memoria disponible en GB

    Returns:
        Diccionario con recomendaciones
    """
    estimated_memory_gb = (df_size * num_modules * 100) / (1024 ** 3)  # Estimación rough

    recommendations = {
        'use_chunking': False,
        'chunk_size': df_size,
        'optimize_memory': False,
        'use_parallel': False,
        'estimated_memory_gb': estimated_memory_gb,
        'warnings': []
    }

    # Si la memoria estimada excede la disponible
    if estimated_memory_gb > available_memory_gb * 0.8:  # 80% de la memoria
        recommendations['use_chunking'] = True
        recommendations['chunk_size'] = max(1000, int(df_size * available_memory_gb / estimated_memory_gb))
        recommendations['optimize_memory'] = True
        recommendations['warnings'].append(
            f"Memoria estimada ({estimated_memory_gb:.1f}GB) excede disponible. Usando chunking."
        )

    # Para datasets grandes, recomendar procesamiento paralelo
    if df_size > 100000 and num_modules > 2:
        recommendations['use_parallel'] = True
        recommendations['warnings'].append(
            "Dataset grande detectado. Considere procesamiento paralelo."
        )

    # Para datasets muy pequeños, desactivar optimizaciones
    if df_size < 1000:
        recommendations['optimize_memory'] = False
        recommendations['warnings'].append(
            "Dataset pequeño. Optimizaciones de memoria no necesarias."
        )

    return recommendations


def export_merge_metadata(merger_instance: ENAHOGeoMerger,
                          output_file: str,
                          include_config: bool = True) -> None:
    """
    Exporta metadatos de configuración del merger.

    Args:
        merger_instance: Instancia del merger
        output_file: Archivo de salida
        include_config: Si incluir configuración completa
    """
    metadata = {
        'export_timestamp': pd.Timestamp.now().isoformat(),
        'merger_version': '2.0.0',
        'configurations': {}
    }

    if include_config:
        # Convertir configuraciones a diccionarios serializables
        metadata['configurations']['geographic'] = {
            'columna_union': merger_instance.geo_config.columna_union,
            'manejo_duplicados': merger_instance.geo_config.manejo_duplicados.value,
            'tipo_validacion': merger_instance.geo_config.tipo_validacion_ubigeo.value,
            'optimizar_memoria': merger_instance.geo_config.optimizar_memoria
        }

        metadata['configurations']['modules'] = {
            'merge_level': merger_instance.module_config.merge_level.value,
            'merge_strategy': merger_instance.module_config.merge_strategy.value,
            'validate_keys': merger_instance.module_config.validate_keys
        }

    # Guardar como JSON
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# =====================================================
# FUNCIONES DE MIGRACIÓN Y COMPATIBILIDAD
# =====================================================

def migrate_from_v1_config(old_config: dict) -> dict:
    """
    Migra configuración de versión 1.x a 2.x.

    Args:
        old_config: Configuración de versión anterior

    Returns:
        Configuración migrada
    """
    warnings.warn(
        "Migrando configuración de versión anterior. "
        "Revise la nueva configuración para aprovechar nuevas funcionalidades.",
        DeprecationWarning
    )

    new_config = {}

    # Mapear configuraciones geográficas
    if 'columna_ubigeo' in old_config:
        new_config['geo_config'] = GeoMergeConfiguration(
            columna_union=old_config['columna_ubigeo'],
            manejo_duplicados=TipoManejoDuplicados(old_config.get('manejo_duplicados', 'first')),
            valor_faltante=old_config.get('valor_faltante', 'DESCONOCIDO')
        )

    # Mapear configuraciones de módulos si existen
    if 'merge_level' in old_config:
        new_config['module_config'] = ModuleMergeConfig(
            merge_level=ModuleMergeLevel(old_config['merge_level']),
            merge_strategy=ModuleMergeStrategy(old_config.get('merge_strategy', 'coalesce'))
        )

    return new_config


def validate_legacy_data(df: pd.DataFrame,
                         expected_structure: str = 'enaho') -> Dict[str, Any]:
    """
    Valida datos con estructuras de versiones anteriores.

    Args:
        df: DataFrame a validar
        expected_structure: Estructura esperada ('enaho', 'geographic', 'general')

    Returns:
        Reporte de validación
    """
    report = {
        'valid': True,
        'structure_type': expected_structure,
        'issues': [],
        'recommendations': []
    }

    if expected_structure == 'enaho':
        # Validar estructura ENAHO básica
        expected_keys = ['conglome', 'vivienda', 'hogar']
        missing_keys = [key for key in expected_keys if key not in df.columns]

        if missing_keys:
            report['valid'] = False
            report['issues'].append(f"Llaves ENAHO faltantes: {missing_keys}")
            report['recommendations'].append(
                "Verificar que el archivo corresponde a microdatos ENAHO"
            )

        # Verificar formato de llaves
        for key in expected_keys:
            if key in df.columns:
                if not pd.api.types.is_numeric_dtype(df[key]) and not df[key].dtype == 'object':
                    report['issues'].append(f"Formato inesperado en columna '{key}'")

    elif expected_structure == 'geographic':
        # Validar estructura geográfica
        merger = ENAHOGeoMerger(verbose=False)
        geo_cols = merger.pattern_detector.detectar_columnas_geograficas(df)

        if not geo_cols:
            report['valid'] = False
            report['issues'].append("No se detectaron columnas geográficas")
            report['recommendations'].append(
                "Verificar que el archivo contiene información geográfica válida"
            )
        else:
            report['detected_geo_columns'] = geo_cols

    return report


def create_migration_report(old_files: List[str],
                            new_structure: bool = True) -> str:
    """
    Crea reporte de migración para archivos antiguos.

    Args:
        old_files: Lista de archivos a migrar
        new_structure: Si usar nueva estructura de módulos

    Returns:
        Reporte de migración
    """
    lines = [
        "📋 REPORTE DE MIGRACIÓN ENAHO MERGER",
        "=" * 40,
        f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Archivos a migrar: {len(old_files)}",
        f"Nueva estructura: {'Sí' if new_structure else 'No'}",
        ""
    ]

    if new_structure:
        lines.extend([
            "🔄 CAMBIOS PRINCIPALES:",
            "• merger.py dividido en módulos especializados",
            "• Nueva configuración con GeoMergeConfiguration y ModuleMergeConfig",
            "• Validadores mejorados con más opciones",
            "• Sistema de estrategias para manejo de duplicados",
            "• Merge entre módulos ENAHO integrado",
            "• Funciones de conveniencia simplificadas",
            ""
        ])

    lines.extend([
        "📁 ARCHIVOS A PROCESAR:",
    ])

    for i, file_path in enumerate(old_files, 1):
        lines.append(f"{i:2d}. {file_path}")

    lines.extend([
        "",
        "⚠️  ACCIONES REQUERIDAS:",
        "1. Actualizar imports: from enaho_analyzer.merger import ...",
        "2. Revisar configuraciones según nueva API",
        "3. Probar funcionalidad con datos de prueba",
        "4. Actualizar scripts existentes según ejemplos",
        "",
        "📖 DOCUMENTACIÓN:",
        "• Ver merger/__init__.py para API completa",
        "• Usar funciones de conveniencia para casos simples",
        "• Consultar examples/ para casos de uso comunes"
    ])

    return "\n".join(lines)


# =====================================================
# FUNCIONES DE TESTING Y DEBUGGING
# =====================================================

def create_test_data(data_type: str = 'enaho',
                     size: int = 1000,
                     include_missing: bool = True,
                     seed: int = 42) -> pd.DataFrame:
    """
    Crea datos de prueba para testing del merger.

    Args:
        data_type: Tipo de datos ('enaho', 'geographic', 'mixed')
        size: Número de registros
        include_missing: Si incluir valores faltantes
        seed: Semilla para reproducibilidad

    Returns:
        DataFrame de prueba
    """
    np.random.seed(seed)

    if data_type == 'enaho':
        # Datos típicos de ENAHO
        data = {
            'conglome': np.random.randint(100000, 999999, size),
            'vivienda': np.random.randint(1, 20, size),
            'hogar': np.random.randint(1, 5, size),
            'codperso': np.random.randint(1, 15, size),
            'mieperho': np.random.randint(1, 12, size),
            'gashog2d': np.random.normal(500, 200, size),
            'inghog2d': np.random.normal(600, 250, size)
        }

    elif data_type == 'geographic':
        # Datos geográficos del Perú
        departamentos = [f"{i:02d}" for i in range(1, 26)]

        data = {
            'ubigeo': [
                f"{np.random.choice(departamentos)}{np.random.randint(1, 99):02d}{np.random.randint(1, 99):02d}"
                for _ in range(size)
            ],
            'departamento': np.random.choice(departamentos, size),
            'nombre_departamento': np.random.choice([
                'LIMA', 'AREQUIPA', 'CUSCO', 'LA LIBERTAD', 'PIURA'
            ], size),
            'latitud': np.random.uniform(-18.5, 0.2, size),
            'longitud': np.random.uniform(-81.5, -68.5, size)
        }

    elif data_type == 'mixed':
        # Datos mixtos (ENAHO + geográfico)
        departamentos = [f"{i:02d}" for i in range(1, 26)]

        data = {
            'conglome': np.random.randint(100000, 999999, size),
            'vivienda': np.random.randint(1, 20, size),
            'hogar': np.random.randint(1, 5, size),
            'ubigeo': [
                f"{np.random.choice(departamentos)}{np.random.randint(1, 99):02d}{np.random.randint(1, 99):02d}"
                for _ in range(size)
            ],
            'valor_variable': np.random.normal(100, 30, size)
        }

    df = pd.DataFrame(data)

    # Introducir valores faltantes si se solicita
    if include_missing:
        for col in df.columns:
            if col not in ['conglome', 'vivienda', 'hogar']:  # Preservar llaves
                missing_mask = np.random.random(size) < 0.05  # 5% missing
                df.loc[missing_mask, col] = np.nan

    return df


def debug_merge_issues(df1: pd.DataFrame, df2: pd.DataFrame,
                       merge_type: str = 'auto',
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Diagnostica problemas potenciales en merge.

    Args:
        df1, df2: DataFrames a analizar
        merge_type: Tipo de merge a diagnosticar
        verbose: Si mostrar información detallada

    Returns:
        Reporte de diagnóstico
    """
    diagnosis = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'merge_type': merge_type,
        'issues_found': [],
        'recommendations': [],
        'dataframe_analysis': {}
    }

    # Análisis básico de DataFrames
    diagnosis['dataframe_analysis'] = {
        'df1': {
            'shape': df1.shape,
            'dtypes': df1.dtypes.value_counts().to_dict(),
            'memory_mb': df1.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_percentage': (df1.isnull().sum().sum() / (df1.shape[0] * df1.shape[1])) * 100
        },
        'df2': {
            'shape': df2.shape,
            'dtypes': df2.dtypes.value_counts().to_dict(),
            'memory_mb': df2.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_percentage': (df2.isnull().sum().sum() / (df2.shape[0] * df2.shape[1])) * 100
        }
    }

    # Detectar tipo de merge si es automático
    if merge_type == 'auto':
        merge_type = auto_detect_merge_type(df1, df2)
        diagnosis['merge_type'] = merge_type

    # Diagnóstico específico por tipo
    if merge_type == 'module':
        # Verificar llaves ENAHO
        enaho_keys = ['conglome', 'vivienda', 'hogar']
        missing_keys_1 = [key for key in enaho_keys if key not in df1.columns]
        missing_keys_2 = [key for key in enaho_keys if key not in df2.columns]

        if missing_keys_1:
            diagnosis['issues_found'].append(f"DF1 missing ENAHO keys: {missing_keys_1}")
        if missing_keys_2:
            diagnosis['issues_found'].append(f"DF2 missing ENAHO keys: {missing_keys_2}")

        # Verificar duplicados en llaves
        if not missing_keys_1:
            duplicates_1 = df1.duplicated(subset=enaho_keys).sum()
            if duplicates_1 > 0:
                diagnosis['issues_found'].append(f"DF1 has {duplicates_1} duplicate keys")

        if not missing_keys_2:
            duplicates_2 = df2.duplicated(subset=enaho_keys).sum()
            if duplicates_2 > 0:
                diagnosis['issues_found'].append(f"DF2 has {duplicates_2} duplicate keys")

    elif merge_type == 'geographic':
        # Verificar columnas geográficas
        merger = ENAHOGeoMerger(verbose=False)
        geo_cols_1 = merger.pattern_detector.detectar_columnas_geograficas(df1)
        geo_cols_2 = merger.pattern_detector.detectar_columnas_geograficas(df2)

        if not geo_cols_1:
            diagnosis['issues_found'].append("DF1 has no detectable geographic columns")
        if not geo_cols_2:
            diagnosis['issues_found'].append("DF2 has no detectable geographic columns")

        common_geo = set(geo_cols_1.keys()) & set(geo_cols_2.keys())
        if not common_geo:
            diagnosis['issues_found'].append("No common geographic levels between DataFrames")

    # Recomendaciones generales
    total_memory = diagnosis['dataframe_analysis']['df1']['memory_mb'] + diagnosis['dataframe_analysis']['df2'][
        'memory_mb']
    if total_memory > 1000:  # 1GB
        diagnosis['recommendations'].append("Consider memory optimization for large datasets")

    if not diagnosis['issues_found']:
        diagnosis['recommendations'].append("No major issues detected. Merge should proceed normally.")

    # Imprimir diagnóstico si es verbose
    if verbose:
        print("🔍 DIAGNÓSTICO DE MERGE")
        print("=" * 30)
        print(f"Tipo de merge: {merge_type}")
        print(f"Memoria total: {total_memory:.1f} MB")

        if diagnosis['issues_found']:
            print(f"\n❌ Problemas encontrados ({len(diagnosis['issues_found'])}):")
            for issue in diagnosis['issues_found']:
                print(f"  • {issue}")

        if diagnosis['recommendations']:
            print(f"\n💡 Recomendaciones:")
            for rec in diagnosis['recommendations']:
                print(f"  • {rec}")

    return diagnosis


def benchmark_merge_performance(df1: pd.DataFrame, df2: pd.DataFrame,
                                configurations: List[Dict],
                                iterations: int = 3) -> pd.DataFrame:
    """
    Benchmarking de rendimiento para diferentes configuraciones.

    Args:
        df1, df2: DataFrames para testing
        configurations: Lista de configuraciones a probar
        iterations: Número de iteraciones por configuración

    Returns:
        DataFrame con resultados de benchmark
    """
    results = []

    for i, config in enumerate(configurations):
        config_name = config.get('name', f'Config_{i + 1}')

        times = []
        memory_usage = []
        success_count = 0

        for iteration in range(iterations):
            try:
                start_time = pd.Timestamp.now()

                # Crear merger con configuración específica
                merger = ENAHOGeoMerger(
                    geo_config=config.get('geo_config'),
                    module_config=config.get('module_config'),
                    verbose=False
                )

                # Ejecutar merge según tipo
                if 'geo_config' in config:
                    result_df, _ = merger.merge_geographic_data(df1, df2)
                else:
                    result = merger.merge_modules(df1, df2, "test1", "test2")
                    result_df = result.merged_df

                end_time = pd.Timestamp.now()

                execution_time = (end_time - start_time).total_seconds()
                memory_mb = result_df.memory_usage(deep=True).sum() / 1024 / 1024

                times.append(execution_time)
                memory_usage.append(memory_mb)
                success_count += 1

            except Exception as e:
                print(f"Error en {config_name}, iteración {iteration + 1}: {str(e)}")
                continue

        if success_count > 0:
            results.append({
                'configuration': config_name,
                'avg_time_seconds': np.mean(times),
                'std_time_seconds': np.std(times),
                'avg_memory_mb': np.mean(memory_usage),
                'success_rate': success_count / iterations,
                'iterations': iterations
            })

    return pd.DataFrame(results).sort_values('avg_time_seconds')


# =====================================================
# EXPORTACIONES ADICIONALES
# =====================================================

__all__ = [
    # Funciones principales de conveniencia
    'quick_geographic_merge',
    'smart_module_merge',
    'validate_and_merge_geography',
    'clean_and_merge_modules',
    'batch_geographic_merge',

    # Funciones de análisis y comparación
    'create_merge_comparison',
    'auto_detect_merge_type',
    'suggest_merge_optimization',

    # Funciones de utilidad
    'export_merge_metadata',
    'migrate_from_v1_config',
    'validate_legacy_data',
    'create_migration_report',

    # Funciones de testing y debugging
    'create_test_data',
    'debug_merge_issues',
    'benchmark_merge_performance'
]