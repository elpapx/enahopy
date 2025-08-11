"""
ENAHO Merger - Merger de Módulos ENAHO
=====================================

Implementación especializada para combinar módulos ENAHO
con validaciones específicas y manejo de conflictos.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..config import ModuleMergeConfig, ModuleMergeLevel, ModuleMergeStrategy, ModuleMergeResult
from ..exceptions import ModuleMergeError, IncompatibleModulesError, MergeKeyError
from .validator import ModuleValidator


class ENAHOModuleMerger:
    """Merger especializado para combinar módulos ENAHO"""

    def __init__(self, config: ModuleMergeConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.validator = ModuleValidator(config, logger)

    def merge_modules(self,
                      left_df: pd.DataFrame,
                      right_df: pd.DataFrame,
                      left_module: str,
                      right_module: str,
                      merge_config: Optional[ModuleMergeConfig] = None) -> ModuleMergeResult:
        """
        Merge entre dos módulos ENAHO con validaciones específicas

        Args:
            left_df: DataFrame del módulo izquierdo
            right_df: DataFrame del módulo derecho
            left_module: Código del módulo izquierdo (ej: "01", "05")
            right_module: Código del módulo derecho
            merge_config: Configuración específica para este merge

        Returns:
            ModuleMergeResult con DataFrame combinado y métricas
        """

        config = merge_config or self.config
        self.logger.info(f"🔗 Iniciando merge: Módulo {left_module} + Módulo {right_module}")

        # 1. Validar DataFrames
        validation_warnings = []
        validation_warnings.extend(self.validator.validate_module_structure(left_df, left_module))
        validation_warnings.extend(self.validator.validate_module_structure(right_df, right_module))

        # 2. Verificar compatibilidad
        compatibility = self.validator.check_module_compatibility(
            left_df, right_df, left_module, right_module, config.merge_level
        )

        if not compatibility['compatible']:
            raise IncompatibleModulesError(
                compatibility.get('error', 'Módulos incompatibles'),
                module1=left_module,
                module2=right_module,
                compatibility_info=compatibility
            )

        # 3. Determinar llaves de merge
        merge_keys = self._get_merge_keys_for_level(config.merge_level)

        # 4. Preparar DataFrames para merge
        left_clean = self._prepare_for_merge(left_df, merge_keys, f"mod_{left_module}")
        right_clean = self._prepare_for_merge(right_df, merge_keys, f"mod_{right_module}")

        # 5. Ejecutar merge
        merged_df = pd.merge(
            left_clean,
            right_clean,
            on=merge_keys,
            how='outer',  # Usar outer para capturar no-matches
            suffixes=config.suffix_conflicts,
            indicator=True
        )

        # 6. Analizar resultado del merge
        merge_stats = self._analyze_merge_result(merged_df)

        # 7. Resolver conflictos si existen
        conflicts_resolved = self._resolve_conflicts(merged_df, config.merge_strategy)

        # 8. Limpiar DataFrame final
        final_df = self._clean_merged_dataframe(merged_df, merge_keys)

        # 9. Calcular score de calidad
        quality_score = self._calculate_merge_quality_score(merge_stats, compatibility)

        # 10. Crear reporte
        merge_report = {
            'modules_merged': f"{left_module} + {right_module}",
            'merge_level': config.merge_level.value,
            'merge_strategy': config.merge_strategy.value,
            'total_records': len(final_df),
            'merge_statistics': merge_stats,
            'compatibility_info': compatibility,
            'quality_score': quality_score
        }

        self.logger.info(f"✅ Merge completado: {len(final_df)} registros finales (Calidad: {quality_score:.1f}%)")

        return ModuleMergeResult(
            merged_df=final_df,
            merge_report=merge_report,
            conflicts_resolved=conflicts_resolved,
            unmatched_left=merge_stats['left_only'],
            unmatched_right=merge_stats['right_only'],
            validation_warnings=validation_warnings,
            quality_score=quality_score
        )

    def merge_multiple_modules(self,
                               modules_dict: Dict[str, pd.DataFrame],
                               base_module: str,
                               merge_config: Optional[ModuleMergeConfig] = None) -> ModuleMergeResult:
        """
        Merge múltiples módulos secuencialmente

        Args:
            modules_dict: Diccionario {codigo_modulo: dataframe}
            base_module: Código del módulo base para iniciar
            merge_config: Configuración de merge

        Returns:
            ModuleMergeResult con todos los módulos combinados
        """

        if base_module not in modules_dict:
            raise ValueError(f"Módulo base '{base_module}' no encontrado")

        # Iniciar con módulo base
        result_df = modules_dict[base_module].copy()
        all_warnings = []
        total_conflicts = 0
        merge_history = [base_module]
        quality_scores = []

        # Merge secuencial con otros módulos
        for module_code, module_df in modules_dict.items():
            if module_code == base_module:
                continue

            self.logger.info(f"🔗 Agregando módulo {module_code}")

            merge_result = self.merge_modules(
                result_df, module_df,
                '+'.join(merge_history), module_code,
                merge_config
            )

            result_df = merge_result.merged_df
            all_warnings.extend(merge_result.validation_warnings)
            total_conflicts += merge_result.conflicts_resolved
            quality_scores.append(merge_result.quality_score)
            merge_history.append(module_code)

        # Calcular calidad promedio
        avg_quality = np.mean(quality_scores) if quality_scores else 100.0

        # Reporte final
        final_report = {
            'modules_sequence': ' → '.join(merge_history),
            'total_modules': len(modules_dict),
            'final_records': len(result_df),
            'total_conflicts_resolved': total_conflicts,
            'average_quality_score': avg_quality,
            'individual_quality_scores': dict(zip(merge_history[1:], quality_scores)),
            'overall_quality_score': self._calculate_overall_quality(result_df)
        }

        return ModuleMergeResult(
            merged_df=result_df,
            merge_report=final_report,
            conflicts_resolved=total_conflicts,
            unmatched_left=0,  # No aplica en merge múltiple
            unmatched_right=0,
            validation_warnings=all_warnings,
            quality_score=avg_quality
        )

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

    def _prepare_for_merge(self, df: pd.DataFrame, merge_keys: List[str],
                           prefix: str) -> pd.DataFrame:
        """Prepara DataFrame para merge"""
        df_clean = df.copy()

        # Verificar que todas las llaves existan
        missing_keys = [key for key in merge_keys if key not in df_clean.columns]
        if missing_keys:
            raise MergeKeyError(
                f"{prefix}: llaves faltantes para merge",
                missing_keys=missing_keys,
                invalid_keys=[]
            )

        # Asegurar que las llaves sean del tipo correcto
        for key in merge_keys:
            # Convertir a string para consistencia
            df_clean[key] = df_clean[key].astype(str)

        # Eliminar registros con llaves nulas
        before_clean = len(df_clean)
        df_clean = df_clean.dropna(subset=merge_keys)
        after_clean = len(df_clean)

        if before_clean != after_clean:
            self.logger.warning(f"{prefix}: {before_clean - after_clean} registros eliminados por llaves nulas")

        return df_clean

    def _analyze_merge_result(self, merged_df: pd.DataFrame) -> Dict[str, int]:
        """Analiza estadísticas del merge"""
        merge_indicator = merged_df['_merge']

        return {
            'both': (merge_indicator == 'both').sum(),
            'left_only': (merge_indicator == 'left_only').sum(),
            'right_only': (merge_indicator == 'right_only').sum(),
            'total': len(merged_df)
        }

    def _resolve_conflicts(self, df: pd.DataFrame, strategy: ModuleMergeStrategy) -> int:
        """Resuelve conflictos entre columnas duplicadas"""
        conflicts_resolved = 0

        # Buscar columnas con sufijos
        suffixes = self.config.suffix_conflicts
        conflict_columns = []

        for col in df.columns:
            if col.endswith(suffixes[0]):
                base_name = col[:-len(suffixes[0])]
                right_col = base_name + suffixes[1]
                if right_col in df.columns:
                    conflict_columns.append((col, right_col, base_name))

        # Resolver cada conflicto
        for left_col, right_col, base_name in conflict_columns:
            try:
                if strategy == ModuleMergeStrategy.COALESCE:
                    df[base_name] = df[left_col].fillna(df[right_col])
                elif strategy == ModuleMergeStrategy.KEEP_LEFT:
                    df[base_name] = df[left_col]
                elif strategy == ModuleMergeStrategy.KEEP_RIGHT:
                    df[base_name] = df[right_col]
                elif strategy == ModuleMergeStrategy.AVERAGE:
                    if pd.api.types.is_numeric_dtype(df[left_col]):
                        df[base_name] = df[[left_col, right_col]].mean(axis=1)
                    else:
                        df[base_name] = df[left_col].fillna(df[right_col])
                elif strategy == ModuleMergeStrategy.CONCATENATE:
                    # Concatenar strings no nulos
                    left_str = df[left_col].astype(str).fillna('')
                    right_str = df[right_col].astype(str).fillna('')
                    combined = left_str + ' ' + right_str
                    df[base_name] = combined.str.strip().replace('', np.nan)
                elif strategy == ModuleMergeStrategy.ERROR:
                    # Verificar si realmente hay conflictos
                    conflicts_mask = (df[left_col].notna() & df[right_col].notna() &
                                      (df[left_col] != df[right_col]))
                    if conflicts_mask.any():
                        n_conflicts = conflicts_mask.sum()
                        raise ModuleMergeError(
                            f"Conflictos detectados en columna '{base_name}': {n_conflicts} registros",
                            modules_involved=[left_col, right_col]
                        )
                    else:
                        df[base_name] = df[left_col].fillna(df[right_col])

                # Eliminar columnas con sufijos
                df.drop([left_col, right_col], axis=1, inplace=True)
                conflicts_resolved += 1

            except Exception as e:
                self.logger.error(f"Error resolviendo conflicto en {base_name}: {str(e)}")
                # Mantener columna izquierda como fallback
                df[base_name] = df[left_col]
                df.drop([left_col, right_col], axis=1, inplace=True)

        return conflicts_resolved

    def _clean_merged_dataframe(self, df: pd.DataFrame, merge_keys: List[str]) -> pd.DataFrame:
        """Limpia DataFrame después del merge"""
        # Eliminar columna indicadora
        if '_merge' in df.columns:
            df.drop('_merge', axis=1, inplace=True)

        # Reordenar columnas: llaves primero
        other_cols = [col for col in df.columns if col not in merge_keys]
        df = df[merge_keys + other_cols]

        return df

    def _calculate_merge_quality_score(self, merge_stats: Dict[str, int],
                                       compatibility_info: Dict[str, any]) -> float:
        """Calcula score de calidad del merge"""
        total = merge_stats['total']
        matched = merge_stats['both']

        if total == 0:
            return 0.0

        # Componentes del score
        match_rate = (matched / total) * 100

        # Penalizar por registros no coincidentes
        unmatched_penalty = ((merge_stats['left_only'] + merge_stats['right_only']) / total) * 20

        # Bonificar por buena compatibilidad previa
        compatibility_bonus = 0
        if 'match_rate_module1' in compatibility_info and 'match_rate_module2' in compatibility_info:
            avg_compatibility = (compatibility_info['match_rate_module1'] +
                                 compatibility_info['match_rate_module2']) / 2
            if avg_compatibility > 90:
                compatibility_bonus = 5
            elif avg_compatibility > 70:
                compatibility_bonus = 2

        return max(0, min(100, match_rate - unmatched_penalty + compatibility_bonus))

    def _calculate_overall_quality(self, df: pd.DataFrame) -> float:
        """Calcula calidad general del DataFrame final"""
        # Combinar métricas de completitud y consistencia
        if df.empty:
            return 0.0

        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

        # Factor de penalización por duplicados
        key_cols = [col for col in ['conglome', 'vivienda', 'hogar'] if col in df.columns]
        if key_cols:
            duplicates = df.duplicated(subset=key_cols).sum()
            duplicate_penalty = (duplicates / len(df)) * 20 if len(df) > 0 else 0
        else:
            duplicate_penalty = 0

        return max(0, completeness - duplicate_penalty)

    def analyze_merge_feasibility(self, modules_dict: Dict[str, pd.DataFrame],
                                  merge_level: ModuleMergeLevel) -> Dict[str, any]:
        """
        Analiza la viabilidad de merge entre múltiples módulos

        Args:
            modules_dict: Diccionario con módulos a analizar
            merge_level: Nivel de merge propuesto

        Returns:
            Análisis de viabilidad completo
        """
        analysis = {
            'feasible': True,
            'merge_level': merge_level.value,
            'modules_analyzed': list(modules_dict.keys()),
            'potential_issues': [],
            'recommendations': [],
            'size_analysis': {},
            'key_analysis': {}
        }

        merge_keys = self._get_merge_keys_for_level(merge_level)

        # Análisis de tamaños
        for module, df in modules_dict.items():
            analysis['size_analysis'][module] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }

        total_rows = sum(info['rows'] for info in analysis['size_analysis'].values())
        max_rows = max(info['rows'] for info in analysis['size_analysis'].values())

        # Análisis de llaves
        for module, df in modules_dict.items():
            missing_keys = [key for key in merge_keys if key not in df.columns]
            if missing_keys:
                analysis['potential_issues'].append(
                    f"Módulo {module}: llaves faltantes {missing_keys}"
                )
                analysis['feasible'] = False

            if not missing_keys:
                # Analizar unicidad de llaves
                key_df = df[merge_keys].copy()
                for key in merge_keys:
                    key_df[key] = key_df[key].astype(str)

                unique_combinations = len(key_df.drop_duplicates())
                total_records = len(df)

                analysis['key_analysis'][module] = {
                    'unique_key_combinations': unique_combinations,
                    'total_records': total_records,
                    'duplication_rate': (total_records - unique_combinations) / total_records * 100
                }

        # Generar recomendaciones
        if analysis['feasible']:
            if total_rows > 1000000:  # 1M registros
                analysis['recommendations'].append(
                    "Dataset grande detectado: considere procesamiento por chunks"
                )

            if max_rows > 500000:  # 500k registros
                analysis['recommendations'].append(
                    "Módulo grande detectado: monitoree uso de memoria durante merge"
                )

            # Analizar tasas de duplicación
            high_dup_modules = [
                module for module, info in analysis['key_analysis'].items()
                if info.get('duplication_rate', 0) > 10
            ]

            if high_dup_modules:
                analysis['recommendations'].append(
                    f"Módulos con alta duplicación: {high_dup_modules}. "
                    f"Considere estrategia de agregación"
                )

        return analysis

    def create_merge_plan(self, modules_dict: Dict[str, pd.DataFrame],
                          target_module: str = "34") -> Dict[str, any]:
        """
        Crea plan de merge optimizado para múltiples módulos

        Args:
            modules_dict: Módulos a fusionar
            target_module: Módulo objetivo (base)

        Returns:
            Plan de merge detallado
        """
        plan = {
            'base_module': target_module,
            'merge_sequence': [],
            'estimated_time_seconds': 0,
            'memory_requirements_mb': 0,
            'optimizations': []
        }

        if target_module not in modules_dict:
            # Elegir módulo base automáticamente (el más grande)
            target_module = max(modules_dict.keys(),
                                key=lambda k: len(modules_dict[k]))
            plan['base_module'] = target_module
            plan['optimizations'].append(f"Módulo base seleccionado automáticamente: {target_module}")

        # Ordenar módulos por tamaño (del más pequeño al más grande)
        other_modules = [(k, len(v)) for k, v in modules_dict.items() if k != target_module]
        other_modules.sort(key=lambda x: x[1])  # Ascendente por tamaño

        plan['merge_sequence'] = [target_module] + [m[0] for m in other_modules]

        # Estimar recursos
        total_rows = sum(len(df) for df in modules_dict.values())
        plan['estimated_time_seconds'] = max(10, total_rows // 10000)  # Estimación básica
        plan['memory_requirements_mb'] = total_rows * len(modules_dict) * 0.1  # Estimación básica

        # Optimizaciones sugeridas
        if len(modules_dict) > 3:
            plan['optimizations'].append("Considere merge por pares para datasets grandes")

        if total_rows > 100000:
            plan['optimizations'].append("Active optimización de memoria")

        return plan