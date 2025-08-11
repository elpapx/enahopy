"""
ENAHO Merger - Clases Principales
================================

Implementación de las clases principales ENAHOGeoMerger con
funcionalidades completas de fusión geográfica y merge de módulos.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

# Importaciones internas
from .config import (
    GeoMergeConfiguration, ModuleMergeConfig, GeoValidationResult,
    ModuleMergeResult, TipoManejoErrores, TipoManejoDuplicados
)
from .exceptions import GeoMergeError, ModuleMergeError
from .geographic.validators import UbigeoValidator, TerritorialValidator, GeoDataQualityValidator
from .geographic.patterns import GeoPatternDetector
from .geographic.strategies import DuplicateStrategyFactory
from .modules.merger import ENAHOModuleMerger
from .modules.validator import ModuleValidator

# Importaciones opcionales del loader principal
try:
    from ..loader import ENAHOConfig, setup_logging, log_performance, CacheManager
except ImportError:
    # Fallback para uso independiente
    from dataclasses import dataclass


    @dataclass(frozen=True)
    class ENAHOConfig:
        cache_dir: str = ".enaho_cache"


    def setup_logging(verbose: bool = True, structured: bool = False, log_file: Optional[str] = None):
        logger = logging.getLogger('enaho_geo_merger')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return logger


    def log_performance(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


    CacheManager = None


class ENAHOGeoMerger:
    """
    Fusionador geográfico avanzado para datos INEI integrado con enaho-analyzer.
    EXTENDIDO con capacidades de merge entre módulos ENAHO.

    Proporciona funcionalidades completas para fusionar datos con información
    geográfica, validación de UBIGEO, detección automática de patrones,
    estrategias flexibles de manejo de duplicados, y merge entre módulos ENAHO.
    """

    def __init__(self,
                 config: Optional[ENAHOConfig] = None,
                 geo_config: Optional[GeoMergeConfiguration] = None,
                 module_config: Optional[ModuleMergeConfig] = None,
                 verbose: bool = True,
                 structured_logging: bool = False,
                 log_file: Optional[str] = None):
        """
        Inicializa el merger geográfico extendido.

        Args:
            config: Configuración ENAHO general
            geo_config: Configuración específica de fusión geográfica
            module_config: Configuración específica de merge entre módulos
            verbose: Si mostrar logs detallados
            structured_logging: Si usar logging estructurado
            log_file: Archivo para logs
        """
        self.config = config or ENAHOConfig()
        self.geo_config = geo_config or GeoMergeConfiguration()
        self.module_config = module_config or ModuleMergeConfig()
        self.logger = setup_logging(verbose, structured_logging, log_file)

        # Inicializar validadores geográficos
        self.ubigeo_validator = UbigeoValidator(self.logger)
        self.territorial_validator = TerritorialValidator(self.logger)
        self.pattern_detector = GeoPatternDetector(self.logger)
        self.quality_validator = GeoDataQualityValidator(self.logger)

        # Inicializar merger de módulos
        self.module_merger = ENAHOModuleMerger(self.module_config, self.logger)

        # Cache para datos geográficos
        self._geo_cache = {}
        if CacheManager and hasattr(self.config, 'cache_dir'):
            try:
                self.cache_manager = CacheManager(self.config.cache_dir)
            except Exception:
                self.cache_manager = None
        else:
            self.cache_manager = None

        self.logger.info("🗺️  ENAHOGeoMerger inicializado (Versión Refactorizada)")
        self.logger.info(f"   Nivel territorial objetivo: {self.geo_config.nivel_territorial_objetivo.value}")
        self.logger.info(f"   Validación UBIGEO: {self.geo_config.tipo_validacion_ubigeo.value}")
        self.logger.info(f"   Merge de módulos habilitado: ✅")

    def _optimize_dataframe_memory(self, df: pd.DataFrame, name: str = "") -> pd.DataFrame:
        """Optimiza el uso de memoria del DataFrame"""
        if not self.geo_config.optimizar_memoria:
            return df

        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Optimizar categóricas para columnas geográficas
        geo_columns = ['departamento', 'provincia', 'distrito', 'centro_poblado']
        for col in df.columns:
            if any(geo_term in col.lower() for geo_term in geo_columns):
                if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')

        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        saved = initial_memory - final_memory

        if saved > 0 and name:
            self.logger.info(f"🔧 {name} memoria optimizada: {saved:.1f}MB ahorrados")

        return df

    @log_performance
    def validate_geographic_data(self, df: pd.DataFrame,
                                 columna_ubigeo: Optional[str] = None) -> GeoValidationResult:
        """
        Valida integralmente un DataFrame con datos geográficos.

        Args:
            df: DataFrame a validar
            columna_ubigeo: Columna con códigos UBIGEO (None para autodetectar)

        Returns:
            Resultado completo de validación
        """
        self.logger.info("🔍 Iniciando validación geográfica completa")

        errors = []
        warnings = []
        quality_metrics = {}

        # Autodetectar columna UBIGEO si no se especifica
        if columna_ubigeo is None:
            columnas_geo = self.pattern_detector.detectar_columnas_geograficas(df)
            if 'ubigeo' in columnas_geo:
                columna_ubigeo = columnas_geo['ubigeo']
            else:
                errors.append("No se encontró columna UBIGEO")
                return GeoValidationResult(
                    is_valid=False, total_records=len(df), valid_ubigeos=0,
                    invalid_ubigeos=len(df), duplicate_ubigeos=0,
                    missing_coordinates=0, territorial_inconsistencies=0,
                    coverage_percentage=0.0, errors=errors, warnings=warnings,
                    quality_metrics=quality_metrics
                )

        total_records = len(df)

        # Validar UBIGEOs
        mask_validos, ubigeo_errors = self.ubigeo_validator.validar_serie_ubigeos(
            df[columna_ubigeo], self.geo_config.tipo_validacion_ubigeo
        )

        valid_ubigeos = mask_validos.sum()
        invalid_ubigeos = total_records - valid_ubigeos
        errors.extend(ubigeo_errors[:10])  # Máximo 10 errores de UBIGEO

        # Detectar duplicados
        duplicates_mask = df[columna_ubigeo].duplicated(keep=False)
        duplicate_ubigeos = duplicates_mask.sum()

        if duplicate_ubigeos > 0:
            warnings.append(f"Se encontraron {duplicate_ubigeos} UBIGEOs duplicados")

        # Validar coordenadas si están disponibles
        missing_coordinates = 0
        if self.geo_config.validar_coordenadas:
            columnas_geo = self.pattern_detector.detectar_columnas_geograficas(df)
            if 'coordenada_x' in columnas_geo and 'coordenada_y' in columnas_geo:
                coord_x_col = columnas_geo['coordenada_x']
                coord_y_col = columnas_geo['coordenada_y']
                missing_coordinates = df[[coord_x_col, coord_y_col]].isnull().any(axis=1).sum()

        # Validar consistencia territorial
        territorial_inconsistencies = 0
        if self.geo_config.validar_consistencia_territorial:
            columnas_geo = self.pattern_detector.detectar_columnas_geograficas(df)
            inconsistencias = self.territorial_validator.validar_jerarquia_territorial(df, columnas_geo)
            territorial_inconsistencies = len(inconsistencias)
            warnings.extend(inconsistencias)

        # Calcular métricas de calidad
        coverage_percentage = (valid_ubigeos / total_records * 100) if total_records > 0 else 0

        quality_metrics = {
            'completeness': coverage_percentage,
            'uniqueness': ((total_records - duplicate_ubigeos) / total_records * 100) if total_records > 0 else 0,
            'validity': (valid_ubigeos / total_records * 100) if total_records > 0 else 0,
            'consistency': (1 - territorial_inconsistencies / total_records) * 100 if total_records > 0 else 100
        }

        # Determinar si es válido
        is_valid = (
                coverage_percentage >= 90 and  # Al menos 90% de cobertura
                quality_metrics['uniqueness'] >= 95 and  # Al menos 95% únicos
                territorial_inconsistencies == 0  # Sin inconsistencias
        )

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
            quality_metrics=quality_metrics
        )

        self.logger.info(f"✅ Validación completada - Cobertura: {coverage_percentage:.1f}%")
        return result

    def _handle_duplicates(self, df: pd.DataFrame, columna_union: str) -> pd.DataFrame:
        """Maneja duplicados usando la estrategia configurada"""
        duplicates_mask = df[columna_union].duplicated(keep=False)

        if not duplicates_mask.any():
            return df

        if self.geo_config.manejo_duplicados == TipoManejoDuplicados.ERROR:
            n_duplicates = duplicates_mask.sum()
            ubigeos_duplicados = df[duplicates_mask][columna_union].unique()
            raise GeoMergeError(
                f"Se encontraron {n_duplicates} duplicados en '{columna_union}'. "
                f"UBIGEOs afectados: {ubigeos_duplicados[:5]}"
            )

        strategy = DuplicateStrategyFactory.create_strategy(
            self.geo_config.manejo_duplicados, self.logger
        )
        return strategy.handle_duplicates(df, columna_union, self.geo_config)

    @log_performance
    def merge_geographic_data(self,
                              df_principal: pd.DataFrame,
                              df_geografia: pd.DataFrame,
                              columnas_geograficas: Optional[Dict[str, str]] = None,
                              columna_union: Optional[str] = None,
                              validate_before_merge: bool = None) -> Tuple[pd.DataFrame, GeoValidationResult]:
        """
        Fusiona datos principales con información geográfica.

        Args:
            df_principal: DataFrame con datos principales
            df_geografia: DataFrame con información geográfica
            columnas_geograficas: Mapeo específico de columnas o None para autodetección
            columna_union: Columna para el merge o None para usar configuración
            validate_before_merge: Si validar antes de fusionar

        Returns:
            Tupla con (DataFrame fusionado, Resultado de validación)
        """
        self.logger.info("🗺️  Iniciando fusión geográfica avanzada")

        # Usar configuración por defecto si no se especifica
        columna_union = columna_union or self.geo_config.columna_union
        validate_before_merge = validate_before_merge if validate_before_merge is not None else self.geo_config.generar_reporte_calidad

        # Validar que existe la columna de unión
        for df, nombre in zip([df_principal, df_geografia], ['principal', 'geografía']):
            if columna_union not in df.columns:
                raise GeoMergeError(f"Columna '{columna_union}' no encontrada en DataFrame {nombre}")

        # Validación previa si está habilitada
        validation_result = None
        if validate_before_merge:
            validation_result = self.validate_geographic_data(df_geografia, columna_union)

            if not validation_result.is_valid and self.geo_config.manejo_errores == TipoManejoErrores.RAISE:
                raise GeoMergeError(f"Validación geográfica falló:\n{validation_result.get_summary_report()}")

        # Autodetectar columnas geográficas si no se especifican
        if columnas_geograficas is None:
            columnas_geograficas = self.pattern_detector.detectar_columnas_geograficas(df_geografia)
            if not columnas_geograficas:
                raise GeoMergeError("No se pudieron detectar columnas geográficas automáticamente")
            self.logger.info(f"📍 Columnas geográficas detectadas: {list(columnas_geograficas.keys())}")

        # Validar que las columnas especificadas existan
        columnas_faltantes = [col for col in columnas_geograficas.values()
                              if col not in df_geografia.columns]
        if columnas_faltantes:
            raise GeoMergeError(f"Columnas no encontradas en df_geografia: {columnas_faltantes}")

        # Optimizar memoria si está habilitado
        if self.geo_config.optimizar_memoria:
            df_geografia = self._optimize_dataframe_memory(df_geografia, "geografía")

        # Manejar duplicados
        df_geografia_limpio = self._handle_duplicates(df_geografia, columna_union)

        # Preparar DataFrame para merge
        columnas_seleccion = [columna_union] + list(columnas_geograficas.values())
        df_geo_para_merge = df_geografia_limpio[columnas_seleccion].copy()

        # Aplicar prefijos/sufijos a columnas
        mapeo_renombre = {}
        for tipo_geo, nombre_columna in columnas_geograficas.items():
            nuevo_nombre = f"{self.geo_config.prefijo_columnas}{tipo_geo}{self.geo_config.sufijo_columnas}"
            if nuevo_nombre != nombre_columna:  # Solo renombrar si es diferente
                mapeo_renombre[nombre_columna] = nuevo_nombre

        if mapeo_renombre:
            df_geo_para_merge = df_geo_para_merge.rename(columns=mapeo_renombre)

        # Realizar merge
        try:
            df_resultado = pd.merge(
                left=df_principal,
                right=df_geo_para_merge,
                on=columna_union,
                how='left',
                validate='m:1'
            )
        except pd.errors.MergeError as e:
            raise GeoMergeError(f"Error en la fusión de datos: {str(e)}")

        # Manejar valores faltantes
        columnas_geo_nuevas = list(mapeo_renombre.values()) if mapeo_renombre else list(columnas_geograficas.values())
        self._handle_missing_values(df_resultado, columnas_geo_nuevas)

        # Optimizar memoria del resultado
        if self.geo_config.optimizar_memoria:
            df_resultado = self._optimize_dataframe_memory(df_resultado, "resultado")

        # Reportar estadísticas
        if self.geo_config.mostrar_estadisticas:
            self._report_merge_statistics(df_principal, df_resultado, columnas_geo_nuevas)

        self.logger.info(f"✅ Fusión geográfica completada: {df_resultado.shape}")

        return df_resultado, validation_result

    def _handle_missing_values(self, df: pd.DataFrame, columnas_geograficas: List[str]) -> None:
        """Maneja valores faltantes según la configuración"""
        if self.geo_config.manejo_errores == TipoManejoErrores.COERCE:
            for col in columnas_geograficas:
                if col in df.columns:
                    df[col] = df[col].fillna(self.geo_config.valor_faltante)

        elif self.geo_config.manejo_errores == TipoManejoErrores.RAISE:
            faltantes = df[columnas_geograficas].isnull().any(axis=1)
            if faltantes.any():
                n_faltantes = faltantes.sum()
                raise GeoMergeError(f"Se encontraron {n_faltantes} registros con información geográfica faltante")

        elif self.geo_config.manejo_errores == TipoManejoErrores.LOG_WARNING:
            for col in columnas_geograficas:
                if col in df.columns:
                    n_missing = df[col].isnull().sum()
                    if n_missing > 0:
                        self.logger.warning(f"⚠️  {n_missing} valores faltantes en columna '{col}'")

    def _report_merge_statistics(self, df_original: pd.DataFrame,
                                 df_resultado: pd.DataFrame, columnas_geo_nuevas: List[str]) -> None:
        """Reporta estadísticas detalladas del merge"""
        n_original = len(df_original)
        n_resultado = len(df_resultado)

        if columnas_geo_nuevas and columnas_geo_nuevas[0] in df_resultado.columns:
            matches_exitosos = df_resultado[columnas_geo_nuevas[0]].notna().sum()
            tasa_match = (matches_exitosos / n_original) * 100

            self.logger.info(f"""
📊 Estadísticas de fusión geográfica:
   • Registros originales: {n_original:,}
   • Registros resultado: {n_resultado:,}
   • Matches exitosos: {matches_exitosos:,} ({tasa_match:.1f}%)
   • Registros sin match: {n_original - matches_exitosos:,}
   • Columnas geográficas añadidas: {len(columnas_geo_nuevas)}
            """)

    def extract_territorial_components(self, df: pd.DataFrame,
                                       columna_ubigeo: str) -> pd.DataFrame:
        """
        Extrae componentes territoriales jerárquicos de códigos UBIGEO.

        Args:
            df: DataFrame con códigos UBIGEO
            columna_ubigeo: Nombre de la columna con UBIGEO

        Returns:
            DataFrame con componentes territoriales extraídos
        """
        self.logger.info(f"🗂️  Extrayendo componentes territoriales de '{columna_ubigeo}'")

        if columna_ubigeo not in df.columns:
            raise GeoMergeError(f"Columna '{columna_ubigeo}' no encontrada")

        componentes = self.ubigeo_validator.extraer_componentes_ubigeo(df[columna_ubigeo])

        # Combinar con DataFrame original
        result = pd.concat([df, componentes.drop('ubigeo', axis=1)], axis=1)

        self.logger.info(f"✅ Componentes extraídos: {list(componentes.columns)}")
        return result

    # =====================================================
    # FUNCIONALIDADES DE MERGE ENTRE MÓDULOS
    # =====================================================

    def merge_modules(self,
                      left_df: pd.DataFrame,
                      right_df: pd.DataFrame,
                      left_module: str,
                      right_module: str,
                      merge_config: Optional[ModuleMergeConfig] = None) -> ModuleMergeResult:
        """
        Merge entre dos módulos ENAHO con validaciones específicas.

        Args:
            left_df: DataFrame del módulo izquierdo
            right_df: DataFrame del módulo derecho
            left_module: Código del módulo izquierdo (ej: "01", "05")
            right_module: Código del módulo derecho
            merge_config: Configuración específica para este merge

        Returns:
            ModuleMergeResult con DataFrame combinado y métricas
        """
        return self.module_merger.merge_modules(
            left_df, right_df, left_module, right_module, merge_config
        )

    def merge_multiple_modules(self,
                               modules_dict: Dict[str, pd.DataFrame],
                               base_module: str = "34",
                               merge_config: Optional[ModuleMergeConfig] = None) -> ModuleMergeResult:
        """
        Merge múltiples módulos secuencialmente.

        Args:
            modules_dict: Diccionario {codigo_modulo: dataframe}
            base_module: Código del módulo base para iniciar
            merge_config: Configuración de merge

        Returns:
            ModuleMergeResult con todos los módulos combinados
        """
        return self.module_merger.merge_multiple_modules(
            modules_dict, base_module, merge_config
        )

    def merge_modules_with_geography(self,
                                     modules_dict: Dict[str, pd.DataFrame],
                                     df_geografia: pd.DataFrame,
                                     base_module: str = "34",
                                     merge_config: Optional[ModuleMergeConfig] = None,
                                     geo_config: Optional[GeoMergeConfiguration] = None) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
        """
        Combina múltiples módulos y luego agrega información geográfica.

        Args:
            modules_dict: Diccionario con módulos ENAHO
            df_geografia: DataFrame con información geográfica
            base_module: Módulo base para iniciar merge
            merge_config: Configuración para merge de módulos
            geo_config: Configuración para merge geográfico

        Returns:
            Tupla con DataFrame final y reporte completo
        """
        self.logger.info("🔗🗺️  Iniciando merge combinado: módulos + geografía")

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
                df_geografia=df_geografia
            )
        finally:
            # Restaurar configuración original
            self.geo_config = original_geo_config

        # 3. Combinar reportes
        combined_report = {
            'module_merge': module_result.merge_report,
            'geographic_merge': {
                'validation': geo_validation.to_dict() if geo_validation else None,
                'final_records': len(geo_result)
            },
            'overall_quality': self._assess_combined_quality(geo_result, module_result),
            'processing_summary': {
                'modules_processed': len(modules_dict),
                'base_module': base_module,
                'final_shape': geo_result.shape,
                'merge_sequence': module_result.merge_report.get('modules_sequence', ''),
                'geographic_coverage': geo_validation.coverage_percentage if geo_validation else 0
            }
        }

        self.logger.info(f"✅ Merge combinado completado: {geo_result.shape}")
        return geo_result, combined_report

    def validate_module_compatibility(self,
                                      modules_dict: Dict[str, pd.DataFrame],
                                      merge_level: str = "hogar") -> Dict[str, Any]:
        """
        Valida compatibilidad entre múltiples módulos.

        Args:
            modules_dict: Diccionario con módulos a validar
            merge_level: Nivel de merge ("hogar", "persona", "vivienda")

        Returns:
            Reporte de compatibilidad detallado
        """
        from .config import ModuleMergeLevel

        level_enum = ModuleMergeLevel(merge_level)
        compatibility_report = {
            'overall_compatible': True,
            'merge_level': merge_level,
            'modules_analyzed': list(modules_dict.keys()),
            'pairwise_compatibility': {},
            'recommendations': [],
            'warnings': [],
            'potential_issues': []
        }

        # Validar cada par de módulos
        module_codes = list(modules_dict.keys())

        for i, module1 in enumerate(module_codes):
            for module2 in module_codes[i + 1:]:
                pair_key = f"{module1}-{module2}"

                compatibility = self.module_merger.validator.check_module_compatibility(
                    modules_dict[module1], modules_dict[module2],
                    module1, module2, level_enum
                )

                compatibility_report['pairwise_compatibility'][pair_key] = compatibility

                if not compatibility['compatible']:
                    compatibility_report['overall_compatible'] = False
                    compatibility_report['potential_issues'].append(
                        f"Módulos {module1} y {module2}: {compatibility.get('error', 'Incompatible')}"
                    )

        # Generar recomendaciones
        if compatibility_report['overall_compatible']:
            compatibility_report['recommendations'].append(
                "✅ Todos los módulos son compatibles para merge"
            )
        else:
            compatibility_report['recommendations'].append(
                "⚠️  Algunos módulos presentan incompatibilidades. Revisar configuración de merge."
            )

            # Sugerir nivel alternativo si es necesario
            if merge_level == "persona":
                compatibility_report['recommendations'].append(
                    "💡 Considere usar nivel 'hogar' para mayor compatibilidad"
                )

        return compatibility_report

    def _assess_combined_quality(self, final_df: pd.DataFrame,
                                 module_result: ModuleMergeResult) -> Dict[str, float]:
        """Evalúa la calidad combinada del resultado final"""

        # Métricas de completitud
        completeness = (1 - final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1])) * 100

        # Métricas de consistencia (basado en llaves primarias)
        key_cols = [col for col in ['conglome', 'vivienda', 'hogar'] if col in final_df.columns]
        if key_cols:
            duplicates = final_df.duplicated(subset=key_cols).sum()
            consistency = (1 - duplicates / len(final_df)) * 100 if len(final_df) > 0 else 100
        else:
            consistency = 100

        # Score de módulos
        module_quality = module_result.quality_score

        # Score combinado
        weights = {'completeness': 0.4, 'consistency': 0.3, 'module_quality': 0.3}
        combined_score = (
                completeness * weights['completeness'] +
                consistency * weights['consistency'] +
                module_quality * weights['module_quality']
        )

        return {
            'completeness': completeness,
            'consistency': consistency,
            'module_quality': module_quality,
            'combined_score': combined_score,
            'quality_grade': self._get_quality_grade(combined_score)
        }

    def _get_quality_grade(self, score: float) -> str:
        """Asigna grado de calidad basado en score"""
        if score >= 90:
            return "A+ (Excelente)"
        elif score >= 80:
            return "A (Muy Bueno)"
        elif score >= 70:
            return "B (Bueno)"
        elif score >= 60:
            return "C (Aceptable)"
        elif score >= 50:
            return "D (Regular)"
        else:
            return "F (Deficiente)"

    # =====================================================
    # MÉTODOS DE ANÁLISIS Y UTILIDADES
    # =====================================================

    def analyze_merge_feasibility(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                  merge_type: str = "geographic") -> Dict[str, Any]:
        """
        Analiza la viabilidad de merge entre dos DataFrames

        Args:
            df1, df2: DataFrames a analizar
            merge_type: Tipo de merge ("geographic" o "module")

        Returns:
            Análisis de viabilidad
        """
        analysis = {
            'feasible': True,
            'merge_type': merge_type,
            'size_compatibility': {},
            'structure_compatibility': {},
            'recommendations': [],
            'potential_issues': []
        }

        # Análisis de tamaños
        size1 = len(df1)
        size2 = len(df2)

        analysis['size_compatibility'] = {
            'df1_size': size1,
            'df2_size': size2,
            'size_ratio': min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0,
            'memory_estimate_mb': (size1 + size2) * len(set(df1.columns) | set(df2.columns)) * 8 / 1024 / 1024
        }

        if merge_type == "geographic":
            # Análisis específico para merge geográfico
            geo_cols1 = self.pattern_detector.detectar_columnas_geograficas(df1)
            geo_cols2 = self.pattern_detector.detectar_columnas_geograficas(df2)

            common_geo = set(geo_cols1.keys()) & set(geo_cols2.keys())

            analysis['structure_compatibility'] = {
                'geo_columns_df1': len(geo_cols1),
                'geo_columns_df2': len(geo_cols2),
                'common_geo_levels': list(common_geo),
                'compatibility_score': len(common_geo) / max(len(geo_cols1), len(geo_cols2), 1)
            }

            if not common_geo:
                analysis['feasible'] = False
                analysis['potential_issues'].append("No hay niveles geográficos comunes")

        elif merge_type == "module":
            # Análisis específico para merge de módulos
            key_cols = ['conglome', 'vivienda', 'hogar']
            missing1 = [col for col in key_cols if col not in df1.columns]
            missing2 = [col for col in key_cols if col not in df2.columns]

            analysis['structure_compatibility'] = {
                'missing_keys_df1': missing1,
                'missing_keys_df2': missing2,
                'key_compatibility': len(missing1) == 0 and len(missing2) == 0
            }

            if missing1 or missing2:
                analysis['feasible'] = False
                analysis['potential_issues'].append(f"Llaves faltantes: df1={missing1}, df2={missing2}")

        # Recomendaciones generales
        if analysis['size_compatibility']['memory_estimate_mb'] > 1000:  # 1GB
            analysis['recommendations'].append("Considere procesamiento por chunks para manejar memoria")

        if analysis['size_compatibility']['size_ratio'] < 0.1:
            analysis['recommendations'].append("Gran diferencia de tamaños - verifique que sea correcto")

        return analysis

    def create_comprehensive_report(self, df: pd.DataFrame,
                                    include_geographic: bool = True,
                                    include_quality: bool = True) -> str:
        """
        Crea reporte integral de un DataFrame

        Args:
            df: DataFrame a analizar
            include_geographic: Si incluir análisis geográfico
            include_quality: Si incluir métricas de calidad

        Returns:
            Reporte formateado como string
        """
        lines = [
            "📊 REPORTE INTEGRAL DE DATOS",
            "=" * 40,
            f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Registros: {len(df):,}",
            f"Columnas: {len(df.columns)}",
            f"Memoria: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
            ""
        ]

        # Análisis geográfico
        if include_geographic:
            geo_cols = self.pattern_detector.detectar_columnas_geograficas(df)
            if geo_cols:
                lines.extend([
                    "🗺️  ANÁLISIS GEOGRÁFICO:",
                    f"Columnas geográficas detectadas: {len(geo_cols)}",
                ])

                for tipo, columna in geo_cols.items():
                    completitud = df[columna].notna().sum() / len(df) * 100
                    lines.append(f"  • {tipo}: {columna} ({completitud:.1f}% completo)")

                # Validación UBIGEO si existe
                if 'ubigeo' in geo_cols:
                    validation = self.validate_geographic_data(df, geo_cols['ubigeo'])
                    lines.extend([
                        "",
                        f"Validación UBIGEO: {'✅ Válido' if validation.is_valid else '❌ Inválido'}",
                        f"Cobertura: {validation.coverage_percentage:.1f}%"
                    ])

        # Métricas de calidad
        if include_quality:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

            lines.extend([
                "",
                "📈 MÉTRICAS DE CALIDAD:",
                f"Completitud general: {completeness:.1f}%",
            ])

            # Top columnas con valores faltantes
            missing_by_col = df.isnull().sum().sort_values(ascending=False)
            top_missing = missing_by_col[missing_by_col > 0].head(5)

            if not top_missing.empty:
                lines.append("Columnas con más faltantes:")
                for col, missing in top_missing.items():
                    pct = missing / len(df) * 100
                    lines.append(f"  • {col}: {missing:,} ({pct:.1f}%)")

        lines.append("")
        return "\n".join(lines)