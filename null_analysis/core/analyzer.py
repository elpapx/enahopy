"""
Analizador principal de valores nulos
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

try:
    from ..loader import setup_logging, CacheManager

    ENAHO_MODULES_AVAILABLE = True
except ImportError:
    ENAHO_MODULES_AVAILABLE = False


    def setup_logging(verbose=True, structured=False, log_file=None):
        import logging
        logger = logging.getLogger('enaho_null_analyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return logger

from ..config import NullAnalysisConfig, AnalysisComplexity
from ..strategies.basic_analysis import BasicNullAnalysis
from ..strategies.advanced_analysis import AdvancedNullAnalysis
from ..visualization.factory import VisualizationFactory
from ..io.exporters import ReportExporter

try:
    from ..merger import detect_geographic_columns
except ImportError:
    def detect_geographic_columns(df):
        return {}


class ENAHONullAnalyzer:
    """
    Analizador avanzado de valores nulos para microdatos del INEI.
    """

    def __init__(self,
                 config: Optional[NullAnalysisConfig] = None,
                 enaho_config: Optional[Any] = None,
                 verbose: bool = True,
                 structured_logging: bool = False,
                 log_file: Optional[str] = None):

        self.config = config or NullAnalysisConfig()
        self.enaho_config = enaho_config
        self.logger = setup_logging(verbose, structured_logging, log_file)

        self._init_analyzers()
        self._init_visualization_factory()
        self._init_report_exporter()
        self._init_cache_system()

        self._validate_optional_dependencies()

        self.logger.info("🔍 ENAHONullAnalyzer inicializado exitosamente")
        self.logger.info(f"   Nivel de complejidad: {self.config.complexity_level.value}")
        self.logger.info(f"   Tipo de visualización: {self.config.visualization_type.value}")

    def _init_analyzers(self):
        """Inicializa estrategias de análisis"""
        self.analyzers = {
            AnalysisComplexity.BASIC: BasicNullAnalysis(self.config, self.logger),
            AnalysisComplexity.STANDARD: BasicNullAnalysis(self.config, self.logger),
            AnalysisComplexity.ADVANCED: AdvancedNullAnalysis(self.config, self.logger),
            AnalysisComplexity.EXPERT: AdvancedNullAnalysis(self.config, self.logger)
        }

    def _init_visualization_factory(self):
        """Inicializa factory de visualizaciones"""
        self.viz_factory = VisualizationFactory(self.config, self.logger)

    def _init_report_exporter(self):
        """Inicializa exportador de reportes"""
        self.report_exporter = ReportExporter(self.config, self.logger)

    def _init_cache_system(self):
        """Inicializa sistema de cache"""
        if self.config.enable_caching and ENAHO_MODULES_AVAILABLE:
            try:
                self.cache_manager = CacheManager(self.enaho_config.cache_dir if self.enaho_config else '.enaho_cache')
            except:
                self.cache_manager = None
        else:
            self.cache_manager = None

    def _validate_optional_dependencies(self):
        """Valida dependencias opcionales y ajusta configuración"""
        try:
            import plotly
        except ImportError:
            if self.config.visualization_type.value == 'interactive':
                self.logger.warning("Plotly no disponible. Cambiando a visualizaciones estáticas.")
                from ..config import VisualizationType
                self.config.visualization_type = VisualizationType.STATIC

    def _generate_cache_key(self, df: pd.DataFrame, **kwargs) -> str:
        """Genera clave de cache única para el análisis"""
        df_info = f"{df.shape}_{df.dtypes.to_string()}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(df_info.encode()).hexdigest()

    def _should_use_sampling(self, df: pd.DataFrame) -> bool:
        """Determina si se debe usar sampling para datasets grandes"""
        return (self.config.use_sampling_for_large_datasets and
                len(df) > self.config.max_sample_size)

    def _create_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea muestra representativa del DataFrame"""
        sample_size = min(self.config.max_sample_size, len(df))
        return df.sample(n=sample_size, random_state=42)

    def analyze_null_patterns(self,
                              df: pd.DataFrame,
                              group_by: Optional[str] = None,
                              geographic_filter: Optional[Dict[str, str]] = None,
                              complexity_level: Optional[AnalysisComplexity] = None,
                              use_cache: bool = None) -> Dict[str, Any]:
        """
        Análisis completo de patrones de valores nulos.
        """

        start_time = time.time()

        if df.empty:
            from ..exceptions import NullAnalysisError
            raise NullAnalysisError("DataFrame vacío")

        analysis_complexity = complexity_level or self.config.complexity_level
        use_cache = use_cache if use_cache is not None else self.config.enable_caching

        cache_key = None
        if use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(
                df, group_by=group_by,
                geographic_filter=geographic_filter,
                complexity=analysis_complexity.value
            )

            cached_result = self.cache_manager.get_metadata(f"null_analysis_{cache_key}")
            if cached_result:
                self.logger.info("✨ Resultado obtenido del cache")
                return cached_result

        self.logger.info(f"🔍 Iniciando análisis de nulos - Complejidad: {analysis_complexity.value}")

        df_filtered = df.copy()
        if geographic_filter and ENAHO_MODULES_AVAILABLE:
            df_filtered = self._apply_geographic_filter(df_filtered, geographic_filter)

        if group_by is None:
            group_by = self._auto_detect_grouping_column(df_filtered)

        df_analysis = df_filtered
        used_sample = False
        if self._should_use_sampling(df_filtered):
            df_analysis = self._create_sample(df_filtered)
            used_sample = True
            self.logger.info(f"📊 Usando muestra de {len(df_analysis):,} registros")

        analyzer = self.analyzers[analysis_complexity]
        analysis_result = analyzer.analyze(df_analysis, group_by=group_by)

        execution_time = time.time() - start_time
        analysis_result.update({
            'execution_time': execution_time,
            'original_shape': df.shape,
            'filtered_shape': df_filtered.shape,
            'analysis_shape': df_analysis.shape,
            'used_sample': used_sample,
            'geographic_filter_applied': geographic_filter is not None,
            'complexity_level': analysis_complexity.value,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

        recommendations = analyzer.get_recommendations(analysis_result)
        analysis_result['recommendations'] = recommendations

        if use_cache and self.cache_manager and cache_key:
            self.cache_manager.set_metadata(f"null_analysis_{cache_key}", analysis_result)

        self.logger.info(f"✅ Análisis completado en {execution_time:.2f} segundos")

        return analysis_result

    def _apply_geographic_filter(self, df: pd.DataFrame,
                                 geographic_filter: Dict[str, str]) -> pd.DataFrame:
        """Aplica filtros geográficos usando integración con merger"""

        try:
            geo_columns = detect_geographic_columns(df)

            df_filtered = df.copy()
            filters_applied = []

            for geo_level, value in geographic_filter.items():
                if geo_level in geo_columns:
                    col_name = geo_columns[geo_level]
                    mask = df_filtered[col_name].str.upper() == value.upper()
                    df_filtered = df_filtered[mask]
                    filters_applied.append(f"{geo_level}={value}")

            if filters_applied:
                self.logger.info(f"🗺️  Filtros geográficos aplicados: {', '.join(filters_applied)}")
                self.logger.info(f"   Registros: {len(df)} → {len(df_filtered)}")

            return df_filtered

        except Exception as e:
            self.logger.warning(f"Error aplicando filtros geográficos: {str(e)}. Continuando sin filtrar.")
            return df

    def _auto_detect_grouping_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detecta automáticamente columna apropiada para agrupación"""

        if ENAHO_MODULES_AVAILABLE:
            try:
                geo_columns = detect_geographic_columns(df)
                for preferred in ['departamento', 'provincia', 'distrito']:
                    if preferred in geo_columns:
                        self.logger.info(f"🗺️  Columna de agrupación detectada: {geo_columns[preferred]}")
                        return geo_columns[preferred]
            except Exception:
                pass

        geographic_patterns = ['departamento', 'provincia', 'distrito', 'region', 'geo_']
        for col in df.columns:
            # FIX: Convertir columna a string antes de aplicar .lower()
            col_str = str(col)
            col_lower = col_str.lower()

            for pattern in geographic_patterns:
                if pattern in col_lower:
                    self.logger.info(f"🗺️  Columna de agrupación detectada: {col}")
                    return col

        # Si hay columnas categóricas con pocos valores únicos
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                n_unique = df[col].nunique()
                if 2 <= n_unique <= 20:  # Rango razonable para agrupación
                    self.logger.info(f"📊 Columna categórica detectada para agrupación: {col}")
                    return col

        return None

    def create_visualizations(self,
                              analysis_result: Dict[str, Any],
                              save_path: Optional[str] = None,
                              show_plots: bool = True) -> Dict[str, Any]:
        """
        Crea visualizaciones del análisis.
        """

        self.logger.info("🎨 Creando visualizaciones...")

        visualizations = self.viz_factory.create_basic_visualizations(analysis_result)

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            saved_files = []

            for viz_name, fig in visualizations.items():
                if hasattr(fig, 'savefig'):
                    file_path = save_path / f"{viz_name}.png"
                    fig.savefig(file_path, dpi=self.config.dpi, bbox_inches='tight')
                    saved_files.append(str(file_path))

                    if not show_plots:
                        import matplotlib.pyplot as plt
                        plt.close(fig)

            for viz_name, fig in visualizations.items():
                if hasattr(fig, 'write_html'):
                    file_path = save_path / f"{viz_name}.html"
                    fig.write_html(str(file_path))
                    saved_files.append(str(file_path))

            visualizations['saved_files'] = saved_files
            self.logger.info(f"💾 {len(saved_files)} visualizaciones guardadas en {save_path}")

        if show_plots:
            import matplotlib.pyplot as plt
            for viz_name, fig in visualizations.items():
                if hasattr(fig, 'show'):
                    plt.show()

        return visualizations

    def generate_comprehensive_report(self,
                                      df: pd.DataFrame,
                                      output_path: str,
                                      group_by: Optional[str] = None,
                                      geographic_filter: Optional[Dict[str, str]] = None,
                                      include_visualizations: bool = True,
                                      complexity_level: Optional[AnalysisComplexity] = None) -> Dict[str, Any]:
        """
        Genera reporte completo con análisis, visualizaciones y exportación.
        """

        self.logger.info("📋 Generando reporte completo...")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        report_info = {
            'output_directory': str(output_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'files_generated': []
        }

        try:
            analysis_result = self.analyze_null_patterns(
                df=df,
                group_by=group_by,
                geographic_filter=geographic_filter,
                complexity_level=complexity_level
            )

            report_info['analysis_completed'] = True
            report_info['analysis_summary'] = {
                'complexity_level': analysis_result['complexity_level'],
                'variables_analyzed': len(
                    analysis_result.get('summary', analysis_result.get('basic_analysis', {}).get('summary', []))),
                'execution_time': analysis_result['execution_time']
            }

            visualizations = {}
            if include_visualizations:
                viz_path = output_path / "visualizations"
                visualizations = self.create_visualizations(
                    analysis_result,
                    save_path=viz_path,
                    show_plots=False
                )
                report_info['visualizations_created'] = True
                report_info['visualization_files'] = visualizations.get('saved_files', [])

            exported_files = self.report_exporter.export_report(
                analysis_result, visualizations, output_path
            )

            report_info['reports_exported'] = exported_files
            report_info['files_generated'].extend(exported_files.values())

            executive_summary = self._create_executive_summary(analysis_result)
            summary_file = output_path / "executive_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(executive_summary)

            report_info['files_generated'].append(str(summary_file))

            self.logger.info(f"✅ Reporte completo generado en {output_path}")
            self.logger.info(f"   📁 {len(report_info['files_generated'])} archivos creados")

            return report_info

        except Exception as e:
            self.logger.error(f"❌ Error generando reporte: {str(e)}")
            report_info['error'] = str(e)
            report_info['success'] = False
            return report_info

    def _create_executive_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Crea resumen ejecutivo del análisis"""

        metrics = analysis_result['metrics']

        summary = f"""
🔍 RESUMEN EJECUTIVO - ANÁLISIS DE VALORES NULOS
===============================================

Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}

📊 MÉTRICAS PRINCIPALES
-----------------------
- Porcentaje de valores faltantes: {metrics.missing_percentage:.1f}%
- Casos completos: {metrics.complete_cases_percentage:.1f}%
- Variables con faltantes: {metrics.variables_with_missing}
- Variables sin faltantes: {metrics.variables_without_missing}
- Score de calidad de datos: {metrics.data_quality_score:.1f}/100

🔍 CLASIFICACIÓN DE PATRONES
-----------------------------
- Patrón detectado: {metrics.missing_data_pattern.value}
- Patrones únicos: {metrics.missing_pattern_count}
- Patrón monótono: {'Sí' if metrics.monotone_missing else 'No'}

💡 RECOMENDACIONES PRINCIPALES
-------------------------------
"""

        recommendations = analysis_result.get('recommendations', [])
        for i, rec in enumerate(recommendations[:5], 1):
            summary += f"{i}. {rec}\n"

        summary += f"""

📈 DETALLES TÉCNICOS
--------------------
- Tipo de análisis: {analysis_result['analysis_type'].title()}
- Tiempo de ejecución: {analysis_result['execution_time']:.2f} segundos
- Registros analizados: {analysis_result['analysis_shape'][0]:,}
- Variables analizadas: {analysis_result['analysis_shape'][1]}
"""

        if analysis_result.get('used_sample'):
            summary += f"• Nota: Se utilizó muestra representativa por tamaño del dataset\n"

        return summary

    def get_data_quality_score(self, df: pd.DataFrame,
                               detailed: bool = False):
        """
        Calcula score rápido de calidad de datos basado en completitud.
        """

        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        complete_cases_percentage = (df.dropna().shape[0] / df.shape[0]) * 100

        completeness_weight = 0.6
        case_completeness_weight = 0.4

        quality_score = (
                (100 - missing_percentage) * completeness_weight +
                complete_cases_percentage * case_completeness_weight
        )

        if not detailed:
            return quality_score

        return {
            'quality_score': quality_score,
            'missing_percentage': missing_percentage,
            'complete_cases_percentage': complete_cases_percentage,
            'variables_with_missing': (df.isnull().sum() > 0).sum(),
            'recommendation': self._get_quality_recommendation(quality_score)
        }

    def _get_quality_recommendation(self, score: float) -> str:
        """Genera recomendación basada en score de calidad"""
        if score >= 90:
            return "✅ Excelente calidad de datos. Dataset listo para análisis."
        elif score >= 75:
            return "🟢 Buena calidad de datos. Revisión menor recomendada."
        elif score >= 60:
            return "🟡 Calidad moderada. Limpieza selectiva recomendada."
        elif score >= 40:
            return "🟠 Calidad baja. Limpieza intensiva necesaria."
        else:
            return "🔴 Calidad muy baja. Revisar fuente de datos y recolección."

    def compare_datasets_nulls(self,
                               datasets: Dict[str, pd.DataFrame],
                               group_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Compara patrones de nulos entre múltiples datasets.
        """

        self.logger.info(f"🔄 Comparando patrones de nulos en {len(datasets)} datasets...")

        comparison_results = {}

        for name, df in datasets.items():
            try:
                result = self.analyze_null_patterns(df, group_by=group_by)
                comparison_results[name] = result
            except Exception as e:
                self.logger.warning(f"Error analizando dataset '{name}': {str(e)}")
                continue

        comparison_summary = {
            'datasets_analyzed': len(comparison_results),
            'individual_results': comparison_results,
            'comparison_metrics': self._create_comparison_metrics(comparison_results),
            'recommendations': self._create_comparison_recommendations(comparison_results)
        }

        return comparison_summary

    def _create_comparison_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Crea métricas comparativas entre datasets"""

        metrics_comparison = {
            'missing_percentages': {},
            'quality_scores': {},
            'complete_cases': {},
            'patterns_detected': {}
        }

        for name, result in results.items():
            metrics = result['metrics']
            metrics_comparison['missing_percentages'][name] = metrics.missing_percentage
            metrics_comparison['quality_scores'][name] = metrics.data_quality_score
            metrics_comparison['complete_cases'][name] = metrics.complete_cases_percentage
            metrics_comparison['patterns_detected'][name] = metrics.missing_data_pattern.value

        import numpy as np
        missing_values = list(metrics_comparison['missing_percentages'].values())
        quality_values = list(metrics_comparison['quality_scores'].values())

        metrics_comparison['summary'] = {
            'best_quality_dataset': max(metrics_comparison['quality_scores'],
                                        key=metrics_comparison['quality_scores'].get),
            'worst_quality_dataset': min(metrics_comparison['quality_scores'],
                                         key=metrics_comparison['quality_scores'].get),
            'average_missing_percentage': np.mean(missing_values),
            'std_missing_percentage': np.std(missing_values),
            'average_quality_score': np.mean(quality_values),
            'quality_range': max(quality_values) - min(quality_values)
        }

        return metrics_comparison

    def _create_comparison_recommendations(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Crea recomendaciones comparativas"""

        recommendations = []

        if not results:
            return ["No hay datasets válidos para comparar"]

        comparison_metrics = self._create_comparison_metrics(results)
        summary = comparison_metrics['summary']

        if summary['quality_range'] > 30:
            recommendations.append(
                f"⚠️  Gran variación en calidad entre datasets "
                f"(rango: {summary['quality_range']:.1f} puntos). "
                f"Mejor: {summary['best_quality_dataset']}, "
                f"Peor: {summary['worst_quality_dataset']}"
            )

        if summary['average_missing_percentage'] > 20:
            recommendations.append(
                f"🔧 Alto porcentaje promedio de faltantes ({summary['average_missing_percentage']:.1f}%). "
                f"Considere estrategia unificada de imputación."
            )

        quality_scores = comparison_metrics['quality_scores']
        low_quality = [name for name, score in quality_scores.items() if score < 60]

        if low_quality:
            recommendations.append(
                f"❌ Datasets con baja calidad requieren atención prioritaria: {', '.join(low_quality)}"
            )

        return recommendations

    def suggest_imputation_strategy(self,
                                    analysis_result: Dict[str, Any],
                                    variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Sugiere estrategia de imputación basada en análisis de patrones.
        """

        metrics = analysis_result['metrics']
        pattern = metrics.missing_data_pattern

        strategies = {
            'recommended_method': None,
            'alternative_methods': [],
            'rationale': '',
            'considerations': []
        }

        from ..config import MissingDataPattern

        if pattern == MissingDataPattern.MCAR:
            strategies['recommended_method'] = 'listwise_deletion'
            strategies['alternative_methods'] = ['mean_imputation', 'median_imputation']
            strategies['rationale'] = 'Datos faltantes completamente aleatorios permiten eliminación segura'

        elif pattern == MissingDataPattern.MAR:
            strategies['recommended_method'] = 'multiple_imputation'
            strategies['alternative_methods'] = ['regression_imputation', 'knn_imputation']
            strategies['rationale'] = 'Patrón MAR requiere métodos que consideren relaciones entre variables'

        elif pattern == MissingDataPattern.MNAR:
            strategies['recommended_method'] = 'domain_specific_modeling'
            strategies['alternative_methods'] = ['pattern_mixture_modeling', 'selection_modeling']
            strategies['rationale'] = 'Patrón MNAR requiere modelado específico del dominio'

        else:
            strategies['recommended_method'] = 'exploratory_analysis'
            strategies['alternative_methods'] = ['multiple_strategies_comparison']
            strategies['rationale'] = 'Patrón incierto requiere análisis exploratorio adicional'

        if metrics.missing_percentage > 50:
            strategies['considerations'].append('Alto porcentaje de faltantes: considere recolección adicional')

        if metrics.monotone_missing:
            strategies['considerations'].append('Patrón monótono: imputación secuencial puede ser efectiva')

        if analysis_result.get('correlations', {}).get('significant_correlations'):
            strategies['considerations'].append('Correlaciones significativas: use imputación multivariada')

        return strategies