"""
Funciones de conveniencia para análisis de valores nulos
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import warnings

from .config import (
    NullAnalysisConfig,
    AnalysisComplexity,
    ExportFormat,
    VisualizationType  # Agregar esta importación
)
from .core.analyzer import ENAHONullAnalyzer

def quick_null_analysis(df: pd.DataFrame,
                       group_by: Optional[str] = None,
                       complexity: str = "standard") -> Dict[str, Any]:
    """
    Función de conveniencia para análisis rápido de nulos.
    """
    # Convertir string a enum apropiadamente
    config = NullAnalysisConfig(
        complexity_level=AnalysisComplexity(complexity),
        visualization_type=VisualizationType.STATIC  # Usar enum en lugar de string
    )

    analyzer = ENAHONullAnalyzer(config=config, verbose=False)
    return analyzer.analyze_null_patterns(df, group_by=group_by)

def get_data_quality_score(df: pd.DataFrame, detailed: bool = False) -> Union[float, Dict[str, Any]]:
    """
    Función de conveniencia para obtener score rápido de calidad de datos.
    """
    analyzer = ENAHONullAnalyzer(verbose=False)
    return analyzer.get_data_quality_score(df, detailed=detailed)

def create_null_visualizations(df: pd.DataFrame,
                             output_path: Optional[str] = None,
                             group_by: Optional[str] = None,
                             interactive: bool = False) -> Dict[str, Any]:
    """
    Función de conveniencia para crear visualizaciones de nulos.
    """
    # Usar enum correcto
    viz_type = VisualizationType.INTERACTIVE if interactive else VisualizationType.STATIC
    config = NullAnalysisConfig(visualization_type=viz_type)

    analyzer = ENAHONullAnalyzer(config=config, verbose=False)

    analysis_result = analyzer.analyze_null_patterns(df, group_by=group_by)

    return analyzer.create_visualizations(
        analysis_result,
        save_path=output_path,
        show_plots=output_path is None
    )

def generate_null_report(df: pd.DataFrame,
                        output_path: str,
                        group_by: Optional[str] = None,
                        geographic_filter: Optional[Dict[str, str]] = None,
                        format_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para generar reporte completo de nulos.
    """
    if format_types is None:
        format_types = ["html", "json"]

    # Convertir strings a enums
    export_formats = [ExportFormat(fmt) for fmt in format_types if fmt in ["html", "json", "xlsx", "md"]]

    config = NullAnalysisConfig(
        export_formats=export_formats,
        complexity_level=AnalysisComplexity.ADVANCED
    )

    analyzer = ENAHONullAnalyzer(config=config)

    return analyzer.generate_comprehensive_report(
        df=df,
        output_path=output_path,
        group_by=group_by,
        geographic_filter=geographic_filter
    )

def compare_null_patterns(datasets: Dict[str, pd.DataFrame],
                          group_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para comparar patrones de nulos entre datasets.
    """

    analyzer = ENAHONullAnalyzer(verbose=False)
    return analyzer.compare_datasets_nulls(datasets, group_by=group_by)


def suggest_imputation_methods(df: pd.DataFrame,
                               variable: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para sugerir métodos de imputación.
    """

    config = NullAnalysisConfig(complexity_level=AnalysisComplexity.ADVANCED)
    analyzer = ENAHONullAnalyzer(config=config, verbose=False)

    analysis_result = analyzer.analyze_null_patterns(df)

    return analyzer.suggest_imputation_strategy(analysis_result, variable=variable)


def validate_data_completeness(df: pd.DataFrame,
                               required_completeness: float = 80.0,
                               required_variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Valida si un DataFrame cumple con requisitos de completitud.
    """

    validation_result = {
        'is_valid': False,
        'overall_completeness': 0.0,
        'fails': [],
        'warnings': [],
        'recommendations': []
    }

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    overall_completeness = ((total_cells - missing_cells) / total_cells) * 100

    validation_result['overall_completeness'] = overall_completeness

    if overall_completeness < required_completeness:
        validation_result['fails'].append(
            f"Completitud general ({overall_completeness:.1f}%) "
            f"menor al requerido ({required_completeness}%)"
        )

    if required_variables:
        for var in required_variables:
            if var not in df.columns:
                validation_result['fails'].append(f"Variable requerida '{var}' no encontrada")
            else:
                var_completeness = ((len(df) - df[var].isnull().sum()) / len(df)) * 100
                if var_completeness < 100:
                    if var_completeness < required_completeness:
                        validation_result['fails'].append(
                            f"Variable '{var}' tiene completitud insuficiente ({var_completeness:.1f}%)"
                        )
                    else:
                        validation_result['warnings'].append(
                            f"Variable '{var}' tiene algunos valores faltantes ({var_completeness:.1f}%)"
                        )

    validation_result['is_valid'] = len(validation_result['fails']) == 0

    if not validation_result['is_valid']:
        validation_result['recommendations'].append(
            "Considere técnicas de imputación o recolección adicional de datos"
        )

        if overall_completeness < 50:
            validation_result['recommendations'].append(
                "Completitud muy baja: revisar calidad de la fuente de datos"
            )

    return validation_result


def detect_missing_patterns_automatically(df: pd.DataFrame,
                                          min_pattern_frequency: int = 10) -> Dict[str, Any]:
    """
    Detecta automáticamente patrones comunes de datos faltantes.
    """

    missing_matrix = df.isnull()

    pattern_strings = missing_matrix.apply(
        lambda row: ''.join(['1' if x else '0' for x in row]), axis=1
    )

    pattern_counts = pattern_strings.value_counts()
    common_patterns = pattern_counts[pattern_counts >= min_pattern_frequency]

    results = {
        'total_unique_patterns': len(pattern_counts),
        'common_patterns': len(common_patterns),
        'pattern_analysis': {},
        'interpretations': []
    }

    column_names = df.columns.tolist()

    for pattern, count in common_patterns.head(10).items():
        pattern_info = {
            'frequency': count,
            'percentage': (count / len(df)) * 100,
            'missing_variables': [
                column_names[i] for i, bit in enumerate(pattern) if bit == '1'
            ],
            'complete_variables': [
                column_names[i] for i, bit in enumerate(pattern) if bit == '0'
            ]
        }

        results['pattern_analysis'][pattern] = pattern_info

        if pattern == '0' * len(column_names):
            results['interpretations'].append(
                f"✅ {count} casos ({pattern_info['percentage']:.1f}%) completamente sin faltantes"
            )
        elif pattern.count('1') == len(column_names):
            results['interpretations'].append(
                f"❌ {count} casos ({pattern_info['percentage']:.1f}%) completamente faltantes"
            )
        elif pattern.count('1') == 1:
            missing_var = pattern_info['missing_variables'][0]
            results['interpretations'].append(
                f"🔍 {count} casos ({pattern_info['percentage']:.1f}%) "
                f"solo faltan en variable '{missing_var}'"
            )
        elif len(pattern_info['missing_variables']) <= 3:
            missing_vars = ', '.join(pattern_info['missing_variables'])
            results['interpretations'].append(
                f"🔗 {count} casos ({pattern_info['percentage']:.1f}%) "
                f"faltan conjuntamente en: {missing_vars}"
            )

    return results


# Compatibilidad con versión anterior
class LegacyNullAnalyzer:
    """Wrapper de compatibilidad con la versión anterior"""

    def __init__(self, config: Optional[NullAnalysisConfig] = None):
        warnings.warn(
            "LegacyNullAnalyzer está deprecated. Use ENAHONullAnalyzer en su lugar.",
            DeprecationWarning,
            stacklevel=2
        )

        self.analyzer = ENAHONullAnalyzer(config=config, verbose=False)

    def diagnostico_nulos_enaho(self, df: pd.DataFrame,
                                desagregado_por: str = 'geo_departamento',
                                incluir_estadisticas: bool = True,
                                filtro_geografico=None,
                                columnas_geo: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """Método de compatibilidad con API anterior"""

        geographic_filter = None
        if filtro_geografico:
            if hasattr(filtro_geografico, 'departamento'):
                geographic_filter = {
                    'departamento': filtro_geografico.departamento,
                    'provincia': getattr(filtro_geografico, 'provincia', None),
                    'distrito': getattr(filtro_geografico, 'distrito', None)
                }
                geographic_filter = {k: v for k, v in geographic_filter.items() if v is not None}

        result = self.analyzer.analyze_null_patterns(
            df,
            group_by=desagregado_por,
            geographic_filter=geographic_filter
        )

        if result['analysis_type'] == 'basic':
            summary = result['summary']
        else:
            summary = result['basic_analysis']['summary']

        desagregado = pd.DataFrame()
        if result.get('group_analysis') is not None:
            desagregado = result['group_analysis']

        legacy_result = {
            'resumen_total': summary,
            'desagregado': desagregado
        }

        if incluir_estadisticas:
            metrics = result['metrics']
            legacy_result['estadisticas'] = {
                'total_celdas': metrics.total_cells,
                'total_nulos': metrics.missing_cells,
                'porcentaje_nulos_global': metrics.missing_percentage,
                'variables_con_nulos': metrics.variables_with_missing,
                'variables_sin_nulos': metrics.variables_without_missing
            }

        return legacy_result


def diagnostico_nulos_enaho(df: pd.DataFrame,
                            desagregado_por: str = 'geo_departamento',
                            filtro_geografico=None,
                            columnas_geo: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Función de compatibilidad con la API anterior.
    DEPRECATED: Use quick_null_analysis() para nuevas implementaciones.
    """
    warnings.warn(
        "diagnostico_nulos_enaho está deprecated. Use quick_null_analysis() en su lugar.",
        DeprecationWarning,
        stacklevel=2
    )

    analyzer = LegacyNullAnalyzer()
    return analyzer.diagnostico_nulos_enaho(
        df, desagregado_por, True, filtro_geografico, columnas_geo
    )