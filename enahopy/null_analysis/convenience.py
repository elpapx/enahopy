"""
Funciones de conveniencia para análisis de valores nulos
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..validation import validate_dataframe_not_empty
from .config import MissingDataPattern  # Agregar si se usa en detect_missing_patterns_automatically
from .config import AnalysisComplexity, ExportFormat, NullAnalysisConfig, VisualizationType


def quick_null_analysis(
    df: pd.DataFrame, group_by: Optional[str] = None, complexity: str = "standard"
) -> Dict[str, Any]:
    """Análisis rápido de nulos con validación mejorada."""
    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F821

    # Validar y convertir complexity de forma segura
    valid_complexities = {c.value for c in AnalysisComplexity}
    if complexity not in valid_complexities:
        raise ValueError(f"Complejidad '{complexity}' no válida. Use: {valid_complexities}")

    complexity_enum = AnalysisComplexity(complexity)

    config = NullAnalysisConfig(
        complexity_level=complexity_enum,  # Usar la variable ya validada
        visualization_type=VisualizationType.STATIC,
    )

    analyzer = ENAHONullAnalyzer(config=config, verbose=False)
    return analyzer.analyze_null_patterns(df, group_by=group_by)


def get_data_quality_score(
    df: pd.DataFrame, detailed: bool = False
) -> Union[float, Dict[str, Any]]:
    """
    Función de conveniencia para obtener score rápido de calidad de datos.
    """
    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F821

    analyzer = ENAHONullAnalyzer(verbose=False)
    return analyzer.get_data_quality_score(df, detailed=detailed)


def create_null_visualizations(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    group_by: Optional[str] = None,
    interactive: bool = False,
) -> Dict[str, Any]:
    """
    Función de conveniencia para crear visualizaciones de nulos.
    """
    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F811

    # Usar enum correcto
    viz_type = VisualizationType.INTERACTIVE if interactive else VisualizationType.STATIC
    config = NullAnalysisConfig(visualization_type=viz_type)

    analyzer = ENAHONullAnalyzer(config=config, verbose=False)

    analysis_result = analyzer.analyze_null_patterns(df, group_by=group_by)

    return analyzer.create_visualizations(
        analysis_result, save_path=output_path, show_plots=output_path is None
    )


def generate_null_report(
    df: pd.DataFrame,
    output_path: str,
    group_by: Optional[str] = None,
    geographic_filter: Optional[Dict[str, str]] = None,
    format_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Genera reporte completo de análisis de valores nulos.

    Args:
        df: Dataframe a analizar.
        output_path: Ruta donde guardar el reporte.
        group_by: Variable a agrupar. opcional
        geographic_filter: Filtros geográficos. opcional
        format_types: Formato de salida.

    Returns:
        Dict con rutas de archivos generados y métricas.

    """
    if format_types is None:
        format_types = ["html", "json"]

    # validar formatos

    valids_formats = ["html", "json", "xlsx", "md"]
    invalid_formats = set(format_types) - set(valids_formats)
    if invalid_formats:
        raise ValueError(
            f"Formatos inválidos: {invalid_formats}. Formatos permitidos: {valids_formats}"
        )

    # Convertir strings a enums
    export_formats = []
    for fmt in format_types:
        try:
            export_formats.append(ExportFormat(fmt))
        except ValueError:
            print(f"formato '{fmt}' omitdo - no válido")
            continue
    if not export_formats:
        export_formats = [ExportFormat("html"), ExportFormat("json")]

    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F811

    config = NullAnalysisConfig(
        export_formats=export_formats, complexity_level=AnalysisComplexity.ADVANCED
    )

    analyzer = ENAHONullAnalyzer(config=config)

    return analyzer.generate_comprehensive_report(
        df=df, output_path=output_path, group_by=group_by, geographic_filter=geographic_filter
    )


def compare_null_patterns(
    datasets: Dict[str, pd.DataFrame], group_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compara patrones de nulos entre múltiples datasets.

    Args:
        datasets: Diccionario {nombre: DataFrame}
        group_by: Variable de agrupación opcional

    Returns:
        Comparación detallada entre datasets
    """
    if not datasets:
        raise ValueError("Se requiere al menos un dataset para comparar")

    if len(datasets) == 1:
        raise ValueError("Se requieren al menos 2 datasets para comparar")

    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F811

    analyzer = ENAHONullAnalyzer(verbose=False)

    results = {}
    metrics_comparison = {}

    # Analizar cada dataset
    for name, df in datasets.items():
        try:
            validate_dataframe_not_empty(df, name=f"Dataset '{name}'")
        except Exception as e:
            results[name] = {"error": str(e)}
            continue

        analysis = analyzer.analyze_null_patterns(df, group_by=group_by)
        results[name] = analysis

        # Extraer métricas para comparación
        if "metrics" in analysis:
            metrics_comparison[name] = {
                "missing_percentage": analysis["metrics"].missing_percentage,
                "complete_cases_percentage": analysis["metrics"].complete_cases_percentage,
                "variables_with_missing": analysis["metrics"].variables_with_missing,
                "data_quality_score": analysis["metrics"].data_quality_score,
            }

    # Calcular diferencias
    if len(metrics_comparison) >= 2:
        names = list(metrics_comparison.keys())
        base_name = names[0]
        base_metrics = metrics_comparison[base_name]

        differences = {}
        for name in names[1:]:
            diff = {}
            for metric_key in base_metrics:
                if metric_key in metrics_comparison[name]:
                    diff[metric_key] = (
                        metrics_comparison[name][metric_key] - base_metrics[metric_key]
                    )
            differences[f"{base_name}_vs_{name}"] = diff
    else:
        differences = {}

    return {
        "individual_analyses": results,
        "metrics_comparison": metrics_comparison,
        "differences": differences,
        "best_quality_dataset": (
            max(metrics_comparison.items(), key=lambda x: x[1].get("data_quality_score", 0))[0]
            if metrics_comparison
            else None
        ),
    }


def suggest_imputation_methods(df: pd.DataFrame, variable: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para sugerir métodos de imputación.
    """
    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F811

    config = NullAnalysisConfig(complexity_level=AnalysisComplexity.ADVANCED)
    analyzer = ENAHONullAnalyzer(config=config, verbose=False)

    analysis_result = analyzer.analyze_null_patterns(df)

    return analyzer.suggest_imputation_strategy(analysis_result, variable=variable)


def validate_data_completeness(
    df: pd.DataFrame,
    required_completeness: float = 80.0,
    required_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Valida si un DataFrame cumple con requisitos de completitud.
    """
    try:
        validate_dataframe_not_empty(df, name="DataFrame de completitud")
    except Exception as e:
        return {
            "is_valid": False,
            "reason": str(e),
            "completeness_score": 0.0,
            "missing_variables": required_variables or [],
            "recommendations": ["Cargar datos antes de validar"],
        }

    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F811

    analyzer = ENAHONullAnalyzer(verbose=False)

    # Calcular métricas básicas
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0

    validation_result = {
        "is_valid": True,
        "completeness_score": completeness,
        "required_completeness": required_completeness,
        "missing_variables": [],
        "variables_below_threshold": {},
        "recommendations": [],
    }

    # Validar completitud global
    if completeness < required_completeness:
        validation_result["is_valid"] = False
        validation_result["recommendations"].append(
            f"Completitud global ({completeness:.1f}%) por debajo del umbral ({required_completeness}%)"
        )

    # Validar variables requeridas
    if required_variables:
        missing_vars = [var for var in required_variables if var not in df.columns]
        if missing_vars:
            validation_result["is_valid"] = False
            validation_result["missing_variables"] = missing_vars
            validation_result["recommendations"].append(
                f"Variables requeridas faltantes: {', '.join(missing_vars)}"
            )

        # Verificar completitud por variable
        for var in required_variables:
            if var in df.columns:
                var_completeness = (1 - df[var].isnull().mean()) * 100
                if var_completeness < required_completeness:
                    validation_result["variables_below_threshold"][var] = var_completeness
                    validation_result["recommendations"].append(
                        f"Variable '{var}' tiene {var_completeness:.1f}% de completitud"
                    )

    # Agregar score de calidad
    validation_result["data_quality_score"] = analyzer.get_data_quality_score(df)

    return validation_result


def analyze_common_missing_patterns(
    df: pd.DataFrame, min_pattern_frequency: int = 10
) -> Dict[str, Any]:
    """
    Analiza patrones comunes de datos faltantes por frecuencia.

    Args:
        df: DataFrame a analizar
        min_pattern_frequency: Frecuencia mínima para considerar un patrón común

    Returns:
        Dict con análisis de patrones comunes
    """

    missing_matrix = df.isnull()

    pattern_strings = missing_matrix.apply(
        lambda row: "".join(["1" if x else "0" for x in row]), axis=1
    )

    pattern_counts = pattern_strings.value_counts()
    common_patterns = pattern_counts[pattern_counts >= min_pattern_frequency]

    results = {
        "total_unique_patterns": len(pattern_counts),
        "common_patterns": len(common_patterns),
        "pattern_analysis": {},
        "interpretations": [],
    }

    column_names = df.columns.tolist()

    for pattern, count in common_patterns.head(10).items():
        pattern_info = {
            "frequency": count,
            "percentage": (count / len(df)) * 100,
            "missing_variables": [column_names[i] for i, bit in enumerate(pattern) if bit == "1"],
            "complete_variables": [column_names[i] for i, bit in enumerate(pattern) if bit == "0"],
        }

        results["pattern_analysis"][pattern] = pattern_info

        if pattern == "0" * len(column_names):
            results["interpretations"].append(
                f"✅ {count} casos ({pattern_info['percentage']:.1f}%) completamente sin faltantes"
            )
        elif pattern.count("1") == len(column_names):
            results["interpretations"].append(
                f"❌ {count} casos ({pattern_info['percentage']:.1f}%) completamente faltantes"
            )
        elif pattern.count("1") == 1:
            missing_var = pattern_info["missing_variables"][0]
            results["interpretations"].append(
                f"🔍 {count} casos ({pattern_info['percentage']:.1f}%) "
                f"solo faltan en variable '{missing_var}'"
            )
        elif len(pattern_info["missing_variables"]) <= 3:
            missing_vars = ", ".join(pattern_info["missing_variables"])
            results["interpretations"].append(
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
            stacklevel=2,
        )

        # Lazy import to avoid circular dependencies
        from . import ENAHONullAnalyzer  # noqa: F811

        self.analyzer = ENAHONullAnalyzer(config=config, verbose=False)

    def diagnostico_nulos_enaho(
        self,
        df: pd.DataFrame,
        desagregado_por: str = "geo_departamento",
        incluir_estadisticas: bool = True,
        filtro_geografico=None,
        columnas_geo: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Metodo de compatibilidad con API anterior"""

        geographic_filter = None
        if filtro_geografico:
            if hasattr(filtro_geografico, "departamento"):
                geographic_filter = {
                    "departamento": filtro_geografico.departamento,
                    "provincia": getattr(filtro_geografico, "provincia", None),
                    "distrito": getattr(filtro_geografico, "distrito", None),
                }
                geographic_filter = {k: v for k, v in geographic_filter.items() if v is not None}

        result = self.analyzer.analyze_null_patterns(
            df, group_by=desagregado_por, geographic_filter=geographic_filter
        )

        if result["analysis_type"] == "basic":
            summary = result["summary"]
        else:
            summary = result["basic_analysis"]["summary"]

        desagregado = pd.DataFrame()
        if result.get("group_analysis") is not None:
            desagregado = result["group_analysis"]

        legacy_result = {"resumen_total": summary, "desagregado": desagregado}

        if incluir_estadisticas:
            metrics = result["metrics"]
            legacy_result["estadisticas"] = {
                "total_celdas": metrics.total_cells,
                "total_nulos": metrics.missing_cells,
                "porcentaje_nulos_global": metrics.missing_percentage,
                "variables_con_nulos": metrics.variables_with_missing,
                "variables_sin_nulos": metrics.variables_without_missing,
            }

        return legacy_result


def diagnostico_nulos_enaho(
    df: pd.DataFrame,
    desagregado_por: str = "geo_departamento",
    filtro_geografico=None,
    columnas_geo: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Función de compatibilidad con la API anterior.
    DEPRECATED: Use quick_null_analysis() para nuevas implementaciones.
    """
    warnings.warn(
        "diagnostico_nulos_enaho está deprecated. Use quick_null_analysis() en su lugar.",
        DeprecationWarning,
        stacklevel=2,
    )

    analyzer = LegacyNullAnalyzer()
    return analyzer.diagnostico_nulos_enaho(
        df, desagregado_por, True, filtro_geografico, columnas_geo
    )


def detect_missing_patterns_automatically(
    df: pd.DataFrame, confidence_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Detecta automáticamente el tipo de patrón de valores faltantes.

    Args:
        df: DataFrame a analizar
        confidence_threshold: Umbral de confianza para clasificación

    Returns:
        Dict con patrón detectado y nivel de confianza
    """
    # Lazy import to avoid circular dependencies
    from . import ENAHONullAnalyzer  # noqa: F811

    analyzer = ENAHONullAnalyzer(
        config=NullAnalysisConfig(complexity_level=AnalysisComplexity.EXPERT), verbose=False
    )

    result = analyzer.analyze_null_patterns(df)

    # Extraer información de patrones
    pattern_info = {
        "detected_pattern": result["metrics"].missing_data_pattern,
        "confidence": 0.0,
        "evidence": [],
        "alternative_patterns": [],
    }

    # Calcular confianza basada en tests estadísticos
    if "statistical_tests" in result:
        mcar_test = result["statistical_tests"].get("simplified_mcar_test", {})
        if mcar_test.get("p_value"):
            if mcar_test["p_value"] > 0.05:
                pattern_info["confidence"] = min(0.95, 1 - mcar_test["p_value"])
                pattern_info["evidence"].append(f"Test MCAR p-value: {mcar_test['p_value']:.4f}")

    # Agregar patrones alternativos si la confianza es baja
    if pattern_info["confidence"] < confidence_threshold:
        pattern_info["alternative_patterns"] = [
            p for p in MissingDataPattern if p != pattern_info["detected_pattern"]
        ]
        pattern_info["recommendation"] = "Considere análisis adicional para confirmar el patrón"

    return pattern_info
