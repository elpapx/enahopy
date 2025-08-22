"""
Configuración y enums para análisis de valores nulos
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MissingDataPattern(Enum):
    """Patrones de datos faltantes según Little & Rubin"""

    MCAR = "missing_completely_at_random"
    MAR = "missing_at_random"
    MNAR = "missing_not_at_random"
    UNKNOWN = "unknown_pattern"


class AnalysisComplexity(Enum):
    """Niveles de complejidad del análisis"""

    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    EXPERT = "expert"


class VisualizationType(Enum):
    """Tipos de visualización disponibles"""

    STATIC = "static"
    INTERACTIVE = "interactive"
    BOTH = "both"


class ExportFormat(Enum):
    """Formatos de exportación de reportes"""

    HTML = "html"
    PDF = "pdf"
    XLSX = "xlsx"
    JSON = "json"
    MARKDOWN = "md"


@dataclass
class NullAnalysisConfig:
    """Configuración avanzada para análisis de valores nulos"""

    # Configuración básica
    complexity_level: AnalysisComplexity = AnalysisComplexity.STANDARD
    visualization_type: VisualizationType = VisualizationType.STATIC
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300

    # Configuración de análisis
    min_occurrence_threshold: float = 0.01
    correlation_threshold: float = 0.3
    cluster_threshold: float = 0.5

    # Configuración de rendimiento
    use_parallel_processing: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 10000
    use_sampling_for_large_datasets: bool = True
    max_sample_size: int = 100000

    # Configuración de visualización
    color_palette: str = "viridis"
    interactive_height: int = 600
    show_progress_bars: bool = True
    annotation_threshold: int = 20

    # Configuración de exportación
    export_formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.HTML])
    include_raw_data: bool = False
    compress_exports: bool = True

    # Configuración de cache
    enable_caching: bool = True
    cache_analysis_results: bool = True
    cache_visualizations: bool = False

    # Configuración estadística
    confidence_level: float = 0.95
    statistical_tests: bool = True
    multiple_testing_correction: str = "bonferroni"


@dataclass
class MissingDataMetrics:
    """Métricas completas de datos faltantes"""

    total_cells: int
    missing_cells: int
    missing_percentage: float
    complete_cases: int
    complete_cases_percentage: float
    variables_with_missing: int
    variables_without_missing: int
    missing_pattern_count: int
    most_common_pattern: str
    missing_data_pattern: MissingDataPattern

    # Métricas estadísticas avanzadas
    little_mcar_test_pvalue: Optional[float] = None
    monotone_missing: bool = False
    missing_clustering_score: Optional[float] = None
    temporal_pattern_detected: bool = False

    # Métricas de calidad
    data_quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convierte métricas a diccionario"""
        return {
            "basic_metrics": {
                "total_cells": self.total_cells,
                "missing_cells": self.missing_cells,
                "missing_percentage": self.missing_percentage,
                "complete_cases": self.complete_cases,
                "complete_cases_percentage": self.complete_cases_percentage,
            },
            "variable_metrics": {
                "variables_with_missing": self.variables_with_missing,
                "variables_without_missing": self.variables_without_missing,
            },
            "pattern_metrics": {
                "missing_pattern_count": self.missing_pattern_count,
                "most_common_pattern": self.most_common_pattern,
                "missing_data_pattern": self.missing_data_pattern.value,
                "monotone_missing": self.monotone_missing,
                "temporal_pattern_detected": self.temporal_pattern_detected,
            },
            "statistical_metrics": {
                "little_mcar_test_pvalue": self.little_mcar_test_pvalue,
                "missing_clustering_score": self.missing_clustering_score,
            },
            "quality_metrics": {
                "data_quality_score": self.data_quality_score,
                "completeness_score": self.completeness_score,
                "consistency_score": self.consistency_score,
            },
        }
