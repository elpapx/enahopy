"""Visualizador de patrones de valores nulos"""

import warnings
from enum import Enum
from typing import Any, Optional, Tuple

import pandas as pd


class VisualizationType(Enum):
    """Tipos de visualización disponibles"""

    MATRIX = "matrix"
    BAR = "bar"
    HEATMAP = "heatmap"
    PATTERN = "pattern"


class NullVisualizer:
    """Visualizador de patrones de valores nulos"""

    def __init__(self, style: str = "seaborn"):
        self.style = style
        self.fig_size = (12, 8)

    def visualize_null_matrix(
        self,
        df: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Matriz de Valores Nulos",
    ) -> Optional[Any]:
        """Visualiza matriz de valores nulos (placeholder)"""
        warnings.warn("Visualización requiere matplotlib/seaborn")
        return None

    def visualize_null_bars(
        self,
        df: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Valores Nulos por Columna",
    ) -> Optional[Any]:
        """Visualiza barras de valores nulos (placeholder)"""
        warnings.warn("Visualización requiere matplotlib/seaborn")
        return None
