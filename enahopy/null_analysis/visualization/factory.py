"""
Factory para crear visualizaciones de análisis de nulos
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..config import NullAnalysisConfig, VisualizationType


class VisualizationFactory:
    """Factory para crear diferentes tipos de visualizaciones"""

    def __init__(self, config: NullAnalysisConfig, logger):
        self.config = config
        self.logger = logger

    def create_basic_visualizations(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Crea visualizaciones básicas"""
        visualizations = {}

        if self.config.visualization_type in [VisualizationType.STATIC, VisualizationType.BOTH]:
            visualizations.update(self._create_static_visualizations(analysis_result))

        if (
            self.config.visualization_type
            in [VisualizationType.INTERACTIVE, VisualizationType.BOTH]
            and PLOTLY_AVAILABLE
        ):
            visualizations.update(self._create_interactive_visualizations(analysis_result))

        return visualizations

    def _create_static_visualizations(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Crea visualizaciones estáticas con matplotlib/seaborn"""
        visualizations = {}

        plt.style.use("default")
        sns.set_palette(self.config.color_palette)

        try:
            fig_bar = self._create_missing_bar_chart(analysis_result)
            visualizations["missing_bar_chart"] = fig_bar

            if analysis_result.get("patterns"):
                fig_heatmap = self._create_missing_heatmap(analysis_result)
                visualizations["missing_heatmap"] = fig_heatmap

            if analysis_result.get("correlations"):
                fig_corr = self._create_correlation_matrix(analysis_result)
                visualizations["correlation_matrix"] = fig_corr

        except Exception as e:
            self.logger.error(f"Error creando visualizaciones estáticas: {str(e)}")
            visualizations["error"] = str(e)

        return visualizations

    def _create_interactive_visualizations(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Crea visualizaciones interactivas con plotly"""
        visualizations = {}

        try:
            fig_interactive = self._create_interactive_missing_chart(analysis_result)
            visualizations["interactive_missing_chart"] = fig_interactive

            if analysis_result.get("patterns"):
                fig_sunburst = self._create_missing_patterns_sunburst(analysis_result)
                visualizations["patterns_sunburst"] = fig_sunburst

        except Exception as e:
            self.logger.error(f"Error creando visualizaciones interactivas: {str(e)}")
            visualizations["interactive_error"] = str(e)

        return visualizations

    def _create_missing_bar_chart(self, analysis_result: Dict[str, Any]) -> plt.Figure:
        """Crea gráfico de barras de valores faltantes"""
        if analysis_result["analysis_type"] == "basic":
            summary = analysis_result["summary"]
        else:
            summary = analysis_result["basic_analysis"]["summary"]

        summary_with_missing = summary[summary["missing_percentage"] > 0].head(20)

        if summary_with_missing.empty:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(
                0.5,
                0.5,
                "No hay valores faltantes en el dataset",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            ax.set_title("Análisis de Valores Faltantes")
            return fig

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1] * 1.2)
        )

        bars1 = ax1.barh(
            summary_with_missing["variable"],
            summary_with_missing["missing_percentage"],
            color=sns.color_palette(self.config.color_palette, len(summary_with_missing)),
        )

        ax1.set_xlabel("Porcentaje de Valores Faltantes (%)")
        ax1.set_title("Variables con Valores Faltantes (Top 20)", fontweight="bold", pad=20)
        ax1.grid(axis="x", alpha=0.3)

        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(
                width + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                ha="left",
                va="center",
                fontweight="bold",
            )

        bars2 = ax2.barh(
            summary_with_missing["variable"],
            summary_with_missing["missing_count"],
            color=sns.color_palette(self.config.color_palette, len(summary_with_missing)),
        )

        ax2.set_xlabel("Cantidad de Valores Faltantes")
        ax2.set_title("Conteo Absoluto de Valores Faltantes")
        ax2.grid(axis="x", alpha=0.3)

        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(
                width + len(str(int(width))) * 100,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                ha="left",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        return fig

    def _create_missing_heatmap(self, analysis_result: Dict[str, Any]) -> plt.Figure:
        """Crea heatmap de patrones de missing data"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        if "correlations" in analysis_result:
            corr_matrix = analysis_result["correlations"]["correlation_matrix"]

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap=self.config.color_palette,
                center=0,
                square=True,
                ax=ax,
            )

            ax.set_title(
                "Correlación entre Patrones de Valores Faltantes", fontweight="bold", pad=20
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Correlaciones no disponibles",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Heatmap de Patrones de Missing Data")

        plt.tight_layout()
        return fig

    def _create_correlation_matrix(self, analysis_result: Dict[str, Any]) -> plt.Figure:
        """Crea matriz de correlaciones específica"""
        correlations = analysis_result.get("correlations", {})

        if "correlation_matrix" not in correlations:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(
                0.5,
                0.5,
                "Matriz de correlaciones no disponible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        corr_matrix = correlations["correlation_matrix"]

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            ax=ax,
        )

        ax.set_title("Correlaciones entre Patrones de Valores Faltantes", fontweight="bold", pad=20)

        plt.tight_layout()
        return fig

    def _create_interactive_missing_chart(self, analysis_result: Dict[str, Any]) -> go.Figure:
        """Crea gráfico interactivo de missing values"""
        if analysis_result["analysis_type"] == "basic":
            summary = analysis_result["summary"]
        else:
            summary = analysis_result["basic_analysis"]["summary"]

        summary_with_missing = summary[summary["missing_percentage"] > 0].head(20)

        if summary_with_missing.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No hay valores faltantes en el dataset",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(title="Análisis de Valores Faltantes")
            return fig

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Porcentaje de Valores Faltantes",
                "Conteo Absoluto de Valores Faltantes",
            ),
            vertical_spacing=0.15,
        )

        fig.add_trace(
            go.Bar(
                x=summary_with_missing["missing_percentage"],
                y=summary_with_missing["variable"],
                orientation="h",
                name="Porcentaje",
                text=[f"{x:.1f}%" for x in summary_with_missing["missing_percentage"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Faltantes: %{x:.1f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=summary_with_missing["missing_count"],
                y=summary_with_missing["variable"],
                orientation="h",
                name="Conteo",
                text=[f"{x:,}" for x in summary_with_missing["missing_count"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Faltantes: %{x:,}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=self.config.interactive_height * 1.5,
            title_text="Análisis Interactivo de Valores Faltantes",
            showlegend=False,
        )

        fig.update_xaxes(title_text="Porcentaje (%)", row=1, col=1)
        fig.update_xaxes(title_text="Cantidad", row=2, col=1)

        return fig

    def _create_missing_patterns_sunburst(self, analysis_result: Dict[str, Any]) -> go.Figure:
        """Crea gráfico sunburst de patrones de missing data"""
        patterns = analysis_result.get("patterns", {})
        most_common = patterns.get("most_common_patterns", {})

        if not most_common:
            fig = go.Figure()
            fig.add_annotation(
                text="No hay patrones de missing data disponibles",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        labels = list(most_common.keys())[:10]
        values = list(most_common.values())[:10]
        parents = [""] * len(labels)

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate="<b>Patrón:</b> %{label}<br><b>Frecuencia:</b> %{value}<extra></extra>",
                maxdepth=2,
            )
        )

        fig.update_layout(
            title="Patrones Más Comunes de Valores Faltantes", height=self.config.interactive_height
        )

        return fig
