"""
Exportadores de reportes para análisis de nulos
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from ..config import ExportFormat, NullAnalysisConfig


class ReportExporter:
    """Exportador de reportes en múltiples formatos"""

    def __init__(self, config: NullAnalysisConfig, logger):
        self.config = config
        self.logger = logger

    def export_report(
        self,
        analysis_result: Dict[str, Any],
        visualizations: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> Dict[str, str]:
        """Exporta reporte completo en formatos especificados"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        for format_type in self.config.export_formats:
            try:
                if format_type == ExportFormat.HTML:
                    file_path = self._export_html_report(
                        analysis_result, visualizations, output_path
                    )
                elif format_type == ExportFormat.JSON:
                    file_path = self._export_json_report(analysis_result, output_path)
                elif format_type == ExportFormat.XLSX:
                    file_path = self._export_excel_report(analysis_result, output_path)
                elif format_type == ExportFormat.MARKDOWN:
                    file_path = self._export_markdown_report(analysis_result, output_path)
                else:
                    continue

                exported_files[format_type.value] = str(file_path)
                self.logger.info(f"Reporte exportado: {file_path}")

            except Exception as e:
                self.logger.error(f"Error exportando {format_type.value}: {str(e)}")
                exported_files[f"{format_type.value}_error"] = str(e)

        return exported_files

    def _export_html_report(
        self, analysis_result: Dict[str, Any], visualizations: Dict[str, Any], output_path: Path
    ) -> Path:
        """Exporta reporte HTML completo"""

        html_content = self._generate_html_content(analysis_result, visualizations)

        file_path = output_path / "null_analysis_report.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return file_path

    def _export_json_report(self, analysis_result: Dict[str, Any], output_path: Path) -> Path:
        """Exporta reporte JSON"""

        json_data = self._prepare_for_json_export(analysis_result)

        file_path = output_path / "null_analysis_report.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        return file_path

    def _export_excel_report(self, analysis_result: Dict[str, Any], output_path: Path) -> Path:
        """Exporta reporte Excel con múltiples hojas"""

        file_path = output_path / "null_analysis_report.xlsx"

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            if analysis_result["analysis_type"] == "basic":
                summary = analysis_result["summary"]
            else:
                summary = analysis_result["basic_analysis"]["summary"]

            summary.to_excel(writer, sheet_name="Resumen_Variables", index=False)

            metrics_df = pd.DataFrame([analysis_result["metrics"].to_dict()])
            metrics_df.to_excel(writer, sheet_name="Metricas_Generales", index=False)

            if analysis_result["analysis_type"] == "advanced":
                if "correlations" in analysis_result:
                    corr_df = analysis_result["correlations"]["correlation_matrix"]
                    corr_df.to_excel(writer, sheet_name="Correlaciones_Missing")

                if analysis_result.get("patterns", {}).get("most_common_patterns"):
                    patterns_df = pd.DataFrame(
                        list(analysis_result["patterns"]["most_common_patterns"].items()),
                        columns=["Patron", "Frecuencia"],
                    )
                    patterns_df.to_excel(writer, sheet_name="Patrones_Comunes", index=False)

        return file_path

    def _export_markdown_report(self, analysis_result: Dict[str, Any], output_path: Path) -> Path:
        """Exporta reporte Markdown"""

        markdown_content = self._generate_markdown_content(analysis_result)

        file_path = output_path / "null_analysis_report.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return file_path

    def _generate_html_content(
        self, analysis_result: Dict[str, Any], visualizations: Dict[str, Any]
    ) -> str:
        """Genera contenido HTML del reporte"""

        html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Análisis de Valores Nulos - INEI</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #3498db; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .recommendation {{ background: #d5f4e6; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Reporte de Análisis de Valores Nulos</h1>
        <p><strong>Fecha de generación:</strong> {fecha}</p>

        <h2>🎯 Resumen Ejecutivo</h2>
        {resumen_metricas}

        <h2>📈 Análisis Detallado</h2>
        {tabla_resumen}

        <h2>💡 Recomendaciones</h2>
        {recomendaciones}

        <h2>🔧 Detalles Técnicos</h2>
        {detalles_tecnicos}
    </div>
</body>
</html>
        """

        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics = analysis_result["metrics"]

        resumen_metricas = f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.missing_percentage:.1f}%</div>
            <div>Porcentaje total de valores faltantes</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.complete_cases_percentage:.1f}%</div>
            <div>Casos completos</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.data_quality_score:.1f}</div>
            <div>Score de calidad de datos</div>
        </div>
        """

        if analysis_result["analysis_type"] == "basic":
            summary = analysis_result["summary"]
        else:
            summary = analysis_result["basic_analysis"]["summary"]

        tabla_html = summary.head(10).to_html(classes="table", escape=False)

        recommendations = []
        if analysis_result["analysis_type"] == "advanced":
            from ..strategies.advanced_analysis import AdvancedNullAnalysis

            recommendations = AdvancedNullAnalysis(self.config, self.logger).get_recommendations(
                analysis_result
            )
        else:
            from ..strategies.basic_analysis import BasicNullAnalysis

            recommendations = BasicNullAnalysis(self.config, self.logger).get_recommendations(
                analysis_result
            )

        recomendaciones_html = "\n".join(
            [f'<div class="recommendation">{rec}</div>' for rec in recommendations]
        )

        detalles = f"""
        <ul>
            <li><strong>Tipo de análisis:</strong> {analysis_result['analysis_type'].title()}</li>
            <li><strong>Variables analizadas:</strong> {len(summary)}</li>
            <li><strong>Patrón de missing data:</strong> {metrics.missing_data_pattern.value}</li>
            <li><strong>Tiempo de ejecución:</strong> {analysis_result.get('execution_time', 0):.2f} segundos</li>
        </ul>
        """

        return html_template.format(
            fecha=fecha,
            resumen_metricas=resumen_metricas,
            tabla_resumen=tabla_html,
            recomendaciones=recomendaciones_html,
            detalles_tecnicos=detalles,
        )

    def _generate_markdown_content(self, analysis_result: Dict[str, Any]) -> str:
        """Genera contenido Markdown del reporte"""

        metrics = analysis_result["metrics"]
        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        markdown_content = f"""# 📊 Reporte de Análisis de Valores Nulos

**Fecha de generación:** {fecha}

## 🎯 Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Porcentaje total de faltantes | {metrics.missing_percentage:.1f}% |
| Casos completos | {metrics.complete_cases_percentage:.1f}% |
| Variables con faltantes | {metrics.variables_with_missing} |
| Variables sin faltantes | {metrics.variables_without_missing} |
| Score de calidad | {metrics.data_quality_score:.1f} |
| Patrón de missing data | {metrics.missing_data_pattern.value} |

## 📈 Variables con Más Valores Faltantes
"""

        if analysis_result["analysis_type"] == "basic":
            summary = analysis_result["summary"]
        else:
            summary = analysis_result["basic_analysis"]["summary"]

        top_missing = summary.head(10)
        for _, row in top_missing.iterrows():
            markdown_content += f"- **{row['variable']}**: {row['missing_percentage']:.1f}% ({row['missing_count']} valores)\n"

        markdown_content += "\n## 💡 Recomendaciones\n\n"

        recommendations = []
        if analysis_result["analysis_type"] == "advanced":
            from ..strategies.advanced_analysis import AdvancedNullAnalysis

            recommendations = AdvancedNullAnalysis(self.config, self.logger).get_recommendations(
                analysis_result
            )
        else:
            from ..strategies.basic_analysis import BasicNullAnalysis

            recommendations = BasicNullAnalysis(self.config, self.logger).get_recommendations(
                analysis_result
            )

        for rec in recommendations:
            markdown_content += f"- {rec}\n"

        markdown_content += f"""
## 🔧 Detalles Técnicos

- **Tipo de análisis:** {analysis_result['analysis_type'].title()}
- **Variables analizadas:** {len(summary)}
- **Tiempo de ejecución:** {analysis_result.get('execution_time', 0):.2f} segundos
- **Configuración:** {self.config.complexity_level.value}
"""

        return markdown_content

    def _prepare_for_json_export(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara datos para exportación JSON"""
        import copy

        clean_result = copy.deepcopy(analysis_result)

        if hasattr(clean_result["metrics"], "to_dict"):
            clean_result["metrics"] = clean_result["metrics"].to_dict()

        def clean_dataframes(obj):
            if isinstance(obj, dict):
                return {k: clean_dataframes(v) for k, v in obj.items()}
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict("records")
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, list):
                return [clean_dataframes(item) for item in obj]
            else:
                return obj

        return clean_dataframes(clean_result)
