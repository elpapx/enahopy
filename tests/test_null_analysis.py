"""
Test Suite Completa - Módulo Null Analysis
===========================================
Tests unitarios e integración para el módulo de análisis de valores nulos.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Importaciones del módulo a testear
from enahopy.null_analysis import (
    ENAHONullAnalyzer,
    analyze_null_patterns,
    generate_null_report,
    NullAnalysisConfig,
    NullAnalysisError,
    PatternDetector,
    NullPatternAnalyzer,
    MissingDataPattern,
    PatternType,
    PatternSeverity,
    ReportGenerator,
    NullAnalysisReport,
    NullVisualizer
)


class TestNullAnalysisConfig(unittest.TestCase):
    """Tests unitarios para configuración de análisis de nulos"""

    def test_default_configuration(self):
        """Verifica configuración por defecto"""
        config = NullAnalysisConfig()

        self.assertEqual(config.complexity_level, 'standard')
        self.assertTrue(config.generate_visualizations)
        self.assertTrue(config.detect_patterns)
        self.assertEqual(config.null_threshold, 0.05)
        self.assertEqual(config.correlation_threshold, 0.3)

    def test_custom_configuration(self):
        """Verifica configuración personalizada"""
        config = NullAnalysisConfig(
            complexity_level='advanced',
            generate_visualizations=False,
            detect_patterns=True,
            null_threshold=0.1,
            correlation_threshold=0.5,
            imputation_strategy='mean',
            report_format='html'
        )

        self.assertEqual(config.complexity_level, 'advanced')
        self.assertFalse(config.generate_visualizations)
        self.assertTrue(config.detect_patterns)
        self.assertEqual(config.null_threshold, 0.1)
        self.assertEqual(config.correlation_threshold, 0.5)
        self.assertEqual(config.imputation_strategy, 'mean')
        self.assertEqual(config.report_format, 'html')

    def test_complexity_levels(self):
        """Verifica niveles de complejidad"""
        levels = ['basic', 'standard', 'advanced', 'expert']

        for level in levels:
            config = NullAnalysisConfig(complexity_level=level)
            self.assertEqual(config.complexity_level, level)

            # Verificar que niveles más altos tienen más características
            if level == 'expert':
                self.assertTrue(config.detect_patterns)
                self.assertTrue(config.generate_visualizations)


class TestPatternDetection(unittest.TestCase):
    """Tests unitarios para detección de patrones de nulos"""

    def setUp(self):
        """Crear datasets con diferentes patrones de nulos"""
        np.random.seed(42)
        n = 1000

        # MCAR - Missing Completely At Random
        self.df_mcar = pd.DataFrame({
            'col1': np.random.randn(n),
            'col2': np.random.randn(n),
            'col3': np.random.randn(n)
        })
        # Insertar nulos aleatoriamente
        for col in self.df_mcar.columns:
            mask = np.random.random(n) < 0.1
            self.df_mcar.loc[mask, col] = np.nan

        # MAR - Missing At Random
        self.df_mar = pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'income': np.random.normal(50000, 20000, n),
            'education': np.random.choice(['High School', 'Bachelor', 'Master'], n)
        })
        # Income missing depends on age
        mask = self.df_mar['age'] < 30
        self.df_mar.loc[mask, 'income'] = np.nan

        # MNAR - Missing Not At Random
        self.df_mnar = pd.DataFrame({
            'salary': np.random.normal(60000, 25000, n),
            'bonus': np.random.normal(10000, 5000, n)
        })
        # High salaries tend to not report bonus
        mask = self.df_mnar['salary'] > 80000
        self.df_mnar.loc[mask[:len(mask) // 2], 'bonus'] = np.nan

        # Monotone pattern
        self.df_monotone = pd.DataFrame({
            'survey_q1': np.random.randn(n),
            'survey_q2': np.random.randn(n),
            'survey_q3': np.random.randn(n),
            'survey_q4': np.random.randn(n)
        })
        # Dropout pattern
        dropout_point = np.random.randint(1, 5, n)
        for i, point in enumerate(dropout_point):
            for j in range(point, 4):
                self.df_monotone.iloc[i, j] = np.nan

    def test_detect_mcar_pattern(self):
        """Verifica detección de patrón MCAR"""
        detector = PatternDetector()
        pattern = detector.detect_pattern(self.df_mcar)

        self.assertEqual(pattern.type, MissingDataPattern.MCAR)
        self.assertGreater(pattern.confidence, 0.7)

    def test_detect_mar_pattern(self):
        """Verifica detección de patrón MAR"""
        detector = PatternDetector()
        pattern = detector.detect_pattern(self.df_mar)

        self.assertEqual(pattern.type, MissingDataPattern.MAR)
        self.assertGreater(pattern.confidence, 0.6)

    def test_detect_mnar_pattern(self):
        """Verifica detección de patrón MNAR"""
        detector = PatternDetector()
        pattern = detector.detect_pattern(self.df_mnar)

        self.assertIn(pattern.type, [MissingDataPattern.MNAR, MissingDataPattern.MAR])

    def test_detect_monotone_pattern(self):
        """Verifica detección de patrón monotónico"""
        detector = PatternDetector()
        pattern = detector.detect_pattern(self.df_monotone)

        self.assertTrue(pattern.is_monotone)
        self.assertGreater(pattern.monotone_confidence, 0.7)

    def test_pattern_severity(self):
        """Verifica clasificación de severidad"""
        detector = PatternDetector()

        # Baja severidad - pocos nulos
        df_low = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, np.nan],
            'col2': [1, 2, 3, 4, 5, 6]
        })
        pattern_low = detector.detect_pattern(df_low)
        self.assertEqual(pattern_low.severity, PatternSeverity.LOW)

        # Alta severidad - muchos nulos
        df_high = pd.DataFrame({
            'col1': [np.nan] * 8 + [1, 2],
            'col2': [np.nan] * 7 + [1, 2, 3]
        })
        pattern_high = detector.detect_pattern(df_high)
        self.assertEqual(pattern_high.severity, PatternSeverity.HIGH)


class TestENAHONullAnalyzer(unittest.TestCase):
    """Tests unitarios para el analizador principal de nulos"""

    def setUp(self):
        """Crear datos de prueba tipo ENAHO"""
        np.random.seed(42)
        n = 500

        self.df_enaho = pd.DataFrame({
            'conglome': [f"{i:06d}" for i in range(n)],
            'vivienda': np.random.choice(['01', '02', '03'], n),
            'hogar': ['1'] * n,
            'factor07': np.random.uniform(0.5, 3, n),
            'ingreso': np.random.normal(3000, 1500, n),
            'gasto': np.random.normal(2500, 1000, n),
            'educacion': np.random.choice([1, 2, 3, 4, np.nan], n, p=[0.2, 0.3, 0.3, 0.15, 0.05]),
            'salud': np.random.choice([0, 1, np.nan], n, p=[0.6, 0.35, 0.05]),
            'edad': np.random.randint(0, 100, n),
            'sexo': np.random.choice([1, 2], n)
        })

        # Agregar nulos con patrón
        mask_ingreso = self.df_enaho['edad'] < 18
        self.df_enaho.loc[mask_ingreso, 'ingreso'] = np.nan

        mask_gasto = np.random.random(n) < 0.1
        self.df_enaho.loc[mask_gasto, 'gasto'] = np.nan

    def test_basic_analysis(self):
        """Verifica análisis básico de nulos"""
        analyzer = ENAHONullAnalyzer(complexity='basic')
        result = analyzer.analyze(self.df_enaho)

        self.assertIn('summary', result)
        self.assertIn('total_nulls', result['summary'])
        self.assertIn('null_percentage', result['summary'])
        self.assertIn('columns_with_nulls', result['summary'])

        # Verificar estadísticas por columna
        self.assertIn('column_stats', result)
        self.assertGreater(len(result['column_stats']), 0)

        for col_stat in result['column_stats']:
            self.assertIn('column', col_stat)
            self.assertIn('null_count', col_stat)
            self.assertIn('null_percentage', col_stat)

    def test_standard_analysis(self):
        """Verifica análisis estándar con correlaciones"""
        analyzer = ENAHONullAnalyzer(complexity='standard')
        result = analyzer.analyze(self.df_enaho)

        # Debe incluir todo lo del análisis básico
        self.assertIn('summary', result)
        self.assertIn('column_stats', result)

        # Más correlación de nulos
        self.assertIn('null_correlations', result)
        corr_matrix = result['null_correlations']
        self.assertIsInstance(corr_matrix, pd.DataFrame)

        # Verificar que la matriz es simétrica
        np.testing.assert_array_almost_equal(
            corr_matrix.values,
            corr_matrix.values.T
        )

    def test_advanced_analysis(self):
        """Verifica análisis avanzado con patrones"""
        analyzer = ENAHONullAnalyzer(complexity='advanced')
        result = analyzer.analyze(self.df_enaho)

        # Debe incluir detección de patrones
        self.assertIn('pattern_analysis', result)
        pattern_info = result['pattern_analysis']

        self.assertIn('detected_pattern', pattern_info)
        self.assertIn('confidence', pattern_info)
        self.assertIn('evidence', pattern_info)

        # Debe incluir análisis de dependencias
        self.assertIn('dependency_analysis', result)

    def test_expert_analysis(self):
        """Verifica análisis experto completo"""
        analyzer = ENAHONullAnalyzer(complexity='expert')
        result = analyzer.analyze(self.df_enaho)

        # Debe incluir todos los componentes
        expected_keys = [
            'summary', 'column_stats', 'null_correlations',
            'pattern_analysis', 'dependency_analysis',
            'imputation_recommendations', 'quality_metrics'
        ]

        for key in expected_keys:
            self.assertIn(key, result)

        # Verificar recomendaciones de imputación
        imputation = result['imputation_recommendations']
        self.assertIn('strategy', imputation)
        self.assertIn('columns', imputation)
        self.assertIn('rationale', imputation)

    def test_threshold_validation(self):
        """Verifica validación de umbrales de nulos"""
        # Crear DataFrame con muchos nulos
        df_high_nulls = self.df_enaho.copy()
        df_high_nulls.iloc[:, 3:] = np.nan  # Hacer nulas la mayoría de columnas

        analyzer = ENAHONullAnalyzer(
            complexity='standard',
            config={'null_threshold': 0.3}
        )

        result = analyzer.analyze(df_high_nulls)

        # Debe marcar columnas que exceden el umbral
        self.assertIn('columns_exceeding_threshold', result)
        self.assertGreater(len(result['columns_exceeding_threshold']), 0)

    def test_geographic_null_analysis(self):
        """Verifica análisis de nulos por geografía"""
        # Agregar columna geográfica
        self.df_enaho['departamento'] = np.random.choice(
            ['Lima', 'Cusco', 'Arequipa'],
            len(self.df_enaho)
        )

        analyzer = ENAHONullAnalyzer(complexity='advanced')
        result = analyzer.analyze_by_geography(
            self.df_enaho,
            geo_column='departamento'
        )

        self.assertIn('Lima', result)
        self.assertIn('Cusco', result)
        self.assertIn('Arequipa', result)

        # Verificar estructura de resultados por departamento
        for dept, dept_result in result.items():
            self.assertIn('null_percentage', dept_result)
            self.assertIn('most_missing', dept_result)


class TestNullVisualization(unittest.TestCase):
    """Tests unitarios para visualización de nulos"""

    def setUp(self):
        """Crear datos y configurar matplotlib"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, 2, 3, np.nan, 5],
            'col3': [1, 2, 3, 4, 5],
            'col4': [np.nan, np.nan, 3, 4, 5]
        })

        self.visualizer = NullVisualizer()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir)
        plt.close('all')

    def test_create_heatmap(self):
        """Verifica creación de heatmap de nulos"""
        fig = self.visualizer.create_heatmap(self.df)

        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

        # Guardar figura
        output_path = Path(self.temp_dir) / "heatmap.png"
        fig.savefig(output_path)
        self.assertTrue(output_path.exists())

    def test_create_bar_chart(self):
        """Verifica creación de gráfico de barras"""
        fig = self.visualizer.create_bar_chart(self.df)

        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

        # Verificar que tiene el número correcto de barras
        ax = fig.get_axes()[0]
        bars = ax.patches
        self.assertEqual(len(bars), len(self.df.columns))

    def test_create_pattern_matrix(self):
        """Verifica creación de matriz de patrones"""
        fig = self.visualizer.create_pattern_matrix(self.df)

        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_create_correlation_plot(self):
        """Verifica creación de gráfico de correlación de nulos"""
        # Crear DataFrame más grande para correlaciones
        df_large = pd.DataFrame(
            np.random.randn(100, 5),
            columns=['A', 'B', 'C', 'D', 'E']
        )
        # Agregar nulos correlacionados
        mask = np.random.random(100) < 0.3
        df_large.loc[mask, 'A'] = np.nan
        df_large.loc[mask, 'B'] = np.nan

        fig = self.visualizer.create_correlation_plot(df_large)

        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_save_all_visualizations(self):
        """Verifica guardado de todas las visualizaciones"""
        output_dir = Path(self.temp_dir) / "visualizations"

        self.visualizer.save_all_visualizations(
            self.df,
            output_dir=str(output_dir)
        )

        # Verificar que se crearon los archivos
        self.assertTrue(output_dir.exists())

        expected_files = [
            'null_heatmap.png',
            'null_bar_chart.png',
            'null_pattern_matrix.png'
        ]

        for file_name in expected_files:
            file_path = output_dir / file_name
            self.assertTrue(file_path.exists(), f"Missing {file_name}")


class TestReportGeneration(unittest.TestCase):
    """Tests unitarios para generación de reportes"""

    def setUp(self):
        """Configurar generador de reportes"""
        self.generator = ReportGenerator()
        self.temp_dir = tempfile.mkdtemp()

        # Crear resultado de análisis mock
        self.analysis_result = {
            'summary': {
                'total_nulls': 150,
                'null_percentage': 15.5,
                'columns_with_nulls': 5
            },
            'column_stats': [
                {'column': 'col1', 'null_count': 50, 'null_percentage': 10.0},
                {'column': 'col2', 'null_count': 100, 'null_percentage': 20.0}
            ],
            'pattern_analysis': {
                'detected_pattern': 'MAR',
                'confidence': 0.85
            }
        }

    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_generate_html_report(self):
        """Verifica generación de reporte HTML"""
        output_path = Path(self.temp_dir) / "report.html"

        self.generator.generate_html_report(
            self.analysis_result,
            output_path=str(output_path)
        )

        self.assertTrue(output_path.exists())

        # Verificar contenido básico
        content = output_path.read_text()
        self.assertIn('<html>', content)
        self.assertIn('Análisis de Valores Nulos', content)
        self.assertIn('15.5', content)  # Porcentaje de nulos

    def test_generate_json_report(self):
        """Verifica generación de reporte JSON"""
        output_path = Path(self.temp_dir) / "report.json"

        self.generator.generate_json_report(
            self.analysis_result,
            output_path=str(output_path)
        )

        self.assertTrue(output_path.exists())

        # Verificar que es JSON válido
        with open(output_path, 'r') as f:
            data = json.load(f)

        self.assertIn('summary', data)
        self.assertIn('column_stats', data)
        self.assertEqual(data['summary']['total_nulls'], 150)

    def test_generate_markdown_report(self):
        """Verifica generación de reporte Markdown"""
        output_path = Path(self.temp_dir) / "report.md"

        self.generator.generate_markdown_report(
            self.analysis_result,
            output_path=str(output_path)
        )

        self.assertTrue(output_path.exists())

        # Verificar contenido Markdown
        content = output_path.read_text()
        self.assertIn('# Análisis de Valores Nulos', content)
        self.assertIn('## Resumen', content)
        self.assertIn('|', content)  # Tabla markdown

    def test_generate_excel_report(self):
        """Verifica generación de reporte Excel"""
        output_path = Path(self.temp_dir) / "report.xlsx"

        self.generator.generate_excel_report(
            self.analysis_result,
            output_path=str(output_path)
        )

        self.assertTrue(output_path.exists())

        # Verificar que se puede leer como Excel
        df_summary = pd.read_excel(output_path, sheet_name='Summary')
        self.assertIsNotNone(df_summary)

    def test_comprehensive_report(self):
        """Verifica generación de reporte comprensivo"""
        report = self.generator.generate_comprehensive_report(
            self.analysis_result,
            include_visualizations=False,
            include_recommendations=True
        )

        self.assertIn('metadata', report)
        self.assertIn('summary', report)
        self.assertIn('details', report)
        self.assertIn('recommendations', report)

        # Verificar metadata
        self.assertIn('generated_at', report['metadata'])
        self.assertIn('analysis_type', report['metadata'])


class TestConvenienceFunctions(unittest.TestCase):
    """Tests para funciones de conveniencia"""

    def setUp(self):
        """Crear DataFrame de prueba"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5, np.nan],
            'col2': [np.nan, 2, 3, 4, 5, 6],
            'col3': [1, 2, 3, 4, 5, 6]
        })

    def test_analyze_null_patterns_function(self):
        """Verifica función analyze_null_patterns"""
        result = analyze_null_patterns(self.df)

        self.assertIsInstance(result, dict)
        self.assertIn('pattern', result)
        self.assertIn('summary', result)

    def test_generate_null_report_function(self):
        """Verifica función generate_null_report"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.html"

            generate_null_report(
                self.df,
                output_path=str(output_path),
                format='html'
            )

            self.assertTrue(output_path.exists())


class TestIntegrationNullAnalysis(unittest.TestCase):
    """Tests de integración para análisis de nulos"""

    def setUp(self):
        """Crear dataset completo tipo ENAHO"""
        np.random.seed(42)
        n = 1000

        # Simular estructura ENAHO completa
        self.df_enaho = pd.DataFrame({
            # Identificadores
            'año': [2023] * n,
            'conglome': [f"{i:06d}" for i in range(n)],
            'vivienda': np.random.choice(['01', '02'], n),
            'hogar': ['1'] * n,

            # Variables demográficas
            'edad': np.random.randint(0, 100, n),
            'sexo': np.random.choice([1, 2], n),
            'estado_civil': np.random.choice([1, 2, 3, 4, 5, 6], n),

            # Variables económicas
            'ingreso_laboral': np.random.lognormal(8, 1, n),
            'ingreso_no_laboral': np.random.lognormal(6, 1.5, n),
            'gasto_alimentos': np.random.lognormal(7, 0.8, n),
            'gasto_educacion': np.random.lognormal(5, 1.2, n),

            # Variables educativas
            'nivel_educativo': np.random.choice(range(1, 12), n),
            'años_educacion': np.random.randint(0, 20, n),

            # Factor de expansión
            'factor07': np.random.uniform(0.5, 3, n)
        })

        # Introducir nulos con diferentes patrones
        # MCAR en estado_civil
        mask_mcar = np.random.random(n) < 0.05
        self.df_enaho.loc[mask_mcar, 'estado_civil'] = np.nan

        # MAR - ingreso no laboral missing para jóvenes
        mask_mar = self.df_enaho['edad'] < 25
        self.df_enaho.loc[mask_mar, 'ingreso_no_laboral'] = np.nan

        # MNAR - gastos en educación missing para niveles educativos bajos
        mask_mnar = self.df_enaho['nivel_educativo'] < 3
        self.df_enaho.loc[mask_mnar[:500], 'gasto_educacion'] = np.nan

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_complete_analysis_workflow(self):
        """Test del flujo completo de análisis"""
        # 1. Análisis básico
        analyzer = ENAHONullAnalyzer(complexity='expert')
        result = analyzer.analyze(self.df_enaho)

        # Verificar estructura completa del resultado
        self.assertIn('summary', result)
        self.assertIn('pattern_analysis', result)
        self.assertIn('imputation_recommendations', result)

        # 2. Generar visualizaciones
        visualizer = NullVisualizer()
        viz_dir = Path(self.temp_dir) / "visualizations"
        visualizer.save_all_visualizations(self.df_enaho, str(viz_dir))

        self.assertTrue(viz_dir.exists())
        self.assertGreater(len(list(viz_dir.glob("*.png"))), 0)

        # 3. Generar reporte
        generator = ReportGenerator()
        report_path = Path(self.temp_dir) / "report.html"
        generator.generate_html_report(result, str(report_path))

        self.assertTrue(report_path.exists())

    def test_analysis_with_weights(self):
        """Test de análisis considerando factores de expansión"""
        analyzer = ENAHONullAnalyzer(
            complexity='advanced',
            weight_column='factor07'
        )

        result = analyzer.analyze(self.df_enaho)

        # Verificar que se consideraron los pesos
        self.assertIn('weighted_analysis', result)
        weighted = result['weighted_analysis']

        self.assertIn('weighted_null_percentage', weighted)
        self.assertIn('unweighted_null_percentage', weighted)

        # Los porcentajes ponderados deben ser diferentes
        self.assertNotEqual(
            weighted['weighted_null_percentage'],
            weighted['unweighted_null_percentage']
        )

    def test_temporal_analysis(self):
        """Test de análisis temporal de nulos"""
        # Crear datos multi-año
        years_data = []
        for year in [2021, 2022, 2023]:
            df_year = self.df_enaho.copy()
            df_year['año'] = year
            # Incrementar nulos cada año
            null_rate = 0.05 * (year - 2020)
            mask = np.random.random(len(df_year)) < null_rate
            df_year.loc[mask, 'ingreso_laboral'] = np.nan
            years_data.append(df_year)

        df_panel = pd.concat(years_data, ignore_index=True)

        analyzer = ENAHONullAnalyzer(complexity='advanced')
        temporal_result = analyzer.analyze_temporal(
            df_panel,
            time_column='año'
        )

        self.assertIn('trend_analysis', temporal_result)
        self.assertIn('yearly_stats', temporal_result)

        # Verificar tendencia creciente de nulos
        yearly = temporal_result['yearly_stats']
        self.assertLess(
            yearly[2021]['null_percentage'],
            yearly[2023]['null_percentage']
        )

    def test_module_specific_analysis(self):
        """Test de análisis específico por módulo ENAHO"""
        # Simular diferentes módulos
        modules = {
            '01': self.df_enaho[['conglome', 'vivienda', 'hogar']].copy(),
            '02': self.df_enaho[['conglome', 'vivienda', 'hogar', 'edad', 'sexo']].copy(),
            '34': self.df_enaho[['conglome', 'vivienda', 'hogar', 'ingreso_laboral', 'gasto_alimentos']].copy()
        }

        # Agregar nulos específicos por módulo
        modules['02'].loc[::10, 'edad'] = np.nan
        modules['34'].loc[::5, 'ingreso_laboral'] = np.nan

        analyzer = ENAHONullAnalyzer(complexity='standard')

        module_results = {}
        for module_name, module_df in modules.items():
            module_results[module_name] = analyzer.analyze(module_df)

        # Verificar que cada módulo tiene su análisis
        self.assertEqual(len(module_results), 3)

        # Módulo 01 no debe tener nulos
        self.assertEqual(
            module_results['01']['summary']['total_nulls'],
            0
        )

        # Módulos 02 y 34 deben tener nulos
        self.assertGreater(
            module_results['02']['summary']['total_nulls'],
            0
        )
        self.assertGreater(
            module_results['34']['summary']['total_nulls'],
            0
        )

    def test_imputation_strategies(self):
        """Test de estrategias de imputación recomendadas"""
        analyzer = ENAHONullAnalyzer(complexity='expert')
        result = analyzer.analyze(self.df_enaho)

        imputation = result['imputation_recommendations']

        # Debe recomendar diferentes estrategias para diferentes patrones
        self.assertIn('strategies_by_column', imputation)
        strategies = imputation['strategies_by_column']

        # Verificar que hay estrategias específicas
        for col, strategy in strategies.items():
            self.assertIn('method', strategy)
            self.assertIn('rationale', strategy)

            # Las estrategias deben ser apropiadas al tipo de dato
            if col in ['edad', 'años_educacion']:
                self.assertIn(strategy['method'], ['mean', 'median', 'knn'])
            elif col in ['sexo', 'estado_civil']:
                self.assertIn(strategy['method'], ['mode', 'forward_fill'])

    def test_export_formats(self):
        """Test de exportación en múltiples formatos"""
        analyzer = ENAHONullAnalyzer(complexity='standard')
        result = analyzer.analyze(self.df_enaho)

        generator = ReportGenerator()

        # Test HTML
        html_path = Path(self.temp_dir) / "report.html"
        generator.generate_html_report(result, str(html_path))
        self.assertTrue(html_path.exists())

        # Test JSON
        json_path = Path(self.temp_dir) / "report.json"
        generator.generate_json_report(result, str(json_path))
        self.assertTrue(json_path.exists())

        # Test Markdown
        md_path = Path(self.temp_dir) / "report.md"
        generator.generate_markdown_report(result, str(md_path))
        self.assertTrue(md_path.exists())

        # Test Excel
        excel_path = Path(self.temp_dir) / "report.xlsx"
        generator.generate_excel_report(result, str(excel_path))
        self.assertTrue(excel_path.exists())


class TestErrorHandling(unittest.TestCase):
    """Tests para manejo de errores en null_analysis"""

    def test_empty_dataframe_error(self):
        """Verifica manejo de DataFrame vacío"""
        analyzer = ENAHONullAnalyzer()
        df_empty = pd.DataFrame()

        with self.assertRaises(ValueError):
            analyzer.analyze(df_empty)

    def test_no_nulls_handling(self):
        """Verifica manejo cuando no hay nulos"""
        df_no_nulls = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df_no_nulls)

        self.assertEqual(result['summary']['total_nulls'], 0)
        self.assertEqual(result['summary']['null_percentage'], 0.0)

    def test_all_nulls_handling(self):
        """Verifica manejo cuando todo es nulo"""
        df_all_nulls = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })

        analyzer = ENAHONullAnalyzer()
        result = analyzer.analyze(df_all_nulls)

        self.assertEqual(result['summary']['null_percentage'], 100.0)
        self.assertIn('warning', result)

    def test_invalid_complexity_level(self):
        """Verifica error con nivel de complejidad inválido"""
        with self.assertRaises(ValueError):
            ENAHONullAnalyzer(complexity='invalid_level')

    def test_invalid_column_reference(self):
        """Verifica manejo de columnas inexistentes"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        analyzer = ENAHONullAnalyzer(weight_column='columna_inexistente')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analyzer.analyze(df)

            # Debe generar advertencia pero no fallar
            self.assertTrue(len(w) > 0)
            self.assertIsNotNone(result)


def run_null_analysis_tests():
    """Ejecutar todos los tests del módulo null_analysis"""
    null_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(null_suite)

    # Resumen de resultados
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS - MÓDULO NULL ANALYSIS")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Éxitos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")

    if result.failures:
        print("\nFallos detectados:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")

    if result.errors:
        print("\nErrores detectados:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_null_analysis_tests()
    exit(0 if success else 1)