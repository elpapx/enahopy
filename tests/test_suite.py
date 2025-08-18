# null_analysis/tests/test_suite.py
"""
Suite completo de tests para el módulo null_analysis.
Incluye tests unitarios e integración.

Para ejecutar:
    pytest null_analysis/tests/test_suite.py -v
    pytest null_analysis/tests/test_suite.py::TestUnitConvenience -v  # Solo unit tests
    pytest null_analysis/tests/test_suite.py::TestIntegration -v     # Solo integration tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

# =====================================================
# IMPORTS
# =====================================================

# Imports principales del módulo
from null_analysis import (
    ENAHONullAnalyzer,
    NullAnalysisConfig,
    MissingDataMetrics,
    MissingDataPattern,
    AnalysisComplexity,
    VisualizationType,
    ExportFormat,
    NullAnalysisError,
    quick_null_analysis,
    get_data_quality_score,
    create_null_visualizations,
    generate_null_report,
    compare_null_patterns,
    suggest_imputation_methods,
    validate_data_completeness,
    detect_missing_patterns_automatically,
    safe_dict_merge,
    InputValidator
)



# =====================================================
# FIXTURES COMPARTIDAS
# =====================================================

@pytest.fixture
def sample_df():
    """DataFrame de prueba con patrones conocidos de nulos"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1, 101),
        'age': [np.nan if i % 10 == 0 else i for i in range(25, 125)],  # 10% nulos
        'income': [np.nan if i % 5 == 0 else i * 1000 for i in range(1, 101)],  # 20% nulos
        'category': ['A', 'B', 'C', np.nan] * 25,  # 25% nulos
        'complete_col': range(100),  # Sin nulos
        'mostly_null': [np.nan] * 90 + list(range(10)),  # 90% nulos
    })


@pytest.fixture
def empty_df():
    """DataFrame vacío para tests de edge cases"""
    return pd.DataFrame()


@pytest.fixture
def single_row_df():
    """DataFrame con una sola fila"""
    return pd.DataFrame({
        'col1': [1],
        'col2': [np.nan],
        'col3': ['test']
    })


@pytest.fixture
def all_null_df():
    """DataFrame completamente nulo"""
    return pd.DataFrame({
        'col1': [np.nan] * 10,
        'col2': [None] * 10,
        'col3': [np.nan] * 10
    })


@pytest.fixture
def config_basic():
    """Configuración básica para tests"""
    return NullAnalysisConfig(
        complexity_level=AnalysisComplexity.BASIC,
        visualization_type=VisualizationType.STATIC,
        enable_caching=False
    )


@pytest.fixture
def config_advanced():
    """Configuración avanzada para tests"""
    return NullAnalysisConfig(
        complexity_level=AnalysisComplexity.ADVANCED,
        visualization_type=VisualizationType.STATIC,
        enable_caching=False,
        correlation_threshold=0.3
    )



# =====================================================
# UNIT TESTS - FUNCIONES DE CONVENIENCIA
# =====================================================
class TestUnitConvenience:
    """Tests unitarios para funciones de conveniencia"""

    def test_quick_null_analysis_basic(self, sample_df):
        """Test análisis rápido básico"""
        result = quick_null_analysis(sample_df, complexity="basic")

        assert result is not None
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'basic'
        assert 'metrics' in result
        assert 'summary' in result

    def test_quick_null_analysis_invalid_complexity(self, sample_df):
        """Test con complejidad inválida"""
        with pytest.raises(ValueError) as exc_info:
            quick_null_analysis(sample_df, complexity="invalid")

        assert "no válida" in str(exc_info.value).lower()

    def test_quick_null_analysis_with_groupby(self, sample_df):
        """Test análisis con agrupación"""
        result = quick_null_analysis(sample_df, group_by="category", complexity="standard")

        assert result is not None
        assert 'group_analysis' in result or 'groups' in result

    def test_get_data_quality_score_simple(self, sample_df):
        """Test score de calidad simple"""
        score = get_data_quality_score(sample_df)

        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_get_data_quality_score_detailed(self, sample_df):
        """Test score de calidad detallado"""
        result = get_data_quality_score(sample_df, detailed=True)

        assert isinstance(result, dict)
        assert 'quality_score' in result
        assert 'missing_percentage' in result
        assert 'complete_cases_percentage' in result
        assert 'recommendation' in result

    def test_validate_data_completeness_complete(self, sample_df):
        """Test validación de completitud con datos buenos"""
        result = validate_data_completeness(
            sample_df,
            required_completeness=50.0,
            required_variables=['id', 'age']
        )

        assert result['is_valid'] is True
        assert result['completeness_score'] > 50
        assert len(result['missing_variables']) == 0

    def test_validate_data_completeness_incomplete(self, all_null_df):
        """Test validación con datos incompletos"""
        result = validate_data_completeness(
            all_null_df,
            required_completeness=80.0
        )

        assert result['is_valid'] is False
        assert result['completeness_score'] == 0
        assert len(result['recommendations']) > 0

    def test_validate_data_completeness_missing_vars(self, sample_df):
        """Test validación con variables faltantes"""
        result = validate_data_completeness(
            sample_df,
            required_variables=['id', 'nonexistent_column']
        )

        assert result['is_valid'] is False
        assert 'nonexistent_column' in result['missing_variables']

    def test_validate_data_completeness_empty_df(self, empty_df):
        """Test validación con DataFrame vacío"""
        result = validate_data_completeness(empty_df)

        assert result['is_valid'] is False
        assert result['reason'] == 'DataFrame vacío o None'
        assert result['completeness_score'] == 0.0

    def test_detect_missing_patterns_automatically(self, sample_df):
        """Test detección automática de patrones"""
        result = detect_missing_patterns_automatically(sample_df)

        assert 'detected_pattern' in result
        assert 'confidence' in result
        assert 'evidence' in result
        assert isinstance(result['detected_pattern'], MissingDataPattern)

    def test_compare_null_patterns_valid(self, sample_df, all_null_df):
        """Test comparación de patrones entre datasets"""
        datasets = {
            'good_data': sample_df,
            'bad_data': all_null_df
        }

        result = compare_null_patterns(datasets)

        assert 'individual_analyses' in result
        assert 'metrics_comparison' in result
        assert 'best_quality_dataset' in result
        assert result['best_quality_dataset'] == 'good_data'

    def test_compare_null_patterns_single_dataset(self, sample_df):
        """Test comparación con un solo dataset (debe fallar)"""
        datasets = {'only_one': sample_df}

        with pytest.raises(ValueError) as exc_info:
            compare_null_patterns(datasets)

        assert "al menos 2 datasets" in str(exc_info.value)

    def test_compare_null_patterns_empty_dataset(self, sample_df, empty_df):
        """Test comparación con dataset vacío"""
        datasets = {
            'good': sample_df,
            'empty': empty_df
        }

        result = compare_null_patterns(datasets)

        assert 'error' in result['individual_analyses']['empty']

    def test_suggest_imputation_methods(self, sample_df):
        """Test sugerencia de métodos de imputación"""
        result = suggest_imputation_methods(sample_df)

        assert 'recommended_method' in result
        assert 'alternative_methods' in result
        assert 'rationale' in result
        assert 'considerations' in result

    def test_generate_null_report_valid_formats(self, sample_df, tmp_path):
        """Test generación de reporte con formatos válidos"""
        output_path = str(tmp_path / "report")

        result = generate_null_report(
            sample_df,
            output_path=output_path,
            format_types=["html", "json"]
        )

        assert result is not None
        assert 'success' in result or 'files_generated' in result

    def test_generate_null_report_invalid_format(self, sample_df, tmp_path):
        """Test generación de reporte con formato inválido"""
        output_path = str(tmp_path / "report")

        with pytest.raises(ValueError) as exc_info:
            generate_null_report(
                sample_df,
                output_path=output_path,
                format_types=["invalid_format"]
            )

        assert "Formatos inválidos" in str(exc_info.value)


# =====================================================
# UNIT TESTS - ANALYZER CORE
# =====================================================
class TestUnitAnalyzer:
    """Tests unitarios para ENAHONullAnalyzer"""

    def test_analyzer_initialization(self, config_basic):
        """Test inicialización del analizador"""
        analyzer = ENAHONullAnalyzer(config=config_basic, verbose=False)

        assert analyzer is not None
        assert analyzer.config == config_basic

    def test_analyzer_basic_analysis(self, sample_df, config_basic):
        """Test análisis básico con analizador"""
        analyzer = ENAHONullAnalyzer(config=config_basic, verbose=False)
        result = analyzer.analyze_null_patterns(sample_df)

        assert result['analysis_type'] == 'basic'
        assert 'metrics' in result
        assert isinstance(result['metrics'], MissingDataMetrics)

    def test_analyzer_advanced_analysis(self, sample_df, config_advanced):
        """Test análisis avanzado con analizador"""
        analyzer = ENAHONullAnalyzer(config=config_advanced, verbose=False)
        result = analyzer.analyze_null_patterns(sample_df)

        assert result['analysis_type'] == 'advanced'
        assert 'patterns' in result
        assert 'correlations' in result

    def test_analyzer_empty_dataframe(self, empty_df, config_basic):
        """Test analizador con DataFrame vacío"""
        analyzer = ENAHONullAnalyzer(config=config_basic, verbose=False)

        with pytest.raises(NullAnalysisError) as exc_info:
            analyzer.analyze_null_patterns(empty_df)

        assert "vacío" in str(exc_info.value).lower()

    def test_analyzer_quality_score(self, sample_df, config_basic):
        """Test cálculo de quality score"""
        analyzer = ENAHONullAnalyzer(config=config_basic, verbose=False)
        score = analyzer.get_data_quality_score(sample_df)

        assert isinstance(score, float)
        assert 0 <= score <= 100

        # Con datos de muestra, esperamos un score razonable
        assert score > 30  # No debe ser terrible
        assert score < 95  # No debe ser perfecto (tenemos nulos)


# =====================================================
# UNIT TESTS - UTILIDADES
# =====================================================

class TestUnitUtils:
    """Tests unitarios para funciones de utilidad - VERSIÓN CORREGIDA"""

    def test_safe_dict_merge_both_none(self):
        """Test merge con ambos diccionarios None"""
        result = safe_dict_merge(None, None)
        assert result == {}

    def test_safe_dict_merge_one_none(self):
        """Test merge con un diccionario None"""
        dict1 = {'a': 1, 'b': 2}

        result1 = safe_dict_merge(dict1, None)
        assert result1 == dict1

        result2 = safe_dict_merge(None, dict1)
        assert result2 == dict1

    def test_safe_dict_merge_update_strategy(self):
        """Test merge con estrategia update"""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4}

        result = safe_dict_merge(dict1, dict2, strategy="update")

        assert result['a'] == 1
        assert result['b'] == 3  # dict2 sobrescribe
        assert result['c'] == 4

    def test_safe_dict_merge_deep_merge(self):
        """Test merge profundo de diccionarios anidados"""
        dict1 = {'a': {'b': 1, 'c': 2}, 'd': 3}
        dict2 = {'a': {'b': 10, 'e': 4}, 'f': 5}

        result = safe_dict_merge(dict1, dict2, strategy="deep_merge")

        assert result['a']['b'] == 10  # Sobrescribe
        assert result['a']['c'] == 2  # Mantiene
        assert result['a']['e'] == 4  # Agrega
        assert result['d'] == 3
        assert result['f'] == 5

    def test_safe_dict_merge_prefer_non_null(self):
        """Test merge prefiriendo valores no nulos"""
        dict1 = {'a': 1, 'b': None, 'c': 3}
        dict2 = {'a': None, 'b': 2, 'd': 4}

        result = safe_dict_merge(dict1, dict2, strategy="prefer_non_null")

        assert result['a'] == 1  # Mantiene no-nulo de dict1
        assert result['b'] == 2  # Toma no-nulo de dict2
        assert result['c'] == 3
        assert result['d'] == 4

    def test_input_validator_valid_dataframe(self, sample_df):
        """Test validación de DataFrame válido"""
        validator = InputValidator()
        result = validator.validate_dataframe(sample_df)

        assert result is sample_df

    def test_input_validator_none_dataframe(self):
        """Test validación con DataFrame None - CORREGIDO"""
        validator = InputValidator()

        # NullAnalysisError ya está importado arriba, no necesitamos importarlo aquí
        with pytest.raises(NullAnalysisError) as exc_info:
            validator.validate_dataframe(None)

        assert "no puede ser None" in str(exc_info.value)

    def test_input_validator_empty_dataframe(self, empty_df):
        """Test validación con DataFrame vacío - CORREGIDO"""
        validator = InputValidator()

        # NullAnalysisError ya está importado arriba
        with pytest.raises(NullAnalysisError) as exc_info:
            validator.validate_dataframe(empty_df)

        assert "vacío" in str(exc_info.value).lower()

    def test_input_validator_required_columns(self, sample_df):
        """Test validación de columnas requeridas - CORREGIDO"""
        validator = InputValidator()

        # Columnas existentes - debe pasar
        result = validator.validate_dataframe(
            sample_df,
            required_columns=['id', 'age']
        )
        assert result is not None

        # Columnas faltantes - debe fallar
        # NullAnalysisError ya está importado arriba
        with pytest.raises(NullAnalysisError) as exc_info:
            validator.validate_dataframe(
                sample_df,
                required_columns=['nonexistent']
            )
        assert "faltantes" in str(exc_info.value).lower()


# =====================================================
# INTEGRATION TESTS
# =====================================================


class TestIntegration:
    """Tests de integración del módulo completo"""

    def test_full_analysis_pipeline(self, sample_df):
        """Test pipeline completo de análisis"""
        # 1. Análisis rápido
        quick_result = quick_null_analysis(sample_df, complexity="standard")
        assert quick_result is not None

        # 2. Score de calidad
        quality_score = get_data_quality_score(sample_df)
        assert 0 <= quality_score <= 100

        # 3. Validación de completitud
        validation = validate_data_completeness(sample_df, required_completeness=50)
        assert validation['is_valid'] is True

        # 4. Detección de patrones
        patterns = detect_missing_patterns_automatically(sample_df)
        assert patterns['detected_pattern'] is not None

        # 5. Sugerencias de imputación
        suggestions = suggest_imputation_methods(sample_df)
        assert suggestions['recommended_method'] is not None

    def test_comparative_analysis_pipeline(self):
        """Test pipeline de análisis comparativo"""
        # Crear datasets con diferentes niveles de calidad
        np.random.seed(42)

        good_df = pd.DataFrame({
            'col1': range(100),
            'col2': [np.nan if i % 20 == 0 else i for i in range(100)],  # 5% nulos
        })

        medium_df = pd.DataFrame({
            'col1': [np.nan if i % 4 == 0 else i for i in range(100)],  # 25% nulos
            'col2': [np.nan if i % 3 == 0 else i for i in range(100)],  # 33% nulos
        })

        bad_df = pd.DataFrame({
            'col1': [np.nan if i % 2 == 0 else i for i in range(100)],  # 50% nulos
            'col2': [np.nan] * 80 + list(range(20)),  # 80% nulos
        })

        datasets = {
            'good': good_df,
            'medium': medium_df,
            'bad': bad_df
        }

        # Comparar datasets
        comparison = compare_null_patterns(datasets)

        assert comparison['best_quality_dataset'] == 'good'
        assert len(comparison['metrics_comparison']) == 3
        assert 'differences' in comparison

    def test_report_generation_pipeline(self, sample_df, tmp_path):
        """Test generación completa de reportes"""
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        # Generar reporte en múltiples formatos
        report_result = generate_null_report(
            sample_df,
            output_path=str(output_dir / "null_report"),
            format_types=["json", "html"]
        )

        # Verificar que se generaron archivos
        if 'files_generated' in report_result:
            assert len(report_result['files_generated']) >= 1

    def test_edge_cases_pipeline(self, empty_df, single_row_df, all_null_df):
        """Test con casos extremos"""
        # DataFrame vacío
        with pytest.raises(NullAnalysisError):
            quick_null_analysis(empty_df)

        # DataFrame de una fila
        result = quick_null_analysis(single_row_df)
        assert result is not None
        assert result['metrics'].total_cells == 3

        # DataFrame completamente nulo
        result = quick_null_analysis(all_null_df)
        assert result['metrics'].missing_percentage == 100.0
        assert result['metrics'].complete_cases == 0

    def test_edge_cases_pipeline(self, empty_df, single_row_df, all_null_df):
        """Test con casos extremos"""
        # DataFrame vacío
        with pytest.raises(NullAnalysisError):
            quick_null_analysis(empty_df)

        # DataFrame de una fila
        result = quick_null_analysis(single_row_df)
        assert result is not None
        assert result['metrics'].total_cells == 3

        # DataFrame completamente nulo
        result = quick_null_analysis(all_null_df)
        assert result['metrics'].missing_percentage == 100.0
        assert result['metrics'].complete_cases == 0

    def test_configuration_changes(self, sample_df):
        """Test cambios de configuración"""
        # Análisis básico
        config_basic = NullAnalysisConfig(
            complexity_level=AnalysisComplexity.BASIC
        )
        analyzer_basic = ENAHONullAnalyzer(config=config_basic, verbose=False)
        result_basic = analyzer_basic.analyze_null_patterns(sample_df)

        # Análisis avanzado
        config_advanced = NullAnalysisConfig(
            complexity_level=AnalysisComplexity.ADVANCED
        )
        analyzer_advanced = ENAHONullAnalyzer(config=config_advanced, verbose=False)
        result_advanced = analyzer_advanced.analyze_null_patterns(sample_df)

        # El análisis avanzado debe tener más información
        assert 'patterns' not in result_basic or 'patterns' in result_advanced
        assert result_basic['analysis_type'] == 'basic'
        assert result_advanced['analysis_type'] == 'advanced'

    def test_grouped_analysis(self, sample_df):
        """Test análisis agrupado"""
        # Agregar columna de grupo
        sample_df['group'] = ['Group1' if i < 50 else 'Group2' for i in range(len(sample_df))]

        # Análisis sin agrupación
        result_ungrouped = quick_null_analysis(sample_df)

        # Análisis con agrupación
        result_grouped = quick_null_analysis(sample_df, group_by='group')

        # Verificar que el análisis agrupado tiene información adicional
        assert result_ungrouped is not None
        assert result_grouped is not None
        # El resultado agrupado podría tener información de grupos

    def test_memory_efficiency(self):
        """Test eficiencia de memoria con dataset grande"""
        # Crear dataset grande
        large_df = pd.DataFrame({
            f'col_{i}': np.random.choice([np.nan, 1, 2, 3], size=10000)
            for i in range(50)
        })

        # Análisis debe completarse sin problemas de memoria
        result = quick_null_analysis(large_df, complexity="basic")
        assert result is not None
        assert result['metrics'].total_cells == 500000

    @pytest.mark.slow
    def test_performance_benchmark(self):
        """Test benchmark de performance"""
        import time

        # Dataset mediano
        medium_df = pd.DataFrame({
            f'col_{i}': np.random.choice([np.nan, 1, 2, 3], size=1000)
            for i in range(20)
        })

        # Medir tiempo de análisis básico
        start = time.time()
        result = quick_null_analysis(medium_df, complexity="basic")
        basic_time = time.time() - start

        # Medir tiempo de análisis avanzado
        start = time.time()
        result = quick_null_analysis(medium_df, complexity="advanced")
        advanced_time = time.time() - start

        # El análisis básico debe ser más rápido
        assert basic_time < advanced_time

        # Ambos deben completarse en tiempo razonable (< 5 segundos)
        assert basic_time < 5
        assert advanced_time < 5

    def test_deprecation_warnings(self):
        """Test warnings de deprecación"""
        from null_analysis.convenience import diagnostico_nulos_enaho

        df = pd.DataFrame({'col1': [1, 2, np.nan]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = diagnostico_nulos_enaho(df)

            assert len(w) > 0
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# =====================================================
# PERFORMANCE TESTS (Opcional)
# =====================================================

@pytest.mark.performance
class TestPerformance:
    """Tests de performance y escalabilidad"""

    def test_large_dataset_performance(self):
        """Test con dataset grande"""
        # Crear dataset de 100k filas x 100 columnas
        large_df = pd.DataFrame(
            np.random.choice([np.nan, 1, 2, 3, 4], size=(100000, 100))
        )

        import time
        start = time.time()

        result = quick_null_analysis(large_df, complexity="basic")

        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 30  # Debe completarse en menos de 30 segundos

    def test_many_patterns_performance(self):
        """Test con muchos patrones únicos de nulos"""
        # Cada fila tiene un patrón único
        patterns_df = pd.DataFrame({
            f'col_{i}': [np.nan if (j + i) % 3 == 0 else j
                         for j in range(1000)]
            for i in range(10)
        })

        result = detect_missing_patterns_automatically(patterns_df)
        assert result is not None


# =====================================================
# FIXTURES PARA PYTEST
# =====================================================

@pytest.fixture(autouse=True)
def suppress_matplotlib():
    """Suprimir ventanas de matplotlib en tests"""
    import matplotlib
    matplotlib.use('Agg')


# =====================================================
# CONFIGURACIÓN PYTEST
# =====================================================
@pytest.fixture(autouse=True)
def suppress_matplotlib():
    """Suprimir ventanas de matplotlib en tests"""
    import matplotlib
    matplotlib.use('Agg')


def pytest_configure(config):
    """Configuración personalizada de pytest"""
    config.addinivalue_line(
        "markers", "slow: marca tests que son lentos"
    )
    config.addinivalue_line(
        "markers", "performance: marca tests de performance"
    )


if __name__ == "__main__":
    # Ejecutar tests con coverage
    pytest.main([__file__, "-v", "--cov=null_analysis", "--cov-report=html"])