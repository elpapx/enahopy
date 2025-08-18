# null_analysis/tests/conftest.py
"""
Configuración compartida de pytest para tests de null_analysis.
Este archivo es automáticamente detectado por pytest.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar el módulo
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================

def pytest_configure(config):
    """Configuración global de pytest"""
    config.addinivalue_line(
        "markers", "slow: tests que tardan más de 1 segundo"
    )
    config.addinivalue_line(
        "markers", "performance: tests de benchmark de performance"
    )
    config.addinivalue_line(
        "markers", "integration: tests de integración"
    )
    config.addinivalue_line(
        "markers", "unit: tests unitarios"
    )


# =====================================================
# FIXTURES GLOBALES
# =====================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configurar ambiente de tests"""
    # Suprimir warnings específicos
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Configurar pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 50)

    # Configurar numpy
    np.random.seed(42)

    yield

    # Limpiar al finalizar
    pd.reset_option('all')


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Resetear seed random para cada test"""
    np.random.seed(42)


@pytest.fixture
def mock_logger():
    """Logger mock para tests"""
    from unittest.mock import Mock
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


# =====================================================
# FIXTURES DE DATOS DE PRUEBA
# =====================================================

@pytest.fixture(scope="module")
def enaho_like_df():
    """DataFrame similar a datos ENAHO"""
    np.random.seed(42)
    n = 1000

    return pd.DataFrame({
        'conglome': np.random.choice(['001', '002', '003', '004', '005'], n),
        'vivienda': np.random.choice(['01', '02', '03'], n),
        'hogar': np.random.choice(['1', '2'], n),
        'p203': np.random.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.1]),  # Parentesco
        'p207': np.random.choice([1, 2, np.nan], n, p=[0.48, 0.48, 0.04]),  # Sexo
        'p208': np.random.randint(0, 100, n),  # Edad
        'p301a': np.random.choice([1, 2, 3, 4, 5, 6, np.nan], n),  # Nivel educativo
        'ingreso': np.where(
            np.random.random(n) > 0.15,
            np.random.lognormal(7, 1.5, n),
            np.nan
        ),
        'gasto': np.where(
            np.random.random(n) > 0.12,
            np.random.lognormal(6.8, 1.3, n),
            np.nan
        ),
        'departamento': np.random.choice(['LIMA', 'AREQUIPA', 'CUSCO', 'PIURA'], n),
        'area': np.random.choice(['1', '2'], n, p=[0.7, 0.3]),  # Urbano/Rural
        'estrato': np.random.choice(range(1, 9), n),
    })


@pytest.fixture
def geographic_df():
    """DataFrame con columnas geográficas"""
    return pd.DataFrame({
        'ubigeo': ['150101', '150102', '040101', '130101'] * 25,
        'departamento': ['LIMA', 'LIMA', 'AREQUIPA', 'CUSCO'] * 25,
        'provincia': ['LIMA', 'LIMA', 'AREQUIPA', 'CUSCO'] * 25,
        'distrito': ['LIMA', 'RIMAC', 'AREQUIPA', 'CUSCO'] * 25,
        'value1': range(100),
        'value2': [np.nan if i % 7 == 0 else i for i in range(100)]
    })


# =====================================================
# pytest.ini
# =====================================================
"""
# Crear archivo pytest.ini en la raíz del proyecto
[pytest]
minversion = 6.0
addopts = -v --tb=short --strict-markers
testpaths = null_analysis/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    performance: marks tests as performance benchmarks
    integration: marks tests as integration tests
    unit: marks tests as unit tests
"""

# =====================================================
# null_analysis/tests/run_tests.py
# =====================================================
"""
Script para ejecutar tests con diferentes configuraciones
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", coverage=False, verbose=True):
    """
    Ejecutar suite de tests

    Args:
        test_type: 'all', 'unit', 'integration', 'performance'
        coverage: Si generar reporte de cobertura
        verbose: Si mostrar output detallado
    """
    cmd = ["pytest"]

    # Agregar archivo de tests
    test_file = Path(__file__).parent / "test_suite.py"
    cmd.append(str(test_file))

    # Configurar tipo de tests
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow and not performance"])

    # Configurar verbosidad
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Configurar coverage
    if coverage:
        cmd.extend([
            "--cov=null_analysis",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Ejecutar tests
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar tests de null_analysis")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "performance", "fast"],
        default="all",
        help="Tipo de tests a ejecutar"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generar reporte de cobertura"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso"
    )

    args = parser.parse_args()

    exit_code = run_tests(
        test_type=args.type,
        coverage=args.coverage,
        verbose=not args.quiet
    )

    sys.exit(exit_code)

# =====================================================
# null_analysis/tests/test_edge_cases.py
# =====================================================
"""
Tests específicos para casos extremos y edge cases
"""

import pytest
import pandas as pd
import numpy as np
from null_analysis import *


class TestEdgeCases:
    """Tests para casos extremos específicos"""

    def test_single_column_dataframe(self):
        """Test con DataFrame de una sola columna"""
        df = pd.DataFrame({'only_col': [1, 2, np.nan, 4, 5]})

        result = quick_null_analysis(df)
        assert result is not None
        assert result['metrics'].variables_with_missing == 1
        assert result['metrics'].variables_without_missing == 0

    def test_no_null_dataframe(self):
        """Test con DataFrame sin valores nulos"""
        df = pd.DataFrame({
            'col1': range(100),
            'col2': list('ABCD' * 25),
            'col3': np.random.randn(100)
        })

        result = quick_null_analysis(df)
        assert result['metrics'].missing_percentage == 0.0
        assert result['metrics'].complete_cases_percentage == 100.0

        score = get_data_quality_score(df)
        assert score == 100.0

    def test_all_null_single_column(self):
        """Test con una columna completamente nula"""
        df = pd.DataFrame({
            'good': range(10),
            'bad': [np.nan] * 10
        })

        result = quick_null_analysis(df)
        assert result['metrics'].variables_with_missing == 1

        # La columna 'bad' debe aparecer con 100% de nulos
        summary = result['summary']
        bad_col = summary[summary['variable'] == 'bad']
        assert bad_col['missing_percentage'].values[0] == 100.0

    def test_mixed_dtypes(self):
        """Test con tipos de datos mixtos"""
        df = pd.DataFrame({
            'int_col': [1, 2, np.nan, 4],
            'float_col': [1.1, np.nan, 3.3, 4.4],
            'str_col': ['a', 'b', None, 'd'],
            'bool_col': [True, False, np.nan, True],
            'date_col': pd.to_datetime(['2021-01-01', np.nan, '2021-03-01', '2021-04-01'])
        })

        result = quick_null_analysis(df)
        assert result is not None
        assert result['metrics'].variables_with_missing == 4

    def test_large_number_of_columns(self):
        """Test con muchas columnas"""
        n_cols = 500
        df = pd.DataFrame({
            f'col_{i}': np.random.choice([np.nan, 1, 2], 100)
            for i in range(n_cols)
        })

        result = quick_null_analysis(df, complexity="basic")
        assert result is not None
        assert len(result['summary']) == n_cols

    def test_special_values(self):
        """Test con valores especiales"""
        df = pd.DataFrame({
            'inf_col': [1, 2, np.inf, -np.inf, 5],
            'nan_col': [1, 2, np.nan, 4, 5],
            'none_col': [1, 2, None, 4, 5]
        })

        result = quick_null_analysis(df)
        assert result is not None
        # np.inf no se considera nulo, pero None y np.nan sí
        assert result['metrics'].variables_with_missing == 2

    def test_duplicate_column_names(self):
        """Test con nombres de columnas duplicados"""
        df = pd.DataFrame({
            'col': [1, 2, 3],
            'col.1': [4, 5, 6]  # Pandas renombra automáticamente
        })
        df.columns = ['col', 'col']  # Forzar duplicados

        # El análisis debe manejar esto sin errores
        result = quick_null_analysis(df)
        assert result is not None

    def test_unicode_column_names(self):
        """Test con nombres de columnas Unicode"""
        df = pd.DataFrame({
            '列1': [1, 2, np.nan],
            'колонка': [4, np.nan, 6],
            '🔍': [7, 8, 9]
        })

        result = quick_null_analysis(df)
        assert result is not None
        assert result['metrics'].variables_with_missing == 2

    def test_very_sparse_data(self):
        """Test con datos muy dispersos (>90% nulos)"""
        df = pd.DataFrame({
            f'col_{i}': [np.nan] * 95 + list(range(5))
            for i in range(10)
        })

        result = quick_null_analysis(df)
        assert result['metrics'].missing_percentage > 90

        validation = validate_data_completeness(df, required_completeness=80)
        assert validation['is_valid'] is False

    def test_memory_stress(self):
        """Test con dataset que estresa la memoria"""
        # Crear dataset grande pero manejable
        df = pd.DataFrame(
            np.random.choice([np.nan, 1], size=(50000, 100), p=[0.3, 0.7])
        )

        # Debe completarse sin error de memoria
        result = quick_null_analysis(df, complexity="basic")
        assert result is not None

    def test_categorical_data(self):
        """Test con datos categóricos"""
        df = pd.DataFrame({
            'cat1': pd.Categorical(['A', 'B', None, 'A', 'B']),
            'cat2': pd.Categorical([1, 2, np.nan, 1, 2])
        })

        result = quick_null_analysis(df)
        assert result is not None
        assert result['metrics'].variables_with_missing == 2


# =====================================================
# null_analysis/tests/test_regression.py
# =====================================================
"""
Tests de regresión para asegurar que cambios futuros no rompan funcionalidad existente
"""

import pytest
import pandas as pd
import numpy as np
import json
from null_analysis import *


class TestRegression:
    """Tests de regresión para mantener compatibilidad"""

    def test_basic_metrics_consistency(self):
        """Verificar que las métricas básicas se mantienen consistentes"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, 2, 3, np.nan, 5]
        })

        result = quick_null_analysis(df, complexity="basic")

        # Valores esperados específicos
        assert result['metrics'].total_cells == 10
        assert result['metrics'].missing_cells == 3
        assert result['metrics'].missing_percentage == 30.0
        assert result['metrics'].complete_cases == 2
        assert result['metrics'].complete_cases_percentage == 40.0

    def test_api_compatibility(self):
        """Verificar que la API antigua sigue funcionando"""
        from null_analysis.convenience import LegacyNullAnalyzer

        df = pd.DataFrame({'col': [1, np.nan, 3]})

        # La API legacy debe funcionar con warning
        with pytest.warns(DeprecationWarning):
            analyzer = LegacyNullAnalyzer()


        with pytest.warns(DeprecationWarning):
            from null_analysis.convenience import diagnostico_nulos_enaho
            result = diagnostico_nulos_enaho(df)
            assert 'resumen_total' in result

    def test_output_structure_stability(self):
        """Verificar que la estructura de salida se mantiene"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan],
            'b': [4, np.nan, 6]
        })

        result = quick_null_analysis(df)

        # Verificar estructura esperada
        expected_keys = ['analysis_type', 'metrics', 'summary']
        for key in expected_keys:
            assert key in result

        # Verificar estructura de metrics
        metrics = result['metrics']
        assert hasattr(metrics, 'total_cells')
        assert hasattr(metrics, 'missing_cells')
        assert hasattr(metrics, 'missing_percentage')

        # Verificar estructura de summary
        summary = result['summary']
        expected_columns = ['variable', 'missing_count', 'missing_percentage']
        for col in expected_columns:
            assert col in summary.columns

    def test_edge_case_backwards_compatibility(self):
        """Verificar compatibilidad con casos extremos anteriores"""
        # DataFrame vacío debe lanzar error como antes
        with pytest.raises(NullAnalysisError):
            quick_null_analysis(pd.DataFrame())

        # DataFrame sin nulos debe dar 100% calidad
        df_perfect = pd.DataFrame({'a': [1, 2, 3]})
        score = get_data_quality_score(df_perfect)
        assert score == 100.0

        # lo contrario
        df_bad = pd.DataFrame({'a': [np.nan, np.nan]})
        score = get_data_quality_score(df_bad)
        assert score == 0.0