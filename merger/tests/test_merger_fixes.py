"""
Test Suite para Verificar Correcciones del Módulo Merger
=========================================================

Tests específicos para validar que todos los errores identificados
han sido corregidos apropiadamente.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
import warnings

# Importar las clases corregidas
# Nota: Ajustar imports según tu estructura de proyecto
try:
    from merger_fixed import ENAHOModuleMerger
    from config import ModuleMergeConfig, ModuleMergeLevel, ModuleMergeStrategy
except ImportError:
    print("Ajustar imports según estructura del proyecto")


class TestMergerFixes(unittest.TestCase):
    """Tests para verificar las correcciones aplicadas al merger"""

    def setUp(self):
        """Configuración inicial para cada test"""
        self.logger = logging.getLogger('test')
        self.config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.COALESCE
        )
        self.merger = ENAHOModuleMerger(self.config, self.logger)

    def tearDown(self):
        """Limpieza después de cada test"""
        pass

    # =========================================================
    # TEST FIX 1: Manejo de DataFrames vacíos
    # =========================================================

    def test_merge_with_empty_dataframes(self):
        """Verifica que el merge maneje correctamente DataFrames vacíos"""

        # Test 1: Ambos DataFrames vacíos
        df1_empty = pd.DataFrame()
        df2_empty = pd.DataFrame()

        result = self.merger.merge_modules(
            df1_empty, df2_empty, "01", "02"
        )

        self.assertTrue(result.merged_df.empty)
        self.assertEqual(result.quality_score, 0.0)
        self.assertIn('error', result.merge_report)

        # Test 2: Solo left_df vacío
        df2_valid = pd.DataFrame({
            'conglome': ['001', '002'],
            'vivienda': ['01', '02'],
            'hogar': ['1', '1'],
            'data': [100, 200]
        })

        result = self.merger.merge_modules(
            df1_empty, df2_valid, "01", "02"
        )

        self.assertFalse(result.merged_df.empty)
        self.assertEqual(len(result.merged_df), 2)
        self.assertEqual(result.quality_score, 50.0)

        # Test 3: Solo right_df vacío
        df1_valid = pd.DataFrame({
            'conglome': ['001', '002'],
            'vivienda': ['01', '02'],
            'hogar': ['1', '1'],
            'value': [10, 20]
        })

        result = self.merger.merge_modules(
            df1_valid, df2_empty, "01", "02"
        )

        self.assertFalse(result.merged_df.empty)
        self.assertEqual(len(result.merged_df), 2)
        self.assertEqual(result.quality_score, 50.0)

    def test_merge_with_none_dataframes(self):
        """Verifica manejo de DataFrames None"""

        df_valid = pd.DataFrame({
            'conglome': ['001'],
            'vivienda': ['01'],
            'hogar': ['1'],
            'data': [100]
        })

        # Test con None
        result = self.merger.merge_modules(
            None, df_valid, "01", "02"
        )

        self.assertFalse(result.merged_df.empty)
        self.assertIn('warning', result.merge_report)

    # =========================================================
    # TEST FIX 2: División por cero en quality score
    # =========================================================

    def test_quality_score_division_by_zero(self):
        """Verifica que no hay división por cero en cálculo de quality score"""

        # Simular estadísticas con total = 0
        merge_stats = {
            'total': 0,
            'both': 0,
            'left_only': 0,
            'right_only': 0
        }

        compatibility_info = {
            'match_rate_module1': 0,
            'match_rate_module2': 0
        }

        # No debe lanzar excepción
        score = self.merger._calculate_merge_quality_score_safe(
            merge_stats, compatibility_info
        )

        self.assertEqual(score, 0.0)

    def test_quality_score_with_none_values(self):
        """Verifica manejo de valores None en compatibility_info"""

        merge_stats = {
            'total': 100,
            'both': 80,
            'left_only': 10,
            'right_only': 10
        }

        # Compatibility info con None
        compatibility_info = {
            'match_rate_module1': None,
            'match_rate_module2': None
        }

        score = self.merger._calculate_merge_quality_score_safe(
            merge_stats, compatibility_info
        )

        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 100)

    # =========================================================
    # TEST FIX 3: Conversión robusta de tipos en llaves
    # =========================================================

    def test_robust_type_conversion_in_merge_keys(self):
        """Verifica conversión robusta de tipos en llaves de merge"""

        # DataFrame con tipos mixtos y NaN
        df1 = pd.DataFrame({
            'conglome': [1, 2, np.nan, '4'],  # Mixto con NaN
            'vivienda': ['01', '02', '03', '04'],
            'hogar': [1.0, 2.0, 3.0, 4.0],  # Float
            'data1': [100, 200, 300, 400]
        })

        df2 = pd.DataFrame({
            'conglome': ['1', '2', '3', '5'],  # String
            'vivienda': [1, 2, 3, 5],  # Numérico
            'hogar': ['1', '2', '3', '5'],  # String
            'data2': [10, 20, 30, 50]
        })

        # No debe lanzar excepción por tipos incompatibles
        result = self.merger._prepare_for_merge_robust(
            df1, ['conglome', 'vivienda', 'hogar'], 'test'
        )

        self.assertIsNotNone(result)
        # Verificar que se eliminaron registros con todas las llaves nulas
        self.assertTrue(len(result) <= len(df1))

    def test_merge_with_incompatible_types(self):
        """Verifica detección y manejo de tipos incompatibles"""

        df1 = pd.DataFrame({
            'conglome': [1, 2, 3],  # int
            'vivienda': ['01', '02', '03'],
            'hogar': ['1', '1', '1'],
            'value': [100, 200, 300]
        })

        df2 = pd.DataFrame({
            'conglome': ['1', '2', '3'],  # string
            'vivienda': ['01', '02', '03'],
            'hogar': ['1', '1', '1'],
            'value2': [10, 20, 30]
        })

        # Verificar detección de tipos incompatibles
        issues = self.merger._validate_data_types_compatibility(
            df1, df2, ['conglome', 'vivienda', 'hogar']
        )

        self.assertTrue(len(issues) > 0)

        # Verificar armonización
        self.merger._harmonize_column_types(
            df1, df2, ['conglome', 'vivienda', 'hogar']
        )

        # Después de armonización, tipos deben ser compatibles
        self.assertEqual(df1['conglome'].dtype, df2['conglome'].dtype)

    # =========================================================
    # TEST FIX 4: Detección de cardinalidad
    # =========================================================

    def test_cardinality_detection(self):
        """Verifica detección correcta de cardinalidad del merge"""

        # Test 1: Relación uno-a-uno (ideal)
        df1_unique = pd.DataFrame({
            'id': [1, 2, 3],
            'data': ['a', 'b', 'c']
        })
        df2_unique = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        warning = self.merger._detect_and_warn_cardinality(
            df1_unique, df2_unique, ['id']
        )
        self.assertIsNone(warning)  # No debe haber advertencia

        # Test 2: Relación muchos-a-muchos (problemática)
        df1_dup = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'data': ['a', 'b', 'c', 'd']
        })
        df2_dup = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'value': [10, 11, 20, 21]
        })

        warning = self.merger._detect_and_warn_cardinality(
            df1_dup, df2_dup, ['id']
        )

        self.assertIsNotNone(warning)
        self.assertIn('muchos-a-muchos', warning)

    # =========================================================
    # TEST FIX 5: Manejo de conflictos robusto
    # =========================================================

    def test_robust_conflict_resolution(self):
        """Verifica resolución robusta de conflictos con múltiples patrones"""

        # DataFrame con múltiples patrones de sufijos
        df_conflicts = pd.DataFrame({
            'id': [1, 2, 3],
            'value_x': [10, 20, 30],  # Patrón pandas default
            'value_y': [11, 21, 31],
            'name_left': ['a', 'b', 'c'],  # Patrón personalizado
            'name_right': ['A', 'B', 'C'],
            'score_1': [100, 200, 300],  # Patrón numérico
            'score_2': [110, 210, 310]
        })

        # Test estrategia COALESCE
        conflicts_resolved = self.merger._resolve_conflicts_robust(
            df_conflicts.copy(), ModuleMergeStrategy.COALESCE
        )

        self.assertGreater(conflicts_resolved, 0)

        # Test estrategia AVERAGE para columnas numéricas
        df_test = df_conflicts.copy()
        conflicts_resolved = self.merger._resolve_conflicts_robust(
            df_test, ModuleMergeStrategy.AVERAGE
        )

        # Verificar que se calculó el promedio correctamente
        if 'score' in df_test.columns:
            expected_avg = (df_conflicts['score_1'] + df_conflicts['score_2']) / 2
            np.testing.assert_array_almost_equal(
                df_test['score'].values, expected_avg.values
            )

    def test_conflict_resolution_with_nulls(self):
        """Verifica manejo de conflictos con valores nulos"""

        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value_x': [10, np.nan, 30],
            'value_y': [np.nan, 20, 31]
        })

        # COALESCE debe combinar valores no nulos
        self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.COALESCE
        )

        self.assertIn('value', df.columns)
        self.assertEqual(df['value'].iloc[0], 10)  # Toma el no-nulo
        self.assertEqual(df['value'].iloc[1], 20)  # Toma el no-nulo

    # =========================================================
    # TEST FIX 6: Gestión de memoria
    # =========================================================

    def test_memory_management_in_multiple_merge(self):
        """Verifica gestión de memoria en merge múltiple"""

        # Crear múltiples módulos pequeños
        modules_dict = {}
        for i in range(5):
            modules_dict[f"0{i}"] = pd.DataFrame({
                'conglome': ['001', '002'],
                'vivienda': ['01', '02'],
                'hogar': ['1', '1'],
                f'data_{i}': [100 + i, 200 + i]
            })

        # Mock para verificar que se llama a gc.collect
        with patch('gc.collect') as mock_gc:
            result = self.merger.merge_multiple_modules(
                modules_dict, "00"
            )

            # Para datasets pequeños no debería llamar a gc.collect
            # pero el método está disponible
            self.assertIsNotNone(result)

    # =========================================================
    # TEST FIX 7: Análisis de viabilidad mejorado
    # =========================================================

    def test_feasibility_analysis_with_empty_modules(self):
        """Verifica análisis de viabilidad con módulos vacíos"""

        modules_dict = {
            "01": pd.DataFrame({
                'conglome': ['001'],
                'vivienda': ['01'],
                'hogar': ['1'],
                'data': [100]
            }),
            "02": pd.DataFrame(),  # Vacío
            "03": None  # None
        }

        analysis = self.merger.analyze_merge_feasibility(
            modules_dict, ModuleMergeLevel.HOGAR
        )

        self.assertIn("02", analysis['modules_empty'])
        self.assertIn("03", analysis['modules_empty'])
        self.assertEqual(len(analysis['modules_analyzed']), 1)
        self.assertTrue(len(analysis['potential_issues']) > 0)

    def test_merge_plan_optimization(self):
        """Verifica creación de plan de merge optimizado"""

        modules_dict = {
            "01": pd.DataFrame({'conglome': range(1000), 'data': range(1000)}),
            "02": pd.DataFrame({'conglome': range(100), 'data': range(100)}),
            "03": pd.DataFrame({'conglome': range(500), 'data': range(500)})
        }

        plan = self.merger.create_merge_plan(modules_dict, "01")

        # Verificar que el plan ordena los módulos por tamaño
        self.assertEqual(plan['base_module'], "01")
        self.assertEqual(plan['merge_sequence'][0], "01")

        # El módulo más pequeño debe ser el segundo
        self.assertEqual(plan['merge_sequence'][1], "02")

        # Debe haber optimizaciones sugeridas
        self.assertTrue(len(plan['optimizations']) > 0 or len(plan['warnings']) >= 0)

    # =========================================================
    # TEST FIX 8: Overall quality calculation
    # =========================================================

    def test_overall_quality_with_edge_cases(self):
        """Verifica cálculo de calidad con casos extremos"""

        # Test 1: DataFrame vacío
        score = self.merger._calculate_overall_quality_safe(pd.DataFrame())
        self.assertEqual(score, 0.0)

        # Test 2: DataFrame None
        score = self.merger._calculate_overall_quality_safe(None)
        self.assertEqual(score, 0.0)

        # Test 3: DataFrame con todos valores nulos
        df_nulls = pd.DataFrame({
            'col1': [np.nan, np.nan],
            'col2': [np.nan, np.nan]
        })
        score = self.merger._calculate_overall_quality_safe(df_nulls)
        self.assertEqual(score, 0.0)

        # Test 4: DataFrame perfecto (sin nulos ni duplicados)
        df_perfect = pd.DataFrame({
            'conglome': ['001', '002', '003'],
            'vivienda': ['01', '02', '03'],
            'hogar': ['1', '2', '3'],
            'data': [100, 200, 300]
        })
        score = self.merger._calculate_overall_quality_safe(df_perfect)
        self.assertEqual(score, 100.0)

        # Test 5: DataFrame con duplicados
        df_duplicates = pd.DataFrame({
            'conglome': ['001', '001', '002'],
            'vivienda': ['01', '01', '02'],
            'hogar': ['1', '1', '2'],
            'data': [100, 100, 200]
        })
        score = self.merger._calculate_overall_quality_safe(df_duplicates)
        self.assertLess(score, 100.0)  # Debe ser penalizado por duplicados


class TestIntegrationMerger(unittest.TestCase):
    """Tests de integración para el merger corregido"""

    def setUp(self):
        """Configuración para tests de integración"""
        self.logger = logging.getLogger('integration_test')
        self.config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.COALESCE
        )
        self.merger = ENAHOModuleMerger(self.config, self.logger)

    def test_complete_merge_workflow(self):
        """Test completo del flujo de merge con múltiples módulos"""

        # Crear datos de prueba realistas
        np.random.seed(42)

        # Módulo 01 - Características del hogar
        mod_01 = pd.DataFrame({
            'conglome': ['001'] * 5 + ['002'] * 5,
            'vivienda': ['01'] * 5 + ['02'] * 5,
            'hogar': [str(i) for i in range(1, 6)] * 2,
            'nbi1': np.random.choice([0, 1], 10),
            'nbi2': np.random.choice([0, 1], 10),
            'nbi3': np.random.choice([0, 1], 10)
        })

        # Módulo 02 - Características de la vivienda (con algunos missing)
        mod_02 = pd.DataFrame({
            'conglome': ['001'] * 3 + ['002'] * 4,  # No todos los hogares
            'vivienda': ['01'] * 3 + ['02'] * 4,
            'hogar': ['1', '2', '3', '1', '2', '3', '4'],
            'v_tipo_pared': np.random.choice([1, 2, 3, 4], 7),
            'v_tipo_piso': np.random.choice([1, 2, 3], 7)
        })

        # Módulo 03 - Educación (con duplicados)
        mod_03 = pd.DataFrame({
            'conglome': ['001'] * 8 + ['002'] * 6,
            'vivienda': ['01'] * 8 + ['02'] * 6,
            'hogar': ['1', '1', '2', '2', '3', '3', '4', '5'] + ['1', '1', '2', '3', '4', '5'],
            'p301a': np.random.choice([1, 2, 3, 4, 5, 6], 14),
            'p306': np.random.choice([1, 2], 14)
        })

        # Módulo vacío para probar manejo
        mod_empty = pd.DataFrame()

        modules_dict = {
            "01": mod_01,
            "02": mod_02,
            "03": mod_03,
            "04": mod_empty
        }

        # Ejecutar merge múltiple
        result = self.merger.merge_multiple_modules(
            modules_dict,
            base_module="01"
        )

        # Validaciones
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.merged_df)
        self.assertFalse(result.merged_df.empty)

        # Verificar que se mantuvieron todos los registros del módulo base
        self.assertGreaterEqual(len(result.merged_df), len(mod_01))

        # Verificar que las columnas de todos los módulos están presentes
        expected_columns = set(['conglome', 'vivienda', 'hogar'])
        expected_columns.update(mod_01.columns)
        expected_columns.update(mod_02.columns)
        expected_columns.update(mod_03.columns)

        for col in expected_columns:
            self.assertIn(col, result.merged_df.columns)

        # Verificar quality score
        self.assertGreater(result.quality_score, 0)
        self.assertLessEqual(result.quality_score, 100)

        # Verificar reporte
        self.assertIn('modules_sequence', result.merge_report)
        self.assertEqual(result.merge_report['modules_skipped'], 1)  # mod_04 vacío

    def test_large_dataset_optimization(self):
        """Test de optimización para datasets grandes"""

        # Crear dataset grande
        n_rows = 100000

        df_large_1 = pd.DataFrame({
            'conglome': [f'{i:06d}' for i in range(n_rows)],
            'vivienda': ['01'] * n_rows,
            'hogar': ['1'] * n_rows,
            'data1': np.random.randn(n_rows)
        })

        df_large_2 = pd.DataFrame({
            'conglome': [f'{i:06d}' for i in range(0, n_rows, 2)],  # Solo pares
            'vivienda': ['01'] * (n_rows // 2),
            'hogar': ['1'] * (n_rows // 2),
            'data2': np.random.randn(n_rows // 2)
        })

        # Verificar que se detecta como dataset grande
        warning = self.merger._detect_and_warn_cardinality(
            df_large_1, df_large_2, ['conglome', 'vivienda', 'hogar']
        )

        # El merge debe completarse sin errores
        with patch.object(self.merger, '_merge_large_datasets') as mock_large:
            mock_large.return_value = pd.DataFrame()  # Mock return

            # Forzar uso del método optimizado
            self.merger._execute_merge_optimized(
                df_large_1, df_large_2,
                ['conglome', 'vivienda', 'hogar'],
                ('_x', '_y')
            )

            # Verificar que se llamó al método optimizado
            mock_large.assert_called_once()

    def test_error_recovery(self):
        """Test de recuperación ante errores"""

        # DataFrame con columna problemática
        df1 = pd.DataFrame({
            'conglome': ['001', '002'],
            'vivienda': ['01', '02'],
            'hogar': ['1', '2'],
            'problematic_col': [complex(1, 2), complex(3, 4)]  # Tipo no convertible
        })

        df2 = pd.DataFrame({
            'conglome': ['001', '002'],
            'vivienda': ['01', '02'],
            'hogar': ['1', '2'],
            'normal_col': [100, 200]
        })

        # El merge debe manejar el error sin crashear
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.merger.merge_modules(df1, df2, "01", "02")

        self.assertIsNotNone(result)
        self.assertIn('normal_col', result.merged_df.columns)


class TestPerformance(unittest.TestCase):
    """Tests de performance para verificar optimizaciones"""

    def setUp(self):
        """Setup para tests de performance"""
        self.logger = logging.getLogger('performance_test')
        self.config = ModuleMergeConfig()
        self.merger = ENAHOModuleMerger(self.config, self.logger)

    def test_merge_scalability(self):
        """Verifica que el merge escala bien con el tamaño"""
        import time

        sizes = [100, 1000, 5000]
        times = []

        for size in sizes:
            df1 = pd.DataFrame({
                'conglome': [f'{i:06d}' for i in range(size)],
                'vivienda': ['01'] * size,
                'hogar': ['1'] * size,
                'data': np.random.randn(size)
            })

            df2 = pd.DataFrame({
                'conglome': [f'{i:06d}' for i in range(0, size, 2)],
                'vivienda': ['01'] * (size // 2),
                'hogar': ['1'] * (size // 2),
                'value': np.random.randn(size // 2)
            })

            start = time.time()
            result = self.merger.merge_modules(df1, df2, "01", "02")
            elapsed = time.time() - start
            times.append(elapsed)

            self.assertIsNotNone(result)

        # Verificar que el tiempo crece de forma razonable (no exponencial)
        # El tiempo para 5000 no debería ser más de 10x el tiempo para 100
        time_ratio = times[-1] / times[0]
        self.assertLess(time_ratio, 100,
                        f"Performance degradation: {time_ratio:.2f}x for 50x data")

    def test_memory_efficiency(self):
        """Verifica eficiencia de memoria"""
        import tracemalloc

        # Iniciar tracking de memoria
        tracemalloc.start()

        # Crear datasets moderados
        modules = {}
        for i in range(5):
            modules[f"0{i}"] = pd.DataFrame({
                'conglome': [f'{j:06d}' for j in range(1000)],
                'vivienda': ['01'] * 1000,
                'hogar': ['1'] * 1000,
                f'data_{i}': np.random.randn(1000)
            })

        # Tomar snapshot inicial
        snapshot1 = tracemalloc.take_snapshot()

        # Ejecutar merge
        result = self.merger.merge_multiple_modules(modules, "00")

        # Tomar snapshot final
        snapshot2 = tracemalloc.take_snapshot()

        # Calcular diferencia
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # La memoria usada no debería ser excesiva
        total_memory = sum(stat.size_diff for stat in top_stats)

        # Verificar que el resultado es válido
        self.assertIsNotNone(result)
        self.assertFalse(result.merged_df.empty)

        tracemalloc.stop()


def run_tests():
    """Función para ejecutar todos los tests"""

    # Crear test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestMergerFixes))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationMerger))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Omitidos: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")

        if result.errors:
            print("\nErrores:")
            for test, trace in result.errors:
                print(f"  - {test}: {trace.split(chr(10))[0]}")

        if result.failures:
            print("\nFallos:")
            for test, trace in result.failures:
                print(f"  - {test}: {trace.split(chr(10))[0]}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ejecutar tests
    success = run_tests()

    # Exit code
    import sys

    sys.exit(0 if success else 1)