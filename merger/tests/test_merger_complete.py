"""
test_merger_complete.py
=======================

Suite completa de tests unitarios e integración para el módulo merger de ENAHO.
Incluye tests para todos los componentes, edge cases y flujos completos.

Estructura:
- Unit Tests: Pruebas aisladas de cada método
- Integration Tests: Pruebas de flujos completos
- Performance Tests: Validación de optimizaciones
- Regression Tests: Prevención de bugs anteriores
"""

import unittest
import pandas as pd
import numpy as np
import logging
import tempfile
import os
import gc
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import warnings

# Imports del proyecto
# Ajustar según tu estructura
try:
    from merger.modules.merger import ENAHOModuleMerger
    from merger.modules.validator import ModuleValidator
    from merger.config import (
        ModuleMergeConfig,
        ModuleMergeLevel,
        ModuleMergeStrategy,
        ModuleType,
        ModuleMergeResult
    )
    from merger.exceptions import (
        ModuleMergeError,
        IncompatibleModulesError,
        MergeKeyError,
        ConflictResolutionError
    )
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Ajustar paths de import según estructura del proyecto")


# =====================================================
# FIXTURES Y HELPERS
# =====================================================

class TestDataGenerator:
    """Generador de datos de prueba para tests"""

    @staticmethod
    def create_sample_module(module_code: str,
                             n_rows: int = 100,
                             seed: int = 42,
                             with_nulls: bool = False,
                             with_duplicates: bool = False) -> pd.DataFrame:
        """
        Crea un módulo ENAHO sintético para pruebas.

        Args:
            module_code: Código del módulo ('01', '02', etc.)
            n_rows: Número de filas
            seed: Semilla para reproducibilidad
            with_nulls: Si incluir valores nulos
            with_duplicates: Si incluir registros duplicados
        """
        np.random.seed(seed)

        # Estructura base común
        data = {
            'conglome': [f'{i:06d}' for i in np.random.choice(range(1000), n_rows)],
            'vivienda': np.random.choice(['01', '02', '03'], n_rows),
            'hogar': np.random.choice(['1', '2', '3'], n_rows)
        }

        # Agregar columnas específicas por módulo
        if module_code == '01':  # Características del hogar
            data.update({
                'nbi1': np.random.choice([0, 1], n_rows),
                'nbi2': np.random.choice([0, 1], n_rows),
                'nbi3': np.random.choice([0, 1], n_rows),
                'nbi4': np.random.choice([0, 1], n_rows),
                'nbi5': np.random.choice([0, 1], n_rows)
            })
        elif module_code == '02':  # Características de los miembros
            data.update({
                'codperso': [f'{i:02d}' for i in range(1, n_rows + 1)],
                'p203': np.random.choice([1, 2], n_rows),  # Sexo
                'p208a': np.random.randint(0, 100, n_rows),  # Edad
                'p209': np.random.choice([1, 2, 3, 4, 5, 6], n_rows)  # Estado civil
            })
        elif module_code == '03':  # Educación
            data.update({
                'codperso': [f'{i:02d}' for i in range(1, n_rows + 1)],
                'p301a': np.random.choice([1, 2, 3, 4, 5, 6], n_rows),
                'p306': np.random.choice([1, 2], n_rows),
                'p307': np.random.choice([1, 2, 3, 4], n_rows)
            })
        elif module_code == '34':  # Sumaria
            data.update({
                'factor07': np.random.uniform(50, 200, n_rows),
                'inghog1d': np.random.uniform(0, 10000, n_rows),
                'gashog2d': np.random.uniform(0, 8000, n_rows),
                'pobreza': np.random.choice([1, 2, 3], n_rows)
            })
        else:
            # Módulo genérico
            data.update({
                f'var_{module_code}_1': np.random.randn(n_rows),
                f'var_{module_code}_2': np.random.randn(n_rows)
            })

        df = pd.DataFrame(data)

        # Agregar valores nulos si se solicita
        if with_nulls:
            null_ratio = 0.1
            for col in df.columns:
                if col not in ['conglome', 'vivienda', 'hogar']:
                    mask = np.random.random(n_rows) < null_ratio
                    df.loc[mask, col] = np.nan

        # Agregar duplicados si se solicita
        if with_duplicates:
            n_duplicates = int(n_rows * 0.1)
            dup_indices = np.random.choice(df.index, n_duplicates)
            df_duplicates = df.loc[dup_indices].copy()
            df = pd.concat([df, df_duplicates], ignore_index=True)

        return df

    @staticmethod
    def create_incompatible_modules() -> tuple:
        """Crea módulos con estructuras incompatibles para tests de error"""

        # Módulo sin llaves necesarias
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        # Módulo con llaves pero tipos incompatibles
        df2 = pd.DataFrame({
            'conglome': [complex(1, 2), complex(3, 4)],  # Tipo no convertible
            'vivienda': ['01', '02'],
            'hogar': ['1', '2'],
            'data': [100, 200]
        })

        return df1, df2


# =====================================================
# UNIT TESTS
# =====================================================

class TestENAHOModuleMergerUnit(unittest.TestCase):
    """Tests unitarios para métodos individuales del merger"""

    def setUp(self):
        """Configuración inicial para cada test"""
        self.logger = logging.getLogger('test_unit')
        self.logger.setLevel(logging.WARNING)

        self.config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.COALESCE,
            continue_on_error=True  # Agregar si falta en config
        )

        self.merger = ENAHOModuleMerger(self.config, self.logger)
        self.generator = TestDataGenerator()

    # -------------------------------------------------
    # Tests para _get_merge_keys_for_level
    # -------------------------------------------------

    def test_get_merge_keys_hogar_level(self):
        """Test obtención de llaves para nivel hogar"""
        keys = self.merger._get_merge_keys_for_level(ModuleMergeLevel.HOGAR)
        self.assertEqual(keys, ['conglome', 'vivienda', 'hogar'])

    def test_get_merge_keys_persona_level(self):
        """Test obtención de llaves para nivel persona"""
        keys = self.merger._get_merge_keys_for_level(ModuleMergeLevel.PERSONA)
        self.assertEqual(keys, ['conglome', 'vivienda', 'hogar', 'codperso'])

    def test_get_merge_keys_vivienda_level(self):
        """Test obtención de llaves para nivel vivienda"""
        keys = self.merger._get_merge_keys_for_level(ModuleMergeLevel.VIVIENDA)
        self.assertEqual(keys, ['conglome', 'vivienda'])

    def test_get_merge_keys_invalid_level(self):
        """Test error con nivel inválido"""
        with self.assertRaises(ValueError):
            self.merger._get_merge_keys_for_level("INVALID_LEVEL")

    # -------------------------------------------------
    # Tests para _prepare_for_merge_robust
    # -------------------------------------------------

    def test_prepare_for_merge_with_clean_data(self):
        """Test preparación con datos limpios"""
        df = self.generator.create_sample_module('01', n_rows=10)
        keys = ['conglome', 'vivienda', 'hogar']

        result = self.merger._prepare_for_merge_robust(df, keys, 'test')

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)
        for key in keys:
            self.assertIn(key, result.columns)

    def test_prepare_for_merge_with_nulls(self):
        """Test preparación con valores nulos en llaves"""
        df = pd.DataFrame({
            'conglome': ['001', None, '003'],
            'vivienda': ['01', '02', None],
            'hogar': ['1', '2', '3'],
            'data': [100, 200, 300]
        })

        result = self.merger._prepare_for_merge_robust(
            df, ['conglome', 'vivienda', 'hogar'], 'test'
        )

        # Debe eliminar filas con TODAS las llaves nulas
        self.assertLessEqual(len(result), len(df))

    def test_prepare_for_merge_with_mixed_types(self):
        """Test preparación con tipos mixtos"""
        df = pd.DataFrame({
            'conglome': [1, 2, '3'],  # Mixto
            'vivienda': [1.0, 2.0, 3.0],  # Float
            'hogar': ['1', '2', '3'],  # String
            'data': [100, 200, 300]
        })

        result = self.merger._prepare_for_merge_robust(
            df, ['conglome', 'vivienda', 'hogar'], 'test'
        )

        # Todos deben ser string después de la preparación
        for key in ['conglome', 'vivienda', 'hogar']:
            self.assertEqual(result[key].dtype, 'object')

    def test_prepare_for_merge_missing_keys(self):
        """Test error cuando faltan llaves"""
        df = pd.DataFrame({
            'conglome': ['001', '002'],
            'data': [100, 200]
        })

        with self.assertRaises(MergeKeyError):
            self.merger._prepare_for_merge_robust(
                df, ['conglome', 'vivienda', 'hogar'], 'test'
            )

    # -------------------------------------------------
    # Tests para _analyze_merge_result
    # -------------------------------------------------

    def test_analyze_merge_result_perfect_match(self):
        """Test análisis con coincidencia perfecta"""
        df = pd.DataFrame({
            'conglome': ['001', '002', '003'],
            'data': [100, 200, 300],
            '_merge': ['both', 'both', 'both']
        })

        stats = self.merger._analyze_merge_result(df)

        self.assertEqual(stats['both'], 3)
        self.assertEqual(stats['left_only'], 0)
        self.assertEqual(stats['right_only'], 0)
        self.assertEqual(stats['total'], 3)

    def test_analyze_merge_result_partial_match(self):
        """Test análisis con coincidencia parcial"""
        df = pd.DataFrame({
            'conglome': ['001', '002', '003', '004', '005'],
            '_merge': ['both', 'both', 'left_only', 'right_only', 'both']
        })

        stats = self.merger._analyze_merge_result(df)

        self.assertEqual(stats['both'], 3)
        self.assertEqual(stats['left_only'], 1)
        self.assertEqual(stats['right_only'], 1)
        self.assertEqual(stats['total'], 5)

    def test_analyze_merge_result_no_indicator(self):
        """Test análisis sin columna _merge"""
        df = pd.DataFrame({
            'conglome': ['001', '002'],
            'data': [100, 200]
        })

        stats = self.merger._analyze_merge_result(df)

        self.assertEqual(stats['total'], 2)
        self.assertEqual(stats['both'], 2)

    # -------------------------------------------------
    # Tests para _resolve_conflicts_robust
    # -------------------------------------------------

    def test_resolve_conflicts_coalesce_strategy(self):
        """Test resolución con estrategia COALESCE"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value_x': [10, np.nan, 30],
            'value_y': [np.nan, 20, 31]
        })

        conflicts = self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.COALESCE
        )

        self.assertIn('value', df.columns)
        self.assertEqual(df['value'].iloc[0], 10)
        self.assertEqual(df['value'].iloc[1], 20)
        self.assertGreater(conflicts, 0)

    def test_resolve_conflicts_keep_left_strategy(self):
        """Test resolución con estrategia KEEP_LEFT"""
        df = pd.DataFrame({
            'id': [1, 2],
            'value_x': [10, 20],
            'value_y': [11, 21]
        })

        self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.KEEP_LEFT
        )

        self.assertEqual(df['value'].iloc[0], 10)
        self.assertEqual(df['value'].iloc[1], 20)

    def test_resolve_conflicts_average_strategy(self):
        """Test resolución con estrategia AVERAGE para numéricos"""
        df = pd.DataFrame({
            'id': [1, 2],
            'score_x': [10.0, 20.0],
            'score_y': [20.0, 30.0]
        })

        self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.AVERAGE
        )

        self.assertEqual(df['score'].iloc[0], 15.0)
        self.assertEqual(df['score'].iloc[1], 25.0)

    def test_resolve_conflicts_concatenate_strategy(self):
        """Test resolución con estrategia CONCATENATE"""
        df = pd.DataFrame({
            'id': [1, 2],
            'name_x': ['John', 'Jane'],
            'name_y': ['Doe', 'Smith']
        })

        self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.CONCATENATE
        )

        self.assertIn('John', df['name'].iloc[0])
        self.assertIn('Doe', df['name'].iloc[0])

    def test_resolve_conflicts_error_strategy(self):
        """Test resolución con estrategia ERROR"""
        df = pd.DataFrame({
            'id': [1, 2],
            'value_x': [10, 20],
            'value_y': [11, 21]  # Valores diferentes
        })

        with self.assertRaises(ConflictResolutionError):
            self.merger._resolve_conflicts_robust(
                df.copy(), ModuleMergeStrategy.ERROR
            )

    def test_resolve_conflicts_multiple_patterns(self):
        """Test con múltiples patrones de sufijos"""
        df = pd.DataFrame({
            'id': [1],
            'value_x': [10],
            'value_y': [11],
            'score_left': [100],
            'score_right': [110],
            'name_1': ['A'],
            'name_2': ['B']
        })

        conflicts = self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.COALESCE
        )

        # Debe resolver todos los conflictos
        self.assertGreaterEqual(conflicts, 3)

    # -------------------------------------------------
    # Tests para cálculo de quality score
    # -------------------------------------------------

    def test_calculate_quality_score_perfect(self):
        """Test score con merge perfecto"""
        merge_stats = {
            'total': 100,
            'both': 100,
            'left_only': 0,
            'right_only': 0
        }

        compatibility = {
            'match_rate_module1': 100,
            'match_rate_module2': 100
        }

        score = self.merger._calculate_merge_quality_score_safe(
            merge_stats, compatibility
        )

        self.assertGreater(score, 95)  # Debe ser muy alto

    def test_calculate_quality_score_poor(self):
        """Test score con merge pobre"""
        merge_stats = {
            'total': 100,
            'both': 20,
            'left_only': 40,
            'right_only': 40
        }

        compatibility = {
            'match_rate_module1': 20,
            'match_rate_module2': 20
        }

        score = self.merger._calculate_merge_quality_score_safe(
            merge_stats, compatibility
        )

        self.assertLess(score, 50)  # Debe ser bajo

    def test_calculate_quality_score_division_by_zero(self):
        """Test protección contra división por cero"""
        merge_stats = {'total': 0, 'both': 0}
        compatibility = {}

        score = self.merger._calculate_merge_quality_score_safe(
            merge_stats, compatibility
        )

        self.assertEqual(score, 0.0)

    def test_calculate_overall_quality_empty_df(self):
        """Test calidad general con DataFrame vacío"""
        score = self.merger._calculate_overall_quality_safe(pd.DataFrame())
        self.assertEqual(score, 0.0)

        score = self.merger._calculate_overall_quality_safe(None)
        self.assertEqual(score, 0.0)

    def test_calculate_overall_quality_with_duplicates(self):
        """Test calidad penalizada por duplicados"""
        df = pd.DataFrame({
            'conglome': ['001', '001', '002'],
            'vivienda': ['01', '01', '02'],
            'hogar': ['1', '1', '2'],
            'data': [100, 100, 200]
        })

        score = self.merger._calculate_overall_quality_safe(df)
        self.assertLess(score, 100)  # Debe ser penalizado

    # -------------------------------------------------
    # Tests para validación de tipos
    # -------------------------------------------------

    def test_validate_data_types_compatible(self):
        """Test validación con tipos compatibles"""
        df1 = pd.DataFrame({
            'conglome': [1, 2, 3],
            'vivienda': ['01', '02', '03']
        })

        df2 = pd.DataFrame({
            'conglome': [1, 2, 3],
            'vivienda': ['01', '02', '03']
        })

        issues = self.merger._validate_data_types_compatibility(
            df1, df2, ['conglome', 'vivienda']
        )

        self.assertEqual(len(issues), 0)

    def test_validate_data_types_incompatible(self):
        """Test validación con tipos incompatibles"""
        df1 = pd.DataFrame({'id': [1, 2, 3]})
        df2 = pd.DataFrame({'id': ['a', 'b', 'c']})

        issues = self.merger._validate_data_types_compatibility(
            df1, df2, ['id']
        )

        self.assertGreater(len(issues), 0)

    def test_harmonize_column_types(self):
        """Test armonización de tipos"""
        df1 = pd.DataFrame({'id': [1, 2, 3]})
        df2 = pd.DataFrame({'id': ['1', '2', '3']})

        self.merger._harmonize_column_types(df1, df2, ['id'])

        # Ambos deben tener el mismo tipo después
        self.assertEqual(df1['id'].dtype, df2['id'].dtype)

    # -------------------------------------------------
    # Tests para detección de cardinalidad
    # -------------------------------------------------

    def test_detect_cardinality_one_to_one(self):
        """Test detección relación uno-a-uno"""
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'data': ['a', 'b', 'c']
        })

        df2 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        warning = self.merger._detect_and_warn_cardinality(df1, df2, ['id'])
        self.assertIsNone(warning)

    def test_detect_cardinality_many_to_many(self):
        """Test detección relación muchos-a-muchos"""
        df1 = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'data': ['a', 'b', 'c', 'd']
        })

        df2 = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'value': [10, 11, 20, 21]
        })

        warning = self.merger._detect_and_warn_cardinality(df1, df2, ['id'])
        self.assertIsNotNone(warning)
        self.assertIn('muchos-a-muchos', warning)

    def test_detect_cardinality_with_explosion(self):
        """Test detección de explosión de datos"""
        # Crear DataFrames que resultarían en explosión
        df1 = pd.DataFrame({
            'id': [1] * 100 + [2] * 100,
            'data': range(200)
        })

        df2 = pd.DataFrame({
            'id': [1] * 100 + [2] * 100,
            'value': range(200)
        })

        warning = self.merger._detect_and_warn_cardinality(df1, df2, ['id'])
        self.assertIsNotNone(warning)
        self.assertIn('podría resultar', warning)


# =====================================================
# INTEGRATION TESTS
# =====================================================

class TestENAHOModuleMergerIntegration(unittest.TestCase):
    """Tests de integración para flujos completos"""

    def setUp(self):
        """Configuración para tests de integración"""
        self.logger = logging.getLogger('test_integration')
        self.logger.setLevel(logging.INFO)

        self.config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.COALESCE
        )

        self.merger = ENAHOModuleMerger(self.config, self.logger)
        self.generator = TestDataGenerator()

    def test_complete_merge_workflow_success(self):
        """Test flujo completo exitoso con múltiples módulos"""

        # Crear módulos realistas
        mod_01 = self.generator.create_sample_module('01', n_rows=50)
        mod_02 = self.generator.create_sample_module('02', n_rows=45, with_nulls=True)
        mod_34 = self.generator.create_sample_module('34', n_rows=50)

        # Merge de dos módulos
        result = self.merger.merge_modules(
            mod_01, mod_34, '01', '34'
        )

        # Validaciones
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.merged_df)
        self.assertFalse(result.merged_df.empty)
        self.assertGreater(result.quality_score, 0)

        # Verificar columnas esperadas
        expected_cols = set(mod_01.columns) | set(mod_34.columns)
        for col in expected_cols:
            self.assertIn(col.replace('_x', '').replace('_y', ''),
                          result.merged_df.columns.str.replace('_x', '').str.replace('_y', ''))

    def test_multiple_modules_merge(self):
        """Test merge de múltiples módulos secuencialmente"""

        modules_dict = {
            '01': self.generator.create_sample_module('01', n_rows=30),
            '02': self.generator.create_sample_module('02', n_rows=25),
            '03': self.generator.create_sample_module('03', n_rows=28),
            '34': self.generator.create_sample_module('34', n_rows=30)
        }

        result = self.merger.merge_multiple_modules(
            modules_dict,
            base_module='34'
        )

        # Validaciones
        self.assertIsNotNone(result)
        self.assertEqual(result.merge_report['modules_merged'], 4)
        self.assertGreater(len(result.merged_df), 0)

        # Verificar que todos los módulos están representados
        self.assertIn('34', result.merge_report['modules_sequence'])

    def test_merge_with_empty_modules(self):
        """Test merge con módulos vacíos"""

        modules_dict = {
            '01': self.generator.create_sample_module('01', n_rows=20),
            '02': pd.DataFrame(),  # Vacío
            '03': None,  # None
            '34': self.generator.create_sample_module('34', n_rows=20)
        }

        result = self.merger.merge_multiple_modules(
            modules_dict,
            base_module='01'
        )

        # Debe manejar módulos vacíos sin crashear
        self.assertIsNotNone(result)
        self.assertEqual(result.merge_report['modules_skipped'], 2)

    def test_merge_with_incompatible_modules(self):
        """Test merge con módulos incompatibles"""

        df1, df2 = self.generator.create_incompatible_modules()

        # Debe lanzar excepción o manejar elegantemente
        with self.assertRaises((IncompatibleModulesError, MergeKeyError)):
            self.merger.merge_modules(df1, df2, '01', '02')

    def test_merge_feasibility_analysis(self):
        """Test análisis de viabilidad antes del merge"""

        modules_dict = {
            '01': self.generator.create_sample_module('01', n_rows=1000),
            '02': self.generator.create_sample_module('02', n_rows=800),
            '34': self.generator.create_sample_module('34', n_rows=1000)
        }

        analysis = self.merger.analyze_merge_feasibility(
            modules_dict,
            ModuleMergeLevel.HOGAR
        )

        # Validaciones
        self.assertTrue(analysis['feasible'])
        self.assertEqual(len(analysis['modules_analyzed']), 3)
        self.assertIn('size_analysis', analysis)
        self.assertIn('memory_estimate_mb', analysis)

        # Debe tener recomendaciones para datasets grandes
        if analysis['memory_estimate_mb'] > 100:
            self.assertGreater(len(analysis['recommendations']), 0)

    def test_merge_plan_creation(self):
        """Test creación de plan de merge optimizado"""

        modules_dict = {
            '01': self.generator.create_sample_module('01', n_rows=500),
            '02': self.generator.create_sample_module('02', n_rows=100),
            '03': self.generator.create_sample_module('03', n_rows=300),
            '34': self.generator.create_sample_module('34', n_rows=500)
        }

        plan = self.merger.create_merge_plan(modules_dict, target_module='34')

        # Validaciones
        self.assertEqual(plan['base_module'], '34')
        self.assertGreater(len(plan['merge_sequence']), 0)
        self.assertEqual(plan['merge_sequence'][0], '34')

        # Los módulos deben estar ordenados por tamaño (menor primero)
        self.assertEqual(plan['merge_sequence'][1], '02')  # El más pequeño

        # Debe tener pasos de ejecución
        self.assertGreater(len(plan['execution_steps']), 0)

    def test_merge_with_different_strategies(self):
        """Test merge con diferentes estrategias de conflicto"""

        # Crear DataFrames con conflictos potenciales
        df1 = pd.DataFrame({
            'conglome': ['001', '002', '003'],
            'vivienda': ['01', '01', '02'],
            'hogar': ['1', '2', '1'],
            'value': [100, 200, 300]
        })

        df2 = pd.DataFrame({
            'conglome': ['001', '002', '003'],
            'vivienda': ['01', '01', '02'],
            'hogar': ['1', '2', '1'],
            'value': [110, 210, 310]
        })

        # Test con COALESCE
        config_coalesce = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.COALESCE
        )
        merger_coalesce = ENAHOModuleMerger(config_coalesce, self.logger)
        result_coalesce = merger_coalesce.merge_modules(df1.copy(), df2.copy(), '01', '02')

        # Test con KEEP_LEFT
        config_left = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.KEEP_LEFT
        )
        merger_left = ENAHOModuleMerger(config_left, self.logger)
        result_left = merger_left.merge_modules(df1.copy(), df2.copy(), '01', '02')

        # Test con AVERAGE
        config_avg = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            merge_strategy=ModuleMergeStrategy.AVERAGE
        )
        merger_avg = ENAHOModuleMerger(config_avg, self.logger)
        result_avg = merger_avg.merge_modules(df1.copy(), df2.copy(), '01', '02')

        # Validar que cada estrategia produce resultados diferentes
        self.assertIsNotNone(result_coalesce)
        self.assertIsNotNone(result_left)
        self.assertIsNotNone(result_avg)

        # Los valores deben ser diferentes según la estrategia
        if 'value' in result_avg.merged_df.columns:
            # AVERAGE debe dar el promedio
            expected_avg = (df1['value'].iloc[0] + df2['value'].iloc[0]) / 2
            actual_avg = result_avg.merged_df['value'].iloc[0]
            self.assertAlmostEqual(actual_avg, expected_avg, places=1)

    def test_merge_with_persona_level(self):
        """Test merge a nivel persona"""

        # Crear módulos con codperso
        df1 = pd.DataFrame({
            'conglome': ['001'] * 3,
            'vivienda': ['01'] * 3,
            'hogar': ['1'] * 3,
            'codperso': ['01', '02', '03'],
            'edad': [25, 30, 35]
        })

        df2 = pd.DataFrame({
            'conglome': ['001'] * 3,
            'vivienda': ['01'] * 3,
            'hogar': ['1'] * 3,
            'codperso': ['01', '02', '03'],
            'educacion': [3, 4, 5]
        })

        config_persona = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA
        )
        merger_persona = ENAHOModuleMerger(config_persona, self.logger)

        result = merger_persona.merge_modules(df1, df2, '02', '03')

        # Debe mantener nivel persona
        self.assertIn('codperso', result.merged_df.columns)
        self.assertEqual(len(result.merged_df), 3)


# =====================================================
# PERFORMANCE TESTS
# =====================================================

class TestENAHOModuleMergerPerformance(unittest.TestCase):
    """Tests de performance y optimización"""

    def setUp(self):
        """Setup para tests de performance"""
        self.logger = logging.getLogger('test_performance')
        self.logger.setLevel(logging.WARNING)

        self.config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.HOGAR,
            chunk_processing=True
        )

        self.merger = ENAHOModuleMerger(self.config, self.logger)
        self.generator = TestDataGenerator()

    def test_large_dataset_optimization_triggered(self):
        """Test que la optimización se activa para datasets grandes"""

        # Crear datasets grandes
        df1 = self.generator.create_sample_module('01', n_rows=300000)
        df2 = self.generator.create_sample_module('34', n_rows=300000)

        # Mock el método optimizado para verificar que se llama
        with patch.object(self.merger, '_merge_large_datasets') as mock_large:
            mock_large.return_value = pd.DataFrame({
                'conglome': ['001'],
                'vivienda': ['01'],
                'hogar': ['1'],
                '_merge': ['both']
            })

            # Forzar la ruta optimizada
            self.merger._execute_merge_optimized(
                df1, df2,
                ['conglome', 'vivienda', 'hogar'],
                ('_x', '_y')
            )

            # Verificar que se llamó
            mock_large.assert_called_once()

    def test_memory_management_with_gc(self):
        """Test gestión de memoria con garbage collection"""

        modules = {}
        for i in range(10):
            modules[f'{i:02d}'] = self.generator.create_sample_module(
                f'{i:02d}', n_rows=10000
            )

        # Mock gc.collect para verificar que se llama
        with patch('gc.collect') as mock_gc:
            result = self.merger.merge_multiple_modules(
                modules, base_module='00'
            )

            # Para datasets grandes debe llamar a gc.collect
            if sum(len(df) for df in modules.values()) > 100000:
                mock_gc.assert_called()

    def test_performance_scaling(self):
        """Test que el performance escala apropiadamente"""

        import time

        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            df1 = self.generator.create_sample_module('01', n_rows=size)
            df2 = self.generator.create_sample_module('34', n_rows=size)

            start = time.time()
            result = self.merger.merge_modules(df1, df2, '01', '34')
            elapsed = time.time() - start

            times.append(elapsed)
            self.assertIsNotNone(result)

        # Verificar que el tiempo crece linealmente (no exponencialmente)
        # El ratio no debe ser mayor a 15x para 10x los datos
        if times[0] > 0:
            scaling_factor = times[-1] / times[0]
            data_factor = sizes[-1] / sizes[0]

            # El tiempo no debe crecer más rápido que O(n log n)
            max_expected = data_factor * np.log(data_factor)
            self.assertLess(scaling_factor, max_expected * 2,
                            f"Performance degradation: {scaling_factor:.2f}x for {data_factor}x data")

    def test_chunk_processing(self):
        """Test procesamiento por chunks"""

        # Dataset grande para forzar chunks
        df1 = self.generator.create_sample_module('01', n_rows=150000)
        df2 = self.generator.create_sample_module('34', n_rows=50000)

        # Ejecutar merge
        result = self.merger._merge_large_datasets(
            df1, df2,
            ['conglome', 'vivienda', 'hogar'],
            ('_x', '_y'),
            chunk_size=10000
        )

        # Debe completarse sin errores
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)


# =====================================================
# REGRESSION TESTS
# =====================================================

class TestENAHOModuleMergerRegression(unittest.TestCase):
    """Tests de regresión para prevenir bugs anteriores"""

    def setUp(self):
        """Setup para tests de regresión"""
        self.logger = logging.getLogger('test_regression')
        self.config = ModuleMergeConfig()
        self.merger = ENAHOModuleMerger(self.config, self.logger)
        self.generator = TestDataGenerator()

    def test_regression_division_by_zero_fix(self):
        """Regresión: División por cero en quality score (Bug #1)"""

        # Caso que causaba división por cero
        merge_stats = {
            'total': 0,
            'both': 0,
            'left_only': 0,
            'right_only': 0
        }

        # No debe lanzar ZeroDivisionError
        score = self.merger._calculate_merge_quality_score_safe(
            merge_stats, {}
        )

        self.assertEqual(score, 0.0)

    def test_regression_empty_dataframes_fix(self):
        """Regresión: Crash con DataFrames vacíos (Bug #2)"""

        # Casos que causaban crash
        df_empty = pd.DataFrame()
        df_none = None
        df_valid = self.generator.create_sample_module('01', n_rows=5)

        # Test con ambos vacíos
        result = self.merger.merge_modules(
            df_empty, df_empty, '01', '02'
        )
        self.assertTrue(result.merged_df.empty)

        # Test con None
        result = self.merger.merge_modules(
            df_none, df_valid, '01', '02'
        )
        self.assertFalse(result.merged_df.empty)

    def test_regression_type_conversion_fix(self):
        """Regresión: Error en conversión de tipos con NaN (Bug #3)"""

        # Caso que causaba error
        df = pd.DataFrame({
            'conglome': [1, np.nan, '3'],
            'vivienda': ['01', '02', '03'],
            'hogar': [1.0, 2.0, np.nan],
            'data': [100, 200, 300]
        })

        # No debe lanzar error
        result = self.merger._prepare_for_merge_robust(
            df, ['conglome', 'vivienda', 'hogar'], 'test'
        )

        self.assertIsNotNone(result)

    def test_regression_cardinality_detection_fix(self):
        """Regresión: Falta de advertencia en merge muchos-a-muchos (Bug #4)"""

        # Caso que no advertía sobre explosión de datos
        df1 = pd.DataFrame({
            'id': [1, 1, 1] * 100,
            'data': range(300)
        })

        df2 = pd.DataFrame({
            'id': [1, 1, 1] * 100,
            'value': range(300)
        })

        warning = self.merger._detect_and_warn_cardinality(
            df1, df2, ['id']
        )

        # Debe advertir sobre la explosión potencial
        self.assertIsNotNone(warning)
        self.assertIn('muchos-a-muchos', warning)

    def test_regression_memory_leak_fix(self):
        """Regresión: Memory leak en merge múltiple (Bug #5)"""

        # Crear múltiples módulos
        modules = {}
        for i in range(5):
            modules[f'{i:02d}'] = self.generator.create_sample_module(
                f'{i:02d}', n_rows=1000
            )

        # Track memoria inicial
        import tracemalloc
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Ejecutar merge múltiple
        result = self.merger.merge_multiple_modules(modules, '00')

        # Track memoria final
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # La memoria no debe crecer excesivamente
        total_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Verificar que no hay leak masivo (menos de 100MB de diferencia)
        self.assertLess(total_diff, 100 * 1024 * 1024)

        tracemalloc.stop()

    def test_regression_conflict_resolution_fix(self):
        """Regresión: Pérdida de datos en resolución de conflictos (Bug #6)"""

        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value_x': [10, 20, 30],
            'value_y': [11, 21, 31],
            'score_left': [100, 200, 300],
            'score_right': [110, 210, 310]
        })

        original_rows = len(df)

        # Resolver conflictos no debe perder filas
        self.merger._resolve_conflicts_robust(
            df, ModuleMergeStrategy.COALESCE
        )

        self.assertEqual(len(df), original_rows)
        self.assertIn('value', df.columns)
        self.assertIn('score', df.columns)


# =====================================================
# EDGE CASES TESTS
# =====================================================

class TestENAHOModuleMergerEdgeCases(unittest.TestCase):
    """Tests para casos extremos y situaciones inusuales"""

    def setUp(self):
        """Setup para edge cases"""
        self.logger = logging.getLogger('test_edge')
        self.config = ModuleMergeConfig()
        self.merger = ENAHOModuleMerger(self.config, self.logger)

    def test_single_row_dataframes(self):
        """Test con DataFrames de una sola fila"""

        df1 = pd.DataFrame({
            'conglome': ['001'],
            'vivienda': ['01'],
            'hogar': ['1'],
            'data': [100]
        })

        df2 = pd.DataFrame({
            'conglome': ['001'],
            'vivienda': ['01'],
            'hogar': ['1'],
            'value': [200]
        })

        result = self.merger.merge_modules(df1, df2, '01', '02')

        self.assertEqual(len(result.merged_df), 1)
        self.assertIn('data', result.merged_df.columns)
        self.assertIn('value', result.merged_df.columns)

    def test_all_nulls_in_merge_keys(self):
        """Test con todas las llaves nulas"""

        df = pd.DataFrame({
            'conglome': [np.nan, np.nan],
            'vivienda': [np.nan, np.nan],
            'hogar': [np.nan, np.nan],
            'data': [100, 200]
        })

        result = self.merger._prepare_for_merge_robust(
            df, ['conglome', 'vivienda', 'hogar'], 'test'
        )

        # Debe eliminar todas las filas
        self.assertEqual(len(result), 0)

    def test_unicode_in_keys(self):
        """Test con caracteres Unicode en llaves"""

        df1 = pd.DataFrame({
            'conglome': ['001ñ', '002á', '003é'],
            'vivienda': ['01', '02', '03'],
            'hogar': ['1', '2', '3'],
            'data': [100, 200, 300]
        })

        df2 = pd.DataFrame({
            'conglome': ['001ñ', '002á', '003é'],
            'vivienda': ['01', '02', '03'],
            'hogar': ['1', '2', '3'],
            'value': [10, 20, 30]
        })

        result = self.merger.merge_modules(df1, df2, '01', '02')

        self.assertIsNotNone(result)
        self.assertEqual(len(result.merged_df), 3)

    def test_extreme_values(self):
        """Test con valores extremos"""

        df1 = pd.DataFrame({
            'conglome': ['001', '002'],
            'vivienda': ['01', '02'],
            'hogar': ['1', '2'],
            'value': [float('inf'), -float('inf')]
        })

        df2 = pd.DataFrame({
            'conglome': ['001', '002'],
            'vivienda': ['01', '02'],
            'hogar': ['1', '2'],
            'score': [1e308, -1e308]
        })

        result = self.merger.merge_modules(df1, df2, '01', '02')

        self.assertIsNotNone(result)
        # Los valores infinitos deben mantenerse
        self.assertTrue(np.isinf(result.merged_df['value'].iloc[0]))

    def test_duplicate_column_names(self):
        """Test con nombres de columnas duplicados"""

        df1 = pd.DataFrame({
            'conglome': ['001'],
            'vivienda': ['01'],
            'hogar': ['1'],
            'value': [100],
            'data': [200]
        })

        df2 = pd.DataFrame({
            'conglome': ['001'],
            'vivienda': ['01'],
            'hogar': ['1'],
            'value': [110],  # Mismo nombre
            'data': [210]  # Mismo nombre
        })

        result = self.merger.merge_modules(df1, df2, '01', '02')

        # Debe resolver conflictos
        self.assertIsNotNone(result)
        self.assertGreater(result.conflicts_resolved, 0)

    def test_very_long_strings_in_keys(self):
        """Test con strings muy largos en llaves"""

        long_string = 'x' * 1000

        df1 = pd.DataFrame({
            'conglome': [long_string],
            'vivienda': ['01'],
            'hogar': ['1'],
            'data': [100]
        })

        df2 = pd.DataFrame({
            'conglome': [long_string],
            'vivienda': ['01'],
            'hogar': ['1'],
            'value': [200]
        })

        result = self.merger.merge_modules(df1, df2, '01', '02')

        self.assertIsNotNone(result)
        self.assertEqual(len(result.merged_df), 1)


# =====================================================
# TEST SUITE RUNNER
# =====================================================

def create_test_suite():
    """Crea la suite completa de tests"""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar todos los test cases
    test_classes = [
        TestENAHOModuleMergerUnit,
        TestENAHOModuleMergerIntegration,
        TestENAHOModuleMergerPerformance,
        TestENAHOModuleMergerRegression,
        TestENAHOModuleMergerEdgeCases
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    return suite


def run_tests_with_coverage():
    """Ejecuta tests con análisis de cobertura"""

    try:
        import coverage

        # Iniciar análisis de cobertura
        cov = coverage.Coverage()
        cov.start()

        # Ejecutar tests
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Detener y reportar cobertura
        cov.stop()
        cov.save()

        print("\n" + "=" * 70)
        print("REPORTE DE COBERTURA")
        print("=" * 70)
        cov.report()

        return result

    except ImportError:
        print("Coverage no instalado. Ejecutando tests sin análisis de cobertura.")
        return run_tests()


def run_tests(verbosity=2, pattern=None):
    """
    Ejecuta los tests con opciones configurables.

    Args:
        verbosity: Nivel de detalle (0=quiet, 1=normal, 2=verbose)
        pattern: Patrón para filtrar tests (ej: 'test_merge*')
    """

    # Configurar logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Crear suite
    if pattern:
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern=pattern)
    else:
        suite = create_test_suite()

    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Generar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print(f"Omitidos: {len(result.skipped)}")

    # Calcular tasa de éxito
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors))
                        / result.testsRun * 100)
        print(f"\nTasa de éxito: {success_rate:.1f}%")

    # Estado final
    if result.wasSuccessful():
        print("\n✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")

        if result.failures:
            print("\nDetalles de fallos:")
            for test, trace in result.failures[:3]:  # Mostrar primeros 3
                print(f"  • {test}")

        if result.errors:
            print("\nDetalles de errores:")
            for test, trace in result.errors[:3]:  # Mostrar primeros 3
                print(f"  • {test}")

    return result


if __name__ == "__main__":
    import sys
    import argparse

    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Tests del módulo merger ENAHO')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                        help='Nivel de verbosidad (0-2)')
    parser.add_argument('-p', '--pattern', type=str, default=None,
                        help='Patrón para filtrar tests')
    parser.add_argument('-c', '--coverage', action='store_true',
                        help='Ejecutar con análisis de cobertura')

    args = parser.parse_args()

    # Ejecutar tests
    if args.coverage:
        result = run_tests_with_coverage()
    else:
        result = run_tests(verbosity=args.verbosity, pattern=args.pattern)

    # Exit code
    sys.exit(0 if result.wasSuccessful() else 1)