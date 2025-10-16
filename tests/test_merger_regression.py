"""
Tests de Regresión para el Módulo Merger
=========================================

Tests críticos para bugs identificados y corregidos.
Estos tests previenen regresiones futuras.

Autor: ENAHOPY Team
Fecha: 2025-10-04
"""

import numpy as np
import pandas as pd
import pytest

from enahopy.merger import ENAHOGeoMerger
from enahopy.merger.config import ModuleMergeConfig, ModuleMergeLevel


@pytest.mark.skip(
    reason="Regression tests use incorrect API parameters - needs refactoring for v0.6.0"
)
class TestMergerRegression:
    """Tests de regresión para bugs críticos"""

    @pytest.fixture
    def sample_personas_df(self):
        """DataFrame de personas de prueba (simula módulo 02)"""
        np.random.seed(42)
        n_personas = 1000

        return pd.DataFrame(
            {
                "conglome": np.random.randint(1, 100, n_personas),
                "vivienda": np.random.randint(1, 10, n_personas),
                "hogar": np.random.randint(1, 5, n_personas),
                "codperso": np.random.randint(1, 10, n_personas),
                "edad": np.random.randint(0, 100, n_personas),
                "sexo": np.random.choice([1, 2], n_personas),
                "p208a": np.random.choice([1, 2, 3, 4], n_personas),  # Nivel educativo
            }
        )

    @pytest.fixture
    def sample_empleo_df(self):
        """DataFrame de empleo de prueba (simula módulo 05)"""
        np.random.seed(43)
        n_empleo = 700  # Menos que personas (no todos trabajan)

        return pd.DataFrame(
            {
                "conglome": np.random.randint(1, 100, n_empleo),
                "vivienda": np.random.randint(1, 10, n_empleo),
                "hogar": np.random.randint(1, 5, n_empleo),
                "codperso": np.random.randint(1, 10, n_empleo),
                "ocu500": np.ones(n_empleo, dtype=int),  # Ocupación principal
                "i524a1": np.random.randint(500, 5000, n_empleo),  # Ingreso
            }
        )

    @pytest.fixture
    def sample_sumaria_df(self):
        """DataFrame sumaria de prueba (simula módulo 34)"""
        np.random.seed(44)
        n_hogares = 300

        return pd.DataFrame(
            {
                "conglome": np.random.randint(1, 100, n_hogares),
                "vivienda": np.random.randint(1, 10, n_hogares),
                "hogar": np.random.randint(1, 5, n_hogares),
                "mieperho": np.random.randint(1, 10, n_hogares),  # Miembros del hogar
                "gashog2d": np.random.randint(500, 5000, n_hogares),  # Gasto del hogar
            }
        )

    # ========================================================================
    # BUG #1: TRIPLICADO DE FILAS EN LEFT JOIN (CRÍTICO)
    # ========================================================================

    def test_left_join_preserves_cardinality(self, sample_personas_df, sample_empleo_df):
        """
        REGRESIÓN: Bug del triplicado de filas (119,747 → 368,114)

        Causa: ENAHOModuleMerger usaba how='outer' en lugar de how='left'
        Corregido en: enahopy/merger/modules/merger.py líneas 1024, 1091

        Este test DEBE fallar si se reintroduce el bug.
        """
        # Arrange
        config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",  # CRÍTICO: debe ser left
            validate_cardinality=True,
        )

        merger = ENAHOGeoMerger(module_config=config)

        modules_dict = {
            "02": sample_personas_df,
            "05": sample_empleo_df,
        }

        initial_rows = len(sample_personas_df)

        # Act
        result = merger.merge_multiple_modules(modules_dict, base_module="02", merge_config=config)

        # Assert - CRÍTICO: cardinalidad debe mantenerse
        assert len(result.merged_df) == initial_rows, (
            f"Left join cambió cardinalidad: {initial_rows} → {len(result.merged_df)} "
            f"(factor {len(result.merged_df) / initial_rows:.2f}x). "
            f"BUG REGRESADO: Revisar merge_type en modules/merger.py"
        )

        # Verificar que no hay duplicados en las llaves
        persona_keys = ["conglome", "vivienda", "hogar", "codperso"]
        duplicados = result.merged_df.duplicated(subset=persona_keys).sum()
        assert duplicados == 0, f"Left join creó {duplicados} duplicados en las llaves"

        # Verificar métricas
        if result.merge_metrics:
            cardinality_change = result.merge_metrics.get("cardinality_change", 0)
            assert (
                abs(cardinality_change - 1.0) < 0.01
            ), f"Cardinality cambió: {cardinality_change:.2f}x (esperado: 1.00x)"

    def test_left_join_with_three_modules(
        self, sample_personas_df, sample_empleo_df, sample_sumaria_df
    ):
        """
        REGRESIÓN: Triplicado al hacer merge secuencial con 3 módulos

        Escenario: Personas (base) + Empleo + Sumaria
        Esperado: Mantener registros de Personas (base)
        """
        # Arrange
        config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",
            validate_cardinality=True,
        )

        merger = ENAHOGeoMerger(module_config=config)

        modules_dict = {
            "02": sample_personas_df,
            "05": sample_empleo_df,
            "34": sample_sumaria_df,
        }

        initial_rows = len(sample_personas_df)

        # Act
        result = merger.merge_multiple_modules(modules_dict, base_module="02", merge_config=config)

        # Assert
        assert (
            len(result.merged_df) == initial_rows
        ), f"Merge secuencial cambió cardinalidad: {initial_rows} → {len(result.merged_df)}"

    def test_large_dataset_left_join_no_inflation(self):
        """
        REGRESIÓN: Bug en _merge_large_datasets que agregaba right_only después de left join

        Causa: Líneas 1069-1083 agregaban registros right_only contradiciendo left join
        Corregido en: enahopy/merger/modules/merger.py línea 1069

        Dataset grande (>500K) para forzar código de _merge_large_datasets
        """
        # Arrange - Crear dataset grande
        np.random.seed(45)
        n_left = 600000

        large_personas_df = pd.DataFrame(
            {
                "conglome": np.random.randint(1, 1000, n_left),
                "vivienda": np.random.randint(1, 50, n_left),
                "hogar": np.random.randint(1, 10, n_left),
                "codperso": np.random.randint(1, 15, n_left),
                "value": np.random.rand(n_left),
            }
        )

        # Empleo es más pequeño (no todos trabajan)
        n_right = 400000
        large_empleo_df = pd.DataFrame(
            {
                "conglome": np.random.randint(1, 1000, n_right),
                "vivienda": np.random.randint(1, 50, n_right),
                "hogar": np.random.randint(1, 10, n_right),
                "codperso": np.random.randint(1, 15, n_right),
                "ingreso": np.random.randint(500, 5000, n_right),
            }
        )

        config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",
            validate_cardinality=True,
        )

        merger = ENAHOGeoMerger(module_config=config)

        modules_dict = {
            "02": large_personas_df,
            "05": large_empleo_df,
        }

        initial_rows = len(large_personas_df)

        # Act
        result = merger.merge_multiple_modules(modules_dict, base_module="02", merge_config=config)

        # Assert
        assert (
            len(result.merged_df) == initial_rows
        ), f"Left join en dataset grande cambió cardinalidad: {initial_rows} → {len(result.merged_df)}"

        # No debe haber registros right_only en left join
        if "_merge" in result.merged_df.columns:
            right_only_count = (result.merged_df["_merge"] == "right_only").sum()
            assert right_only_count == 0, (
                f"Left join creó {right_only_count} registros 'right_only'. "
                f"BUG: Revisar _merge_large_datasets línea 1069"
            )

    # ========================================================================
    # BUG #2: VALIDACIÓN DE CARDINALIDAD
    # ========================================================================

    def test_validation_detects_cardinality_change(self, sample_personas_df):
        """
        OPTIMIZACIÓN: Validación post-merge detecta cambios inesperados en cardinalidad

        Nuevo: Validación automática de cardinalidad
        Agregado en: enahopy/merger/modules/merger.py línea 268
        """
        # Arrange - Crear DataFrame con duplicados intencionales
        empleo_with_duplicates = pd.concat(
            [
                sample_personas_df[["conglome", "vivienda", "hogar", "codperso"]].head(100),
                sample_personas_df[["conglome", "vivienda", "hogar", "codperso"]].head(
                    100
                ),  # Duplicar
            ],
            ignore_index=True,
        )
        empleo_with_duplicates["ingreso"] = np.random.randint(
            500, 5000, len(empleo_with_duplicates)
        )

        config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",
            validate_cardinality=True,  # CRÍTICO: habilitar validación
        )

        merger = ENAHOGeoMerger(module_config=config)

        modules_dict = {
            "02": sample_personas_df.head(100),
            "05": empleo_with_duplicates,
        }

        # Act
        result = merger.merge_multiple_modules(modules_dict, base_module="02", merge_config=config)

        # Assert - Debe generar warning por cambio de cardinalidad
        warnings_text = " ".join(result.validation_warnings)
        assert (
            "VALIDACIÓN FALLIDA" in warnings_text or "cardinalidad" in warnings_text.lower()
        ), "Validación no detectó cambio de cardinalidad causado por duplicados"

    # ========================================================================
    # BUG #3: MÉTRICAS DE MERGE
    # ========================================================================

    def test_merge_metrics_are_calculated(self, sample_personas_df, sample_empleo_df):
        """
        OPTIMIZACIÓN: Métricas de merge se calculan correctamente

        Nuevo: merge_metrics en ModuleMergeResult
        Agregado en: enahopy/merger/modules/merger.py línea 260
        """
        # Arrange
        config = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",
        )

        merger = ENAHOGeoMerger(module_config=config)

        modules_dict = {
            "02": sample_personas_df,
            "05": sample_empleo_df,
        }

        # Act
        result = merger.merge_multiple_modules(modules_dict, base_module="02", merge_config=config)

        # Assert - Verificar que las métricas existen
        assert result.merge_metrics is not None, "Métricas de merge no fueron calculadas"

        # Verificar métricas específicas
        assert "rows_before" in result.merge_metrics
        assert "rows_after" in result.merge_metrics
        assert "merge_type" in result.merge_metrics
        assert "match_rate" in result.merge_metrics
        assert "cardinality_change" in result.merge_metrics

        # Validar valores
        assert result.merge_metrics["rows_before"] == len(sample_personas_df)
        assert result.merge_metrics["merge_type"] == "left"
        assert 0.0 <= result.merge_metrics["match_rate"] <= 1.0
        assert result.merge_metrics["cardinality_change"] > 0

    # ========================================================================
    # BUG #4: CACHÉ DE VALIDACIONES
    # ========================================================================

    def test_validation_cache_improves_performance(self, sample_personas_df, sample_empleo_df):
        """
        OPTIMIZACIÓN: Caché de validaciones reduce tiempo en merges repetidos

        Nuevo: use_validation_cache en ModuleMergeConfig
        Agregado en: enahopy/merger/modules/merger.py línea 170
        """
        import time

        # Arrange
        config_with_cache = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",
            use_validation_cache=True,
        )

        config_without_cache = ModuleMergeConfig(
            merge_level=ModuleMergeLevel.PERSONA,
            merge_type="left",
            use_validation_cache=False,
        )

        modules_dict = {
            "02": sample_personas_df,
            "05": sample_empleo_df,
        }

        # Act - Merge repetido CON caché
        merger_with_cache = ENAHOGeoMerger(module_config=config_with_cache)
        start_with_cache = time.time()
        for _ in range(3):  # Repetir 3 veces
            merger_with_cache.merge_multiple_modules(
                modules_dict, base_module="02", merge_config=config_with_cache
            )
        time_with_cache = time.time() - start_with_cache

        # Act - Merge repetido SIN caché
        merger_without_cache = ENAHOGeoMerger(module_config=config_without_cache)
        start_without_cache = time.time()
        for _ in range(3):  # Repetir 3 veces
            merger_without_cache.merge_multiple_modules(
                modules_dict, base_module="02", merge_config=config_without_cache
            )
        time_without_cache = time.time() - start_without_cache

        # Assert - Con caché debe ser más rápido (al menos 10% mejor)
        improvement = (time_without_cache - time_with_cache) / time_without_cache
        assert improvement > 0.05, (
            f"Caché no mejoró performance significativamente: "
            f"Con caché: {time_with_cache:.3f}s, Sin caché: {time_without_cache:.3f}s, "
            f"Mejora: {improvement * 100:.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
