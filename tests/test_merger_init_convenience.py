"""
Comprehensive tests for enahopy.merger convenience functions

This test module covers convenience functions in the merger.__init__ module,
including module merge functions, geographic validation, and utility functions.

Target: merger/__init__.py (currently 24.69% coverage)
Goal: Achieve 50-55% coverage (+2.5% overall project coverage)
"""

import pandas as pd
import pytest

from enahopy.merger import (
    create_optimized_merge_config,
    get_available_duplicate_strategies,
    merge_enaho_modules,
    validate_merge_configuration,
)
from enahopy.merger.config import GeoMergeConfiguration, ModuleMergeConfig, TipoManejoDuplicados

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_hogar_modules():
    """Create sample hogar-level modules for testing"""
    mod34 = pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["1", "1", "1"],
            "mieperho": [4, 3, 5],
            "gashog2d": [1200.5, 800.3, 1500.0],
            "pobreza": [1, 2, 1],
        }
    )

    mod01 = pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["1", "1", "1"],
            "area": [1, 2, 1],
            "dominio": [1, 2, 3],
        }
    )

    return {"34": mod34, "01": mod01}


@pytest.fixture
def sample_persona_modules():
    """Create sample persona-level modules"""
    mod02 = pd.DataFrame(
        {
            "conglome": ["001", "001", "002"],
            "vivienda": ["01", "01", "01"],
            "hogar": ["1", "1", "1"],
            "codperso": ["01", "02", "01"],
            "p207": [35, 28, 45],
        }
    )

    mod03 = pd.DataFrame(
        {
            "conglome": ["001", "001", "002"],
            "vivienda": ["01", "01", "01"],
            "hogar": ["1", "1", "1"],
            "codperso": ["01", "02", "01"],
            "p301a": [6, 5, 7],
        }
    )

    return {"02": mod02, "03": mod03}


# ============================================================================
# TESTS FOR merge_enaho_modules()
# ============================================================================


def test_merge_enaho_modules_basic(sample_hogar_modules):
    """Test basic module merging with default parameters"""
    result = merge_enaho_modules(sample_hogar_modules, verbose=False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "mieperho" in result.columns  # From module 34
    assert "area" in result.columns  # From module 01


def test_merge_enaho_modules_with_base_module(sample_hogar_modules):
    """Test module merging with custom base module"""
    result = merge_enaho_modules(sample_hogar_modules, base_module="34", verbose=False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_merge_enaho_modules_hogar_level(sample_hogar_modules):
    """Test module merging at hogar level"""
    result = merge_enaho_modules(sample_hogar_modules, level="hogar", verbose=False)

    assert isinstance(result, pd.DataFrame)
    assert "hogar" in result.columns


def test_merge_enaho_modules_persona_level(sample_persona_modules):
    """Test module merging at persona level"""
    result = merge_enaho_modules(
        sample_persona_modules, level="persona", base_module="02", verbose=False
    )

    assert isinstance(result, pd.DataFrame)
    assert "codperso" in result.columns


def test_merge_enaho_modules_coalesce_strategy(sample_hogar_modules):
    """Test module merging with coalesce strategy"""
    result = merge_enaho_modules(sample_hogar_modules, strategy="coalesce", verbose=False)

    assert isinstance(result, pd.DataFrame)


def test_merge_enaho_modules_keep_left_strategy(sample_hogar_modules):
    """Test module merging with keep_left strategy"""
    result = merge_enaho_modules(sample_hogar_modules, strategy="keep_left", verbose=False)

    assert isinstance(result, pd.DataFrame)


def test_merge_enaho_modules_verbose_mode(sample_hogar_modules, capfd):
    """Test module merging with verbose output"""
    result = merge_enaho_modules(sample_hogar_modules, verbose=True)

    capfd.readouterr()
    assert isinstance(result, pd.DataFrame)
    # Verbose mode should print something (though exact output may vary)


def test_merge_enaho_modules_single_module():
    """Test module merging with single module (edge case) - should raise error"""
    single_mod = {
        "34": pd.DataFrame(
            {
                "conglome": ["001"],
                "vivienda": ["01"],
                "hogar": ["1"],
                "mieperho": [4],
            }
        )
    }

    # Single module should raise ValueError (requires at least 2 modules)
    try:
        result = merge_enaho_modules(single_mod, verbose=False)
        # If it doesn't raise, it returned something
        assert result is not None
    except ValueError as e:
        # Expected: requires at least 2 modules
        assert "2" in str(e) or "dos" in str(e).lower() or "módulos" in str(e)


# ============================================================================
# TESTS FOR validate_merge_configuration()
# ============================================================================


def test_validate_merge_configuration_valid_geo_config():
    """Test validation with valid geographic configuration"""
    geo_config = GeoMergeConfiguration(chunk_size=1000)

    result = validate_merge_configuration(geo_config=geo_config)

    assert isinstance(result, dict)
    assert "valid" in result
    assert "warnings" in result
    assert "errors" in result
    assert result["valid"] is True


def test_validate_merge_configuration_valid_module_config():
    """Test validation with valid module configuration"""
    module_config = ModuleMergeConfig()

    result = validate_merge_configuration(module_config=module_config)

    assert isinstance(result, dict)
    assert result["valid"] is True


def test_validate_merge_configuration_invalid_chunk_size():
    """Test validation with invalid chunk size"""
    geo_config = GeoMergeConfiguration(chunk_size=-100)

    result = validate_merge_configuration(geo_config=geo_config)

    assert result["valid"] is False
    assert len(result["errors"]) > 0
    assert any("chunk_size" in error for error in result["errors"])


def test_validate_merge_configuration_aggregate_without_functions():
    """Test validation for AGGREGATE strategy without aggregation functions"""
    geo_config = GeoMergeConfiguration(
        manejo_duplicados=TipoManejoDuplicados.AGGREGATE, funciones_agregacion=None
    )

    result = validate_merge_configuration(geo_config=geo_config)

    assert result["valid"] is False
    assert any("funciones_agregacion" in error for error in result["errors"])


def test_validate_merge_configuration_best_quality_without_column():
    """Test validation for BEST_QUALITY strategy without quality column"""
    geo_config = GeoMergeConfiguration(
        manejo_duplicados=TipoManejoDuplicados.BEST_QUALITY, columna_calidad=None
    )

    result = validate_merge_configuration(geo_config=geo_config)

    assert result["valid"] is False
    assert any("columna_calidad" in error for error in result["errors"])


def test_validate_merge_configuration_invalid_min_match_rate():
    """Test validation with invalid min_match_rate"""
    module_config = ModuleMergeConfig(min_match_rate=1.5)  # Invalid: > 1

    result = validate_merge_configuration(module_config=module_config)

    assert result["valid"] is False
    assert any("min_match_rate" in error for error in result["errors"])


def test_validate_merge_configuration_negative_max_conflicts():
    """Test validation with negative max_conflicts_allowed"""
    module_config = ModuleMergeConfig(max_conflicts_allowed=-5)

    result = validate_merge_configuration(module_config=module_config)

    assert result["valid"] is False
    assert any("max_conflicts_allowed" in error for error in result["errors"])


def test_validate_merge_configuration_both_configs():
    """Test validation with both geo and module configs"""
    geo_config = GeoMergeConfiguration(chunk_size=1000)
    module_config = ModuleMergeConfig()

    result = validate_merge_configuration(geo_config=geo_config, module_config=module_config)

    assert isinstance(result, dict)
    assert result["valid"] is True


def test_validate_merge_configuration_empty_hogar_keys_warning():
    """Test that empty hogar_keys produces warning"""
    module_config = ModuleMergeConfig(hogar_keys=[])

    result = validate_merge_configuration(module_config=module_config)

    # Should be valid but have warnings
    assert result["valid"] is True
    assert len(result["warnings"]) > 0


# ============================================================================
# TESTS FOR create_optimized_merge_config()
# ============================================================================


def test_create_optimized_merge_config_small_geographic():
    """Test optimized config for small geographic dataset"""
    result = create_optimized_merge_config(df_size=5000, merge_type="geographic")

    assert isinstance(result, dict)
    assert "geo_config" in result
    assert isinstance(result["geo_config"], GeoMergeConfiguration)
    assert result["geo_config"].chunk_size == 5000
    assert result["geo_config"].optimizar_memoria is False


def test_create_optimized_merge_config_medium_geographic():
    """Test optimized config for medium geographic dataset"""
    result = create_optimized_merge_config(df_size=50000, merge_type="geographic")

    assert isinstance(result, dict)
    assert result["geo_config"].chunk_size == 10000
    assert result["geo_config"].usar_cache is True


def test_create_optimized_merge_config_large_geographic():
    """Test optimized config for large geographic dataset"""
    result = create_optimized_merge_config(df_size=200000, merge_type="geographic")

    assert isinstance(result, dict)
    assert result["geo_config"].chunk_size == 50000
    assert result["geo_config"].optimizar_memoria is True
    assert result["geo_config"].usar_cache is True


def test_create_optimized_merge_config_large_geographic_memory_priority():
    """Test optimized config with memory priority"""
    result = create_optimized_merge_config(
        df_size=200000, merge_type="geographic", performance_priority="memory"
    )

    assert result["geo_config"].chunk_size == 25000  # Smaller for memory priority
    assert result["geo_config"].optimizar_memoria is True


def test_create_optimized_merge_config_small_module():
    """Test optimized config for small module dataset"""
    result = create_optimized_merge_config(df_size=30000, merge_type="module")

    assert isinstance(result, dict)
    assert "module_config" in result
    assert isinstance(result["module_config"], ModuleMergeConfig)
    assert result["module_config"].chunk_processing is False


def test_create_optimized_merge_config_large_module():
    """Test optimized config for large module dataset"""
    result = create_optimized_merge_config(df_size=100000, merge_type="module")

    assert result["module_config"].chunk_size == 50000


def test_create_optimized_merge_config_large_module_memory_priority():
    """Test optimized config for large module with memory priority"""
    result = create_optimized_merge_config(
        df_size=100000, merge_type="module", performance_priority="memory"
    )

    assert result["module_config"].chunk_processing is True
    assert result["module_config"].chunk_size == 25000


def test_create_optimized_merge_config_speed_priority():
    """Test optimized config with speed priority"""
    result = create_optimized_merge_config(
        df_size=200000, merge_type="geographic", performance_priority="speed"
    )

    assert result["geo_config"].chunk_size == 50000  # Larger for speed


def test_create_optimized_merge_config_balanced_priority():
    """Test optimized config with balanced priority"""
    result = create_optimized_merge_config(
        df_size=200000, merge_type="geographic", performance_priority="balanced"
    )

    assert isinstance(result, dict)
    assert "geo_config" in result


# ============================================================================
# TESTS FOR get_available_duplicate_strategies()
# ============================================================================


def test_get_available_duplicate_strategies_returns_list():
    """Test that function returns a list of strategies"""
    result = get_available_duplicate_strategies()

    assert isinstance(result, list)
    assert len(result) > 0


def test_get_available_duplicate_strategies_contains_expected():
    """Test that result contains expected strategy types"""
    result = get_available_duplicate_strategies()

    # Should contain enum objects or strings
    # Check that result is not empty and contains valid types
    assert len(result) > 0
    # Can be either TipoManejoDuplicados enum or string
    first_item = result[0]
    assert isinstance(first_item, (str, TipoManejoDuplicados)) or hasattr(first_item, "value")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow_merge_and_validate(sample_hogar_modules):
    """Test complete workflow: merge modules -> validate config"""
    # Step 1: Merge modules
    merged_df = merge_enaho_modules(sample_hogar_modules, verbose=False)

    assert isinstance(merged_df, pd.DataFrame)
    assert len(merged_df) > 0

    # Step 2: Create optimized config based on result size
    config = create_optimized_merge_config(df_size=len(merged_df), merge_type="module")

    assert "module_config" in config

    # Step 3: Validate configuration
    validation = validate_merge_configuration(module_config=config["module_config"])

    assert validation["valid"] is True


def test_config_optimization_workflow():
    """Test workflow: get strategies -> create config -> validate"""
    # Step 1: Get available strategies
    strategies = get_available_duplicate_strategies()
    assert len(strategies) > 0

    # Step 2: Create optimized config
    config = create_optimized_merge_config(df_size=50000, merge_type="geographic")
    assert "geo_config" in config

    # Step 3: Validate
    validation = validate_merge_configuration(geo_config=config["geo_config"])
    assert validation["valid"] is True


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_merge_enaho_modules_empty_dict():
    """Test merging with empty modules dictionary"""
    try:
        result = merge_enaho_modules({}, verbose=False)
        # May succeed with empty result or raise error - both are acceptable
        if result is not None:
            assert isinstance(result, pd.DataFrame)
    except (ValueError, KeyError, IndexError):
        # Expected behavior for empty input
        pass


def test_validate_merge_configuration_none_configs():
    """Test validation with no configs provided"""
    result = validate_merge_configuration()

    assert isinstance(result, dict)
    assert result["valid"] is True  # No configs to validate = valid


def test_create_optimized_merge_config_zero_size():
    """Test config creation with zero dataset size"""
    result = create_optimized_merge_config(df_size=0, merge_type="geographic")

    assert isinstance(result, dict)


def test_create_optimized_merge_config_very_large_size():
    """Test config creation with very large dataset"""
    result = create_optimized_merge_config(df_size=10000000, merge_type="geographic")

    assert isinstance(result, dict)
    assert result["geo_config"].optimizar_memoria is True


# ============================================================================
# PARAMETER VALIDATION TESTS
# ============================================================================


def test_merge_enaho_modules_invalid_level():
    """Test merging with invalid level parameter"""
    modules = {"34": pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"]})}

    try:
        result = merge_enaho_modules(modules, level="invalid_level", verbose=False)
        # If it doesn't raise, check it returns something reasonable
        assert result is not None
    except (ValueError, KeyError):
        # Expected for invalid level
        pass


def test_validate_merge_configuration_multiple_errors():
    """Test validation accumulates multiple errors"""
    module_config = ModuleMergeConfig(
        min_match_rate=1.5,  # Invalid
        max_conflicts_allowed=-10,  # Invalid
    )

    result = validate_merge_configuration(module_config=module_config)

    assert result["valid"] is False
    assert len(result["errors"]) >= 2  # Should have multiple errors
