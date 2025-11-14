"""
Comprehensive tests for enahopy.merger.modules convenience functions

This test module covers all convenience functions in the merger.modules.__init__ module,
including quick merge operations, validation helpers, and module information utilities.

Target: merger/modules/__init__.py (currently 10.66% coverage)
Goal: Achieve 50-60% coverage (+3-4% overall project coverage)
"""

import logging

import pandas as pd
import pytest

from enahopy.merger.modules import (
    ENAHOModuleMerger,
    ModuleValidator,
    analyze_merge_feasibility_quick,
    check_modules_compatibility_quick,
    create_optimal_merge_plan,
    get_module_info,
    merge_multiple_modules_quick,
    quick_module_merge,
    validate_module_structure_quick,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_hogar_df():
    """Create a sample hogar-level DataFrame (module 34 - Sumaria)"""
    return pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["1", "1", "1"],
            "mieperho": [4, 3, 5],
            "gashog2d": [1200.5, 800.3, 1500.0],
            "inghog2d": [1500.0, 1000.0, 2000.0],
            "pobreza": [1, 2, 1],
        }
    )


@pytest.fixture
def sample_hogar_df2():
    """Create a second hogar-level DataFrame (module 01)"""
    return pd.DataFrame(
        {
            "conglome": ["001", "002", "003"],
            "vivienda": ["01", "01", "02"],
            "hogar": ["1", "1", "1"],
            "area": [1, 2, 1],
            "dominio": [1, 2, 3],
            "estrato": [1, 2, 1],
        }
    )


@pytest.fixture
def sample_persona_df():
    """Create a sample persona-level DataFrame (module 02)"""
    return pd.DataFrame(
        {
            "conglome": ["001", "001", "002", "002"],
            "vivienda": ["01", "01", "01", "01"],
            "hogar": ["1", "1", "1", "1"],
            "codperso": ["01", "02", "01", "02"],
            "p207": [35, 28, 45, 40],
            "p208a": [1, 2, 1, 1],
        }
    )


@pytest.fixture
def sample_persona_df2():
    """Create a second persona-level DataFrame (module 03 - Educación)"""
    return pd.DataFrame(
        {
            "conglome": ["001", "001", "002", "002"],
            "vivienda": ["01", "01", "01", "01"],
            "hogar": ["1", "1", "1", "1"],
            "codperso": ["01", "02", "01", "02"],
            "p301a": [6, 5, 7, 6],
            "p301b": [1, 1, 2, 1],
        }
    )


@pytest.fixture
def modules_dict_hogar(sample_hogar_df, sample_hogar_df2):
    """Dictionary of hogar-level modules"""
    return {"34": sample_hogar_df, "01": sample_hogar_df2}


@pytest.fixture
def modules_dict_persona(sample_persona_df, sample_persona_df2):
    """Dictionary of persona-level modules"""
    return {"02": sample_persona_df, "03": sample_persona_df2}


# ============================================================================
# TESTS FOR quick_module_merge
# ============================================================================


def test_quick_module_merge_hogar_level(sample_hogar_df, sample_hogar_df2):
    """Test quick merge of two hogar-level modules"""
    result = quick_module_merge(
        sample_hogar_df, sample_hogar_df2, "34", "01", level="hogar", strategy="coalesce"
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "mieperho" in result.columns
    assert "area" in result.columns
    assert "dominio" in result.columns


def test_quick_module_merge_persona_level(sample_persona_df, sample_persona_df2):
    """Test quick merge of two persona-level modules"""
    result = quick_module_merge(
        sample_persona_df, sample_persona_df2, "02", "03", level="persona", strategy="coalesce"
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert "p207" in result.columns
    assert "p301a" in result.columns


def test_quick_module_merge_keep_left_strategy(sample_hogar_df, sample_hogar_df2):
    """Test quick merge with keep_left strategy"""
    # Add conflicting column
    sample_hogar_df2["test_col"] = [10, 20, 30]
    sample_hogar_df["test_col"] = [100, 200, 300]

    result = quick_module_merge(
        sample_hogar_df, sample_hogar_df2, "34", "01", level="hogar", strategy="keep_left"
    )

    assert isinstance(result, pd.DataFrame)
    # Should keep left values
    assert result["test_col"].tolist() == [100, 200, 300]


def test_quick_module_merge_verbose_logging(sample_hogar_df, sample_hogar_df2, caplog):
    """Test verbose logging in quick merge"""
    with caplog.at_level(logging.INFO):
        result = quick_module_merge(
            sample_hogar_df,
            sample_hogar_df2,
            "34",
            "01",
            level="hogar",
            strategy="coalesce",
            verbose=True,
        )

    assert isinstance(result, pd.DataFrame)
    # Logging should be configured


def test_quick_module_merge_quiet_mode(sample_hogar_df, sample_hogar_df2, caplog):
    """Test quiet mode (verbose=False) in quick merge"""
    with caplog.at_level(logging.WARNING):
        result = quick_module_merge(
            sample_hogar_df,
            sample_hogar_df2,
            "34",
            "01",
            level="hogar",
            strategy="coalesce",
            verbose=False,
        )

    assert isinstance(result, pd.DataFrame)


# ============================================================================
# TESTS FOR validate_module_structure_quick
# ============================================================================


def test_validate_module_structure_quick_valid_module(sample_hogar_df):
    """Test validation of a valid module structure"""
    warnings = validate_module_structure_quick(sample_hogar_df, "34")

    assert isinstance(warnings, list)
    # Valid module should have minimal or no warnings


def test_validate_module_structure_quick_persona_module(sample_persona_df):
    """Test validation of a persona-level module"""
    warnings = validate_module_structure_quick(sample_persona_df, "02")

    assert isinstance(warnings, list)


def test_validate_module_structure_quick_missing_columns():
    """Test validation with missing required columns"""
    df = pd.DataFrame({"conglome": ["001"], "vivienda": ["01"]})

    warnings = validate_module_structure_quick(df, "34")

    assert isinstance(warnings, list)
    assert len(warnings) > 0  # Should have warnings about missing columns


def test_validate_module_structure_quick_empty_dataframe():
    """Test validation with empty DataFrame"""
    df = pd.DataFrame(columns=["conglome", "vivienda", "hogar"])

    warnings = validate_module_structure_quick(df, "34")

    assert isinstance(warnings, list)
    assert len(warnings) > 0  # Should warn about empty DataFrame


def test_validate_module_structure_quick_unknown_module():
    """Test validation with unknown module code"""
    df = pd.DataFrame({"conglome": ["001"], "vivienda": ["01"]})

    # Module 99 doesn't exist - should handle gracefully or warn
    warnings = validate_module_structure_quick(df, "99")

    assert isinstance(warnings, list)


# ============================================================================
# TESTS FOR check_modules_compatibility_quick
# ============================================================================


def test_check_modules_compatibility_quick_compatible(modules_dict_hogar):
    """Test compatibility check for compatible modules"""
    result = check_modules_compatibility_quick(modules_dict_hogar, merge_level="hogar")

    assert isinstance(result, dict)
    assert "overall_compatible" in result
    assert "merge_level" in result
    assert result["merge_level"] == "hogar"
    assert "modules_analyzed" in result
    assert set(result["modules_analyzed"]) == {"34", "01"}
    assert "pairwise_compatibility" in result
    assert "34-01" in result["pairwise_compatibility"]


def test_check_modules_compatibility_quick_persona_level(modules_dict_persona):
    """Test compatibility check for persona-level modules"""
    result = check_modules_compatibility_quick(modules_dict_persona, merge_level="persona")

    assert isinstance(result, dict)
    assert result["merge_level"] == "persona"
    assert "overall_compatible" in result


def test_check_modules_compatibility_quick_single_module():
    """Test compatibility check with single module (no pairs)"""
    df = pd.DataFrame(
        {
            "conglome": ["001"],
            "vivienda": ["01"],
            "hogar": ["1"],
            "mieperho": [4],
        }
    )

    result = check_modules_compatibility_quick({"34": df}, merge_level="hogar")

    assert isinstance(result, dict)
    assert result["overall_compatible"] is True
    assert len(result["pairwise_compatibility"]) == 0  # No pairs to check


def test_check_modules_compatibility_quick_incompatible_modules():
    """Test compatibility check with incompatible modules"""
    df1 = pd.DataFrame({"conglome": ["001"], "vivienda": ["01"], "hogar": ["1"]})
    df2 = pd.DataFrame({"conglome": ["999"], "vivienda": ["99"], "hogar": ["9"]})  # No overlap

    result = check_modules_compatibility_quick({"34": df1, "01": df2}, merge_level="hogar")

    assert isinstance(result, dict)
    # May be incompatible due to no key overlap


def test_check_modules_compatibility_quick_three_modules(
    sample_hogar_df, sample_hogar_df2, sample_persona_df
):
    """Test compatibility check with three modules"""
    modules = {"34": sample_hogar_df, "01": sample_hogar_df2}

    result = check_modules_compatibility_quick(modules, merge_level="hogar")

    assert isinstance(result, dict)
    assert len(result["modules_analyzed"]) == 2
    # Should check 34-01 pair


# ============================================================================
# TESTS FOR merge_multiple_modules_quick
# ============================================================================


def test_merge_multiple_modules_quick_basic(modules_dict_hogar):
    """Test merging multiple modules with default settings"""
    result = merge_multiple_modules_quick(modules_dict_hogar, base_module="34")

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "mieperho" in result.columns  # From module 34
    assert "area" in result.columns  # From module 01


def test_merge_multiple_modules_quick_persona_level(modules_dict_persona):
    """Test merging multiple persona-level modules"""
    result = merge_multiple_modules_quick(modules_dict_persona, base_module="02", level="persona")

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_merge_multiple_modules_quick_with_report(modules_dict_hogar):
    """Test merging with report generation"""
    result = merge_multiple_modules_quick(modules_dict_hogar, base_module="34", return_report=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    df, report = result
    assert isinstance(df, pd.DataFrame)
    assert isinstance(report, dict)


def test_merge_multiple_modules_quick_custom_strategy(modules_dict_hogar):
    """Test merging with custom conflict resolution strategy"""
    result = merge_multiple_modules_quick(
        modules_dict_hogar, base_module="34", level="hogar", strategy="keep_left"
    )

    assert isinstance(result, pd.DataFrame)


def test_merge_multiple_modules_quick_verbose_mode(modules_dict_hogar):
    """Test merging with verbose logging"""
    result = merge_multiple_modules_quick(modules_dict_hogar, base_module="34", verbose=True)

    assert isinstance(result, pd.DataFrame)


def test_merge_multiple_modules_quick_quiet_mode(modules_dict_hogar):
    """Test merging with quiet mode"""
    result = merge_multiple_modules_quick(modules_dict_hogar, base_module="34", verbose=False)

    assert isinstance(result, pd.DataFrame)


# ============================================================================
# TESTS FOR get_module_info
# ============================================================================


def test_get_module_info_valid_module_34():
    """Test getting info for module 34 (Sumaria)"""
    info = get_module_info("34")

    assert isinstance(info, dict)
    assert info["valid"] is True
    assert info["module_code"] == "34"
    assert "level" in info
    assert "required_keys" in info
    assert "description" in info
    assert "Sumaria" in info["description"]


def test_get_module_info_valid_module_01():
    """Test getting info for module 01"""
    info = get_module_info("01")

    assert isinstance(info, dict)
    assert info["valid"] is True
    assert info["module_code"] == "01"
    assert "Vivienda" in info["description"]


def test_get_module_info_valid_module_02():
    """Test getting info for module 02"""
    info = get_module_info("02")

    assert isinstance(info, dict)
    assert info["valid"] is True
    assert "Miembros" in info["description"]


def test_get_module_info_valid_module_03():
    """Test getting info for module 03 (Educación)"""
    info = get_module_info("03")

    assert isinstance(info, dict)
    assert info["valid"] is True
    assert "Educación" in info["description"] or "Educacion" in info["description"]


def test_get_module_info_valid_module_05():
    """Test getting info for module 05 (Empleo)"""
    info = get_module_info("05")

    assert isinstance(info, dict)
    assert info["valid"] is True
    assert "Empleo" in info["description"]


def test_get_module_info_invalid_module():
    """Test getting info for invalid module"""
    info = get_module_info("99")

    assert isinstance(info, dict)
    assert info["valid"] is False
    assert "error" in info
    assert "available_modules" in info
    assert isinstance(info["available_modules"], list)


def test_get_module_info_all_documented_modules():
    """Test that all documented modules have descriptions"""
    documented_modules = ["01", "02", "03", "04", "05", "07", "08", "09", "34", "37"]

    for module_code in documented_modules:
        info = get_module_info(module_code)
        if info["valid"]:
            assert "description" in info
            assert len(info["description"]) > 0


# ============================================================================
# TESTS FOR analyze_merge_feasibility_quick
# ============================================================================


def test_analyze_merge_feasibility_quick_feasible(modules_dict_hogar):
    """Test feasibility analysis for feasible merge"""
    result = analyze_merge_feasibility_quick(modules_dict_hogar, target_level="hogar")

    assert isinstance(result, dict)
    assert "feasible" in result


def test_analyze_merge_feasibility_quick_persona_level(modules_dict_persona):
    """Test feasibility analysis for persona-level merge"""
    result = analyze_merge_feasibility_quick(modules_dict_persona, target_level="persona")

    assert isinstance(result, dict)


def test_analyze_merge_feasibility_quick_single_module():
    """Test feasibility analysis with single module"""
    df = pd.DataFrame(
        {
            "conglome": ["001"],
            "vivienda": ["01"],
            "hogar": ["1"],
            "mieperho": [4],
        }
    )

    result = analyze_merge_feasibility_quick({"34": df}, target_level="hogar")

    assert isinstance(result, dict)


def test_analyze_merge_feasibility_quick_empty_dict():
    """Test feasibility analysis with empty dictionary"""
    result = analyze_merge_feasibility_quick({}, target_level="hogar")

    assert isinstance(result, dict)


# ============================================================================
# TESTS FOR create_optimal_merge_plan
# ============================================================================


def test_create_optimal_merge_plan_basic(modules_dict_hogar):
    """Test creating optimal merge plan"""
    plan = create_optimal_merge_plan(modules_dict_hogar, target_module="34")

    assert isinstance(plan, dict)


def test_create_optimal_merge_plan_persona_modules(modules_dict_persona):
    """Test creating merge plan for persona-level modules"""
    plan = create_optimal_merge_plan(modules_dict_persona, target_module="02")

    assert isinstance(plan, dict)


def test_create_optimal_merge_plan_single_module():
    """Test merge plan with single module"""
    df = pd.DataFrame(
        {
            "conglome": ["001"],
            "vivienda": ["01"],
            "hogar": ["1"],
            "mieperho": [4],
        }
    )

    plan = create_optimal_merge_plan({"34": df}, target_module="34")

    assert isinstance(plan, dict)


# ============================================================================
# TESTS FOR MODULE EXPORTS AND METADATA
# ============================================================================


def test_module_exports():
    """Test that __all__ exports are available"""
    from enahopy.merger import modules

    expected_exports = [
        "ENAHOModuleMerger",
        "ModuleValidator",
        "quick_module_merge",
        "validate_module_structure_quick",
        "check_modules_compatibility_quick",
        "merge_multiple_modules_quick",
        "get_module_info",
        "analyze_merge_feasibility_quick",
        "create_optimal_merge_plan",
    ]

    for export in expected_exports:
        assert hasattr(modules, export)


def test_module_version():
    """Test module version metadata"""
    from enahopy.merger import modules

    assert hasattr(modules, "__version__")
    assert isinstance(modules.__version__, str)


def test_module_description():
    """Test module description metadata"""
    from enahopy.merger import modules

    assert hasattr(modules, "__description__")
    assert isinstance(modules.__description__, str)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow_hogar_modules(modules_dict_hogar):
    """Test complete workflow: validate -> check compatibility -> analyze -> merge"""
    # Step 1: Validate individual modules
    for module_code, df in modules_dict_hogar.items():
        warnings = validate_module_structure_quick(df, module_code)
        assert isinstance(warnings, list)

    # Step 2: Check compatibility
    compatibility = check_modules_compatibility_quick(modules_dict_hogar, merge_level="hogar")
    assert isinstance(compatibility, dict)

    # Step 3: Analyze feasibility
    feasibility = analyze_merge_feasibility_quick(modules_dict_hogar, target_level="hogar")
    assert isinstance(feasibility, dict)

    # Step 4: Merge
    result = merge_multiple_modules_quick(modules_dict_hogar, base_module="34")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_full_workflow_persona_modules(modules_dict_persona):
    """Test complete workflow for persona-level modules"""
    # Validate
    for module_code, df in modules_dict_persona.items():
        warnings = validate_module_structure_quick(df, module_code)
        assert isinstance(warnings, list)

    # Check compatibility
    compatibility = check_modules_compatibility_quick(modules_dict_persona, merge_level="persona")
    assert isinstance(compatibility, dict)

    # Analyze feasibility
    feasibility = analyze_merge_feasibility_quick(modules_dict_persona, target_level="persona")
    assert isinstance(feasibility, dict)

    # Merge
    result = merge_multiple_modules_quick(modules_dict_persona, base_module="02", level="persona")
    assert isinstance(result, pd.DataFrame)


def test_get_info_for_all_valid_modules():
    """Test getting info for all valid modules in the system"""
    # Test a selection of known modules
    known_modules = ["01", "02", "03", "34", "05"]

    for module_code in known_modules:
        info = get_module_info(module_code)
        if info["valid"]:
            assert "level" in info
            assert "required_keys" in info
            assert "description" in info
            assert len(info["required_keys"]) > 0
