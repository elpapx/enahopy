# Phase 2 Coverage Analysis - Detailed Gap Identification

**Date:** November 16, 2025
**Session:** Coverage Improvement Phase 2
**Analyst:** AI Engineer (Claude Code)

---

## Executive Summary

This document provides detailed analysis of coverage gaps in the three target modules for Phase 2 coverage improvement. Based on source code review and existing test analysis, specific test cases are recommended to achieve 85%+ coverage in each module.

**Target Modules:**
1. `enahopy/null_analysis/__init__.py` - Current: 68.56% → Target: 85%+
2. `enahopy/null_analysis/convenience.py` - Current: 73.86% → Target: 85%+
3. `enahopy/merger/core.py` - Current: 77.61% → Target: 85%+

**Expected Overall Gain:** +2-3% (59.03% → 61-62%)

---

## Module 1: null_analysis/__init__.py

### Current Coverage: 68.56% (193 statements, 63 missing)

### Identified Coverage Gaps

#### **Gap 1: Import Fallback Paths (Lines 18-27, 42-49, 56-61, 77-78)**
**Missing Lines:** Import error handling when submodules unavailable

**Test Strategy:**
```python
def test_null_analyzer_import_fallback():
    """Test fallback when NullAnalyzer import fails"""
    # Mock ImportError for core.analyzer

def test_patterns_unavailable_fallback():
    """Test PATTERNS_AVAILABLE=False path"""
    # Mock ImportError for patterns module

def test_reports_unavailable_fallback():
    """Test REPORTS_AVAILABLE=False path"""
    # Mock ImportError for reports module
```

**Priority:** Medium (improves robustness testing)
**Estimated Coverage Gain:** +2-3%

#### **Gap 2: CONVENIENCE_AVAILABLE False Path (Lines 96-105)**
**Missing Lines:** Fallback function definitions when convenience module unavailable

**Test Strategy:**
```python
def test_convenience_unavailable_uses_fallback_functions():
    """Test that fallback calculate_null_percentage works"""
    # Import with mocked unavailable convenience module
    # Test fallback calculate_null_percentage (lines 108-128)
    # Test fallback find_columns_with_nulls (lines 130-145)
    # Test fallback get_null_summary (lines 147-169)
```

**Priority:** High (tests actual fallback code paths)
**Estimated Coverage Gain:** +3-4%

#### **Gap 3: Visualization Error Handling (Lines 314-320)**
**Missing Lines:** Exception handling in analyze() for visualization generation

**Test Strategy:**
```python
def test_analyze_visualization_exception_handling(monkeypatch):
    """Test analyze handles visualization errors gracefully"""
    def mock_visualizer_raises(*args, **kwargs):
        raise RuntimeError("Visualization failed")

    analyzer = ENAHONullAnalyzer()
    if analyzer.visualizer:
        monkeypatch.setattr(analyzer.visualizer, "visualize_null_matrix", mock_visualizer_raises)

    result = analyzer.analyze(df, include_visualizations=True)
    # Should handle exception and continue
```

**Priority:** High (error path coverage)
**Estimated Coverage Gain:** +1-2%

#### **Gap 4: Report Generation Error Paths (Lines 564-587)**
**Missing Lines:** Multiple exception handlers in generate_null_report()

**Test Strategy:**
```python
def test_generate_null_report_osserror_handling(tmp_path, monkeypatch):
    """Test OSError handling when saving report (lines 569-574)"""
    # Create read-only directory or invalid path

def test_generate_null_report_attribute_error_handling(monkeypatch):
    """Test AttributeError when report.save() doesn't exist (lines 576-581)"""
    # Mock report object without save() method

def test_generate_null_report_unexpected_error_logging(monkeypatch):
    """Test unexpected error logging (lines 583-587)"""
    # Mock to raise unexpected exception
```

**Priority:** High (error handling coverage)
**Estimated Coverage Gain:** +2-3%

#### **Gap 5: ADVANCED_IMPUTATION_AVAILABLE False Path (Lines 200-204)**
**Missing Lines:** Fallback when ML imputation modules unavailable

**Test Strategy:**
```python
def test_advanced_imputation_unavailable_fallback():
    """Test behavior when ML imputation not available"""
    # Mock ImportError for strategies module
    # Verify flags are set to False
```

**Priority:** Low (ML modules typically excluded)
**Estimated Coverage Gain:** +0.5%

### Recommended Test Cases (10-15 tests)

1. ✅ **test_import_fallback_null_analyzer** - Test NullAnalyzer fallback class
2. ✅ **test_patterns_import_failure** - Test PATTERNS_AVAILABLE=False
3. ✅ **test_reports_import_failure** - Test REPORTS_AVAILABLE=False
4. ✅ **test_utils_import_failure** - Test UTILS_AVAILABLE=False
5. ✅ **test_convenience_fallback_calculate_null_percentage** - Test fallback function
6. ✅ **test_convenience_fallback_find_columns_with_nulls** - Test fallback function
7. ✅ **test_convenience_fallback_get_null_summary** - Test fallback function
8. ✅ **test_analyze_visualization_exception** - Test visualization error handling
9. ✅ **test_generate_report_osserror** - Test OS error in report saving
10. ✅ **test_generate_report_attributeerror** - Test missing save() method
11. ✅ **test_generate_report_unexpected_error** - Test unexpected exceptions
12. ✅ **test_analyze_with_report_has_recommendations** - Test recommendations inclusion
13. ✅ **test_get_imputation_recommendations_no_metrics** - Test empty analysis input
14. ✅ **test_analyze_null_patterns_with_geographic_filter** - Test geographic filtering
15. ✅ **test_generate_comprehensive_report_metadata** - Test metadata structure

**Expected Coverage After Tests:** 68.56% → 82-85%
**Expected Overall Gain:** +0.9-1.0%

---

## Module 2: null_analysis/convenience.py

### Current Coverage: 73.86% (175 statements, 42 missing)

### Identified Coverage Gaps

#### **Gap 1: Format Validation in generate_null_report() (Lines 110-118)**
**Missing Lines:** Format enum conversion and fallback logic

**Test Strategy:**
```python
def test_generate_null_report_format_conversion():
    """Test format string to enum conversion (lines 112-116)"""
    # Test valid formats: html, json, xlsx, md
    # Test invalid format handling with print statement

def test_generate_null_report_empty_formats_fallback():
    """Test fallback when all formats invalid (lines 117-118)"""
    # Pass all invalid formats
    # Verify fallback to ["html", "json"]
```

**Priority:** Medium
**Estimated Coverage Gain:** +1-2%

#### **Gap 2: Difference Calculation in compare_null_patterns() (Lines 182-198)**
**Missing Lines:** Multi-dataset comparison difference calculation

**Test Strategy:**
```python
def test_compare_null_patterns_difference_calculation():
    """Test difference calculation between datasets (lines 182-196)"""
    df1 = pd.DataFrame({"A": [1, None, 3], "B": [1, 2, 3]})  # 16.7% missing
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [None, None, 3]})  # 33.3% missing
    df3 = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})  # 0% missing

    result = compare_null_patterns({"ds1": df1, "ds2": df2, "ds3": df3})

    # Should have differences calculated
    assert "differences" in result
    assert len(result["differences"]) >= 2  # ds1 vs ds2, ds1 vs ds3
```

**Priority:** High
**Estimated Coverage Gain:** +2-3%

#### **Gap 3: Variable Completeness Check (Lines 283-289)**
**Missing Lines:** Per-variable completeness validation in validate_data_completeness()

**Test Strategy:**
```python
def test_validate_data_completeness_variable_thresholds():
    """Test per-variable completeness checking (lines 283-289)"""
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],  # 100% complete
        "B": [1, None, None, 4, 5],  # 60% complete
        "C": [None, None, None, None, 5]  # 20% complete
    })

    result = validate_data_completeness(
        df,
        required_completeness=80.0,
        required_variables=["A", "B", "C"]
    )

    # B and C should be below threshold
    assert "variables_below_threshold" in result
    assert "B" in result["variables_below_threshold"]
    assert "C" in result["variables_below_threshold"]
    assert result["variables_below_threshold"]["B"] == 60.0
```

**Priority:** High
**Estimated Coverage Gain:** +1-2%

#### **Gap 4: Pattern Interpretation Logic (Lines 329-358)**
**Missing Lines:** Common missing pattern interpretations in analyze_common_missing_patterns()

**Test Strategy:**
```python
def test_analyze_common_missing_patterns_all_complete():
    """Test interpretation for all-complete pattern (lines 339-342)"""
    df = pd.DataFrame({"A": list(range(20)), "B": list(range(20))})
    result = analyze_common_missing_patterns(df, min_pattern_frequency=15)

    # Should detect and interpret all-complete pattern
    assert any("completamente sin faltantes" in interp for interp in result["interpretations"])

def test_analyze_common_missing_patterns_all_missing():
    """Test interpretation for all-missing pattern (lines 343-346)"""
    df = pd.DataFrame({"A": [None]*20, "B": [None]*20})
    result = analyze_common_missing_patterns(df, min_pattern_frequency=15)

    # Should detect all-missing pattern
    assert any("completamente faltantes" in interp for interp in result["interpretations"])

def test_analyze_common_missing_patterns_single_variable():
    """Test single-variable missing pattern (lines 347-352)"""
    df = pd.DataFrame({
        "A": [None]*20,  # Always missing
        "B": list(range(20))  # Never missing
    })
    result = analyze_common_missing_patterns(df, min_pattern_frequency=15)

    # Should detect single-variable pattern
    assert any("solo faltan en variable" in interp for interp in result["interpretations"])

def test_analyze_common_missing_patterns_joint_missing():
    """Test joint missing pattern (lines 353-358)"""
    df = pd.DataFrame({
        "A": [None, None, 1]*7,  # Missing in pairs
        "B": [None, None, 2]*7,  # Missing in pairs
        "C": list(range(21))  # Never missing
    })
    result = analyze_common_missing_patterns(df, min_pattern_frequency=5)

    # Should detect joint missing pattern
    assert any("faltan conjuntamente" in interp for interp in result["interpretations"])
```

**Priority:** Medium
**Estimated Coverage Gain:** +2-3%

#### **Gap 5: Statistical Tests in detect_missing_patterns_automatically() (Lines 479-493)**
**Missing Lines:** Confidence calculation and alternative patterns

**Test Strategy:**
```python
def test_detect_missing_patterns_with_statistical_tests():
    """Test confidence calculation from statistical tests (lines 480-493)"""
    # Create data with specific pattern
    df = pd.DataFrame({
        "A": [1, None, 3, None, 5, None]*10,
        "B": [None, 2, 3, None, 5, None]*10
    })

    result = detect_missing_patterns_automatically(df, confidence_threshold=0.90)

    # Should include evidence and alternative patterns
    assert "evidence" in result
    assert "alternative_patterns" in result
    if result["confidence"] < 0.90:
        assert "recommendation" in result
```

**Priority:** Medium
**Estimated Coverage Gain:** +1-2%

### Recommended Test Cases (8-10 tests)

1. ✅ **test_generate_null_report_format_enum_conversion** - Test format handling
2. ✅ **test_generate_null_report_all_invalid_formats** - Test fallback
3. ✅ **test_compare_null_patterns_three_datasets** - Test multi-dataset differences
4. ✅ **test_validate_data_completeness_per_variable_check** - Test variable thresholds
5. ✅ **test_analyze_common_patterns_all_complete_interpretation** - Test interpretation
6. ✅ **test_analyze_common_patterns_all_missing_interpretation** - Test interpretation
7. ✅ **test_analyze_common_patterns_single_var_interpretation** - Test interpretation
8. ✅ **test_analyze_common_patterns_joint_interpretation** - Test interpretation
9. ✅ **test_detect_patterns_with_low_confidence** - Test alternative patterns
10. ✅ **test_detect_patterns_with_statistical_evidence** - Test evidence inclusion

**Expected Coverage After Tests:** 73.86% → 85-88%
**Expected Overall Gain:** +0.7-0.8%

---

## Module 3: merger/core.py

### Current Coverage: 77.61% (546 statements, 101 missing)

**Note:** merger/core.py is large (25K+ tokens). Analysis based on test file review and common patterns.

### Identified Coverage Gaps

#### **Gap 1: Configuration Edge Cases**
**Test Strategy:**
- Invalid chunk sizes (zero, negative)
- Missing required configuration fields
- Extreme threshold values

#### **Gap 2: Duplicate Strategy Error Paths**
**Test Strategy:**
- BEST_QUALITY without quality column
- AGGREGATE with invalid functions
- AGGREGATE with empty functions

#### **Gap 3: Quality Metrics Edge Cases**
**Test Strategy:**
- Zero division handling
- Empty DataFrame quality calculation
- Quality recommendations for edge scores

#### **Gap 4: Merge Operation Error Paths**
**Test Strategy:**
- Mismatched column types
- All-NaN keys
- Empty DataFrames
- Duplicate column names

#### **Gap 5: Validation Error Handling**
**Test Strategy:**
- Missing ubigeo column
- All-NaN ubigeo values
- Invalid coordinate ranges
- Territorial inconsistencies

### Recommended Test Cases (10-12 tests)

Based on existing test_merger_core_coverage.py, add:

1. ✅ **test_chunk_processing_with_large_dataset** - Test chunking logic
2. ✅ **test_merge_with_memory_optimization** - Test memory optimization paths
3. ✅ **test_aggregate_with_mixed_types** - Test aggregation type handling
4. ✅ **test_quality_metrics_with_partial_data** - Test incomplete data scenarios
5. ✅ **test_territorial_validation_with_inconsistencies** - Test inconsistency detection
6. ✅ **test_coordinate_validation_boundary_cases** - Test lat/lon boundaries
7. ✅ **test_merge_report_generation** - Test report creation paths
8. ✅ **test_cache_hit_and_miss_paths** - Test cache logic
9. ✅ **test_logging_in_verbose_mode** - Test verbose logging paths
10. ✅ **test_early_exit_with_warnings** - Test warning generation
11. ✅ **test_validate_merge_column_types_mismatch** - Test type validation
12. ✅ **test_module_merge_with_conflicts** - Test conflict resolution

**Expected Coverage After Tests:** 77.61% → 85-87%
**Expected Overall Gain:** +0.4-0.5%

---

## Summary and Next Steps

### Phase 2 Target Summary

| Module | Current | Target | Tests Needed | Est. Gain (Module) | Est. Gain (Overall) |
|--------|---------|--------|--------------|-------------------|---------------------|
| null_analysis/__init__.py | 68.56% | 85%+ | 10-15 | +16% | +1.0% |
| null_analysis/convenience.py | 73.86% | 85%+ | 8-10 | +11% | +0.7% |
| merger/core.py | 77.61% | 85%+ | 10-12 | +7% | +0.4% |
| **TOTAL** | 59.03% | 61-62% | **28-37** | - | **+2.1%** |

### Recommended Execution Order

**Session 1: null_analysis/__init__.py (High ROI)**
- Focus on error handling and fallback paths
- 10-15 tests, estimated 2 hours
- Gain: +1.0% overall coverage

**Session 2: null_analysis/convenience.py (Medium ROI)**
- Focus on pattern interpretation and validation
- 8-10 tests, estimated 1.5 hours
- Gain: +0.7% overall coverage

**Session 3: merger/core.py (Moderate ROI)**
- Focus on edge cases and error paths
- 10-12 tests, estimated 2 hours
- Gain: +0.4% overall coverage

### Success Criteria

- ✅ All three modules reach 85%+ coverage
- ✅ Overall coverage reaches 61-62% (from 59.03%)
- ✅ No regression in existing test pass rate
- ✅ All new tests follow existing patterns and conventions

---

**Document Status:** ✅ Complete
**Ready for Implementation:** Yes
**Next Action:** Begin Session 1 - null_analysis/__init__.py test implementation
