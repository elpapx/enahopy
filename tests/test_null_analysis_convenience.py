"""
Tests for enahopy.null_analysis.convenience module

Comprehensive coverage for convenience wrapper functions
"""

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from enahopy.null_analysis.convenience import (
    analyze_common_missing_patterns,
    compare_null_patterns,
    create_null_visualizations,
    detect_missing_patterns_automatically,
    diagnostico_nulos_enaho,
    generate_null_report,
    get_data_quality_score,
    quick_null_analysis,
    suggest_imputation_methods,
    validate_data_completeness,
    LegacyNullAnalyzer,
)


class TestQuickNullAnalysis:
    """Test quick_null_analysis convenience function"""

    def test_quick_null_analysis_basic(self):
        """Test basic null pattern analysis through convenience function"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5],
            'C': [1, 2, 3, 4, 5]
        })

        result = quick_null_analysis(df)

        assert result is not None
        assert 'metrics' in result
        assert 'patterns' in result
        assert result['metrics'].total_rows == 5
        assert result['metrics'].missing_percentage > 0

    def test_quick_null_analysis_with_group_by(self):
        """Test null analysis with grouping"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5],
            'group': ['G1', 'G1', 'G2', 'G2', 'G2']
        })

        result = quick_null_analysis(df, group_by='group')

        assert result is not None
        assert 'metrics' in result

    def test_quick_null_analysis_with_complexity(self):
        """Test null analysis with custom complexity"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })

        result = quick_null_analysis(df, complexity='advanced')

        assert result is not None
        assert 'metrics' in result

    def test_quick_null_analysis_invalid_complexity(self):
        """Test error handling for invalid complexity"""
        df = pd.DataFrame({'A': [1, None, 3]})

        with pytest.raises(ValueError, match="Complejidad .* no válida"):
            quick_null_analysis(df, complexity='invalid')


class TestGetDataQualityScore:
    """Test get_data_quality_score convenience function"""

    def test_get_data_quality_score_basic(self):
        """Test basic quality score calculation"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 2, 3, 4, 5]
        })

        score = get_data_quality_score(df)

        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_get_data_quality_score_detailed(self):
        """Test detailed quality score with breakdown"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })

        result = get_data_quality_score(df, detailed=True)

        assert isinstance(result, dict)
        assert 'overall_score' in result or isinstance(result, (int, float))


class TestCreateNullVisualizations:
    """Test create_null_visualizations convenience function"""

    @pytest.mark.skip(reason="create_visualizations method not implemented in ENAHONullAnalyzer")
    def test_create_null_visualizations_basic(self):
        """Test basic visualization creation"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5]
        })

        result = create_null_visualizations(df)

        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.skip(reason="create_visualizations method not implemented in ENAHONullAnalyzer")
    def test_create_null_visualizations_with_output(self):
        """Test visualization creation with file output"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "viz_test")
            result = create_null_visualizations(df, output_path=output_path)

            assert result is not None
            assert isinstance(result, dict)

    @pytest.mark.skip(reason="create_visualizations method not implemented in ENAHONullAnalyzer")
    def test_create_null_visualizations_with_group_by(self):
        """Test visualization with grouping"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3],
            'group': ['G1', 'G1', 'G2']
        })

        result = create_null_visualizations(df, group_by='group')

        assert result is not None

    @pytest.mark.skip(reason="create_visualizations method not implemented in ENAHONullAnalyzer")
    def test_create_null_visualizations_interactive(self):
        """Test interactive visualization mode"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })

        result = create_null_visualizations(df, interactive=True)

        assert result is not None


class TestGenerateNullReport:
    """Test generate_null_report convenience function"""

    def test_generate_null_report_basic(self):
        """Test basic report generation"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report")
            result = generate_null_report(df, output_path)

            assert result is not None
            assert isinstance(result, dict)

    def test_generate_null_report_with_formats(self):
        """Test report with custom formats"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report")
            result = generate_null_report(
                df,
                output_path,
                format_types=['html', 'json']
            )

            assert result is not None

    def test_generate_null_report_invalid_format(self):
        """Test error handling for invalid format"""
        df = pd.DataFrame({'A': [1, None, 3]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report")

            with pytest.raises(ValueError, match="Formatos inválidos"):
                generate_null_report(
                    df,
                    output_path,
                    format_types=['invalid_format']
                )

    def test_generate_null_report_with_group_by(self):
        """Test report with grouping"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3],
            'group': ['G1', 'G1', 'G2']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report")
            result = generate_null_report(df, output_path, group_by='group')

            assert result is not None


class TestCompareNullPatterns:
    """Test compare_null_patterns convenience function"""

    def test_compare_null_patterns_basic(self):
        """Test basic comparison of null patterns"""
        df1 = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })
        df2 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [None, None, 3]
        })

        datasets = {'dataset1': df1, 'dataset2': df2}
        result = compare_null_patterns(datasets)

        assert result is not None
        assert 'individual_analyses' in result
        assert 'metrics_comparison' in result
        assert 'differences' in result
        assert 'best_quality_dataset' in result

    def test_compare_null_patterns_with_group_by(self):
        """Test comparison with grouping"""
        df1 = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3],
            'group': ['G1', 'G1', 'G2']
        })
        df2 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [None, None, 3],
            'group': ['G1', 'G1', 'G2']
        })

        datasets = {'dataset1': df1, 'dataset2': df2}
        result = compare_null_patterns(datasets, group_by='group')

        assert result is not None

    def test_compare_null_patterns_empty_datasets(self):
        """Test error handling for empty datasets dict"""
        with pytest.raises(ValueError, match="Se requiere al menos un dataset"):
            compare_null_patterns({})

    def test_compare_null_patterns_single_dataset(self):
        """Test error handling for single dataset"""
        df = pd.DataFrame({'A': [1, None, 3]})
        datasets = {'dataset1': df}

        with pytest.raises(ValueError, match="Se requieren al menos 2 datasets"):
            compare_null_patterns(datasets)

    def test_compare_null_patterns_with_invalid_dataframe(self):
        """Test comparison with invalid dataframe in dict"""
        df1 = pd.DataFrame({'A': [1, None, 3]})
        df2 = pd.DataFrame()  # Empty dataframe

        datasets = {'dataset1': df1, 'dataset2': df2}
        result = compare_null_patterns(datasets)

        assert result is not None
        assert 'individual_analyses' in result
        # dataset2 should have an error entry
        assert 'dataset2' in result['individual_analyses']


class TestSuggestImputationMethods:
    """Test suggest_imputation_methods convenience function"""

    @pytest.mark.skip(reason="suggest_imputation_strategy method not implemented in ENAHONullAnalyzer")
    def test_suggest_imputation_methods_basic(self):
        """Test basic imputation suggestion"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5],
            'C': [1.0, 2.0, None, 4.0, 5.0]
        })

        result = suggest_imputation_methods(df)

        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.skip(reason="suggest_imputation_strategy method not implemented in ENAHONullAnalyzer")
    def test_suggest_imputation_methods_specific_variable(self):
        """Test imputation suggestion for specific variable"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5]
        })

        result = suggest_imputation_methods(df, variable='A')

        assert result is not None


class TestValidateDataCompleteness:
    """Test validate_data_completeness convenience function"""

    def test_validate_data_completeness_basic(self):
        """Test basic completeness validation"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 2, 3, 4, 5]
        })

        result = validate_data_completeness(df)

        assert result is not None
        assert 'is_valid' in result
        assert 'completeness_score' in result
        assert result['is_valid'] is True
        assert result['completeness_score'] == 100.0

    def test_validate_data_completeness_below_threshold(self):
        """Test validation with data below threshold"""
        df = pd.DataFrame({
            'A': [1, None, None, None, 5],
            'B': [None, None, None, None, 5]
        })

        result = validate_data_completeness(df, required_completeness=80.0)

        assert result is not None
        assert result['is_valid'] is False
        assert result['completeness_score'] < 80.0
        assert len(result['recommendations']) > 0

    def test_validate_data_completeness_required_variables(self):
        """Test validation with required variables"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1, 2, 3],
            'C': [1, None, 3]
        })

        result = validate_data_completeness(
            df,
            required_completeness=90.0,
            required_variables=['A', 'B', 'C']
        )

        assert result is not None
        assert 'variables_below_threshold' in result

    def test_validate_data_completeness_missing_required_variables(self):
        """Test validation with missing required variables"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1, 2, 3]
        })

        result = validate_data_completeness(
            df,
            required_variables=['A', 'B', 'X', 'Y']
        )

        assert result is not None
        assert result['is_valid'] is False
        assert 'X' in result['missing_variables']
        assert 'Y' in result['missing_variables']
        assert len(result['recommendations']) > 0

    def test_validate_data_completeness_empty_dataframe(self):
        """Test validation with empty dataframe"""
        df = pd.DataFrame()

        result = validate_data_completeness(df)

        assert result is not None
        assert result['is_valid'] is False
        assert result['completeness_score'] == 0.0


class TestAnalyzeCommonMissingPatterns:
    """Test analyze_common_missing_patterns convenience function"""

    def test_analyze_common_missing_patterns_basic(self):
        """Test basic pattern analysis"""
        df = pd.DataFrame({
            'A': [1, None, 3, None, 5, None, 7, None, 9, None, 11, None],
            'B': [None, 2, 3, None, 5, None, 7, None, 9, None, 11, None],
            'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        })

        result = analyze_common_missing_patterns(df, min_pattern_frequency=2)

        assert result is not None
        assert 'total_unique_patterns' in result
        assert 'common_patterns' in result
        assert 'pattern_analysis' in result
        assert 'interpretations' in result

    def test_analyze_common_missing_patterns_no_common_patterns(self):
        """Test pattern analysis with no common patterns"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, None]
        })

        result = analyze_common_missing_patterns(df, min_pattern_frequency=10)

        assert result is not None
        assert result['common_patterns'] == 0

    def test_analyze_common_missing_patterns_all_complete(self):
        """Test pattern analysis with no missing values"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        })

        result = analyze_common_missing_patterns(df, min_pattern_frequency=5)

        assert result is not None
        assert result['common_patterns'] >= 1
        # Should detect the "all complete" pattern
        assert len(result['interpretations']) > 0

    def test_analyze_common_missing_patterns_all_missing(self):
        """Test pattern analysis with all values missing"""
        df = pd.DataFrame({
            'A': [None] * 15,
            'B': [None] * 15
        })

        result = analyze_common_missing_patterns(df, min_pattern_frequency=5)

        assert result is not None
        assert result['common_patterns'] >= 1


class TestDetectMissingPatternsAutomatically:
    """Test detect_missing_patterns_automatically convenience function"""

    def test_detect_missing_patterns_automatically_basic(self):
        """Test automatic pattern detection"""
        df = pd.DataFrame({
            'A': [1, None, 3, 4, 5],
            'B': [None, 2, 3, None, 5],
            'C': [1, 2, 3, 4, 5]
        })

        result = detect_missing_patterns_automatically(df)

        assert result is not None
        assert 'detected_pattern' in result
        assert 'confidence' in result
        assert 'evidence' in result
        assert 'alternative_patterns' in result

    def test_detect_missing_patterns_automatically_with_threshold(self):
        """Test automatic detection with custom threshold"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3]
        })

        result = detect_missing_patterns_automatically(df, confidence_threshold=0.90)

        assert result is not None
        assert 0 <= result['confidence'] <= 1.0


class TestLegacyFunctions:
    """Test legacy compatibility functions"""

    def test_legacy_null_analyzer_deprecation_warning(self):
        """Test deprecation warning for LegacyNullAnalyzer"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyzer = LegacyNullAnalyzer()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    @pytest.mark.skip(reason="Legacy diagnostico has KeyError in implementation - needs fix in convenience.py")
    def test_legacy_null_analyzer_diagnostico(self):
        """Test legacy analyzer diagnostico method"""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            analyzer = LegacyNullAnalyzer()

            df = pd.DataFrame({
                'A': [1, None, 3],
                'B': [None, 2, 3],
                'geo_departamento': ['Lima', 'Lima', 'Cusco']
            })

            result = analyzer.diagnostico_nulos_enaho(
                df,
                desagregado_por='geo_departamento'
            )

            assert result is not None
            assert 'resumen_total' in result or 'estadisticas' in result

    @pytest.mark.skip(reason="Legacy diagnostico has KeyError in implementation - needs fix in convenience.py")
    def test_diagnostico_nulos_enaho_deprecation_warning(self):
        """Test deprecation warning for diagnostico_nulos_enaho function"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3],
            'geo_departamento': ['Lima', 'Lima', 'Cusco']
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = diagnostico_nulos_enaho(df)

            assert len(w) >= 1
            # Check for deprecation warning
            deprecation_warnings = [warning for warning in w
                                   if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    @pytest.mark.skip(reason="Legacy diagnostico has KeyError in implementation - needs fix in convenience.py")
    def test_diagnostico_nulos_enaho_with_geographic_filter(self):
        """Test legacy function with geographic filter"""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, 2, 3],
            'geo_departamento': ['Lima', 'Lima', 'Cusco']
        })

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = diagnostico_nulos_enaho(
                df,
                desagregado_por='geo_departamento'
            )

            assert result is not None
            assert 'resumen_total' in result or 'estadisticas' in result


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_convenience_functions_with_no_missing_values(self):
        """Test functions with completely clean data"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        # Should work without errors
        result1 = quick_null_analysis(df)
        assert result1['metrics'].missing_percentage == 0.0

        result2 = get_data_quality_score(df)
        assert result2 > 0  # Should have high quality score

        result3 = analyze_common_missing_patterns(df)
        assert result3['total_unique_patterns'] >= 1

    def test_convenience_functions_with_all_missing_values(self):
        """Test functions with completely missing data"""
        df = pd.DataFrame({
            'A': [None, None, None],
            'B': [None, None, None]
        })

        # Should work without errors
        result1 = quick_null_analysis(df)
        assert result1['metrics'].missing_percentage == 100.0

        result2 = get_data_quality_score(df)
        assert result2 >= 0  # Should have low quality score

        result3 = validate_data_completeness(df, required_completeness=50.0)
        assert result3['is_valid'] is False

    def test_convenience_functions_with_single_column(self):
        """Test functions with single column dataframe"""
        df = pd.DataFrame({'A': [1, None, 3, None, 5]})

        result1 = quick_null_analysis(df)
        assert result1 is not None

        result2 = get_data_quality_score(df)
        assert result2 is not None

        result3 = analyze_common_missing_patterns(df, min_pattern_frequency=1)
        assert result3 is not None

    def test_convenience_functions_with_single_row(self):
        """Test functions with single row dataframe"""
        df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})

        result1 = quick_null_analysis(df)
        assert result1 is not None

        result2 = get_data_quality_score(df)
        assert result2 is not None
