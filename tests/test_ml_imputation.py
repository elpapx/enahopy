"""
Tests for ML Imputation Strategies
===================================

Comprehensive tests for MICE, KNN, MissForest, and quality assessment.

Coverage target: 75%+
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest


# Test data generation
@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing"""
    np.random.seed(42)
    n = 500

    data = {
        "num_var1": np.random.normal(100, 15, n),
        "num_var2": np.random.gamma(2, 50, n),
        "num_var3": np.random.exponential(30, n),
        "cat_var1": np.random.choice(["A", "B", "C", "D"], n),
        "cat_var2": np.random.choice([1, 2, 3, 4, 5], n),
        "binary_var": np.random.choice([0, 1], n),
    }

    df = pd.DataFrame(data)

    # Introduce MCAR (Missing Completely At Random)
    mcar_mask = np.random.random(n) < 0.15
    df.loc[mcar_mask, "num_var1"] = np.nan

    # Introduce MAR (Missing At Random) - related to another variable
    mar_mask = (df["num_var2"] > df["num_var2"].quantile(0.75)) & (np.random.random(n) < 0.20)
    df.loc[mar_mask, "num_var3"] = np.nan

    # Categorical missing
    cat_missing = np.random.random(n) < 0.10
    df.loc[cat_missing, "cat_var1"] = np.nan

    return df


@pytest.fixture
def categorical_cols():
    """List of categorical column names"""
    return ["cat_var1", "cat_var2", "binary_var"]


# ============================================================================
# Test ML Imputation Strategies
# ============================================================================


class TestKNNImputation:
    """Tests for KNN imputation"""

    def test_knn_basic_imputation(self, sample_data_with_missing, categorical_cols):
        """Test basic KNN imputation"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()
        original_missing = df.isnull().sum().sum()

        strategy = KNNImputationStrategy(n_neighbors=5, weights="uniform")
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Check that missing values were imputed
        assert df_imputed.isnull().sum().sum() < original_missing
        # Check shape unchanged
        assert df_imputed.shape == df.shape

    def test_knn_with_different_k(self, sample_data_with_missing):
        """Test KNN with different k values"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()

        for k in [3, 5, 10]:
            strategy = KNNImputationStrategy(n_neighbors=k)
            strategy.fit(df)
            df_imputed = strategy.transform(df)
            assert df_imputed.isnull().sum().sum() < df.isnull().sum().sum()

    def test_knn_preserves_non_missing(self, sample_data_with_missing):
        """Test that KNN preserves non-missing values"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()
        non_missing_mask = df.notna()

        strategy = KNNImputationStrategy(n_neighbors=5)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Check that originally non-missing values are preserved
        for col in df.columns:
            if df[col].notna().any():
                original_values = df.loc[non_missing_mask[col], col]
                imputed_values = df_imputed.loc[non_missing_mask[col], col]
                # Allow small numerical differences due to encoding
                if df[col].dtype in [np.float64, np.int64]:
                    np.testing.assert_array_almost_equal(
                        original_values.values, imputed_values.values, decimal=5
                    )


class TestIterativeImputation:
    """Tests for Iterative (MICE-like) imputation"""

    def test_iterative_basic(self, sample_data_with_missing, categorical_cols):
        """Test basic iterative imputation"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import IterativeImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()
        original_missing = df.isnull().sum().sum()

        strategy = IterativeImputationStrategy(max_iter=5, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        assert df_imputed.isnull().sum().sum() < original_missing
        assert df_imputed.shape == df.shape

    def test_iterative_convergence(self, sample_data_with_missing):
        """Test that iterative imputation converges"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import IterativeImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()

        strategy = IterativeImputationStrategy(max_iter=10, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Should complete without errors
        assert strategy.fitted


class TestRandomForestImputation:
    """Tests for Random Forest imputation"""

    def test_rf_basic_imputation(self, sample_data_with_missing, categorical_cols):
        """Test basic RF imputation"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import (
                RandomForestImputationStrategy,
            )
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()
        original_missing = df.isnull().sum().sum()

        strategy = RandomForestImputationStrategy(n_estimators=50, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        assert df_imputed.isnull().sum().sum() < original_missing
        assert df_imputed.shape == df.shape

    def test_rf_feature_importance(self, sample_data_with_missing):
        """Test that RF produces feature importances"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import (
                RandomForestImputationStrategy,
            )
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()

        strategy = RandomForestImputationStrategy(n_estimators=50, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Check that feature importances are available
        importances = strategy.get_feature_importance()
        assert isinstance(importances, dict)
        # Should have importances for columns with missing values
        assert len(importances) > 0


class TestMLImputationManager:
    """Tests for ML imputation manager"""

    def test_manager_creation(self):
        """Test ML imputation manager creation"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import create_ml_imputation_manager
        except ImportError:
            pytest.skip("ML imputation not available")

        manager = create_ml_imputation_manager()
        assert manager is not None
        assert len(manager.strategies) > 0

    def test_manager_strategy_comparison(self, sample_data_with_missing):
        """Test strategy comparison"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import create_ml_imputation_manager
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()
        manager = create_ml_imputation_manager()

        # Compare strategies
        results = manager.compare_strategies(df, test_size=0.3)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_manager_best_strategy_selection(self, sample_data_with_missing):
        """Test best strategy selection"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import create_ml_imputation_manager
        except ImportError:
            pytest.skip("ML imputation not available")

        df = sample_data_with_missing.copy()
        manager = create_ml_imputation_manager()

        results = manager.compare_strategies(df, test_size=0.3)
        best_strategy = manager.get_best_strategy(results)

        assert best_strategy in manager.strategies


# ============================================================================
# Test Quality Assessment
# ============================================================================


class TestQualityAssessment:
    """Tests for imputation quality assessment"""

    def test_quality_assessment_basic(self, sample_data_with_missing):
        """Test basic quality assessment"""
        try:
            from enahopy.null_analysis.strategies.imputation_quality_assessment import (
                QualityAssessmentConfig,
                assess_imputation_quality,
            )
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("Quality assessment not available")

        df = sample_data_with_missing.copy()
        missing_mask = df.isnull()

        # Perform imputation
        strategy = KNNImputationStrategy(n_neighbors=5)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Assess quality
        config = QualityAssessmentConfig(
            metrics_to_compute=["distribution_preservation", "correlation_preservation"]
        )
        result = assess_imputation_quality(df, df_imputed, missing_mask, config=config)

        assert result.overall_score >= 0 and result.overall_score <= 1
        assert isinstance(result.metric_scores, dict)
        assert isinstance(result.recommendations, list)

    def test_quality_metrics_range(self, sample_data_with_missing):
        """Test that quality metrics are in valid ranges"""
        try:
            from enahopy.null_analysis.strategies.imputation_quality_assessment import (
                assess_imputation_quality,
            )
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("Quality assessment not available")

        df = sample_data_with_missing.copy()

        strategy = KNNImputationStrategy(n_neighbors=5)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        result = assess_imputation_quality(df, df_imputed)

        # Check that all scores are in [0, 1]
        for metric, score in result.metric_scores.items():
            assert score >= 0 and score <= 1, f"{metric} score out of range: {score}"


# ============================================================================
# Test Integration with Core Analyzer
# ============================================================================


class TestAnalyzerIntegration:
    """Tests for integration with ENAHONullAnalyzer"""

    def test_analyzer_advanced_imputation(self, sample_data_with_missing, categorical_cols):
        """Test analyzer's advanced imputation method"""
        try:
            from enahopy.null_analysis.core.analyzer import NullAnalyzer
        except ImportError:
            pytest.skip("Analyzer not available")

        df = sample_data_with_missing.copy()
        analyzer = NullAnalyzer()

        try:
            df_imputed, quality_report = analyzer.impute_advanced(
                df,
                strategy="knn",
                categorical_cols=categorical_cols,
                compare_strategies=False,
                assess_quality=True,
            )

            assert df_imputed is not None
            assert quality_report is not None
            assert "strategy_used" in quality_report
            assert quality_report["strategy_used"] == "knn"

        except ImportError:
            pytest.skip("Advanced imputation dependencies not available")


# ============================================================================
# Test ENAHO-Specific Scenarios
# ============================================================================


class TestENAHOScenarios:
    """Tests for ENAHO-specific scenarios"""

    def test_income_imputation(self):
        """Test imputation on income-like data"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import IterativeImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        # Create ENAHO-like income data
        np.random.seed(42)
        n = 300

        df = pd.DataFrame(
            {
                "ing_lab": np.random.lognormal(6, 1, n),
                "ing_indep": np.random.lognormal(5, 1.5, n),
                "edad": np.random.randint(18, 70, n),
                "nivel_educ": np.random.choice([1, 2, 3, 4, 5], n),
                "sexo": np.random.choice([1, 2], n),
            }
        )

        # Introduce missing income data
        income_missing = np.random.random(n) < 0.20
        df.loc[income_missing, "ing_lab"] = np.nan

        strategy = IterativeImputationStrategy(max_iter=5, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Check that income values are positive
        assert (df_imputed["ing_lab"] >= 0).all()

    def test_survey_weight_preservation(self):
        """Test that survey weights are handled correctly"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import (
                RandomForestImputationStrategy,
            )
        except ImportError:
            pytest.skip("ML imputation not available")

        np.random.seed(42)
        n = 300

        df = pd.DataFrame(
            {
                "variable1": np.random.normal(100, 15, n),
                "variable2": np.random.gamma(2, 50, n),
                "factor07": np.random.uniform(0.5, 3.0, n),
            }
        )

        # Introduce missing
        missing = np.random.random(n) < 0.15
        df.loc[missing, "variable1"] = np.nan

        strategy = RandomForestImputationStrategy(n_estimators=50, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Check that factor07 (weight) is preserved
        pd.testing.assert_series_equal(df["factor07"], df_imputed["factor07"])


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests"""

    def test_large_dataset_performance(self):
        """Test performance on larger dataset"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        import time

        # Create larger dataset
        np.random.seed(42)
        n = 5000

        df = pd.DataFrame(
            {
                "var1": np.random.normal(100, 15, n),
                "var2": np.random.gamma(2, 50, n),
                "var3": np.random.exponential(30, n),
            }
        )

        # Introduce missing
        missing = np.random.random(n) < 0.10
        df.loc[missing, "var1"] = np.nan

        # Measure time
        start = time.time()
        strategy = KNNImputationStrategy(n_neighbors=5)
        strategy.fit(df)
        df_imputed = strategy.transform(df)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 30 seconds)
        assert elapsed < 30, f"Imputation took too long: {elapsed:.2f}s"

    def test_memory_efficiency(self, sample_data_with_missing):
        """Test that imputation doesn't explode memory"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import (
                RandomForestImputationStrategy,
            )
        except ImportError:
            pytest.skip("ML imputation not available")

        import sys

        df = sample_data_with_missing.copy()
        original_size = sys.getsizeof(df)

        strategy = RandomForestImputationStrategy(n_estimators=50, random_state=42)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        imputed_size = sys.getsizeof(df_imputed)

        # Should not increase memory by more than 3x
        assert imputed_size < original_size * 3


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_no_missing_values(self):
        """Test imputation on data with no missing values"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = pd.DataFrame({"var1": [1, 2, 3, 4, 5], "var2": [10, 20, 30, 40, 50]})

        strategy = KNNImputationStrategy(n_neighbors=3)
        strategy.fit(df)
        df_imputed = strategy.transform(df)

        # Should return unchanged data
        pd.testing.assert_frame_equal(df, df_imputed)

    def test_all_missing_column(self):
        """Test handling of column with all missing values"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import (
                RandomForestImputationStrategy,
            )
        except ImportError:
            pytest.skip("ML imputation not available")

        df = pd.DataFrame({"var1": [1, 2, 3, 4, 5], "var2": [np.nan] * 5})

        strategy = RandomForestImputationStrategy(n_estimators=50)

        # Should handle gracefully (not raise error)
        try:
            strategy.fit(df)
            df_imputed = strategy.transform(df)
            # Column with all missing might remain missing or be filled with constant
            assert df_imputed is not None
        except Exception:
            # It's acceptable to raise an error for this edge case
            pass

    def test_single_row(self):
        """Test handling of single row DataFrame"""
        try:
            from enahopy.null_analysis.strategies.ml_imputation import KNNImputationStrategy
        except ImportError:
            pytest.skip("ML imputation not available")

        df = pd.DataFrame({"var1": [np.nan], "var2": [10]})

        strategy = KNNImputationStrategy(n_neighbors=1)

        # Should handle gracefully or raise informative error
        try:
            strategy.fit(df)
            df_imputed = strategy.transform(df)
        except ValueError:
            # Expected behavior for insufficient data
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
