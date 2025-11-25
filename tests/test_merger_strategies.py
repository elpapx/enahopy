"""
Tests for enahopy.merger.strategies module

Tests coverage for:
- MergeStrategy base class
- StandardMergeStrategy implementation
- Strategy pattern usage
"""

from abc import ABC

import numpy as np
import pandas as pd
import pytest

from enahopy.merger.strategies import MergeStrategy, StandardMergeStrategy


class TestMergeStrategyBase:
    """Test cases for MergeStrategy abstract base class"""

    def test_merge_strategy_is_abstract(self):
        """Test that MergeStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MergeStrategy()

    def test_merge_strategy_is_abc(self):
        """Test that MergeStrategy is an ABC"""
        assert issubclass(MergeStrategy, ABC)

    def test_merge_method_is_abstract(self):
        """Test that merge method is abstract"""
        assert hasattr(MergeStrategy, "merge")
        assert getattr(MergeStrategy.merge, "__isabstractmethod__", False)

    def test_custom_strategy_must_implement_merge(self):
        """Test that custom strategies must implement merge method"""

        class IncompletStrategy(MergeStrategy):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompletStrategy()


class TestStandardMergeStrategy:
    """Test cases for StandardMergeStrategy implementation"""

    @pytest.fixture
    def strategy(self):
        """Create a StandardMergeStrategy instance"""
        return StandardMergeStrategy()

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrames for testing"""
        left_df = pd.DataFrame(
            {"key": [1, 2, 3, 4], "value_left": ["a", "b", "c", "d"], "common": [10, 20, 30, 40]}
        )

        right_df = pd.DataFrame(
            {"key": [2, 3, 4, 5], "value_right": ["w", "x", "y", "z"], "common": [25, 35, 45, 55]}
        )

        return left_df, right_df

    def test_strategy_instantiation(self, strategy):
        """Test that StandardMergeStrategy can be instantiated"""
        assert isinstance(strategy, StandardMergeStrategy)
        assert isinstance(strategy, MergeStrategy)

    def test_merge_left_join_default(self, strategy, sample_data):
        """Test default merge behavior (left join)"""
        left_df, right_df = sample_data

        result = strategy.merge(left_df, right_df, keys=["key"])

        # Default should be left join
        assert len(result) == 4
        assert list(result["key"]) == [1, 2, 3, 4]

        # Check that right values are present where keys match
        assert pd.notna(result.loc[result["key"] == 2, "value_right"].iloc[0])
        assert pd.isna(result.loc[result["key"] == 1, "value_right"].iloc[0])

    def test_merge_inner_join(self, strategy, sample_data):
        """Test inner join merge"""
        left_df, right_df = sample_data

        result = strategy.merge(left_df, right_df, keys=["key"], how="inner")

        # Inner join should only keep matching keys
        assert len(result) == 3
        assert set(result["key"]) == {2, 3, 4}

    def test_merge_outer_join(self, strategy, sample_data):
        """Test outer join merge"""
        left_df, right_df = sample_data

        result = strategy.merge(left_df, right_df, keys=["key"], how="outer")

        # Outer join should keep all keys
        assert len(result) == 5
        assert set(result["key"]) == {1, 2, 3, 4, 5}

    def test_merge_right_join(self, strategy, sample_data):
        """Test right join merge"""
        left_df, right_df = sample_data

        result = strategy.merge(left_df, right_df, keys=["key"], how="right")

        # Right join should keep all right keys
        assert len(result) == 4
        assert set(result["key"]) == {2, 3, 4, 5}

    def test_merge_with_multiple_keys(self, strategy):
        """Test merge with multiple key columns"""
        left_df = pd.DataFrame(
            {"key1": [1, 1, 2, 2], "key2": ["a", "b", "a", "b"], "val_l": [10, 20, 30, 40]}
        )

        right_df = pd.DataFrame(
            {"key1": [1, 1, 2, 2], "key2": ["a", "b", "a", "b"], "val_r": [100, 200, 300, 400]}
        )

        result = strategy.merge(left_df, right_df, keys=["key1", "key2"])

        assert len(result) == 4
        assert "val_l" in result.columns
        assert "val_r" in result.columns

    def test_merge_with_empty_left(self, strategy):
        """Test merge behavior with empty left DataFrame"""
        left_df = pd.DataFrame({"key": [], "value": []})
        right_df = pd.DataFrame({"key": [1, 2], "value": [10, 20]})

        result = strategy.merge(left_df, right_df, keys=["key"])

        assert len(result) == 0

    def test_merge_with_empty_right(self, strategy):
        """Test merge behavior with empty right DataFrame"""
        left_df = pd.DataFrame({"key": [1, 2], "value": [10, 20]})
        right_df = pd.DataFrame({"key": [], "value": []})

        result = strategy.merge(left_df, right_df, keys=["key"])

        # Left join with empty right should keep all left rows with NaN for right columns
        assert len(result) == 2
        assert pd.isna(result["value_y"]).all()

    def test_merge_with_duplicates(self, strategy):
        """Test merge behavior with duplicate keys"""
        left_df = pd.DataFrame({"key": [1, 1, 2], "value": ["a", "b", "c"]})

        right_df = pd.DataFrame({"key": [1, 1, 2], "value": ["x", "y", "z"]})

        result = strategy.merge(left_df, right_df, keys=["key"])

        # Should create cartesian product for duplicates
        assert len(result) == 5  # (2*2) for key=1 + 1 for key=2

    def test_merge_with_suffixes(self, strategy, sample_data):
        """Test merge with custom suffixes for overlapping columns"""
        left_df, right_df = sample_data

        result = strategy.merge(left_df, right_df, keys=["key"], suffixes=("_L", "_R"))

        # Check that suffixes are applied
        assert "common_L" in result.columns
        assert "common_R" in result.columns

    def test_merge_with_indicator(self, strategy, sample_data):
        """Test merge with indicator column"""
        left_df, right_df = sample_data

        result = strategy.merge(left_df, right_df, keys=["key"], indicator=True)

        # Check that indicator column exists
        assert "_merge" in result.columns
        assert set(result["_merge"].unique()) <= {"left_only", "both", "right_only"}

    def test_merge_with_nan_keys(self, strategy):
        """Test merge behavior with NaN in key columns"""
        left_df = pd.DataFrame({"key": [1, 2, np.nan, 4], "value_left": ["a", "b", "c", "d"]})

        right_df = pd.DataFrame({"key": [2, 3, np.nan, 4], "value_right": ["w", "x", "y", "z"]})

        result = strategy.merge(left_df, right_df, keys=["key"])

        # pandas typically doesn't match NaN keys in merge
        # Check that we get expected behavior
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2  # At least keys 2 and 4 should match

    def test_merge_preserves_dtypes(self, strategy):
        """Test that merge preserves data types"""
        left_df = pd.DataFrame(
            {"key": [1, 2, 3], "int_col": [10, 20, 30], "float_col": [1.1, 2.2, 3.3]}
        )

        right_df = pd.DataFrame({"key": [2, 3, 4], "str_col": ["a", "b", "c"]})

        result = strategy.merge(left_df, right_df, keys=["key"])

        assert result["int_col"].dtype == left_df["int_col"].dtype
        assert result["float_col"].dtype == left_df["float_col"].dtype

    def test_merge_with_validate_parameter(self, strategy, sample_data):
        """Test merge with validate parameter"""
        left_df, right_df = sample_data

        # Should work fine with unique keys in this example
        result = strategy.merge(left_df, right_df, keys=["key"], validate="one_to_one")

        assert isinstance(result, pd.DataFrame)

    def test_strategy_imports_correctly(self):
        """Test that strategy classes can be imported from package"""
        # Test imports from main module
        from enahopy.merger.strategies import MergeStrategy, StandardMergeStrategy

        assert MergeStrategy is not None
        assert StandardMergeStrategy is not None

        # Test imports from submodules
        from enahopy.merger.strategies.base import MergeStrategy as Base
        from enahopy.merger.strategies.standard import StandardMergeStrategy as Standard

        assert Base is not None
        assert Standard is not None


class TestCustomMergeStrategy:
    """Test cases for creating custom merge strategies"""

    def test_custom_strategy_implementation(self):
        """Test that custom strategies can be implemented"""

        class CustomStrategy(MergeStrategy):
            def merge(self, left, right, keys, **kwargs):
                # Custom merge logic: only keep rows where key is even
                merged = pd.merge(left, right, on=keys, how="inner")
                return merged[merged[keys[0]] % 2 == 0]

        strategy = CustomStrategy()
        assert isinstance(strategy, MergeStrategy)

        left_df = pd.DataFrame({"key": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})

        right_df = pd.DataFrame({"key": [2, 3, 4, 5], "value": ["w", "x", "y", "z"]})

        result = strategy.merge(left_df, right_df, keys=["key"])

        # Should only have even keys
        assert set(result["key"]) == {2, 4}

    def test_strategy_pattern_flexibility(self):
        """Test that strategy pattern provides flexibility"""

        class LeftOnlyStrategy(MergeStrategy):
            """Strategy that only returns left DataFrame columns"""

            def merge(self, left, right, keys, **kwargs):
                return left

        class RightOnlyStrategy(MergeStrategy):
            """Strategy that only returns right DataFrame columns"""

            def merge(self, right, left, keys, **kwargs):
                return right

        left_df = pd.DataFrame({"key": [1, 2], "left_val": ["a", "b"]})
        right_df = pd.DataFrame({"key": [1, 2], "right_val": ["x", "y"]})

        left_strategy = LeftOnlyStrategy()
        result_left = left_strategy.merge(left_df, right_df, ["key"])
        assert "right_val" not in result_left.columns

        right_strategy = RightOnlyStrategy()
        result_right = right_strategy.merge(right_df, left_df, ["key"])
        assert "left_val" not in result_right.columns


class TestStrategiesModuleImports:
    """Test module-level imports and exports"""

    def test_module_has_all(self):
        """Test that __all__ is defined"""
        from enahopy.merger import strategies

        assert hasattr(strategies, "__all__")
        assert "MergeStrategy" in strategies.__all__
        assert "StandardMergeStrategy" in strategies.__all__

    def test_module_exports_correct_classes(self):
        """Test that module exports correct classes"""
        from enahopy.merger import strategies

        assert hasattr(strategies, "MergeStrategy")
        assert hasattr(strategies, "StandardMergeStrategy")

    def test_wildcard_import(self):
        """Test that wildcard import works correctly"""
        # This simulates: from enahopy.merger.strategies import *
        import enahopy.merger.strategies as strategies_module

        exported_names = strategies_module.__all__
        assert len(exported_names) == 2
        assert all(hasattr(strategies_module, name) for name in exported_names)
