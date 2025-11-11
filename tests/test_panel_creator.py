"""
Tests for Panel Data Creator
Tests the enahopy.merger.panel.creator module
"""

import pandas as pd
import pytest

from enahopy.merger.panel import PanelCreator, PanelDataResult, create_panel_data


class TestPanelDataResult:
    """Tests for PanelDataResult dataclass"""

    def test_panel_data_result_creation(self):
        """Test creating PanelDataResult instance"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = PanelDataResult(
            panel_df=df,
            n_periods=2,
            n_individuals=100,
            balanced=True,
            attrition_rate=0.05,
            metadata={"periods": ["2022", "2023"]},
        )

        assert isinstance(result.panel_df, pd.DataFrame)
        assert result.n_periods == 2
        assert result.n_individuals == 100
        assert result.balanced is True
        assert result.attrition_rate == 0.05
        assert result.metadata == {"periods": ["2022", "2023"]}


class TestPanelCreator:
    """Tests for PanelCreator class"""

    @pytest.fixture
    def simple_data_dict(self):
        """Create simple multi-period test data"""
        data_2022 = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["001", "001", "002"],
                "hogar": ["01", "01", "01"],
                "ingreso": [1000, 1500, 2000],
            }
        )

        data_2023 = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["001", "001", "002"],
                "hogar": ["01", "01", "01"],
                "ingreso": [1100, 1600, 2100],
            }
        )

        return {"2022": data_2022, "2023": data_2023}

    @pytest.fixture
    def unbalanced_data_dict(self):
        """Create unbalanced panel with attrition"""
        data_2022 = pd.DataFrame(
            {
                "household_id": [1, 2, 3, 4],
                "income": [1000, 1500, 2000, 2500],
            }
        )

        # Household 4 dropped out
        data_2023 = pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "income": [1100, 1600, 2100],
            }
        )

        return {"2022": data_2022, "2023": data_2023}

    def test_panel_creator_initialization(self):
        """Test PanelCreator can be initialized"""
        creator = PanelCreator()
        assert creator is not None
        assert creator.logger is not None

    def test_create_balanced_panel(self, simple_data_dict):
        """Test creating a balanced panel dataset"""
        creator = PanelCreator()
        id_vars = ["conglome", "vivienda", "hogar"]

        result = creator.create_panel(simple_data_dict, id_vars=id_vars, time_var="año")

        # Check result structure
        assert isinstance(result, PanelDataResult)
        assert isinstance(result.panel_df, pd.DataFrame)

        # Check panel has both periods
        assert result.n_periods == 2
        assert "año" in result.panel_df.columns

        # Check all periods present
        assert set(result.panel_df["año"].unique()) == {"2022", "2023"}

        # Check balanced panel
        assert result.balanced

        # Check total rows (3 households × 2 years = 6)
        assert len(result.panel_df) == 6

        # Check individuals
        assert result.n_individuals == 3

    def test_create_unbalanced_panel(self, unbalanced_data_dict):
        """Test creating unbalanced panel with attrition"""
        creator = PanelCreator()
        id_vars = ["household_id"]

        result = creator.create_panel(unbalanced_data_dict, id_vars=id_vars, time_var="year")

        # Panel should be unbalanced
        assert not result.balanced

        # Check attrition rate (1 household dropped out of 4 = 25%)
        assert result.attrition_rate == 0.25

        # Total rows: 4 + 3 = 7
        assert len(result.panel_df) == 7

        # Check year column added
        assert "year" in result.panel_df.columns
        assert set(result.panel_df["year"].unique()) == {"2022", "2023"}

    def test_panel_three_periods(self):
        """Test panel with three time periods"""
        data_dict = {
            "2021": pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            "2022": pd.DataFrame({"id": [1, 2], "value": [15, 25]}),
            "2023": pd.DataFrame({"id": [1, 2], "value": [20, 30]}),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        assert result.n_periods == 3
        assert result.balanced
        assert len(result.panel_df) == 6  # 2 individuals × 3 years
        assert result.attrition_rate == 0.0

    def test_panel_metadata(self, simple_data_dict):
        """Test panel metadata is correctly populated"""
        creator = PanelCreator()
        result = creator.create_panel(
            simple_data_dict, id_vars=["conglome", "vivienda", "hogar"], time_var="año"
        )

        assert "periods" in result.metadata
        assert result.metadata["periods"] == ["2022", "2023"]

    def test_panel_preserves_columns(self, simple_data_dict):
        """Test that all original columns are preserved"""
        creator = PanelCreator()
        result = creator.create_panel(
            simple_data_dict, id_vars=["conglome", "vivienda", "hogar"], time_var="año"
        )

        expected_columns = ["conglome", "vivienda", "hogar", "ingreso", "año"]
        assert set(result.panel_df.columns) == set(expected_columns)

    def test_panel_with_composite_id(self):
        """Test panel creation with composite identifier"""
        data_dict = {
            "2022": pd.DataFrame(
                {
                    "region": ["A", "A", "B"],
                    "household": [1, 2, 1],
                    "income": [1000, 1500, 2000],
                }
            ),
            "2023": pd.DataFrame(
                {
                    "region": ["A", "A", "B"],
                    "household": [1, 2, 1],
                    "income": [1100, 1600, 2100],
                }
            ),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["region", "household"], time_var="year")

        assert result.n_individuals == 3
        assert result.balanced
        assert len(result.panel_df) == 6

    def test_panel_single_period_edge_case(self):
        """Test panel creation with single period (edge case)"""
        data_dict = {"2022": pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})}

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        assert result.n_periods == 1
        assert result.balanced
        assert len(result.panel_df) == 3
        # With single period, attrition is 0
        assert result.attrition_rate == 0.0

    def test_panel_empty_dataframe(self):
        """Test panel creation with empty dataframes"""
        data_dict = {
            "2022": pd.DataFrame({"id": [], "value": []}),
            "2023": pd.DataFrame({"id": [], "value": []}),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        assert result.n_periods == 2
        assert result.n_individuals == 0
        assert len(result.panel_df) == 0

    def test_attrition_calculation(self):
        """Test attrition rate calculation accuracy"""
        # Start with 10 households, end with 7 (30% attrition)
        data_dict = {
            "2022": pd.DataFrame({"id": list(range(1, 11)), "value": [100] * 10}),
            "2023": pd.DataFrame({"id": list(range(1, 8)), "value": [110] * 7}),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        assert result.attrition_rate == pytest.approx(0.3, abs=0.01)
        assert not result.balanced


class TestConvenienceFunction:
    """Tests for create_panel_data convenience function"""

    def test_create_panel_data_returns_dataframe(self):
        """Test convenience function returns DataFrame directly"""
        data_dict = {
            "2022": pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            "2023": pd.DataFrame({"id": [1, 2], "value": [15, 25]}),
        }

        result = create_panel_data(data_dict, id_vars=["id"], time_var="year")

        assert isinstance(result, pd.DataFrame)
        assert "year" in result.columns
        assert len(result) == 4  # 2 individuals × 2 years

    def test_convenience_function_equivalent_to_class(self):
        """Test convenience function produces same result as class method"""
        data_dict = {
            "2022": pd.DataFrame({"id": [1, 2], "value": [10, 20]}),
            "2023": pd.DataFrame({"id": [1, 2], "value": [15, 25]}),
        }

        # Using convenience function
        result_convenience = create_panel_data(data_dict, id_vars=["id"], time_var="year")

        # Using class directly
        creator = PanelCreator()
        result_class = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        # Should be equivalent
        pd.testing.assert_frame_equal(result_convenience, result_class.panel_df)


class TestPanelCreatorEdgeCases:
    """Edge case tests for panel creator"""

    def test_mismatched_columns_between_periods(self):
        """Test handling when different periods have different columns"""
        # 2022 has extra column that 2023 doesn't have
        data_dict = {
            "2022": pd.DataFrame({"id": [1, 2], "value": [10, 20], "extra": [100, 200]}),
            "2023": pd.DataFrame({"id": [1, 2], "value": [15, 25]}),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        # Should handle gracefully with NaN for missing values
        assert len(result.panel_df) == 4
        assert "extra" in result.panel_df.columns
        # 2023 rows should have NaN in 'extra' column
        assert result.panel_df[result.panel_df["year"] == "2023"]["extra"].isna().all()

    def test_large_attrition(self):
        """Test panel with severe attrition (90%)"""
        data_dict = {
            "2022": pd.DataFrame({"id": list(range(1, 101)), "value": [100] * 100}),
            "2023": pd.DataFrame({"id": list(range(1, 11)), "value": [110] * 10}),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        assert result.attrition_rate == pytest.approx(0.9, abs=0.01)
        assert not result.balanced
        assert result.n_individuals == 100  # Unique individuals across all periods

    def test_panel_with_duplicate_ids_within_period(self):
        """Test panel when there are duplicate IDs within a single period"""
        # This represents data quality issues - multiple observations per household
        data_dict = {
            "2022": pd.DataFrame(
                {
                    "id": [1, 1, 2],  # Duplicate ID
                    "value": [10, 11, 20],
                }
            ),
            "2023": pd.DataFrame({"id": [1, 2], "value": [15, 25]}),
        }

        creator = PanelCreator()
        result = creator.create_panel(data_dict, id_vars=["id"], time_var="year")

        # Should still create panel, but n_individuals counts unique IDs
        assert result.n_individuals == 2
        assert len(result.panel_df) == 5  # 3 + 2 rows
