"""
Integration Tests for End-to-End ENAHO Workflows
Tests complete pipelines: download → read → merge → analyze
"""

import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path

from enahopy.loader import ENAHODataDownloader
from enahopy.loader.core.config import ENAHOConfig
from enahopy.merger import ENAHOModuleMerger
from enahopy.merger.config import ModuleMergeConfig, ModuleMergeLevel
from enahopy.merger.geographic.merger import GeographicMerger
from enahopy.null_analysis import ENAHONullAnalyzer
from enahopy.merger.panel import PanelCreator


class TestDownloadReadWorkflow:
    """Test download → read workflow"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_downloader_initialization_workflow(self, temp_cache_dir):
        """Test initializing downloader with config"""
        config = ENAHOConfig(cache_dir=temp_cache_dir, default_max_workers=2)

        downloader = ENAHODataDownloader(config=config, verbose=False)

        assert downloader is not None
        assert downloader.config == config
        assert downloader.config.cache_dir == temp_cache_dir

    def test_cache_hit_workflow(self, temp_cache_dir):
        """Test that cache system works in realistic workflow"""
        config = ENAHOConfig(cache_dir=temp_cache_dir)

        downloader = ENAHODataDownloader(config=config, verbose=False)

        # Simulate cache entry
        cache = downloader.cache_manager
        test_metadata = {"module": "01", "year": "2022", "format": "csv"}
        cache.set_metadata("test_key", test_metadata)

        # Verify cache retrieval
        retrieved = cache.get_metadata("test_key")
        assert retrieved is not None
        assert retrieved["module"] == "01"


class TestMergerWorkflows:
    """Test various merger workflows"""

    @pytest.fixture
    def sample_household_data(self):
        """Create sample household-level data"""
        return pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["001", "001", "002"],
                "hogar": ["01", "01", "01"],
                "mieperho": [3, 4, 5],
                "estrato": [1, 2, 1],
            }
        )

    @pytest.fixture
    def sample_person_data(self):
        """Create sample person-level data"""
        return pd.DataFrame(
            {
                "conglome": ["001", "001", "001", "002", "002", "003"],
                "vivienda": ["001", "001", "001", "001", "001", "002"],
                "hogar": ["01", "01", "01", "01", "01", "01"],
                "codperso": ["01", "02", "03", "01", "02", "01"],
                "edad": [45, 42, 15, 38, 12, 55],
                "p207": [1, 2, 1, 1, 1, 2],  # Gender
            }
        )

    @pytest.fixture
    def sample_sumaria_data(self):
        """Create sample sumaria (summary) data"""
        return pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["001", "001", "002"],
                "hogar": ["01", "01", "01"],
                "inghog2d": [2500.0, 3200.0, 1800.0],
                "gashog2d": [2000.0, 2800.0, 1500.0],
                "pobreza": [3, 3, 2],  # 1=pobre extremo, 2=pobre, 3=no pobre
            }
        )

    @pytest.fixture
    def sample_ubigeo_data(self):
        """Create sample UBIGEO geographic data"""
        return pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "010103"],
                "NOMBDEP": ["Amazonas", "Amazonas", "Amazonas"],
                "NOMBPROV": ["Chachapoyas", "Chachapoyas", "Chachapoyas"],
                "NOMBDIST": ["Chachapoyas", "Asunción", "Balsas"],
            }
        )

    def test_module_merge_hogar_level(self, sample_household_data, sample_sumaria_data):
        """Test merging modules at household level"""
        import logging
        config = ModuleMergeConfig(merge_level=ModuleMergeLevel.HOGAR)
        logger = logging.getLogger('test_merger')
        merger = ENAHOModuleMerger(config, logger)

        result = merger.merge_modules(
            left_df=sample_household_data,
            right_df=sample_sumaria_data,
            left_module="01",
            right_module="34",
            merge_config=config,
        )

        # Check merge success
        assert result.merged_df is not None
        assert len(result.merged_df) == 3  # Should preserve all 3 households

        # Check key columns preserved
        assert "conglome" in result.merged_df.columns
        assert "inghog2d" in result.merged_df.columns  # From sumaria
        assert "mieperho" in result.merged_df.columns  # From household

    def test_module_merge_persona_level(self, sample_person_data):
        """Test merging modules at person level"""
        import logging
        # Create additional person-level data (e.g., employment)
        employment_data = pd.DataFrame(
            {
                "conglome": ["001", "001", "001", "002", "002", "003"],
                "vivienda": ["001", "001", "001", "001", "001", "002"],
                "hogar": ["01", "01", "01", "01", "01", "01"],
                "codperso": ["01", "02", "03", "01", "02", "01"],
                "ocu500": [1, 1, 2, 1, 2, 1],  # Occupation status
                "i524a1": [1000, 1500, 0, 1200, 0, 800],  # Income
            }
        )

        config = ModuleMergeConfig(merge_level=ModuleMergeLevel.PERSONA)
        logger = logging.getLogger('test_merger')
        merger = ENAHOModuleMerger(config, logger)

        result = merger.merge_modules(
            left_df=sample_person_data,
            right_df=employment_data,
            left_module="02",
            right_module="05",
            merge_config=config,
        )

        # Check merge success
        assert result.merged_df is not None
        assert len(result.merged_df) == 6  # All 6 persons

        # Check data preserved
        assert "edad" in result.merged_df.columns
        assert "ocu500" in result.merged_df.columns
        assert "i524a1" in result.merged_df.columns

    def test_geographic_merge_workflow(self, sample_sumaria_data):
        """Test merging geographic UBIGEO data"""
        # Add ubigeo column to sumaria data
        sumaria_with_ubigeo = sample_sumaria_data.copy()
        sumaria_with_ubigeo["ubigeo"] = ["010101", "010102", "010103"]

        ubigeo_data = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102", "010103"],
                "NOMBDEP": ["Amazonas", "Amazonas", "Amazonas"],
                "NOMBPROV": ["Chachapoyas", "Chachapoyas", "Chachapoyas"],
                "NOMBDIST": ["Chachapoyas", "Asunción", "Balsas"],
            }
        )

        merger = GeographicMerger()
        result_df, report = merger.merge(sumaria_with_ubigeo, ubigeo_data, columna_union="ubigeo")

        # Check merge success
        assert result_df is not None
        assert len(result_df) == 3

        # Check geographic columns added
        assert "NOMBDEP" in result_df.columns
        assert "NOMBPROV" in result_df.columns
        assert "NOMBDIST" in result_df.columns

        # Check report
        assert report["output_rows"] == 3
        # Note: merge_type might not be in report, check what's available
        assert "output_rows" in report


class TestCompleteAnalysisWorkflow:
    """Test complete end-to-end analysis workflows"""

    @pytest.fixture
    def complete_household_dataset(self):
        """Create complete household dataset with multiple modules merged"""
        return pd.DataFrame(
            {
                "conglome": ["001", "002", "003", "004"],
                "vivienda": ["001", "001", "002", "002"],
                "hogar": ["01", "01", "01", "01"],
                "mieperho": [3, 4, 5, 2],
                "inghog2d": [2500.0, 3200.0, 1800.0, 1200.0],
                "gashog2d": [2000.0, 2800.0, 1500.0, 1000.0],
                "pobreza": [3, 3, 2, 1],
                "ubigeo": ["010101", "010102", "010103", "010104"],
                "NOMBDEP": ["Amazonas", "Amazonas", "Amazonas", "Amazonas"],
                "factor07": [100.0, 120.0, 90.0, 110.0],  # Expansion factor
            }
        )

    def test_poverty_analysis_workflow(self, complete_household_dataset):
        """Test complete poverty analysis workflow"""
        df = complete_household_dataset

        # Calculate poverty rate
        poverty_rate = (df["pobreza"] <= 2).mean()

        # Calculate weighted poverty rate
        weighted_poor = (df[df["pobreza"] <= 2]["factor07"]).sum()
        total_weighted = df["factor07"].sum()
        weighted_poverty_rate = weighted_poor / total_weighted

        # Basic assertions
        assert 0 <= poverty_rate <= 1
        assert 0 <= weighted_poverty_rate <= 1

        # Calculate average income by poverty status
        poor_income = df[df["pobreza"] <= 2]["inghog2d"].mean()
        non_poor_income = df[df["pobreza"] == 3]["inghog2d"].mean()

        assert poor_income < non_poor_income  # Poor should have lower income

    def test_geographic_inequality_workflow(self, complete_household_dataset):
        """Test geographic inequality analysis"""
        df = complete_household_dataset

        # Calculate income by department
        dept_income = df.groupby("NOMBDEP")["inghog2d"].agg(["mean", "median", "count"])

        assert len(dept_income) > 0
        assert "mean" in dept_income.columns
        assert "median" in dept_income.columns

        # Calculate Gini coefficient (simplified)
        incomes = df["inghog2d"].sort_values().values
        n = len(incomes)
        cumsum = incomes.cumsum()
        gini = (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n if cumsum[-1] > 0 else 0

        assert 0 <= gini <= 1  # Gini must be between 0 and 1

    def test_null_analysis_integration(self, complete_household_dataset):
        """Test null analysis integration in workflow"""
        df = complete_household_dataset.copy()

        # Introduce some missing values
        df.loc[0, "inghog2d"] = None
        df.loc[1, "gashog2d"] = None

        # Run null analysis (default config)
        analyzer = ENAHONullAnalyzer(verbose=False)
        result = analyzer.analyze(df)

        # Check analysis results
        assert result is not None
        assert "summary" in result
        # The summary has 'null_values' key, not 'total_nulls'
        assert result["summary"]["null_values"] == 2

    def test_panel_creation_workflow(self):
        """Test panel data creation workflow"""
        # Create two-period household data
        data_2022 = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["001", "001", "002"],
                "hogar": ["01", "01", "01"],
                "inghog2d": [2500.0, 3200.0, 1800.0],
            }
        )

        data_2023 = pd.DataFrame(
            {
                "conglome": ["001", "002", "003"],
                "vivienda": ["001", "001", "002"],
                "hogar": ["01", "01", "01"],
                "inghog2d": [2600.0, 3300.0, 1900.0],
            }
        )

        # Create panel
        creator = PanelCreator()
        result = creator.create_panel(
            {"2022": data_2022, "2023": data_2023},
            id_vars=["conglome", "vivienda", "hogar"],
            time_var="año",
        )

        # Analyze income growth
        panel_df = result.panel_df
        growth = (
            panel_df[panel_df["año"] == "2023"]["inghog2d"].values
            - panel_df[panel_df["año"] == "2022"]["inghog2d"].values
        )

        assert len(growth) == 3
        assert all(growth > 0)  # All households grew income


class TestMultiModuleMergeWorkflow:
    """Test merging multiple modules in sequence"""

    def test_three_module_merge_workflow(self):
        """Test merging three modules: household + sumaria + person aggregate"""
        import logging
        # Module 01: Household characteristics
        df_hogar = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["01", "01"],
                "mieperho": [3, 4],
            }
        )

        # Module 34: Sumaria (income/expenditure)
        df_sumaria = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["01", "01"],
                "inghog2d": [2500.0, 3200.0],
            }
        )

        # Aggregated person data
        df_person_agg = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "vivienda": ["001", "001"],
                "hogar": ["01", "01"],
                "n_ocupados": [2, 3],  # Number employed
                "edad_prom": [35.5, 28.3],  # Average age
            }
        )

        config = ModuleMergeConfig(merge_level=ModuleMergeLevel.HOGAR)
        logger = logging.getLogger('test_merger')
        merger = ENAHOModuleMerger(config, logger)

        # Merge step 1: hogar + sumaria
        result1 = merger.merge_modules(df_hogar, df_sumaria, "01", "34", config)

        # Merge step 2: result + person aggregate
        result2 = merger.merge_modules(
            result1.merged_df, df_person_agg, "merged", "person_agg", config
        )

        final_df = result2.merged_df

        # Check all data preserved
        assert len(final_df) == 2
        assert "mieperho" in final_df.columns
        assert "inghog2d" in final_df.columns
        assert "n_ocupados" in final_df.columns
        assert "edad_prom" in final_df.columns


class TestErrorHandlingInWorkflows:
    """Test error handling in realistic workflows"""

    def test_merge_with_missing_key_columns(self):
        """Test merge behavior when key columns are missing"""
        import logging
        df1 = pd.DataFrame({"conglome": ["001"], "value": [100]})

        df2 = pd.DataFrame({"different_key": ["001"], "value2": [200]})

        config = ModuleMergeConfig(merge_level=ModuleMergeLevel.HOGAR)
        logger = logging.getLogger('test_merger')
        merger = ENAHOModuleMerger(config, logger)

        # This should handle the error gracefully
        with pytest.raises(Exception):  # Expected to raise some exception
            merger.merge_modules(df1, df2, "01", "02", config)

    def test_geographic_merge_with_invalid_ubigeo(self):
        """Test geographic merge with invalid UBIGEO codes"""
        df_data = pd.DataFrame(
            {
                "conglome": ["001", "002"],
                "ubigeo": ["999999", "888888"],  # Invalid codes
                "value": [100, 200],
            }
        )

        df_ubigeo = pd.DataFrame(
            {
                "ubigeo": ["010101", "010102"],  # Valid codes that don't match
                "NOMBDEP": ["Amazonas", "Amazonas"],
            }
        )

        merger = GeographicMerger()
        result_df, report = merger.merge(df_data, df_ubigeo, columna_union="ubigeo")

        # Should still return result (with NaN for unmatched)
        assert result_df is not None
        assert len(result_df) == 2

        # Department names should be missing (NaN)
        assert result_df["NOMBDEP"].isna().all()


class TestPerformanceWorkflows:
    """Test workflows with larger datasets"""

    def test_large_person_dataset_merge(self):
        """Test merging larger person-level datasets"""
        import logging
        # Create 1000 person records
        n_persons = 1000

        df_persons = pd.DataFrame(
            {
                "conglome": ["001"] * n_persons,
                "vivienda": ["001"] * n_persons,
                "hogar": ["01"] * n_persons,
                "codperso": [f"{i:02d}" for i in range(1, n_persons + 1)],
                "edad": [i % 80 + 1 for i in range(n_persons)],
            }
        )

        df_employment = pd.DataFrame(
            {
                "conglome": ["001"] * n_persons,
                "vivienda": ["001"] * n_persons,
                "hogar": ["01"] * n_persons,
                "codperso": [f"{i:02d}" for i in range(1, n_persons + 1)],
                "ocu500": [1 if i % 2 == 0 else 2 for i in range(n_persons)],
            }
        )

        config = ModuleMergeConfig(merge_level=ModuleMergeLevel.PERSONA)
        logger = logging.getLogger('test_merger')
        merger = ENAHOModuleMerger(config, logger)

        result = merger.merge_modules(df_persons, df_employment, "02", "05", config)

        # Should handle large dataset efficiently
        assert len(result.merged_df) == n_persons
        assert "edad" in result.merged_df.columns
        assert "ocu500" in result.merged_df.columns
