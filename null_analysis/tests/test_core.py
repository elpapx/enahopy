# null_analysis/tests/test_core.py
"""Tests para el analizador principal"""

import pytest
import pandas as pd
import numpy as np
from ..core.analyzer import ENAHONullAnalyzer
from ..config import NullAnalysisConfig, AnalysisComplexity


class TestENAHONullAnalyzer:

    @pytest.fixture
    def sample_df(self):
        """DataFrame de prueba con patrones de nulos"""
        np.random.seed(42)
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, 2, 3, np.nan, 5],
            'col3': [1, 2, 3, 4, 5],
            'col4': [np.nan, np.nan, np.nan, 4, 5]
        })
        return df

    def test_basic_analysis(self, sample_df):
        """Test análisis básico"""
        config = NullAnalysisConfig(complexity_level=AnalysisComplexity.BASIC)
        analyzer = ENAHONullAnalyzer(config=config, verbose=False)

        result = analyzer.analyze_null_patterns(sample_df)

        assert result['analysis_type'] == 'basic'
        assert 'metrics' in result
        assert result['metrics'].missing_percentage > 0

    def test_advanced_analysis(self, sample_df):
        """Test análisis avanzado"""
        config = NullAnalysisConfig(complexity_level=AnalysisComplexity.ADVANCED)
        analyzer = ENAHONullAnalyzer(config=config, verbose=False)

        result = analyzer.analyze_null_patterns(sample_df)

        assert result['analysis_type'] == 'advanced'
        assert 'patterns' in result
        assert 'correlations' in result

    def test_quality_score(self, sample_df):
        """Test cálculo de score de calidad"""
        analyzer = ENAHONullAnalyzer(verbose=False)
        score = analyzer.get_data_quality_score(sample_df)

        assert 0 <= score <= 100
        assert isinstance(score, float)

    def test_empty_dataframe(self):
        """Test con DataFrame vacío"""
        from ..exceptions import NullAnalysisError

        analyzer = ENAHONullAnalyzer(verbose=False)
        empty_df = pd.DataFrame()

        with pytest.raises(NullAnalysisError):
            analyzer.analyze_null_patterns(empty_df)