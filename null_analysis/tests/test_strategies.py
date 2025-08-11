# null_analysis/tests/test_strategies.py
"""Tests para estrategias de análisis"""

import pytest
import pandas as pd
import numpy as np
from ..strategies.basic_analysis import BasicNullAnalysis
from ..strategies.advanced_analysis import AdvancedNullAnalysis
from ..config import NullAnalysisConfig


class TestStrategies:

    @pytest.fixture
    def config(self):
        return NullAnalysisConfig()

    @pytest.fixture
    def logger(self):
        import logging
        return logging.getLogger('test')

    def test_basic_strategy(self, config, logger):
        """Test estrategia básica"""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.nan, 2, 3]
        })

        strategy = BasicNullAnalysis(config, logger)
        result = strategy.analyze(df)

        assert result['analysis_type'] == 'basic'
        assert 'summary' in result
        assert len(result['summary']) == 2

    def test_recommendations(self, config, logger):
        """Test generación de recomendaciones"""
        df = pd.DataFrame({
            'a': [np.nan] * 10,  # 100% missing
            'b': [1] * 10  # 0% missing
        })

        strategy = BasicNullAnalysis(config, logger)
        result = strategy.analyze(df)
        recommendations = strategy.get_recommendations(result)

        assert len(recommendations) > 0
        assert any('50%' in rec for rec in recommendations)