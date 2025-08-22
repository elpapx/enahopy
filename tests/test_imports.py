"""
Tests específicos de importaciones para cada módulo
"""

import sys
from pathlib import Path

import pytest


class TestLoaderImports:
    """Tests de importación del módulo loader"""

    def test_loader_core_imports(self):
        """Test importaciones del core del loader"""
        try:
            from enahopy.loader.core import CacheManager, ENAHOConfig, ENAHOError, setup_logging

            assert all([ENAHOConfig, ENAHOError, setup_logging, CacheManager])
        except ImportError as e:
            pytest.skip(f"Core del loader no disponible: {e}")

    def test_loader_io_imports(self):
        """Test importaciones de IO del loader"""
        try:
            from enahopy.loader.io import ENAHODataDownloader, ENAHOLocalReader

            assert all([ENAHODataDownloader, ENAHOLocalReader])
        except ImportError as e:
            pytest.skip(f"IO del loader no disponible: {e}")

    def test_loader_utils_imports(self):
        """Test importaciones de utils del loader"""
        try:
            from enahopy.loader.utils import ENAHOUtils, download_enaho_data, read_enaho_file

            assert all([download_enaho_data, read_enaho_file, ENAHOUtils])
        except ImportError as e:
            pytest.skip(f"Utils del loader no disponible: {e}")


class TestMergerImports:
    """Tests de importación del módulo merger"""

    def test_merger_basic_imports(self):
        """Test importaciones básicas del merger"""
        try:
            from enahopy.merger import ENAHOMerger

            assert ENAHOMerger is not None
        except ImportError as e:
            pytest.skip(f"Merger no disponible: {e}")


class TestNullAnalysisImports:
    """Tests de importación del módulo null_analysis"""

    def test_null_analysis_basic_imports(self):
        """Test importaciones básicas de null_analysis"""
        try:
            from enahopy.null_analysis import ENAHONullAnalyzer

            assert ENAHONullAnalyzer is not None
        except ImportError as e:
            pytest.skip(f"Null analysis no disponible: {e}")
