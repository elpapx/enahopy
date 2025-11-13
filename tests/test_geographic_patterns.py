"""
Tests for Geographic Pattern Detection Module

Tests coverage for enahopy/merger/geographic/patterns.py
"""

import logging
import unittest

import numpy as np
import pandas as pd

from enahopy.merger.config import NivelTerritorial
from enahopy.merger.geographic.patterns import GeoPatternDetector


class TestGeoPatternDetectorCore(unittest.TestCase):
    """Test core geographic pattern detection functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_geo_patterns")
        self.detector = GeoPatternDetector(self.logger)

    def test_init(self):
        """Test GeoPatternDetector initialization"""
        assert self.detector.logger == self.logger
        assert isinstance(self.detector._detection_cache, dict)
        assert len(self.detector._detection_cache) == 0

    def test_detectar_columnas_geograficas_basic(self):
        """Test basic geographic column detection"""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "150103"],
                "departamento": ["LIMA", "LIMA", "LIMA"],
                "provincia": ["LIMA", "LIMA", "LIMA"],
                "other_column": [1, 2, 3],
            }
        )

        result = self.detector.detectar_columnas_geograficas(df)

        assert isinstance(result, dict)
        assert "ubigeo" in result
        assert "departamento" in result
        assert result["ubigeo"] == "ubigeo"
        assert result["departamento"] == "departamento"

    def test_detectar_columnas_geograficas_with_cache(self):
        """Test that detection uses cache on second call"""
        df = pd.DataFrame({"ubigeo": ["150101", "150102"], "dep": ["15", "15"]})

        # First call - should detect and cache
        result1 = self.detector.detectar_columnas_geograficas(df, confianza_minima=0.8)

        # Second call with same df and confidence - should use cache
        result2 = self.detector.detectar_columnas_geograficas(df, confianza_minima=0.8)

        assert result1 == result2
        # Cache should have one entry
        assert len(self.detector._detection_cache) == 1

    def test_detectar_columnas_geograficas_different_confidence(self):
        """Test detection with different confidence thresholds"""
        df = pd.DataFrame(
            {"ub": ["150101", "150102"], "dep": ["15", "15"]}  # Shorter name, lower confidence
        )

        # High confidence - may not find short names
        result_high = self.detector.detectar_columnas_geograficas(df, confianza_minima=0.9)

        # Lower confidence - should find more
        result_low = self.detector.detectar_columnas_geograficas(df, confianza_minima=0.6)

        # Lower confidence should find same or more columns
        assert len(result_low) >= len(result_high)

    def test_detectar_columnas_geograficas_empty_dataframe(self):
        """Test detection with empty DataFrame"""
        df = pd.DataFrame()

        result = self.detector.detectar_columnas_geograficas(df)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_detectar_columnas_geograficas_no_geo_columns(self):
        """Test detection when no geographic columns exist"""
        df = pd.DataFrame(
            {"column_a": [1, 2, 3], "column_b": ["x", "y", "z"], "value": [10.5, 20.3, 30.1]}
        )

        result = self.detector.detectar_columnas_geograficas(df)

        assert isinstance(result, dict)
        # Should find nothing or very few matches
        assert len(result) <= 1  # Maybe false positives, but unlikely

    def test_detectar_columnas_geograficas_case_insensitive(self):
        """Test that detection is case-insensitive"""
        df = pd.DataFrame(
            {
                "UBIGEO": ["150101", "150102"],
                "Departamento": ["LIMA", "CUSCO"],
                "PROVINCIA": ["LIMA", "CUSCO"],
            }
        )

        result = self.detector.detectar_columnas_geograficas(df)

        assert "ubigeo" in result
        assert "departamento" in result


class TestSimilarityScoring(unittest.TestCase):
    """Test similarity scoring methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_similarity")
        self.detector = GeoPatternDetector(self.logger)

    def test_calculate_similarity_score_exact_match(self):
        """Test similarity score for exact column name match"""
        df = pd.DataFrame({"ubigeo": ["150101", "150102", "150103"]})

        score = self.detector._calculate_similarity_score("ubigeo", "ubigeo", df, "ubigeo")

        # Exact match should have high score
        assert score > 0.9
        assert score <= 1.0

    def test_calculate_similarity_score_partial_match(self):
        """Test similarity score for partial match"""
        df = pd.DataFrame({"ubigeo_col": ["150101", "150102"]})

        score = self.detector._calculate_similarity_score("ubigeo", "ubigeo_col", df, "ubigeo_col")

        # Partial match should have good score
        assert score > 0.7
        assert score < 1.0

    def test_calculate_similarity_score_no_match(self):
        """Test similarity score when no match"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        score = self.detector._calculate_similarity_score(
            "ubigeo", "other_column", df, "other_column"
        )

        # No match should have zero or very low score
        assert score == 0.0

    def test_fuzzy_match_with_underscores(self):
        """Test fuzzy matching with underscores"""
        # Should match despite underscores
        assert self.detector._fuzzy_match("ubigeo", "ubi_geo")
        assert self.detector._fuzzy_match("departamento", "depar_tamento")

    def test_fuzzy_match_with_hyphens(self):
        """Test fuzzy matching with hyphens"""
        # Should match despite hyphens
        assert self.detector._fuzzy_match("ubigeo", "ubi-geo")

    def test_fuzzy_match_no_match(self):
        """Test fuzzy matching when strings don't match"""
        assert not self.detector._fuzzy_match("ubigeo", "provincia")
        assert not self.detector._fuzzy_match("departamento", "distrito")


class TestContentValidation(unittest.TestCase):
    """Test content validation methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_content")
        self.detector = GeoPatternDetector(self.logger)

    def test_analyze_column_content_empty_series(self):
        """Test content analysis with empty series"""
        serie = pd.Series([], dtype=object)

        score = self.detector._analyze_column_content(serie, "ubigeo")

        assert score == 0.0

    def test_analyze_column_content_all_null(self):
        """Test content analysis with all null values"""
        serie = pd.Series([None, np.nan, pd.NA])

        score = self.detector._analyze_column_content(serie, "ubigeo")

        assert score == 0.0

    def test_validate_ubigeo_content_valid(self):
        """Test UBIGEO content validation with valid codes"""
        muestra = pd.Series(["150101", "150102", "080101", "010101"])

        score = self.detector._validate_ubigeo_content(muestra)

        # Valid UBIGEOs should have high score
        assert score > 0.7
        assert score <= 1.0

    def test_validate_ubigeo_content_invalid(self):
        """Test UBIGEO content validation with invalid codes"""
        muestra = pd.Series(["abc", "xyz", "123", "999"])

        score = self.detector._validate_ubigeo_content(muestra)

        # Invalid UBIGEOs should have low score
        assert score < 0.5

    def test_validate_ubigeo_content_mixed(self):
        """Test UBIGEO content validation with mixed valid/invalid"""
        muestra = pd.Series(["150101", "invalid", "080101", "abc", "010101"])

        score = self.detector._validate_ubigeo_content(muestra)

        # Mixed should have medium score
        assert 0.3 < score < 0.9

    def test_validate_territorial_content_departamento_codes(self):
        """Test territorial content validation for department codes"""
        muestra = pd.Series(["15", "08", "01", "25"])

        score = self.detector._validate_territorial_content(muestra, "departamento")

        # Valid department codes should score well
        assert score > 0.5

    def test_validate_territorial_content_departamento_names(self):
        """Test territorial content validation for department names"""
        muestra = pd.Series(["LIMA", "CUSCO", "AREQUIPA", "LORETO"])

        score = self.detector._validate_territorial_content(muestra, "departamento")

        # Valid department names should score high
        assert score > 0.6

    def test_validate_territorial_content_provincia(self):
        """Test territorial content validation for provincia"""
        muestra = pd.Series(["01", "02", "10", "1501"])

        score = self.detector._validate_territorial_content(muestra, "provincia")

        # Valid provincia codes should score well
        assert score > 0.4

    def test_validate_territorial_content_distrito(self):
        """Test territorial content validation for distrito"""
        muestra = pd.Series(["01", "02", "150101", "080101"])

        score = self.detector._validate_territorial_content(muestra, "distrito")

        # Valid distrito codes should score well
        assert score > 0.4

    def test_validate_conglomerado_content_valid(self):
        """Test conglomerado content validation with valid format"""
        muestra = pd.Series(["15010100001", "08010100001", "25010100001"])

        score = self.detector._validate_conglomerado_content(muestra)

        # Valid conglomerado should have high score
        assert score > 0.7

    def test_validate_conglomerado_content_invalid(self):
        """Test conglomerado content validation with invalid format"""
        muestra = pd.Series(["abc", "123", "short"])

        score = self.detector._validate_conglomerado_content(muestra)

        # Invalid conglomerado should have low score
        assert score < 0.5

    def test_validate_coordinate_content_valid_x(self):
        """Test coordinate content validation for valid X coordinates"""
        muestra = pd.Series([-77.0428, -77.0282, -77.0454])  # Lima area

        score = self.detector._validate_coordinate_content(muestra, "coordenada_x")

        # Valid Peru longitude should score high
        assert score > 0.8

    def test_validate_coordinate_content_valid_y(self):
        """Test coordinate content validation for valid Y coordinates"""
        muestra = pd.Series([-12.0464, -12.0262, -12.0532])  # Lima area

        score = self.detector._validate_coordinate_content(muestra, "coordenada_y")

        # Valid Peru latitude should score high
        assert score > 0.8

    def test_validate_coordinate_content_invalid_x(self):
        """Test coordinate content validation for invalid X coordinates"""
        muestra = pd.Series([100.0, 200.0, -150.0])  # Outside Peru

        score = self.detector._validate_coordinate_content(muestra, "coordenada_x")

        # Invalid longitude gets neutral score (0.5)
        assert score == 0.5

    def test_validate_coordinate_content_invalid_y(self):
        """Test coordinate content validation for invalid Y coordinates"""
        muestra = pd.Series([50.0, -90.0, 80.0])  # Outside Peru

        score = self.detector._validate_coordinate_content(muestra, "coordenada_y")

        # Invalid latitude gets neutral score (0.5)
        assert score == 0.5


class TestTerritorialLevelSuggestion(unittest.TestCase):
    """Test territorial level suggestion"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_territorial")
        self.detector = GeoPatternDetector(self.logger)

    def test_sugerir_nivel_territorial_ubigeo(self):
        """Test suggest territorial level with UBIGEO"""
        columnas = {"ubigeo": "ubigeo_col"}

        nivel = self.detector.sugerir_nivel_territorial(columnas)

        assert nivel == NivelTerritorial.DISTRITO

    def test_sugerir_nivel_territorial_distrito(self):
        """Test suggest territorial level with distrito"""
        columnas = {"distrito": "distrito_col"}

        nivel = self.detector.sugerir_nivel_territorial(columnas)

        assert nivel == NivelTerritorial.DISTRITO

    def test_sugerir_nivel_territorial_provincia(self):
        """Test suggest territorial level with provincia"""
        columnas = {"provincia": "prov_col", "departamento": "dep_col"}

        nivel = self.detector.sugerir_nivel_territorial(columnas)

        assert nivel == NivelTerritorial.PROVINCIA

    def test_sugerir_nivel_territorial_departamento(self):
        """Test suggest territorial level with only departamento"""
        columnas = {"departamento": "dep_col"}

        nivel = self.detector.sugerir_nivel_territorial(columnas)

        assert nivel == NivelTerritorial.DEPARTAMENTO

    def test_sugerir_nivel_territorial_empty(self):
        """Test suggest territorial level with no columns"""
        columnas = {}

        nivel = self.detector.sugerir_nivel_territorial(columnas)

        # Default is DISTRITO when no columns detected
        assert nivel == NivelTerritorial.DISTRITO


class TestGeographicCompleteness(unittest.TestCase):
    """Test geographic completeness analysis"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_completeness")
        self.detector = GeoPatternDetector(self.logger)

    def test_analyze_geographic_completeness_complete(self):
        """Test completeness analysis with all geographic columns"""
        df = pd.DataFrame(
            {
                "ubigeo": ["150101", "150102", "080101"],
                "departamento": ["LIMA", "LIMA", "CUSCO"],
                "provincia": ["LIMA", "LIMA", "CUSCO"],
                "distrito": ["LIMA", "ANCON", "CUSCO"],
            }
        )

        columnas = {
            "ubigeo": "ubigeo",
            "departamento": "departamento",
            "provincia": "provincia",
            "distrito": "distrito",
        }

        result = self.detector.analyze_geographic_completeness(df, columnas)

        assert isinstance(result, dict)
        # Returns dict with percentages per column type
        assert "ubigeo" in result or "departamento" in result
        assert len(result) > 0

    def test_analyze_geographic_completeness_partial(self):
        """Test completeness analysis with some missing columns"""
        df = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["LIMA", "LIMA"]})

        columnas = {"ubigeo": "ubigeo", "departamento": "departamento"}

        result = self.detector.analyze_geographic_completeness(df, columnas)

        assert isinstance(result, dict)
        # Returns dict with percentages per column type
        assert "ubigeo" in result
        assert "departamento" in result

    def test_analyze_geographic_completeness_empty(self):
        """Test completeness analysis with no columns"""
        df = pd.DataFrame({"other": [1, 2, 3]})

        columnas = {}

        result = self.detector.analyze_geographic_completeness(df, columnas)

        assert isinstance(result, dict)
        # Empty dict when no columns
        assert len(result) == 0


class TestDetectPatternsInData(unittest.TestCase):
    """Test detect_geographic_patterns_in_data method"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_patterns")
        self.detector = GeoPatternDetector(self.logger)

    def test_detect_patterns_with_ubigeo(self):
        """Test pattern detection with UBIGEO column"""
        df = pd.DataFrame({"ubigeo": ["150101", "150102", "080101", "080102", "150103"]})

        columnas = {"ubigeo": "ubigeo"}

        result = self.detector.detect_geographic_patterns_in_data(df, columnas)

        assert isinstance(result, dict)
        assert "distribucion_territorial" in result
        assert "concentracion_geografica" in result

    def test_detect_patterns_with_departamento(self):
        """Test pattern detection with departamento column"""
        df = pd.DataFrame({"departamento": ["LIMA", "LIMA", "CUSCO", "CUSCO", "AREQUIPA"]})

        columnas = {"departamento": "departamento"}

        result = self.detector.detect_geographic_patterns_in_data(df, columnas)

        assert isinstance(result, dict)
        assert "distribucion_territorial" in result

    def test_detect_patterns_empty_columns(self):
        """Test pattern detection with no geographic columns"""
        df = pd.DataFrame({"other": [1, 2, 3]})

        columnas = {}

        result = self.detector.detect_geographic_patterns_in_data(df, columnas)

        assert isinstance(result, dict)
        # Should still return structure with empty patterns
        assert "distribucion_territorial" in result


class TestSuggestMergeStrategy(unittest.TestCase):
    """Test suggest_merge_strategy method"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_strategy")
        self.detector = GeoPatternDetector(self.logger)

    def test_suggest_strategy_basic(self):
        """Test basic merge strategy suggestion"""
        df1 = pd.DataFrame({"ubigeo": ["150101", "150102"], "data1": [10, 20]})

        df2 = pd.DataFrame({"ubigeo": ["150101", "150102"], "data2": [30, 40]})

        columnas1 = {"ubigeo": "ubigeo"}
        columnas2 = {"ubigeo": "ubigeo"}

        result = self.detector.suggest_merge_strategy(df1, df2, columnas1, columnas2)

        assert isinstance(result, dict)
        # Returns columna_union_sugerida, nivel_recomendado, etc.
        assert "columna_union_sugerida" in result or "nivel_recomendado" in result

    def test_suggest_strategy_different_levels(self):
        """Test strategy suggestion with different territorial levels"""
        df1 = pd.DataFrame({"ubigeo": ["150101", "150102"], "data1": [10, 20]})

        df2 = pd.DataFrame({"departamento": ["15", "15"], "data2": [30, 40]})

        columnas1 = {"ubigeo": "ubigeo"}
        columnas2 = {"departamento": "departamento"}

        result = self.detector.suggest_merge_strategy(df1, df2, columnas1, columnas2)

        assert isinstance(result, dict)


class TestGenerateDetectionReport(unittest.TestCase):
    """Test generate_detection_report method"""

    def setUp(self):
        """Set up test fixtures"""
        self.logger = logging.getLogger("test_report")
        self.detector = GeoPatternDetector(self.logger)

    def test_generate_report_with_detections(self):
        """Test report generation with detected columns"""
        df = pd.DataFrame({"ubigeo": ["150101", "150102"], "departamento": ["LIMA", "LIMA"]})

        columnas_detectadas = {"ubigeo": "ubigeo", "departamento": "departamento"}

        report = self.detector.generate_detection_report(df, columnas_detectadas)

        assert isinstance(report, str)
        assert len(report) > 0
        # Should mention detected columns
        assert "ubigeo" in report.lower() or "geográf" in report.lower()

    def test_generate_report_empty_detections(self):
        """Test report generation with no detections"""
        df = pd.DataFrame({"other": [1, 2, 3]})

        columnas_detectadas = {}

        report = self.detector.generate_detection_report(df, columnas_detectadas)

        assert isinstance(report, str)
        assert len(report) > 0


if __name__ == "__main__":
    unittest.main()
