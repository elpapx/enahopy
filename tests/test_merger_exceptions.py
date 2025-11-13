"""
Tests for enahopy.merger.exceptions module

Comprehensive coverage for all custom exception classes
"""

import pytest

from enahopy.merger.exceptions import (  # Geographic exceptions; Module merge exceptions; Data quality exceptions; Configuration exceptions; Base merger exception
    ConfigurationError,
    ConflictResolutionError,
    DataQualityError,
    DuplicateHandlingError,
    GeoMergeError,
    IncompatibleModulesError,
    MergeKeyError,
    MergerError,
    MergeValidationError,
    ModuleMergeError,
    ModuleValidationError,
    TerritorialInconsistencyError,
    UbigeoValidationError,
    ValidationThresholdError,
)


class TestGeographicExceptions:
    """Test geographic merge exceptions"""

    def test_geo_merge_error_basic(self):
        """Test GeoMergeError basic instantiation"""
        error = GeoMergeError("Test error message")

        assert str(error) == "Test error message"
        assert error.error_code is None
        assert error.context == {}

    def test_geo_merge_error_with_code(self):
        """Test GeoMergeError with error code"""
        error = GeoMergeError("Test error", error_code="GEO_001")

        assert error.error_code == "GEO_001"

    def test_geo_merge_error_with_context(self):
        """Test GeoMergeError with context"""
        error = GeoMergeError("Test error", error_code="GEO_001", file="test.csv", line=42)

        assert error.context["file"] == "test.csv"
        assert error.context["line"] == 42

    def test_geo_merge_error_raising(self):
        """Test that GeoMergeError can be raised and caught"""
        with pytest.raises(GeoMergeError, match="Geographic merge failed"):
            raise GeoMergeError("Geographic merge failed")

    def test_ubigeo_validation_error_basic(self):
        """Test UbigeoValidationError basic instantiation"""
        error = UbigeoValidationError("Invalid UBIGEO codes")

        assert str(error) == "Invalid UBIGEO codes"
        assert error.invalid_ubigeos == []

    def test_ubigeo_validation_error_with_codes(self):
        """Test UbigeoValidationError with invalid codes"""
        invalid_codes = ["ABC123", "999999", "000000"]
        error = UbigeoValidationError("Invalid codes", invalid_ubigeos=invalid_codes)

        assert error.invalid_ubigeos == invalid_codes
        assert len(error.invalid_ubigeos) == 3

    def test_ubigeo_validation_error_with_context(self):
        """Test UbigeoValidationError with context"""
        error = UbigeoValidationError(
            "Format errors", invalid_ubigeos=["ABC"], count=3, percentage=0.5
        )

        assert error.context["count"] == 3
        assert error.context["percentage"] == 0.5

    def test_territorial_inconsistency_error_basic(self):
        """Test TerritorialInconsistencyError basic instantiation"""
        error = TerritorialInconsistencyError("Territorial mismatch")

        assert str(error) == "Territorial mismatch"
        assert error.inconsistencies == []

    def test_territorial_inconsistency_error_with_data(self):
        """Test TerritorialInconsistencyError with inconsistencies"""
        inconsistencies = [
            {"ubigeo": "150101", "dept": "15", "expected_dept": "14"},
            {"ubigeo": "150102", "prov": "02", "expected_prov": "01"},
        ]
        error = TerritorialInconsistencyError(
            "Inconsistent territories", inconsistencies=inconsistencies, count=2
        )

        assert len(error.inconsistencies) == 2
        assert error.context["count"] == 2

    def test_duplicate_handling_error_basic(self):
        """Test DuplicateHandlingError basic instantiation"""
        error = DuplicateHandlingError("Cannot handle duplicates")

        assert str(error) == "Cannot handle duplicates"
        assert error.duplicates_info == {}

    def test_duplicate_handling_error_with_info(self):
        """Test DuplicateHandlingError with duplicate info"""
        dup_info = {
            "total_duplicates": 10,
            "affected_keys": ["150101", "150102"],
            "strategy": "KEEP_FIRST",
        }
        error = DuplicateHandlingError("Duplicate handling failed", duplicates_info=dup_info)

        assert error.duplicates_info["total_duplicates"] == 10
        assert len(error.duplicates_info["affected_keys"]) == 2


class TestModuleMergeExceptions:
    """Test module merge exceptions"""

    def test_module_merge_error_basic(self):
        """Test ModuleMergeError basic instantiation"""
        error = ModuleMergeError("Module merge failed")

        assert str(error) == "Module merge failed"
        assert error.modules_involved == []

    def test_module_merge_error_with_modules(self):
        """Test ModuleMergeError with modules"""
        modules = ["01", "02", "05"]
        error = ModuleMergeError("Merge failed", modules_involved=modules)

        assert error.modules_involved == modules

    def test_module_merge_error_with_context(self):
        """Test ModuleMergeError with context"""
        error = ModuleMergeError(
            "Merge failed", modules_involved=["01", "02"], merge_type="left", records_lost=100
        )

        assert error.context["merge_type"] == "left"
        assert error.context["records_lost"] == 100

    def test_module_validation_error_basic(self):
        """Test ModuleValidationError basic instantiation"""
        error = ModuleValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert error.module_code is None
        assert error.validation_failures == []

    def test_module_validation_error_with_failures(self):
        """Test ModuleValidationError with validation failures"""
        failures = ["Missing required column: conglome", "Invalid data type in column: vivienda"]
        error = ModuleValidationError(
            "Validation failed", module_code="01", validation_failures=failures
        )

        assert error.module_code == "01"
        assert len(error.validation_failures) == 2

    def test_incompatible_modules_error_basic(self):
        """Test IncompatibleModulesError basic instantiation"""
        error = IncompatibleModulesError("Modules incompatible")

        assert str(error) == "Modules incompatible"
        assert error.module1 is None
        assert error.module2 is None
        assert error.compatibility_issues == []

    def test_incompatible_modules_error_full(self):
        """Test IncompatibleModulesError with full details"""
        compat_issues = ["Different merge levels", "Module1 level: HOGAR", "Module2 level: PERSONA"]
        error = IncompatibleModulesError(
            "Cannot merge",
            module1="01",
            module2="02",
            compatibility_issues=compat_issues,
            suggestion="Use appropriate merge keys",
        )

        assert error.module1 == "01"
        assert error.module2 == "02"
        assert len(error.compatibility_issues) == 3
        assert "Different merge levels" in error.compatibility_issues
        assert error.context["suggestion"] == "Use appropriate merge keys"

    def test_merge_key_error_basic(self):
        """Test MergeKeyError basic instantiation"""
        error = MergeKeyError("Missing merge key")

        # MergeKeyError inherits from KeyError which adds quotes to string representation
        assert "Missing merge key" in str(error)

    def test_merge_key_error_is_key_error(self):
        """Test that MergeKeyError is also a KeyError"""
        error = MergeKeyError("Key not found")

        assert isinstance(error, ModuleMergeError)
        assert isinstance(error, KeyError)

    def test_merge_key_error_with_context(self):
        """Test MergeKeyError with context"""
        error = MergeKeyError(
            "Key missing",
            modules_involved=["01", "02"],
            missing_key="conglome",
            available_keys=["vivienda", "hogar"],
        )

        assert error.context["missing_key"] == "conglome"
        assert len(error.context["available_keys"]) == 2

    def test_conflict_resolution_error_basic(self):
        """Test ConflictResolutionError basic instantiation"""
        error = ConflictResolutionError("Cannot resolve conflict")

        assert str(error) == "Cannot resolve conflict"

    def test_conflict_resolution_error_with_details(self):
        """Test ConflictResolutionError with conflict details"""
        error = ConflictResolutionError(
            "Column conflict",
            modules_involved=["01", "05"],
            conflicting_column="p207",
            strategies_tried=["keep_left", "keep_right", "coalesce"],
        )

        assert error.context["conflicting_column"] == "p207"
        assert len(error.context["strategies_tried"]) == 3


class TestDataQualityExceptions:
    """Test data quality exceptions"""

    def test_data_quality_error_basic(self):
        """Test DataQualityError basic instantiation"""
        error = DataQualityError("Data quality issue detected")

        assert str(error) == "Data quality issue detected"

    def test_data_quality_error_with_metrics(self):
        """Test DataQualityError with quality metrics"""
        error = DataQualityError(
            "Low quality data", completeness=0.65, validity=0.80, threshold=0.90
        )

        assert error.context["completeness"] == 0.65
        assert error.context["validity"] == 0.80

    def test_validation_threshold_error_basic(self):
        """Test ValidationThresholdError basic instantiation"""
        error = ValidationThresholdError("Threshold exceeded")

        assert str(error) == "Threshold exceeded"

    def test_validation_threshold_error_with_values(self):
        """Test ValidationThresholdError with threshold values"""
        error = ValidationThresholdError(
            "Match rate too low", metric="match_rate", actual_value=0.75, threshold_value=0.95
        )

        assert error.context["metric"] == "match_rate"
        assert error.context["actual_value"] == 0.75
        assert error.context["threshold_value"] == 0.95


class TestConfigurationExceptions:
    """Test configuration exceptions"""

    def test_configuration_error_basic(self):
        """Test ConfigurationError basic instantiation"""
        error = ConfigurationError("Invalid configuration")

        assert str(error) == "Invalid configuration"

    def test_configuration_error_with_details(self):
        """Test ConfigurationError with configuration details"""
        error = ConfigurationError(
            "Invalid chunk_size", parameter="chunk_size", provided_value=0, valid_range="1-1000000"
        )

        assert error.context["parameter"] == "chunk_size"
        assert error.context["provided_value"] == 0

    def test_merge_validation_error_basic(self):
        """Test MergeValidationError basic instantiation"""
        error = MergeValidationError("Merge validation failed")

        assert str(error) == "Merge validation failed"

    def test_merge_validation_error_with_checks(self):
        """Test MergeValidationError with validation checks"""
        error = MergeValidationError(
            "Pre-merge validation failed",
            validation_type="pre-merge",
            failed_checks=["column_overlap", "data_types"],
            modules_involved=["01", "02"],
            passed_checks=["row_counts", "encoding"],
        )

        # Check attributes
        assert error.validation_type == "pre-merge"
        assert len(error.failed_checks) == 2
        assert error.context.get("modules_involved") == ["01", "02"]
        assert error.context.get("passed_checks") == ["row_counts", "encoding"]


class TestBaseMergerException:
    """Test base MergerError exception"""

    def test_merger_error_basic(self):
        """Test MergerError basic instantiation"""
        error = MergerError("Generic merger error")

        assert str(error) == "Generic merger error"

    def test_merger_error_is_exception(self):
        """Test that MergerError is an Exception"""
        error = MergerError("Test")

        assert isinstance(error, Exception)

    def test_merger_error_raising(self):
        """Test that MergerError can be raised"""
        with pytest.raises(MergerError):
            raise MergerError("Something went wrong")


class TestExceptionHierarchy:
    """Test exception inheritance and hierarchy"""

    def test_geo_exceptions_inherit_from_enaho_error(self):
        """Test that geographic exceptions inherit from ENAHOError"""
        from enahopy.merger.exceptions import ENAHOError

        assert issubclass(GeoMergeError, ENAHOError)
        assert issubclass(TerritorialInconsistencyError, GeoMergeError)
        assert issubclass(DuplicateHandlingError, GeoMergeError)

    def test_validation_exceptions_hierarchy(self):
        """Test validation exception hierarchy"""
        from enahopy.merger.exceptions import ENAHOValidationError

        assert issubclass(UbigeoValidationError, ENAHOValidationError)
        assert issubclass(ModuleValidationError, ENAHOValidationError)
        assert issubclass(ValidationThresholdError, ENAHOValidationError)

    def test_module_exceptions_hierarchy(self):
        """Test module exception hierarchy"""
        from enahopy.merger.exceptions import ENAHOError

        assert issubclass(ModuleMergeError, ENAHOError)
        assert issubclass(IncompatibleModulesError, ModuleMergeError)
        assert issubclass(ConflictResolutionError, ModuleMergeError)
        assert issubclass(MergeValidationError, ModuleMergeError)

    def test_merge_key_error_multiple_inheritance(self):
        """Test that MergeKeyError inherits from both ModuleMergeError and KeyError"""
        assert issubclass(MergeKeyError, ModuleMergeError)
        assert issubclass(MergeKeyError, KeyError)

    def test_catching_base_exception(self):
        """Test that derived exceptions can be caught by base exception"""
        from enahopy.merger.exceptions import ENAHOError

        with pytest.raises(ENAHOError):
            raise GeoMergeError("Test error")

        with pytest.raises(ENAHOError):
            raise ModuleMergeError("Test error")

        with pytest.raises(ENAHOError):
            raise DataQualityError("Test error")


class TestExceptionUsagePatterns:
    """Test common exception usage patterns"""

    def test_exception_chaining(self):
        """Test exception chaining with from clause"""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise GeoMergeError("Wrapped error", original_error=str(e)) from e
        except GeoMergeError as ge:
            assert "Wrapped error" in str(ge)
            assert ge.__cause__ is not None

    def test_exception_context_dict_access(self):
        """Test accessing context as dictionary"""
        error = DataQualityError("Quality issue", metric1=10, metric2=20, metric3=30)

        assert "metric1" in error.context
        assert error.context.get("metric1") == 10
        assert list(error.context.keys()) == ["metric1", "metric2", "metric3"]

    def test_exception_with_empty_lists(self):
        """Test that empty lists are handled correctly"""
        error = UbigeoValidationError("Test", invalid_ubigeos=[])
        assert error.invalid_ubigeos == []

        error2 = ModuleValidationError("Test", validation_failures=[])
        assert error2.validation_failures == []

    def test_exception_with_none_values(self):
        """Test that None values are handled correctly"""
        error = IncompatibleModulesError("Test", module1=None, module2=None)
        assert error.module1 is None
        assert error.module2 is None


class TestExceptionUtilityFunctions:
    """Test utility functions for exception reporting and diagnostics"""

    def test_format_exception_details_basic(self):
        """Test format_exception_details with basic exception"""
        from enahopy.merger.exceptions import GeoMergeError, format_exception_details

        error = GeoMergeError("Test error", error_code="GEO001", key1="value1")
        details = format_exception_details(error)

        assert details["exception_type"] == "GeoMergeError"
        assert details["message"] == "Test error"
        assert details["error_code"] == "GEO001"
        assert "context" in details
        assert details["context"]["key1"] == "value1"

    def test_format_exception_details_ubigeo_validation(self):
        """Test format_exception_details with UbigeoValidationError"""
        from enahopy.merger.exceptions import UbigeoValidationError, format_exception_details

        error = UbigeoValidationError(
            "Invalid UBIGEOs", invalid_ubigeos=["999999", "888888"], source="validation"
        )
        details = format_exception_details(error)

        assert details["exception_type"] == "UbigeoValidationError"
        # invalid_ubigeos is stored as an attribute, check the error object itself
        assert error.invalid_ubigeos == ["999999", "888888"]
        assert details["context"]["source"] == "validation"

    def test_format_exception_details_module_validation(self):
        """Test format_exception_details with ModuleValidationError"""
        from enahopy.merger.exceptions import ModuleValidationError, format_exception_details

        error = ModuleValidationError(
            "Validation failed",
            module_code="05",
            validation_failures=["missing_column", "wrong_dtype"],
        )
        details = format_exception_details(error)

        assert details["exception_type"] == "ModuleValidationError"
        # module_code is stored as an attribute, check the error object itself
        assert error.module_code == "05"
        assert error.validation_failures == ["missing_column", "wrong_dtype"]

    def test_format_exception_details_incompatible_modules(self):
        """Test format_exception_details with IncompatibleModulesError"""
        from enahopy.merger.exceptions import IncompatibleModulesError, format_exception_details

        error = IncompatibleModulesError(
            "Modules incompatible",
            module1="01",
            module2="05",
            compatibility_issues=["different years"],
        )
        details = format_exception_details(error)

        assert details["exception_type"] == "IncompatibleModulesError"
        # module1/module2 are stored as attributes, check the error object itself
        assert error.module1 == "01"
        assert error.module2 == "05"
        assert error.compatibility_issues == ["different years"]

    def test_format_exception_details_validation_threshold(self):
        """Test format_exception_details with ValidationThresholdError"""
        from enahopy.merger.exceptions import ValidationThresholdError, format_exception_details

        error = ValidationThresholdError(
            "Threshold exceeded", threshold_type="quality", expected=0.8, actual=0.6
        )
        details = format_exception_details(error)

        assert details["exception_type"] == "ValidationThresholdError"
        # threshold_type, expected, actual are in context
        assert details["context"]["threshold_type"] == "quality"
        assert details["context"]["expected"] == 0.8
        assert details["context"]["actual"] == 0.6

    def test_create_merge_error_report_basic(self):
        """Test create_merge_error_report with basic exception"""
        from enahopy.merger.exceptions import GeoMergeError, create_merge_error_report

        error = GeoMergeError("Merge failed")
        report = create_merge_error_report(error)

        assert "ENAHOPY ERROR REPORT" in report
        assert "GeoMergeError" in report
        assert "Merge failed" in report

    def test_create_merge_error_report_with_operation_context(self):
        """Test create_merge_error_report with operation context"""
        from enahopy.merger.exceptions import DataQualityError, create_merge_error_report

        error = DataQualityError("Low quality data")
        context = {"module": "05", "operation": "geographic_merge", "records_affected": 150}
        report = create_merge_error_report(error, operation_context=context)

        assert "Operation Context:" in report
        assert "module: 05" in report
        assert "operation: geographic_merge" in report
        assert "records_affected: 150" in report

    def test_create_merge_error_report_with_exception_context(self):
        """Test create_merge_error_report with exception context"""
        from enahopy.merger.exceptions import DuplicateHandlingError, create_merge_error_report

        error = DuplicateHandlingError(
            "Duplicates found", duplicate_count=25, strategy_used="BEST_QUALITY"
        )
        report = create_merge_error_report(error)

        assert "Exception Context:" in report
        # duplicate_count is a special attribute, not in context
        # Only strategy_used appears in the exception context dict
        assert "strategy_used: BEST_QUALITY" in report

    def test_create_merge_error_report_with_recommendations_ubigeo(self):
        """Test create_merge_error_report includes recommendations for UbigeoValidationError"""
        from enahopy.merger.exceptions import UbigeoValidationError, create_merge_error_report

        error = UbigeoValidationError("Invalid UBIGEO format")
        report = create_merge_error_report(error)

        assert "Recomendaciones:" in report
        assert "Verificar formato de códigos UBIGEO" in report
        assert "6 dígitos" in report

    def test_create_merge_error_report_with_recommendations_incompatible(self):
        """Test create_merge_error_report includes recommendations for IncompatibleModulesError"""
        from enahopy.merger.exceptions import IncompatibleModulesError, create_merge_error_report

        error = IncompatibleModulesError("Modules cannot be merged", module1="01", module2="05")
        report = create_merge_error_report(error)

        assert "Recomendaciones:" in report
        assert "Verificar que los módulos sean del mismo año" in report

    def test_create_merge_error_report_with_recommendations_duplicate(self):
        """Test create_merge_error_report includes recommendations for DuplicateHandlingError"""
        from enahopy.merger.exceptions import DuplicateHandlingError, create_merge_error_report

        error = DuplicateHandlingError("Cannot handle duplicates")
        report = create_merge_error_report(error)

        assert "Recomendaciones:" in report
        assert "estrategia de manejo de duplicados" in report
        assert "BEST_QUALITY" in report or "AGGREGATE" in report

    def test_create_merge_error_report_with_recommendations_data_quality(self):
        """Test create_merge_error_report includes recommendations for DataQualityError"""
        from enahopy.merger.exceptions import DataQualityError, create_merge_error_report

        error = DataQualityError("Data quality issues detected")
        report = create_merge_error_report(error)

        assert "Recomendaciones:" in report
        assert "completitud" in report or "consistencia" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
