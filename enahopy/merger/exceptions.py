"""
ENAHO Merger - Exception Compatibility Layer
==========================================

This module imports exceptions from the unified hierarchy for backward compatibility.
All new code should import directly from enahopy.exceptions.

DEPRECATED: Import from enahopy.exceptions instead.
"""

import warnings

# Import from unified exception hierarchy
from enahopy.exceptions import (
    DataQualityError,
    DuplicateHandlingError,
    ENAHOConfigError,
    ENAHOError,
    ENAHOMergeError,
    ENAHOValidationError,
    GeoMergeError,
    IncompatibleModulesError,
    MergeKeyError,
    ModuleMergeError,
    TerritorialInconsistencyError,
    UbigeoValidationError,
)

# Deprecated warnings for backward compatibility
warnings.warn(
    "Importing from enahopy.merger.exceptions is deprecated. "
    "Import from enahopy.exceptions instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Backward compatibility wrapper classes with old interfaces
class MergerError(ENAHOMergeError):
    """Base merger exception - alias for ENAHOMergeError"""


class ConfigurationError(ENAHOConfigError):
    """Configuration error - alias for ENAHOConfigError"""


class ModuleValidationError(ENAHOValidationError):
    """Module validation error with backward-compatible attributes"""

    def __init__(self, message: str, module_code=None, validation_failures=None, **kwargs):
        super().__init__(message, validation_failures=validation_failures, **kwargs)
        self.module_code = module_code


class MergeValidationError(ModuleMergeError):
    """Merge validation error with backward-compatible attributes"""

    def __init__(
        self,
        message: str,
        merge_type=None,
        validation_type=None,
        failed_checks=None,
        validation_details=None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.merge_type = merge_type
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []
        self.validation_details = validation_details or {}


class ValidationThresholdError(ENAHOValidationError):
    """Validation threshold error - alias for ENAHOValidationError"""


class ConflictResolutionError(ModuleMergeError):
    """Conflict resolution error - alias for ModuleMergeError"""


# Import utility functions from unified exception hierarchy
from enahopy.exceptions import (
    create_error_report,
    format_exception_for_logging,
    get_error_recommendations,
)

# Backward compatibility aliases for utility functions
format_exception_details = format_exception_for_logging
create_merge_error_report = create_error_report

# Export all for backward compatibility
__all__ = [
    "ENAHOError",
    "ENAHOValidationError",
    "ENAHOConfigError",
    "DataQualityError",
    "GeoMergeError",
    "UbigeoValidationError",
    "TerritorialInconsistencyError",
    "DuplicateHandlingError",
    "ENAHOMergeError",
    "ModuleMergeError",
    "IncompatibleModulesError",
    "MergeKeyError",
    # Aliases
    "MergerError",
    "ConfigurationError",
    "ModuleValidationError",
    "MergeValidationError",
    "ValidationThresholdError",
    "ConflictResolutionError",
    # Utility functions
    "format_exception_details",
    "create_merge_error_report",
    "get_error_recommendations",
]
