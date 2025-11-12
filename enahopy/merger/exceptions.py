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
    ENAHOError,
    ENAHOValidationError,
    ENAHOConfigError,
    DataQualityError,
    GeoMergeError,
    UbigeoValidationError,
    TerritorialInconsistencyError,
    DuplicateHandlingError,
    ENAHOMergeError,
    ModuleMergeError,
    IncompatibleModulesError,
    MergeKeyError,
)

# Deprecated warnings for backward compatibility
warnings.warn(
    "Importing from enahopy.merger.exceptions is deprecated. "
    "Import from enahopy.exceptions instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Aliases for backward compatibility
MergerError = ENAHOMergeError  # Base merger exception
ConfigurationError = ENAHOConfigError
ModuleValidationError = ENAHOValidationError
MergeValidationError = ENAHOMergeError
ValidationThresholdError = ENAHOValidationError
ConflictResolutionError = ENAHOMergeError

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
]
