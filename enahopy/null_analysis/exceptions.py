"""
ENAHO Null Analysis - Exception Compatibility Layer
=================================================

This module imports exceptions from the unified hierarchy for backward compatibility.
All new code should import directly from enahopy.exceptions.

DEPRECATED: Import from enahopy.exceptions instead.
"""

import warnings

# Import from unified exception hierarchy
from enahopy.exceptions import (
    ENAHOError,
    ENAHONullAnalysisError,
    ENAHOValidationError,
    ImputationError,
    PatternDetectionError,
    VisualizationError,
)

# Deprecated warnings for backward compatibility
warnings.warn(
    "Importing from enahopy.null_analysis.exceptions is deprecated. "
    "Import from enahopy.exceptions instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias for backward compatibility
NullAnalysisError = ENAHONullAnalysisError
ValidationError = ENAHOValidationError

# Export all for backward compatibility
__all__ = [
    "ENAHOError",
    "ENAHONullAnalysisError",
    "NullAnalysisError",  # Alias for compatibility
    "PatternDetectionError",
    "VisualizationError",
    "ImputationError",
    "ValidationError",  # Alias for compatibility
]
