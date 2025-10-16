"""
Analysis strategies for null patterns and advanced ML-based imputation
"""

from .advanced_analysis import AdvancedNullAnalysis
from .basic_analysis import BasicNullAnalysis, INullAnalysisStrategy

# Advanced ML Imputation imports
try:
    from .advanced_ml_imputation import (
        AutoencoderImputer,
        ImputationConfig,
        ImputationResult,
        MICEImputer,
        MissForestImputer,
        MLImputationManager,
        compare_imputation_methods,
        create_advanced_imputer,
        create_ml_imputation_manager,
    )

    ML_IMPUTATION_AVAILABLE = True
except ImportError as e:
    ML_IMPUTATION_AVAILABLE = False

    # Create dummy classes to maintain API compatibility
    class MICEImputer:
        pass

    class MissForestImputer:
        pass

    class AutoencoderImputer:
        pass

    class ImputationConfig:
        pass

    class ImputationResult:
        pass

    class MLImputationManager:
        pass

    def create_advanced_imputer(*args, **kwargs):
        raise ImportError("Advanced ML imputation not available. Missing dependencies.")

    def compare_imputation_methods(*args, **kwargs):
        raise ImportError("Advanced ML imputation not available. Missing dependencies.")

    def create_ml_imputation_manager(*args, **kwargs):
        raise ImportError("Advanced ML imputation not available. Missing dependencies.")


# ENAHO Pattern-Aware Imputation imports
try:
    from .enaho_pattern_imputation import (
        ENAHOImputationConfig,
        ENAHOMissingPattern,
        ENAHOPatternDetector,
        ENAHOPatternImputer,
        create_enaho_pattern_imputer,
    )

    ENAHO_PATTERN_AVAILABLE = True
except ImportError as e:
    ENAHO_PATTERN_AVAILABLE = False

    class ENAHOMissingPattern:
        pass

    class ENAHOImputationConfig:
        pass

    class ENAHOPatternDetector:
        pass

    class ENAHOPatternImputer:
        pass

    def create_enaho_pattern_imputer(*args, **kwargs):
        raise ImportError("ENAHO pattern imputation not available. Missing dependencies.")


# Quality Assessment imports
try:
    from .imputation_quality_assessment import (
        ImputationQualityAssessor,
        QualityAssessmentConfig,
        QualityAssessmentResult,
        QualityMetricType,
        assess_imputation_quality,
    )

    QUALITY_ASSESSMENT_AVAILABLE = True
except ImportError as e:
    QUALITY_ASSESSMENT_AVAILABLE = False

    class QualityMetricType:
        pass

    class QualityAssessmentConfig:
        pass

    class QualityAssessmentResult:
        pass

    class ImputationQualityAssessor:
        pass

    def assess_imputation_quality(*args, **kwargs):
        raise ImportError("Quality assessment not available. Missing dependencies.")


__all__ = [
    # Basic analysis
    "INullAnalysisStrategy",
    "BasicNullAnalysis",
    "AdvancedNullAnalysis",
    # ML Imputation
    "MICEImputer",
    "MissForestImputer",
    "AutoencoderImputer",
    "ImputationConfig",
    "ImputationResult",
    "MLImputationManager",
    "create_advanced_imputer",
    "compare_imputation_methods",
    "create_ml_imputation_manager",
    # ENAHO Pattern Imputation
    "ENAHOMissingPattern",
    "ENAHOImputationConfig",
    "ENAHOPatternDetector",
    "ENAHOPatternImputer",
    "create_enaho_pattern_imputer",
    # Quality Assessment
    "QualityMetricType",
    "QualityAssessmentConfig",
    "QualityAssessmentResult",
    "ImputationQualityAssessor",
    "assess_imputation_quality",
    # Availability flags
    "ML_IMPUTATION_AVAILABLE",
    "ENAHO_PATTERN_AVAILABLE",
    "QUALITY_ASSESSMENT_AVAILABLE",
]
