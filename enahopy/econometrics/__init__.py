"""
ENAHO Advanced Econometrics Package - ANALYZE Phase
===================================================

Comprehensive econometric analysis toolkit for ENAHO policy research including:
- Advanced poverty and inequality analysis
- Causal inference methods
- Policy impact evaluation
- Microsimulation models
- Regression analysis with survey weights
- Decomposition methods
"""

# Core econometric modules
try:
    from .poverty_analysis import (
        AdvancedPovertyAnalyzer,
        MultidimensionalPoverty,
        PovertyDecomposition,
        PovertyTrends,
        create_poverty_analyzer,
    )

    POVERTY_ANALYSIS_AVAILABLE = True
except ImportError:
    POVERTY_ANALYSIS_AVAILABLE = False
    AdvancedPovertyAnalyzer = None
    PovertyDecomposition = None
    PovertyTrends = None
    MultidimensionalPoverty = None
    create_poverty_analyzer = None

try:
    from .inequality_analysis import (
        AdvancedInequalityAnalyzer,
        InequalityDecomposition,
        SocialMobility,
        create_inequality_analyzer,
    )

    INEQUALITY_ANALYSIS_AVAILABLE = True
except ImportError:
    INEQUALITY_ANALYSIS_AVAILABLE = False
    AdvancedInequalityAnalyzer = None
    InequalityDecomposition = None
    SocialMobility = None
    create_inequality_analyzer = None

try:
    from .causal_inference import (
        CausalAnalyzer,
        DifferenceInDifferences,
        InstrumentalVariables,
        PropensityScoreMatching,
        RegressionDiscontinuity,
        create_causal_analyzer,
    )

    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False
    CausalAnalyzer = None
    PropensityScoreMatching = None
    RegressionDiscontinuity = None
    DifferenceInDifferences = None
    InstrumentalVariables = None
    create_causal_analyzer = None

try:
    from .policy_evaluation import (
        Microsimulation,
        PolicyImpactEvaluator,
        PolicyScenarios,
        create_policy_evaluator,
    )

    POLICY_EVALUATION_AVAILABLE = True
except ImportError:
    POLICY_EVALUATION_AVAILABLE = False
    PolicyImpactEvaluator = None
    Microsimulation = None
    PolicyScenarios = None
    create_policy_evaluator = None

try:
    from .survey_methods import (
        ComplexSampleAnalysis,
        DesignEffects,
        SurveyRegression,
        SurveyWeights,
        create_survey_analyzer,
    )

    SURVEY_METHODS_AVAILABLE = True
except ImportError:
    SURVEY_METHODS_AVAILABLE = False
    SurveyRegression = None
    ComplexSampleAnalysis = None
    SurveyWeights = None
    DesignEffects = None
    create_survey_analyzer = None


def get_econometrics_status() -> dict:
    """Get status of all econometrics components"""
    return {
        "poverty_analysis": POVERTY_ANALYSIS_AVAILABLE,
        "inequality_analysis": INEQUALITY_ANALYSIS_AVAILABLE,
        "causal_inference": CAUSAL_INFERENCE_AVAILABLE,
        "policy_evaluation": POLICY_EVALUATION_AVAILABLE,
        "survey_methods": SURVEY_METHODS_AVAILABLE,
    }


def show_econometrics_status():
    """Display econometrics components status"""
    status = get_econometrics_status()

    print("ENAHO Advanced Econometrics Status:")
    print("-" * 40)

    components = {
        "Poverty Analysis": status["poverty_analysis"],
        "Inequality Analysis": status["inequality_analysis"],
        "Causal Inference": status["causal_inference"],
        "Policy Evaluation": status["policy_evaluation"],
        "Survey Methods": status["survey_methods"],
    }

    for name, available in components.items():
        symbol = "[OK]" if available else "[X]"
        status_text = "Available" if available else "Not Available"
        print(f"{symbol} {name}: {status_text}")


def create_econometrics_suite(logger=None):
    """
    Create complete econometrics analysis suite

    Args:
        logger: Optional logger instance

    Returns:
        Dictionary with all available econometric tools
    """
    suite = {}

    if POVERTY_ANALYSIS_AVAILABLE:
        suite["poverty_analyzer"] = create_poverty_analyzer(logger)

    if INEQUALITY_ANALYSIS_AVAILABLE:
        suite["inequality_analyzer"] = create_inequality_analyzer(logger)

    if CAUSAL_INFERENCE_AVAILABLE:
        suite["causal_analyzer"] = create_causal_analyzer(logger)

    if POLICY_EVALUATION_AVAILABLE:
        suite["policy_evaluator"] = create_policy_evaluator(logger)

    if SURVEY_METHODS_AVAILABLE:
        suite["survey_analyzer"] = create_survey_analyzer(logger)

    return suite


# Export all available components
__all__ = ["get_econometrics_status", "show_econometrics_status", "create_econometrics_suite"]

if POVERTY_ANALYSIS_AVAILABLE:
    __all__.extend(
        [
            "AdvancedPovertyAnalyzer",
            "PovertyDecomposition",
            "PovertyTrends",
            "MultidimensionalPoverty",
            "create_poverty_analyzer",
        ]
    )

if INEQUALITY_ANALYSIS_AVAILABLE:
    __all__.extend(
        [
            "AdvancedInequalityAnalyzer",
            "InequalityDecomposition",
            "SocialMobility",
            "create_inequality_analyzer",
        ]
    )

if CAUSAL_INFERENCE_AVAILABLE:
    __all__.extend(
        [
            "CausalAnalyzer",
            "PropensityScoreMatching",
            "RegressionDiscontinuity",
            "DifferenceInDifferences",
            "InstrumentalVariables",
            "create_causal_analyzer",
        ]
    )

if POLICY_EVALUATION_AVAILABLE:
    __all__.extend(
        ["PolicyImpactEvaluator", "Microsimulation", "PolicyScenarios", "create_policy_evaluator"]
    )

if SURVEY_METHODS_AVAILABLE:
    __all__.extend(
        [
            "SurveyRegression",
            "ComplexSampleAnalysis",
            "SurveyWeights",
            "DesignEffects",
            "create_survey_analyzer",
        ]
    )
