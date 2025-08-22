"""
Excepciones específicas para análisis de nulos
"""

try:
    from ..loader import ENAHOError
except ImportError:

    class ENAHOError(Exception):
        pass


class NullAnalysisError(ENAHOError):
    """Error específico del análisis de nulos"""

    pass


class ValidationError(NullAnalysisError):
    """Error de validación de parámetros o datos"""

    pass


class VisualizationError(NullAnalysisError):
    """Error en la generación de visualizaciones"""

    pass


class PatternDetectionError(NullAnalysisError):
    """Error en la detección de patrones"""

    pass
