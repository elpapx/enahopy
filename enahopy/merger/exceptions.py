"""

ENAHO Merger - Excepciones específicas

=====================================



Excepciones personalizadas para el sistema de fusión geográfica

y merge entre módulos ENAHO.

"""

try:
    from ..loader import ENAHOError, ENAHOValidationError

except ImportError:
    # Fallback para uso independiente

    class ENAHOError(Exception):
        """Excepción base para errores de ENAHO"""

        pass

    class ENAHOValidationError(ENAHOError):
        """Error de validación de parámetros"""

        pass


# =====================================================

# EXCEPCIONES GEOGRÁFICAS

# =====================================================


class GeoMergeError(ENAHOError):
    """Error específico durante operaciones de fusión geográfica"""

    def __init__(self, message: str, error_code: str = None, **context):
        super().__init__(message)

        self.error_code = error_code

        self.context = context


class UbigeoValidationError(ENAHOValidationError):
    """Error de validación de códigos UBIGEO"""

    def __init__(self, message: str, invalid_ubigeos: list = None, **context):
        super().__init__(message)

        self.invalid_ubigeos = invalid_ubigeos or []

        self.context = context


class TerritorialInconsistencyError(GeoMergeError):
    """Error de inconsistencia territorial"""

    def __init__(self, message: str, inconsistencies: list = None, **context):
        super().__init__(message)

        self.inconsistencies = inconsistencies or []

        self.context = context


class DuplicateHandlingError(GeoMergeError):
    """Error durante el manejo de duplicados geográficos"""

    def __init__(self, message: str, duplicates_info: dict = None, **context):
        super().__init__(message)

        self.duplicates_info = duplicates_info or {}

        self.context = context


# =====================================================

# EXCEPCIONES DE MÓDULOS

# =====================================================


class ModuleMergeError(ENAHOError):
    """Error específico durante merge entre módulos ENAHO"""

    def __init__(self, message: str, modules_involved: list = None, **context):
        super().__init__(message)

        self.modules_involved = modules_involved or []

        self.context = context


class ModuleValidationError(ENAHOValidationError):
    """Error de validación de estructura de módulos"""

    def __init__(
        self, message: str, module_code: str = None, validation_failures: list = None, **context
    ):
        super().__init__(message)

        self.module_code = module_code

        self.validation_failures = validation_failures or []

        self.context = context


class IncompatibleModulesError(ModuleMergeError):
    """Error cuando módulos no son compatibles para merge"""

    def __init__(
        self,
        message: str,
        module1: str = None,
        module2: str = None,
        compatibility_info: dict = None,
        **context,
    ):
        super().__init__(message)

        self.module1 = module1

        self.module2 = module2

        self.compatibility_info = compatibility_info or {}

        self.context = context


class MergeKeyError(ModuleMergeError):
    """Error relacionado con llaves de merge"""

    def __init__(
        self, message: str, missing_keys: list = None, invalid_keys: list = None, **context
    ):
        super().__init__(message)

        self.missing_keys = missing_keys or []

        self.invalid_keys = invalid_keys or []

        self.context = context


class ConflictResolutionError(ModuleMergeError):
    """Error durante la resolución de conflictos en merge"""

    def __init__(self, message: str, conflicts: list = None, strategy_used: str = None, **context):
        super().__init__(message)

        self.conflicts = conflicts or []

        self.strategy_used = strategy_used

        self.context = context


# =====================================================

# EXCEPCIONES DE CALIDAD Y VALIDACIÓN

# =====================================================


class DataQualityError(ENAHOError):
    """Error relacionado con calidad de datos"""

    def __init__(self, message: str, quality_metrics: dict = None, **context):
        super().__init__(message)

        self.quality_metrics = quality_metrics or {}

        self.context = context


class ValidationThresholdError(ENAHOValidationError):
    """Error cuando no se cumplen umbrales de validación"""

    def __init__(
        self,
        message: str,
        threshold_type: str = None,
        expected: float = None,
        actual: float = None,
        **context,
    ):
        super().__init__(message)

        self.threshold_type = threshold_type

        self.expected = expected

        self.actual = actual

        self.context = context


class ConfigurationError(ENAHOError):
    """Error en la configuración de merge"""

    def __init__(
        self, message: str, config_section: str = None, invalid_values: dict = None, **context
    ):
        super().__init__(message)

        self.config_section = config_section

        self.invalid_values = invalid_values or {}

        self.context = context


# =====================================================

# UTILIDADES PARA MANEJO DE EXCEPCIONES

# =====================================================


def format_exception_details(exception: Exception) -> dict:
    """

    Formatea detalles de excepción para logging estructurado



    Args:

        exception: Excepción a formatear



    Returns:

        Diccionario con detalles estructurados

    """

    details = {
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "module": getattr(exception, "__module__", "unknown"),
    }

    # Agregar contexto específico si está disponible

    if hasattr(exception, "context"):
        details["context"] = exception.context

    if hasattr(exception, "error_code"):
        details["error_code"] = exception.error_code

    # Atributos específicos por tipo de excepción

    if isinstance(exception, UbigeoValidationError):
        details["invalid_ubigeos"] = exception.invalid_ubigeos

    elif isinstance(exception, ModuleValidationError):
        details["module_code"] = exception.module_code

        details["validation_failures"] = exception.validation_failures

    elif isinstance(exception, IncompatibleModulesError):
        details["modules"] = [exception.module1, exception.module2]

        details["compatibility_info"] = exception.compatibility_info

    elif isinstance(exception, ValidationThresholdError):
        details["threshold_info"] = {
            "type": exception.threshold_type,
            "expected": exception.expected,
            "actual": exception.actual,
        }

    return details


def create_merge_error_report(exception: Exception, operation_context: dict = None) -> str:
    """

    Crea reporte detallado de error para operaciones de merge



    Args:

        exception: Excepción ocurrida

        operation_context: Contexto de la operación



    Returns:

        Reporte formateado como string

    """

    report_lines = [
        "=== REPORTE DE ERROR EN MERGE ===",
        f"Tipo: {type(exception).__name__}",
        f"Mensaje: {str(exception)}",
    ]

    # Agregar contexto de operación

    if operation_context:
        report_lines.append("\nContexto de operación:")

        for key, value in operation_context.items():
            report_lines.append(f"  {key}: {value}")

    # Detalles específicos de la excepción

    if hasattr(exception, "context") and exception.context:
        report_lines.append("\nDetalles adicionales:")

        for key, value in exception.context.items():
            report_lines.append(f"  {key}: {value}")

    # Recomendaciones específicas

    recommendations = _get_error_recommendations(exception)

    if recommendations:
        report_lines.append("\nRecomendaciones:")

        for rec in recommendations:
            report_lines.append(f"  • {rec}")

    return "\n".join(report_lines)


def _get_error_recommendations(exception: Exception) -> list:
    """Genera recomendaciones específicas según el tipo de error"""

    if isinstance(exception, UbigeoValidationError):
        return [
            "Verificar formato de códigos UBIGEO (6 dígitos)",
            "Validar que departamentos estén en el rango 01-25",
            "Revisar consistencia entre provincia y distrito",
        ]

    elif isinstance(exception, IncompatibleModulesError):
        return [
            "Verificar que los módulos sean del mismo año",
            "Considerar usar merge a nivel hogar en lugar de persona",
            "Revisar llaves de identificación en ambos módulos",
        ]

    elif isinstance(exception, DuplicateHandlingError):
        return [
            "Especificar estrategia de manejo de duplicados apropiada",
            "Verificar columna de calidad si usa BEST_QUALITY",
            "Considerar usar AGGREGATE con funciones específicas",
        ]

    elif isinstance(exception, DataQualityError):
        return [
            "Revisar completitud de datos antes del merge",
            "Validar consistencia de llaves de unión",
            "Considerar filtros de calidad adicionales",
        ]

    return [
        "Revisar configuración de merge",
        "Validar estructura de DataFrames de entrada",
        "Consultar documentación para parámetros correctos",
    ]


# =====================================================
# EXCEPCIONES DE VALIDACIÓN DE MERGE
# =====================================================


class MergeValidationError(ModuleMergeError):
    """Error de validación durante el proceso de merge"""

    def __init__(
        self,
        message: str,
        validation_type: str = None,
        failed_checks: list = None,
        **context,
    ):
        super().__init__(message)
        self.validation_type = validation_type
        self.failed_checks = failed_checks or []
        self.context = context


# Alias para compatibilidad


class MergerError(Exception):
    """Error base para operaciones de merge"""

    pass
