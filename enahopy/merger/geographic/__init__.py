"""
ENAHO Merger - Submódulo Geográfico
==================================

Exportaciones del submódulo de funcionalidades geográficas:
validadores, detectores de patrones y estrategias de duplicados.
"""

from .validators import (
    UbigeoValidator,
    TerritorialValidator,
    GeoDataQualityValidator
)

from .patterns import GeoPatternDetector

from .strategies import (
    DuplicateHandlingStrategy,
    FirstLastStrategy,
    AggregateStrategy,
    BestQualityStrategy,
    KeepAllStrategy,
    MostRecentStrategy,
    DuplicateStrategyFactory
)


# Funciones de conveniencia específicas del submódulo geográfico
def validate_ubigeo_series(series, validation_type='structural', logger=None):
    """
    Función de conveniencia para validar serie de UBIGEOs

    Args:
        series: Serie de pandas con códigos UBIGEO
        validation_type: Tipo de validación ('basic', 'structural')
        logger: Logger opcional

    Returns:
        Tupla (mask_validos, errores)
    """
    if logger is None:
        import logging
        logger = logging.getLogger('ubigeo_validator')

    from ..config import TipoValidacionUbigeo
    validator = UbigeoValidator(logger)

    tipo_enum = TipoValidacionUbigeo(validation_type)
    return validator.validar_serie_ubigeos(series, tipo_enum)


def detect_geo_columns_quick(df, confidence_threshold=0.8):
    """
    Función de conveniencia para detección rápida de columnas geográficas

    Args:
        df: DataFrame a analizar
        confidence_threshold: Umbral de confianza

    Returns:
        Diccionario con columnas detectadas
    """
    import logging
    logger = logging.getLogger('geo_detector')

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # Silencioso por defecto

    detector = GeoPatternDetector(logger)
    return detector.detectar_columnas_geograficas(df, confidence_threshold)


def create_duplicate_strategy(strategy_name, logger=None):
    """
    Función de conveniencia para crear estrategias de duplicados

    Args:
        strategy_name: Nombre de la estrategia ('first', 'last', 'aggregate', etc.)
        logger: Logger opcional

    Returns:
        Instancia de la estrategia
    """
    if logger is None:
        import logging
        logger = logging.getLogger('duplicate_strategy')

    from ..config import TipoManejoDuplicados
    strategy_enum = TipoManejoDuplicados(strategy_name)
    return DuplicateStrategyFactory.create_strategy(strategy_enum, logger)


def comprehensive_geo_validation(df, ubigeo_column=None, **kwargs):
    """
    Función de conveniencia para validación geográfica integral

    Args:
        df: DataFrame a validar
        ubigeo_column: Columna UBIGEO (None para autodetectar)
        **kwargs: Argumentos adicionales

    Returns:
        Diccionario con resultados de validación
    """
    import logging
    logger = logging.getLogger('geo_validator')

    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    validator = GeoDataQualityValidator(logger)

    # Detectar columnas geográficas si no se especifica UBIGEO
    if ubigeo_column is None:
        geo_columns = detect_geo_columns_quick(df)
        if 'ubigeo' in geo_columns:
            ubigeo_column = geo_columns['ubigeo']
        else:
            geo_columns = {}
    else:
        geo_columns = {'ubigeo': ubigeo_column}

    return validator.comprehensive_validation(
        df, geo_columns, kwargs.get('validation_config', None)
    )


# Exportaciones públicas del submódulo
__all__ = [
    # Clases principales
    'UbigeoValidator',
    'TerritorialValidator',
    'GeoDataQualityValidator',
    'GeoPatternDetector',

    # Estrategias de duplicados
    'DuplicateHandlingStrategy',
    'FirstLastStrategy',
    'AggregateStrategy',
    'BestQualityStrategy',
    'KeepAllStrategy',
    'MostRecentStrategy',
    'DuplicateStrategyFactory',

    # Funciones de conveniencia
    'validate_ubigeo_series',
    'detect_geo_columns_quick',
    'create_duplicate_strategy',
    'comprehensive_geo_validation'
]

# Metadatos del submódulo
__version__ = "2.0.0"
__description__ = "Geographic validation and pattern detection for INEI data"