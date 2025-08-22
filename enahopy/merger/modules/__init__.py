"""
ENAHO Merger - Submódulo de Módulos ENAHO
========================================

Exportaciones del submódulo especializado en merge entre módulos ENAHO:
merger, validador y funciones de compatibilidad.
"""

from .merger import ENAHOModuleMerger
from .validator import ModuleValidator


# Funciones de conveniencia específicas del submódulo de módulos
def quick_module_merge(
    df1, df2, module1_code, module2_code, level="hogar", strategy="coalesce", **kwargs
):
    """
    Función de conveniencia para merge rápido entre dos módulos

    Args:
        df1, df2: DataFrames de los módulos
        module1_code, module2_code: Códigos de los módulos
        level: Nivel de merge ('hogar', 'persona', 'vivienda')
        strategy: Estrategia de conflictos ('coalesce', 'keep_left', etc.)
        **kwargs: Argumentos adicionales

    Returns:
        DataFrame resultado del merge
    """
    import logging

    from ..config import ModuleMergeConfig, ModuleMergeLevel, ModuleMergeStrategy

    logger = logging.getLogger("module_merger")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO if kwargs.get("verbose", True) else logging.WARNING)

    config = ModuleMergeConfig(
        merge_level=ModuleMergeLevel(level), merge_strategy=ModuleMergeStrategy(strategy)
    )

    merger = ENAHOModuleMerger(config, logger)
    result = merger.merge_modules(df1, df2, module1_code, module2_code)

    return result.merged_df


def validate_module_structure_quick(df, module_code):
    """
    Función de conveniencia para validación rápida de estructura de módulo

    Args:
        df: DataFrame del módulo
        module_code: Código del módulo ENAHO

    Returns:
        Lista de advertencias encontradas
    """
    import logging

    from ..config import ModuleMergeConfig

    logger = logging.getLogger("module_validator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    config = ModuleMergeConfig()  # Configuración por defecto
    validator = ModuleValidator(config, logger)

    return validator.validate_module_structure(df, module_code)


def check_modules_compatibility_quick(modules_dict, merge_level="hogar"):
    """
    Función de conveniencia para verificar compatibilidad entre múltiples módulos

    Args:
        modules_dict: Diccionario {codigo_modulo: dataframe}
        merge_level: Nivel de merge a evaluar

    Returns:
        Diccionario con resultado de compatibilidad
    """
    import logging

    from ..config import ModuleMergeConfig, ModuleMergeLevel

    logger = logging.getLogger("compatibility_checker")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    config = ModuleMergeConfig(merge_level=ModuleMergeLevel(merge_level))
    validator = ModuleValidator(config, logger)

    compatibility_report = {
        "overall_compatible": True,
        "merge_level": merge_level,
        "modules_analyzed": list(modules_dict.keys()),
        "pairwise_compatibility": {},
        "issues": [],
    }

    # Verificar compatibilidad por pares
    module_codes = list(modules_dict.keys())
    level_enum = ModuleMergeLevel(merge_level)

    for i, module1 in enumerate(module_codes):
        for module2 in module_codes[i + 1 :]:
            compatibility = validator.check_module_compatibility(
                modules_dict[module1], modules_dict[module2], module1, module2, level_enum
            )

            pair_key = f"{module1}-{module2}"
            compatibility_report["pairwise_compatibility"][pair_key] = compatibility

            if not compatibility["compatible"]:
                compatibility_report["overall_compatible"] = False
                compatibility_report["issues"].append(
                    f"Módulos {module1} y {module2}: {compatibility.get('error', 'Incompatible')}"
                )

    return compatibility_report


def merge_multiple_modules_quick(
    modules_dict, base_module="34", level="hogar", strategy="coalesce", **kwargs
):
    """
    Función de conveniencia para merge múltiple con configuración automática

    Args:
        modules_dict: Diccionario {codigo_modulo: dataframe}
        base_module: Módulo base para iniciar
        level: Nivel de merge
        strategy: Estrategia de conflictos
        **kwargs: Argumentos adicionales

    Returns:
        DataFrame con todos los módulos combinados
    """
    import logging

    from ..config import ModuleMergeConfig, ModuleMergeLevel, ModuleMergeStrategy

    logger = logging.getLogger("multi_module_merger")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO if kwargs.get("verbose", True) else logging.WARNING)

    config = ModuleMergeConfig(
        merge_level=ModuleMergeLevel(level), merge_strategy=ModuleMergeStrategy(strategy)
    )

    merger = ENAHOModuleMerger(config, logger)
    result = merger.merge_multiple_modules(modules_dict, base_module)

    if kwargs.get("return_report", False):
        return result.merged_df, result.merge_report
    else:
        return result.merged_df


def get_module_info(module_code):
    """
    Función de conveniencia para obtener información sobre un módulo ENAHO

    Args:
        module_code: Código del módulo

    Returns:
        Diccionario con información del módulo
    """
    from ..config import ModuleMergeConfig

    config = ModuleMergeConfig()

    if module_code not in config.module_validations:
        return {
            "valid": False,
            "error": f"Módulo {module_code} no reconocido",
            "available_modules": list(config.module_validations.keys()),
        }

    module_info = config.module_validations[module_code]

    return {
        "valid": True,
        "module_code": module_code,
        "level": module_info["level"].value,
        "required_keys": module_info["required_keys"],
        "description": _get_module_description(module_code),
    }


def _get_module_description(module_code):
    """Obtiene descripción del módulo ENAHO"""
    descriptions = {
        "01": "Características de la Vivienda y del Hogar",
        "02": "Características de los Miembros del Hogar",
        "03": "Educación",
        "04": "Salud",
        "05": "Empleo e Ingresos",
        "07": "Ingresos del Hogar",
        "08": "Gastos del Hogar",
        "09": "Programas Sociales",
        "34": "Sumaria (Resumen del Hogar)",
        "37": "Gobierno Electrónico",
    }
    return descriptions.get(module_code, "Módulo ENAHO")


def analyze_merge_feasibility_quick(modules_dict, target_level="hogar"):
    """
    Función de conveniencia para análisis rápido de viabilidad de merge

    Args:
        modules_dict: Diccionario de módulos
        target_level: Nivel objetivo de merge

    Returns:
        Diccionario con análisis de viabilidad
    """
    import logging

    from ..config import ModuleMergeConfig, ModuleMergeLevel

    logger = logging.getLogger("feasibility_analyzer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    config = ModuleMergeConfig(merge_level=ModuleMergeLevel(target_level))
    merger = ENAHOModuleMerger(config, logger)

    return merger.analyze_merge_feasibility(modules_dict, ModuleMergeLevel(target_level))


def create_optimal_merge_plan(modules_dict, target_module="34"):
    """
    Función de conveniencia para crear plan óptimo de merge

    Args:
        modules_dict: Diccionario de módulos
        target_module: Módulo objetivo

    Returns:
        Plan de merge optimizado
    """
    import logging

    from ..config import ModuleMergeConfig

    logger = logging.getLogger("merge_planner")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    config = ModuleMergeConfig()
    merger = ENAHOModuleMerger(config, logger)

    return merger.create_merge_plan(modules_dict, target_module)


# Exportaciones públicas del submódulo
__all__ = [
    # Clases principales
    "ENAHOModuleMerger",
    "ModuleValidator",
    # Funciones de conveniencia
    "quick_module_merge",
    "validate_module_structure_quick",
    "check_modules_compatibility_quick",
    "merge_multiple_modules_quick",
    "get_module_info",
    "analyze_merge_feasibility_quick",
    "create_optimal_merge_plan",
]

# Metadatos del submódulo
__version__ = "2.0.0"
__description__ = "ENAHO module merging and validation system"
