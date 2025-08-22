"""
ENAHO Validation Module
======================

Validador específico para parámetros y datos ENAHO.
Incluye validación de años, módulos, directorios
y archivos según estándares INEI.
"""

import logging
from pathlib import Path
from typing import List, Union

from ...core.config import ENAHOConfig
from ...core.exceptions import ENAHOValidationError


class ENAHOValidator:
    """Validador de parámetros para ENAHO"""

    def __init__(self, config: ENAHOConfig):
        self.config = config

    def validate_years(self, years: List[str], is_panel: bool) -> None:
        """
        Valida que los años estén disponibles

        Args:
            years: Lista de años a validar
            is_panel: Si es dataset panel o transversal

        Raises:
            ENAHOValidationError: Si hay años inválidos
        """
        if not years:
            raise ENAHOValidationError("Debe especificar al menos un año", "EMPTY_YEARS")

        year_map = self.config.YEAR_MAP_PANEL if is_panel else self.config.YEAR_MAP_TRANSVERSAL
        invalid_years = [year for year in years if year not in year_map]

        if invalid_years:
            dataset_type = "panel" if is_panel else "corte transversal"
            raise ENAHOValidationError(
                f"Años {invalid_years} no disponibles para {dataset_type}",
                "INVALID_YEARS",
                invalid_years=invalid_years,
                valid_years=list(year_map.keys()),
                dataset_type=dataset_type,
            )

    def validate_modules(self, modules: List[str]) -> List[str]:
        """
        Valida la lista de módulos

        Args:
            modules: Lista de módulos a validar

        Returns:
            Lista de módulos normalizados

        Raises:
            ENAHOValidationError: Si hay módulos inválidos
        """
        if not modules:
            raise ENAHOValidationError("Debe especificar al menos un módulo", "EMPTY_MODULES")

        # Normalizar módulos (agregar 0 si es necesario)
        normalized_modules = []
        for module in modules:
            if not isinstance(module, str):
                raise ENAHOValidationError(
                    f"Módulo debe ser string: {module}", "INVALID_MODULE_TYPE"
                )

            module = module.strip().zfill(2)  # Asegura formato 01, 02, etc.
            normalized_modules.append(module)

        # Validar contra módulos disponibles
        invalid_modules = [m for m in normalized_modules if m not in self.config.AVAILABLE_MODULES]
        if invalid_modules:
            raise ENAHOValidationError(
                f"Módulos {invalid_modules} no están disponibles",
                "INVALID_MODULES",
                invalid_modules=invalid_modules,
                available_modules=list(self.config.AVAILABLE_MODULES.keys()),
            )

        return normalized_modules

    def validate_output_dir(self, output_dir: Union[str, Path]) -> Path:
        """
        Valida y crea el directorio de salida

        Args:
            output_dir: Directorio de salida

        Returns:
            Path del directorio validado

        Raises:
            ENAHOValidationError: Si hay problemas con el directorio
        """
        output_path = Path(output_dir)

        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ENAHOValidationError(
                f"Sin permisos para crear directorio: {output_path}", "PERMISSION_ERROR"
            )
        except OSError as e:
            raise ENAHOValidationError(f"Error creando directorio: {e}", "DIRECTORY_ERROR")

        return output_path

    def validate_file_exists(self, file_path: Union[str, Path]) -> Path:
        """
        Valida que un archivo exista

        Args:
            file_path: Ruta al archivo

        Returns:
            Path del archivo validado

        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        return path


__all__ = ["ENAHOValidator"]
