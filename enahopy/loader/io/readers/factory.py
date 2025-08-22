"""
Reader Factory Module
====================

Factory pattern para crear el reader apropiado según
la extensión del archivo. Manejo centralizado de
formatos soportados y no soportados.
"""

import logging
from pathlib import Path

from ...core.exceptions import UnsupportedFormatError
from ...io.base import IReader
from .csv import CSVReader
from .parquet import ParquetReader
from .spss import SPSSReader
from .stata import StataReader


class ReaderFactory:
    """Fábrica para crear el lector apropiado según la extensión del archivo."""

    @staticmethod
    def create_reader(file_path: Path, logger: logging.Logger) -> IReader:
        """
        Crea el reader apropiado para el archivo dado.

        Args:
            file_path: Ruta al archivo
            logger: Logger para mensajes

        Returns:
            Reader específico para el formato

        Raises:
            UnsupportedFormatError: Si el formato no es soportado
        """
        extension = file_path.suffix.lower()
        logger.info(f"Detectado formato de archivo: {extension}")

        reader_map = {
            ".sav": SPSSReader,
            ".por": SPSSReader,
            ".dta": StataReader,
            ".parquet": ParquetReader,
            ".csv": CSVReader,
            ".txt": CSVReader,
        }

        reader_class = reader_map.get(extension)

        if not reader_class:
            supported_formats = list(reader_map.keys())
            raise UnsupportedFormatError(
                f"Formato de archivo no soportado: {extension}. "
                f"Formatos soportados: {supported_formats}"
            )

        logger.info(f"Usando lector: {reader_class.__name__}")
        return reader_class(file_path, logger)


__all__ = ["ReaderFactory"]
