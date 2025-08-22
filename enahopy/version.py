"""
Version information for enahopy

Este archivo centraliza la información de versión para evitar
problemas de importación circular y facilitar el mantenimiento.
"""

__version__ = "0.1.2"
__version_info__ = (0, 1, 2)


def get_version():
    """Retorna la versión como string"""
    return __version__


def get_version_tuple():
    """Retorna la versión como tupla"""
    return __version_info__
