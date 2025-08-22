"""
Tests básicos para verificar la instalación de enahopy
"""

import sys
from pathlib import Path

import pytest


def test_import():
    """Test que el paquete se puede importar"""
    import enahopy

    assert enahopy is not None
    assert hasattr(enahopy, "__version__")


def test_version():
    """Test que la versión está definida correctamente"""
    import enahopy

    assert isinstance(enahopy.__version__, str)
    version_parts = enahopy.__version__.split(".")
    assert len(version_parts) == 3
    # Verificar que cada parte es un número
    for part in version_parts:
        assert part.isdigit()


def test_loader_module():
    """Test que el módulo loader está disponible"""
    try:
        from enahopy.loader import ENAHODataDownloader

        assert ENAHODataDownloader is not None
        print("✅ Módulo loader disponible")
    except ImportError as e:
        pytest.skip(f"Módulo loader no disponible: {e}")


def test_loader_config():
    """Test que la configuración del loader funciona"""
    try:
        from enahopy.loader.core import ENAHOConfig

        config = ENAHOConfig()
        assert config.base_url is not None
        assert len(config.AVAILABLE_MODULES) > 0
        print(f"✅ Configuración cargada: {len(config.AVAILABLE_MODULES)} módulos")
    except ImportError as e:
        pytest.skip(f"Configuración no disponible: {e}")


def test_merger_module():
    """Test que el módulo merger está disponible"""
    try:
        from enahopy.merger import ENAHOMerger

        assert ENAHOMerger is not None
        print("✅ Módulo merger disponible")
    except ImportError as e:
        pytest.skip(f"Módulo merger no disponible: {e}")


def test_null_analysis_module():
    """Test que el módulo null_analysis está disponible"""
    try:
        from enahopy.null_analysis import ENAHONullAnalyzer

        assert ENAHONullAnalyzer is not None
        print("✅ Módulo null_analysis disponible")
    except ImportError as e:
        pytest.skip(f"Módulo null_analysis no disponible: {e}")


def test_package_structure():
    """Test que la estructura del paquete es correcta"""
    import enahopy

    # Verificar que __all__ está definido
    assert hasattr(enahopy, "__all__")

    # Verificar metadatos básicos
    expected_attrs = ["__version__"]
    for attr in expected_attrs:
        assert hasattr(enahopy, attr), f"Falta atributo: {attr}"

    print(f"✅ Estructura del paquete correcta")
    print(f"   Versión: {enahopy.__version__}")
    print(f"   Exports: {len(enahopy.__all__)} elementos")
