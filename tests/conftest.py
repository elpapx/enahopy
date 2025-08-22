"""
Configuración global para pytest de enahopy
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Agregar el directorio padre al path para poder importar enahopy
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================
# FIXTURES BÁSICAS
# ============================================


@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de ejemplo para pruebas"""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "conglome": np.random.randint(100000, 999999, 100),
            "vivienda": np.random.randint(1, 20, 100),
            "hogar": np.random.randint(1, 3, 100),
            "p203": np.random.choice([1, 2, np.nan], 100, p=[0.4, 0.4, 0.2]),
            "p207": np.random.choice([1, 2], 100),
            "p208a": np.random.randint(0, 99, 100),
            "factor07": np.random.uniform(50, 500, 100),
        }
    )


@pytest.fixture
def temp_directory():
    """Crea un directorio temporal para pruebas"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_enaho_config():
    """Crea una configuración mock para pruebas"""

    class MockConfig:
        base_url = "https://test.inei.gob.pe"
        cache_dir = "./test_cache"
        verbose = False
        cache_ttl_hours = 1
        AVAILABLE_MODULES = {
            "01": "Características de la Vivienda",
            "02": "Características de los Miembros del Hogar",
        }
        YEAR_MAP_TRANSVERSAL = {"2023": "952", "2022": "950"}

    retur
