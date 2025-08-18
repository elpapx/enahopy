"""
Test suite for enahopy package
"""

import pytest
from pathlib import Path

# Agregar el directorio padre al path para importaciones
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))