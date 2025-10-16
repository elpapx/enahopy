# setup_test.py
"""
Script de configuración para el entorno de testing
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_test_environment():
    """Configurar el entorno completo de testing"""

    print("=" * 60)
    print("CONFIGURACIÓN DEL ENTORNO DE TESTING - ENAHOPY")
    print("=" * 60)

    # 1. Instalar dependencias de testing
    print("\n1. Instalando dependencias de testing...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"]
        )
        print("   ✅ Dependencias instaladas")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error instalando dependencias: {e}")
        return False

    # 2. Crear estructura de directorios
    print("\n2. Creando estructura de directorios...")
    directories = [
        "tests",
        "tests/fixtures",
        "tests/data",
        "tests/unit",
        "tests/integration",
        "htmlcov",
        "reports",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")

    # 3. Configurar pre-commit hooks
    print("\n3. Configurando pre-commit hooks...")
    try:
        subprocess.check_call(["pre-commit", "install"])
        print("   ✅ Pre-commit hooks instalados")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️ Pre-commit no disponible (opcional)")

    # 4. Verificar instalación
    print("\n4. Verificando instalación...")

    # Verificar pytest
    try:
        result = subprocess.run(["pytest", "--version"], capture_output=True, text=True)
        print(f"   ✅ pytest: {result.stdout.strip()}")
    except FileNotFoundError:
        print("   ❌ pytest no encontrado")

    # Verificar coverage
    try:
        result = subprocess.run(["coverage", "--version"], capture_output=True, text=True)
        print(f"   ✅ coverage: {result.stdout.strip()}")
    except FileNotFoundError:
        print("   ❌ coverage no encontrado")

    print("\n" + "=" * 60)
    print("✅ ENTORNO DE TESTING CONFIGURADO CORRECTAMENTE")
    print("=" * 60)

    return True


def create_test_data():
    """Crear datos de prueba para los tests"""

    print("\nCreando datos de prueba...")

    import numpy as np
    import pandas as pd

    # Crear archivo de prueba ENAHO simulado
    np.random.seed(42)
    n = 100

    test_data = pd.DataFrame(
        {
            "conglome": [f"{i:06d}" for i in range(n)],
            "vivienda": np.random.choice(["01", "02"], n),
            "hogar": ["1"] * n,
            "factor07": np.random.uniform(0.5, 3, n),
            "ingreso": np.random.normal(3000, 1500, n),
            "gasto": np.random.normal(2500, 1000, n),
            "ubigeo": np.random.choice(["150101", "130101", "080801"], n),
        }
    )

    # Guardar en diferentes formatos
    data_dir = Path("tests/data")

    # CSV
    test_data.to_csv(data_dir / "test_enaho.csv", index=False)
    print(f"   ✅ test_enaho.csv creado")

    # Parquet
    try:
        test_data.to_parquet(data_dir / "test_enaho.parquet", index=False)
        print(f"   ✅ test_enaho.parquet creado")
    except:
        print(f"   ⚠️ No se pudo crear archivo parquet")

    # Stata
    try:
        test_data.to_stata(data_dir / "test_enaho.dta", write_index=False)
        print(f"   ✅ test_enaho.dta creado")
    except:
        print(f"   ⚠️ No se pudo crear archivo stata")

    print("\n✅ Datos de prueba creados")


if __name__ == "__main__":
    # Ejecutar configuración
    success = setup_test_environment()

    if success:
        create_test_data()

        print("\n" + "=" * 60)
        print("PRÓXIMOS PASOS:")
        print("=" * 60)
        print("1. Ejecutar tests unitarios:")
        print("   python -m pytest tests/")
        print("\n2. Ejecutar con cobertura:")
        print("   pytest --cov=enahopy --cov-report=html")
        print("\n3. Ejecutar test runner completo:")
        print("   python run_tests.py --coverage")
        print("\n4. Ejecutar en múltiples entornos:")
        print("   tox")
        print("=" * 60)

"""
.pre-commit-config.yaml
=======================
Configuración para pre-commit hooks
"""

PRE_COMMIT_CONFIG = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""

"""
Makefile
========
Comandos útiles para desarrollo y testing
"""

MAKEFILE = """
.PHONY: help test coverage lint format install clean

help:
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias"
	@echo "  make test       - Ejecutar tests"
	@echo "  make coverage   - Ejecutar tests con cobertura"
	@echo "  make lint       - Verificar estilo de código"
	@echo "  make format     - Formatear código"
	@echo "  make clean      - Limpiar archivos temporales"

install:
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	python setup_test.py

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

coverage:
	pytest --cov=enahopy --cov-report=html --cov-report=term

lint:
	flake8 enahopy tests
	pylint enahopy
	mypy enahopy

format:
	black enahopy tests
	isort enahopy tests

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	sphinx-build -b html docs docs/_build

all: format lint test coverage
"""
