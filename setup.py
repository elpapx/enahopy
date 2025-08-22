"""
Setup configuration for enahopy package
"""

from pathlib import Path

from setuptools import find_packages, setup

# Leer el contenido del README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Leer las dependencias desde requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()
    # Filtrar comentarios y líneas vacías
    required = [line for line in required if line and not line.startswith("#")]

setup(
    name="enahopy",
    version="0.1.2",
    author="Alonso Camacho Abadie",
    author_email="pcamacho447@gmail.com",
    description="Librería Python para análisis de microdatos ENAHO del INEI (Perú)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elpapx/enahopy",
    project_urls={
        "Bug Tracker": "https://github.com/elpapx/enahopy/issues",
        "Documentation": "https://github.com/elpapx/enahopy/docs",
        "Source Code": "https://github.com/elpapx/enahopy",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "full": [
            "pyreadstat>=1.1.0",
            "dask[complete]>=2021.0.0",
            "geopandas>=0.10.0",
            "plotly>=5.0.0",
            "scikit-learn>=1.0.0",
            "statsmodels>=0.12.0",
            "missingno>=0.5.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "enahopy=enahopy.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "enaho",
        "peru",
        "inei",
        "survey",
        "microdata",
        "encuesta",
        "hogares",
        "estadistica",
        "analisis",
    ],
)
