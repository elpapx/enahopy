<p align="center">
  <img src="assets/enahopy_logo.png" alt="enahopy" width="500"/>
</p>

<h1 align="center">enahopy</h1>

<p align="center">
  <em>Professional Python toolkit for analyzing Peru's ENAHO household survey data</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/elpapx/enahopy/actions/workflows/ci.yml"><img src="https://github.com/elpapx/enahopy/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline"></a>
  <a href="https://codecov.io/gh/elpapx/enahopy"><img src="https://codecov.io/gh/elpapx/enahopy/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  <a href="#-why-enahopy">Why</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-supported-modules">Modules</a> •
  <a href="docs/">Documentation</a> •
  <a href="examples/">Examples</a>
</p>

---

## 🎯 Why enahopy?

Transform Peru's ENAHO survey data from raw ZIP files to analysis-ready pandas DataFrames in **3 lines of code**.

**Before enahopy** (50+ lines of boilerplate):
```python
# Download ZIP from INEI website
# Extract DBF files manually
# Handle multiple encodings (CP1252/UTF-8)
# Merge modules with proper keys
# Apply factores de expansión correctly
# Handle missing data...
# (50+ more lines)
```

**With enahopy** (3 lines):
```python
import enahopy as enaho
loader = enaho.ENAHOLoader(year=2022)
df = loader.load_module("01")  # ¡Listo! 🎉
```

---

## 📦 Installation

### Basic installation
```bash
pip install enahopy
```

### With all features
```bash
pip install enahopy[all]
```

---

## 🚀 Quick Start

### Example 1: Download and Load Data
```python
from enahopy.loader import ENAHODataDownloader

# Initialize downloader
downloader = ENAHODataDownloader(verbose=True)

# Download housing characteristics data
data = downloader.download(
    modules=['01'],
    years=['2024'],
    output_dir='./data',
    load_dta=True
)

df_hogar = data[('2024', '01')]['enaho01-2024-100']
print(f"✓ Loaded {len(df_hogar):,} households")
```

### Example 2: Weighted Statistics (Professional)
```python
import pandas as pd
import numpy as np

# Load sumaria module with poverty indicators
df_sumaria = data[('2024', '34')]['sumaria-2024']

# ✅ CORRECT: Weighted statistics using factores de expansión
factor = df_sumaria['factor07']  # Expansion factor

# Calculate weighted poverty rate
pobreza_rate = (
    (df_sumaria['pobreza'] <= 2) * factor  # 1=extreme poor, 2=poor
).sum() / factor.sum() * 100

print(f"Tasa de pobreza (ponderada): {pobreza_rate:.2f}%")

# Weighted analysis by geographic domain
def weighted_stats(group):
    w = group['factor07']
    return pd.Series({
        'pobreza_pct': ((group['pobreza'] <= 2) * w).sum() / w.sum() * 100,
        'ingreso_promedio': np.average(group['inghog2d'], weights=w)
    })

analisis_geografico = df_sumaria.groupby('dominio').apply(weighted_stats)
print("\nIndicadores por Dominio (ponderado):")
print(analisis_geografico)
```

**[📚 See full tutorials with notebooks →](examples/)**

---

## ✨ Key Features

- 🎯 **One-line data loading** from INEI servers or local files
- 🔢 **60+ ENAHO modules** supported (all modules from 01 to 100, years 2015-2024)
- ⚖️ **Expansion factors** (factor07) for proper population estimates
- 🔗 **Smart module merging** at household/person/dwelling level
- 💾 **Intelligent caching** (save bandwidth & time on repeated downloads)
- 🧹 **Automatic data cleaning** (encodings, dtypes, nulls)
- 📊 **Multiple formats**: DBF, SPSS (.sav), Stata (.dta), CSV, Parquet
- 🗺️ **Geographic integration** with UBIGEO (departamento/provincia/distrito)
- 🕳️ **Missing data analysis** with ML-powered imputation strategies
- 🐍 **100% Python** - No R or external dependencies required

---

## 📦 Supported ENAHO Modules

### Most Common Modules

| Module | Description | Level | Years |
|--------|-------------|-------|-------|
| `01` | Características de la vivienda y del hogar | Hogar | 2015-2024 |
| `02` | Características de los miembros del hogar | Persona | 2015-2024 |
| `03` | Educación | Persona | 2015-2024 |
| `04` | Salud | Persona | 2015-2024 |
| `05` | Empleo e ingresos | Persona | 2015-2024 |
| `34` | Programas sociales, alimentación | Hogar | 2015-2024 |
| `37` | Gastos del hogar | Hogar | 2015-2024 |
| `85` | Sumaria de pobreza (línea de pobreza) | Hogar | 2015-2024 |
| `sumaria` | Indicadores agregados (gasto, ingreso, pobreza) | Hogar | 2015-2024 |

### Additional Modules Available

The library supports **all ENAHO modules** (01-100) across years 2015-2024, including:
- **Labor market**: Modules 05, 18 (informal sector)
- **Income & expenditure**: Modules 37, 85, sumaria
- **Social programs**: Module 34 (Juntos, Qali Warma, Pensión 65)
- **Housing infrastructure**: Module 01 (water, sanitation, electricity)
- **Education**: Module 03 (enrollment, literacy, school completion)
- **Health**: Module 04 (insurance, morbidity, healthcare access)

**[📋 See complete module reference →](docs/modules.md)**

---

## 💡 Real-World Examples

### Advanced: Poverty & Labor Market Analysis

Complete pipeline merging 6 modules to analyze the relationship between monetary poverty and labor market conditions.

**Files:**
- 📓 [`examples/investigacion/analisis_pob_mon_lab.ipynb`](examples/investigacion/analisis_pob_mon_lab.ipynb) - Interactive notebook
- 🐍 [`examples/investigacion/analisis_pob_mon_lab.py`](examples/investigacion/analisis_pob_mon_lab.py) - Reusable Python module

Key features demonstrated:
- Multi-module parallel downloads (01, 02, 03, 04, 05, 34)
- Smart module merging at household & person level
- Proper use of expansion factors (factor07)
- Labor informality indicators
- Weighted geographic analysis

**[🎓 Read the full tutorial on Medium →](https://medium.com/@pcamacho447)**

---

## 📚 More Examples

### Additional Use Cases

1. **[Geographic Inequality Analysis](examples/02_geographic_inequality_analysis.py)**
   - Merge with UBIGEO data
   - Regional poverty comparisons
   - Weighted statistics by department

2. **[Multi-module Household Analysis](examples/03_multimodule_analysis.py)**
   - Combine housing + education + health data
   - Create composite indicators
   - Panel analysis across years

3. **[ML-Powered Missing Data Imputation](examples/04_advanced_ml_imputation_demo.py)**
   - Detect missing patterns (MCAR, MAR, MNAR)
   - KNN and Random Forest imputation
   - Quality assessment metrics

4. **[Housing Quality Dashboard](examples/medium/caracteristicas_del_hogar.ipynb)**
   - Interactive visualizations
   - NBI (Necesidades Básicas Insatisfechas) analysis
   - Geographic disparities

**[📓 Browse all examples →](examples/)**

---

## 📖 Documentation

- **[Getting Started](docs/getting_started.rst)** - Installation & first steps
- **[User Guide](docs/tutorials/)** - Step-by-step tutorials
- **[API Reference](docs/api/)** - Complete API documentation
- **[FAQ](docs/faq.rst)** - Common questions answered
- **[Troubleshooting](docs/troubleshooting.rst)** - Solutions to common issues

---

## 🏗️ Package Architecture

```
enahopy/
├── loader/              # Data download and loading
│   ├── core/           # Configuration and exceptions
│   ├── io/             # Format readers (DTA, SAV, CSV, Parquet) and downloaders
│   └── utils/          # Utilities and helpers
├── merger/             # Module and geographic merging
│   ├── geographic/     # UBIGEO handling and validation
│   ├── modules/        # ENAHO module merging (01, 02, 05, 34, sumaria)
│   └── strategies/     # Merge strategies (household, person, panel)
└── null_analysis/      # Missing data analysis
    ├── core/          # Analysis engine and classification
    ├── patterns/      # Pattern detection (MCAR, MAR, MNAR)
    ├── strategies/    # Imputation strategies (mean, KNN, ML)
    └── reports/       # Report generation and visualizations
```

---

## 🔧 Advanced Configuration

### Cache & Performance

```python
from enahopy.loader import ENAHOConfig, ENAHODataDownloader

config = ENAHOConfig(
    cache_dir='.enaho_cache',
    enable_cache=True,
    max_workers=4,           # Parallel downloads
    chunk_size=50000,
    enable_validation=True
)

downloader = ENAHODataDownloader(config=config)

# First run: ~30 seconds (downloads from INEI)
# Second run: <1 second (reads from local cache)
```

### Strict Validation in Mergers

```python
from enahopy.merger import MergerConfig, ENAHOMerger

config = MergerConfig(
    validate_merge=True,
    strict_mode=True,
    allow_duplicates=False,
    validate_ubigeo=True
)

merger = ENAHOMerger(config=config)
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/elpapx/enahopy.git
cd enahopy

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=enahopy

# Code quality checks
black enahopy/ tests/
flake8 enahopy/
isort enahopy/ tests/
```

### CI/CD Status

All PRs are automatically validated:
- ✅ **Quality Checks**: black, flake8, isort
- ✅ **Multi-platform Tests**: Ubuntu, Windows, macOS
- ✅ **Python Matrix**: 3.8, 3.9, 3.10, 3.11, 3.12
- ✅ **Coverage**: Minimum 40% required
- ✅ **Build Validation**: PyPI packaging

---

## 📈 Roadmap

**Upcoming features:**
- [ ] ENDES support (Demographic and Family Health Survey)
- [ ] ENAPRES integration (National Survey of Budget Programs)
- [ ] Interactive Streamlit dashboard
- [ ] R format exports (RData, feather)
- [ ] Longitudinal analysis (multi-year panels)
- [ ] REST API for web services

---

## 👤 Author

**Pablo Camacho**

- 📝 **Medium**: [@pcamacho447](https://medium.com/@pcamacho447) - Tutorials and use cases
- 💻 **GitHub**: [@elpapx](https://github.com/elpapx)
- 📧 **Email**: pcamacho447@gmail.com

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **INEI (Instituto Nacional de Estadística e Informática)** for making microdata publicly available
- Peru's social research and data science community
- All contributors and users of this project

---

<p align="center">
  <strong>Made with ❤️ for social researchers and data scientists in Peru</strong>
</p>

<p align="center">
  <a href="https://en.wikipedia.org/wiki/Peru"><img src="https://img.shields.io/badge/Made%20in-Peru-red.svg" alt="Made in Peru"></a>
</p>
