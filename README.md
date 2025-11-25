<p align="center">
  <img src="assets/logo.jpg" alt="enahopy" width="500"/>
</p>

<h1 align="center">enahopy</h1>

<p align="center">
  <em>Kit de herramientas profesional en Python para analizar datos de la encuesta ENAHO del Perú</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/elpapx/enahopy/actions/workflows/ci.yml"><img src="https://github.com/elpapx/enahopy/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline"></a>
  <a href="https://codecov.io/gh/elpapx/enahopy"><img src="https://codecov.io/gh/elpapx/enahopy/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  <a href="#-por-qué-enahopy">Por qué</a> •
  <a href="#-instalación">Instalación</a> •
  <a href="#-inicio-rápido">Inicio Rápido</a> •
  <a href="#-características">Características</a> •
  <a href="#-módulos-soportados">Módulos</a> •
  <a href="examples/">Ejemplos</a>
</p>

---

## 🎯 ¿Por qué enahopy?

Transforma los datos de la encuesta ENAHO del Perú desde archivos ZIP sin procesar a DataFrames de pandas listos para análisis en **3 líneas de código**.

**Antes de enahopy** (50+ líneas de código repetitivo):
```python
# Descargar ZIP del sitio web de INEI
# Extraer archivos DBF manualmente
# Manejar múltiples codificaciones (CP1252/UTF-8)
# Unir módulos con claves apropiadas
# Aplicar factores de expansión correctamente
# Manejar datos faltantes...
# (50+ líneas más)
```

**Con enahopy** (3 líneas):
```python
import enahopy as enaho
loader = enaho.ENAHOLoader(year=2022)
df = loader.load_module("01")  # ¡Listo! 🎉
```

---

## 📦 Instalación

### Instalación básica
```bash
pip install enahopy
```

### Con todas las funcionalidades
```bash
pip install enahopy[all]
```

---

## 🚀 Inicio Rápido

### Ejemplo 1: Descargar y Cargar Datos
```python
from enahopy.loader import ENAHODataDownloader

# Inicializar descargador
downloader = ENAHODataDownloader(verbose=True)

# Descargar datos de características de la vivienda
data = downloader.download(
    modules=['01'],
    years=['2024'],
    output_dir='./data',
    load_dta=True
)

df_hogar = data[('2024', '01')]['enaho01-2024-100']
print(f"✓ Cargados {len(df_hogar):,} hogares")
```

### Ejemplo 2: Estadísticas Ponderadas (Profesional)
```python
import pandas as pd
import numpy as np

# Cargar módulo sumaria con indicadores de pobreza
df_sumaria = data[('2024', '34')]['sumaria-2024']

# ✅ CORRECTO: Estadísticas ponderadas usando factores de expansión
factor = df_sumaria['factor07']  # Factor de expansión

# Calcular tasa de pobreza ponderada
pobreza_rate = (
    (df_sumaria['pobreza'] <= 2) * factor  # 1=pobreza extrema, 2=pobre
).sum() / factor.sum() * 100

print(f"Tasa de pobreza (ponderada): {pobreza_rate:.2f}%")

# Análisis ponderado por dominio geográfico
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

**[📚 Ver tutoriales completos con notebooks →](examples/)**

---

## ✨ Características Principales

- 🎯 **Carga de datos en una línea** desde servidores INEI o archivos locales
- 🔢 **60+ módulos ENAHO** soportados (todos los módulos del 01 al 100, años 2015-2024)
- ⚖️ **Factores de expansión** (factor07) para estimaciones poblacionales apropiadas
- 🔗 **Unión inteligente de módulos** a nivel de hogar/persona/vivienda
- 💾 **Caché inteligente** (ahorra ancho de banda y tiempo en descargas repetidas)
- 🧹 **Limpieza automática de datos** (codificaciones, tipos de datos, nulos)
- 📊 **Múltiples formatos**: DBF, SPSS (.sav), Stata (.dta), CSV, Parquet
- 🗺️ **Integración geográfica** con UBIGEO (departamento/provincia/distrito)
- 🕳️ **Análisis de datos faltantes** con estrategias de imputación potenciadas por ML
- 🐍 **100% Python** - No requiere R ni dependencias externas

---

## 📦 Módulos ENAHO Soportados

### Módulos Más Comunes

| Módulo | Descripción | Nivel | Años |
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

### Módulos Adicionales Disponibles

La librería soporta **todos los módulos ENAHO** (01-100) a través de los años 2015-2024, incluyendo:
- **Mercado laboral**: Módulos 05, 18 (sector informal)
- **Ingresos y gastos**: Módulos 37, 85, sumaria
- **Programas sociales**: Módulo 34 (Juntos, Qali Warma, Pensión 65)
- **Infraestructura de vivienda**: Módulo 01 (agua, saneamiento, electricidad)
- **Educación**: Módulo 03 (matrícula, alfabetización, culminación escolar)
- **Salud**: Módulo 04 (seguro, morbilidad, acceso a servicios de salud)

---

## 💡 Ejemplos

Encuentra notebooks y scripts completos en el directorio [`examples/`](examples/):

### 📁 Investigación
- **[Análisis de Pobreza Monetaria y Mercado Laboral](examples/investigacion/)** - Pipeline completo uniendo 6 módulos ENAHO
  - [`analisis_pob_mon_lab.ipynb`](examples/investigacion/analisis_pob_mon_lab.ipynb) - Notebook interactivo
  - [`analisis_pob_mon_lab.py`](examples/investigacion/analisis_pob_mon_lab.py) - Script reutilizable
  - Uso apropiado de factores de expansión (factor07)
  - Análisis de informalidad laboral y pobreza

### 📁 Medium
- **[Características del Hogar](examples/medium/caracteristicas_del_hogar.ipynb)** - Dashboard de calidad de vivienda
  - Visualizaciones interactivas
  - Análisis NBI (Necesidades Básicas Insatisfechas)
  - Disparidades geográficas

**[🎓 Lee más tutoriales en Medium →](https://medium.com/@pcamacho447)**

---

## 🏗️ Arquitectura del Paquete

```
enahopy/
├── loader/              # Descarga y carga de datos
│   ├── core/           # Configuración y excepciones
│   ├── io/             # Lectores de formato (DTA, SAV, CSV, Parquet) y descargadores
│   └── utils/          # Utilidades y helpers
├── merger/             # Unión de módulos y geográfica
│   ├── geographic/     # Manejo y validación de UBIGEO
│   ├── modules/        # Unión de módulos ENAHO (01, 02, 05, 34, sumaria)
│   └── strategies/     # Estrategias de unión (hogar, persona, panel)
└── null_analysis/      # Análisis de datos faltantes
    ├── core/          # Motor de análisis y clasificación
    ├── patterns/      # Detección de patrones (MCAR, MAR, MNAR)
    ├── strategies/    # Estrategias de imputación (media, KNN, ML)
    └── reports/       # Generación de reportes y visualizaciones
```

---

## 🔧 Configuración Avanzada

### Caché y Rendimiento

```python
from enahopy.loader import ENAHOConfig, ENAHODataDownloader

config = ENAHOConfig(
    cache_dir='.enaho_cache',
    enable_cache=True,
    max_workers=4,           # Descargas paralelas
    chunk_size=50000,
    enable_validation=True
)

downloader = ENAHODataDownloader(config=config)

# Primera ejecución: ~30 segundos (descarga desde INEI)
# Segunda ejecución: <1 segundo (lee desde caché local)
```

### Validación Estricta en Uniones

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

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Ver [CONTRIBUTING.md](CONTRIBUTING.md) para lineamientos.

### Configuración de Desarrollo

```bash
# Clonar repositorio
git clone https://github.com/elpapx/enahopy.git
cd enahopy

# Instalar en modo desarrollo
pip install -e .[dev]

# Instalar hooks de pre-commit
pre-commit install

# Ejecutar tests
pytest tests/ -v --cov=enahopy

# Verificaciones de calidad de código
black enahopy/ tests/
flake8 enahopy/
isort enahopy/ tests/
```

### Estado CI/CD

Todos los PRs son automáticamente validados:
- ✅ **Verificaciones de Calidad**: black, flake8, isort
- ✅ **Tests Multi-plataforma**: Ubuntu, Windows, macOS
- ✅ **Matriz de Python**: 3.8, 3.9, 3.10, 3.11, 3.12
- ✅ **Cobertura**: Mínimo 40% requerido
- ✅ **Validación de Build**: Empaquetado PyPI

---

## 📈 Hoja de Ruta

**Próximas funcionalidades:**
- [ ] Soporte para ENDES (Encuesta Demográfica y de Salud Familiar)
- [ ] Integración con ENAPRES (Encuesta Nacional de Programas Estratégicos)
- [ ] Dashboard interactivo con Streamlit
- [ ] Exportación a formatos R (RData, feather)
- [ ] Análisis longitudinal (paneles multi-año)
- [ ] API REST para servicios web

---

## 👤 Autor

**Pablo Camacho**

- 📝 **Medium**: [@pcamacho447](https://medium.com/@pcamacho447) - Tutoriales y casos de uso
- 💻 **GitHub**: [@elpapx](https://github.com/elpapx)
- 📧 **Email**: pcamacho447@gmail.com

---

## 📄 Licencia

Licencia MIT - ver [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- **INEI (Instituto Nacional de Estadística e Informática)** por hacer los microdatos públicamente disponibles
- Comunidad de investigación social y ciencia de datos del Perú
- Todos los contribuidores y usuarios de este proyecto

---

<p align="center">
  <strong>Hecho con ❤️ para investigadores sociales y científicos de datos en el Perú</strong>
</p>

<p align="center">
  <a href="https://en.wikipedia.org/wiki/Peru"><img src="https://img.shields.io/badge/Hecho%20en-Perú-red.svg" alt="Hecho en Perú"></a>
</p>
