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

## Por qué enahopy

Transforma los datos de la encuesta ENAHO del Perú desde archivos ZIP sin procesar a DataFrames de pandas listos para análisis en **en unas cuantas líneas de código**.

**Antes de enahopy** (muchos procesos diferentes):
```python
# Descargar ZIP del sitio web de INEI
# Extraer archivos DBF manualmente
# Manejar múltiples codificaciones (CP1252/UTF-8)
# Unir módulos con claves apropiadas
```

**Descarga con enahopy** (con enahopy en unas cuantas líneas):
```python
from enahopy.loader import ENAHODataDownloader

# Módulos a descargar
modulos_interes = {
    "01": "Caracteristica de la vivienda y del hogar",
    "34": "Sumarias ( Variables Calculadas )",
}

downloader = ENAHODataDownloader(verbose=True)

# Descarga múltiple
data_multi = downloader.download(
    modules=list(modulos_interes.keys()), # ["01", "34"] también funciona
    years=["2024"],                   # puedes descargar multiples años
    output_dir=r"\examples\medium\data",
    decompress=True,                  # Descomprime archivos ZIP  
    only_dta=True,                    # Descarga solo archivos dta  
    load_dta=True,                    # Carga datos en DataFrame pandas
    parallel=True,                    # ¡Descarga paralela!
    max_workers=2,                    # Puedes decidir cuantas
    verbose=False                     # Desactiva mensajes de estado  
)
```


---

## 📦 Instalación

### Instalación básica
```bash
pip install enahopy
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

### Ejemplo 2: Estadísticas Ponderadas 
```python
import pandas as pd
import numpy as np
from enahopy.loader import ENAHODataDownloader

# Inicializar descargador
downloader = ENAHODataDownloader(verbose=True)

# Descargar datos de características de la vivienda
data = downloader.download(
    modules=['34'],         # Puedes descargar multiples modulos    
    years=['2024'],         # Puedes descargar multiples años
    output_dir='./data',    # todo al mismo tiempo y en la misma carpeta
    load_dta=True           # Y cargarlo de paso en DataFrame pandas
)

# Cargar módulo sumaria con indicadores de pobreza
df_sumaria = data[('2024', '34')]['sumaria-2024'] # Una vez cargado, pasamos a trabajar

# Estadísticas ponderadas usando factores de expansión
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

### Ejemplo 3: Proceso completo incluido merge entre modulos
```
# ========== USANDO ENAHOPY ENAHO LOADER ==========
from enahopy.loader import ENAHODataDownloader
from enahopy.loader.io import ENAHOLocalReader # si necesitas leer el archivo descargado


# ========== USANDO ENAHOPY's ENAHOModuleMerger ==========
from enahopy.merger import ENAHOModuleMerger
from enahopy.merger.config import ModuleMergeConfig, ModuleMergeLevel
import logging


# ========== USANDO ENAHOPY ENAHO NULL_ANALYSIS ==========
from enahopy.null_analysis import ENAHONullAnalyzer

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')


# ========== CONFIGURACIÓN ENAHOModuleMerger ==========

# Configurar el merger para nivel individual (persona)
config = ModuleMergeConfig(merge_level=ModuleMergeLevel.PERSONA)
logger = logging.getLogger('enaho_merger')
merger = ENAHOModuleMerger(config, logger)


# Configurar el merger para nivel hogar
config_hogar = ModuleMergeConfig(merge_level=ModuleMergeLevel.HOGAR)
merger_hogar = ENAHOModuleMerger(config_hogar, logger)

# ========== DESCARGA DE DATOS ==========

# Seleccionar los módulos a descargar
modulos_interes = {
    "01": "Caracteristica de la vivienda y del hogar",
    "34": "Sumarias ( Variables Calculadas )",
}

# Inicializar descargador
downloader = ENAHODataDownloader(verbose=True)


# Iniciar descarga
data_multi = downloader.download(
    modules=list(modulos_interes.keys()),
    years=["2024"],
    output_dir=r"\examples\medium\data",
    decompress=True,
    only_dta=True,
    load_dta=True,
    parallel=True,                    
    max_workers=2,                  
    verbose=False
)

# ========== CARGA DE DATOS ==========

# Filtramos las preguntas que requerimos para nuestra investigación
sumaria_vars = ['conglome', 'vivienda', 'hogar', 'ubigeo',
    'pobreza','inghog2d', 'mieperho', 'dominio', 'estrato',
    'factor07']

carac_hogar_vars = [ 'conglome','vivienda', 'hogar', 'p101',
    'p102', 'p103', 'p103a', 'p104b1', 'p110', 'p111a', 'i105b',
    'nbi1', 'nbi2', 'nbi3', 'nbi4', 'nbi5']

# Filtramos los datasets que hemos descargado
data_caracteristica_vivienda = data_multi[('2024', '01')]['enaho01-2024-100']
data_sumaria = data_multi[('2024', '34')]['sumaria-2024']

# Filtrados los datasets, filtramos las variables que nos interesan
data_carac_viv = data_caracteristica_vivienda[carac_hogar_vars]
data_sum = data_sumaria[sumaria_vars]

# ========== MERGE DE DATOS ==========
print("\n" + "=" * 70)
print(" PASO 3: UNIR CON SUMARIA (BASE A NIVEL HOGAR) ".center(70))
print("=" * 70 + "\n")

print("Uniendo datos agregados con sumaria (módulo 34)...")
print(f"   Base (sumaria): {data_sum.shape[0]:,} hogares")
print(f"   Datos agregados: {data_carac_viv.shape[0]:,} hogares")


# Realizar el merge usando enahopy
print("\n Usando ENAHOModuleMerger de enahopy para fusionar sumaria con caracteristicas de la vivienda y hogar...")
merge_result_hogar = merger_hogar.merge_modules(
    left_df=data_sum,           # Sumaria como base (left)
    right_df=data_carac_viv,         # Datos agregados (right)
    left_module='34',           # Módulo sumaria
    right_module='01',    # Identificador para datos agregados
    merge_config=config_hogar
)
```

**[📚 Ver tutoriales completos con notebooks →](examples/)**

---

## ✨ Características Principales

- 🎯 **Carga de datos en una línea** desde servidores INEI o archivos locales
- 🔢 **20+ módulos ENAHO** soportados (todos los módulos del 01 al 100)
- 🔗 **Unión inteligente de módulos** a nivel de vivienda/hogar/persona
- 💾 **Caché inteligente** (ahorra ancho de banda y tiempo en descargas repetidas)
- 🧹 **Limpieza automática de datos** (codificaciones, tipos de datos, nulos)
- 📊 **Múltiples formatos**: DBF, SPSS (.sav), Stata (.dta), CSV, Parquet
- 🗺️ **Integración geográfica** con UBIGEO (departamento/provincia/distrito)


---

## 📦 Módulos ENAHO Soportados

### Módulos Más Comunes

| Módulo | Descripción                                | 
|--------|--------------------------------------------|
| `01`   | Características de la vivienda y del hogar |  
| `02`   | Características de los miembros del hogar  |  
| `03`   | Educación                                  |  
| `04`   | Salud                                      |  
| `05`   | Empleo e ingresos                          |  
| `22`   | Producción Agrícola                        |  
| `34`   | Sumaria (Variables Calculadas)             |
| `37`   | Programas Sociales (Miembros del Hogar)    |  
| `85`   | Gobernabilidad, Democracia y Transparencia | 
  

La librería soporta **todos los módulos ENAHO** (01-100) a través de los años que proporciona el INEI. Tanto como 
corte transversal.

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
- ✅ **Cobertura**: Mínimo 60% requerido
- ✅ **Validación de Build**: Empaquetado PyPI

---

## 📈 Hoja de Ruta

**Próximas funcionalidades:**
- Diseño Muestral
- Metadata
- Análisis longitudinal (paneles multi-año)
- Análisis avanzado

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
