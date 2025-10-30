# ENAHOPY 🇵🇪

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/elpapx/enahopy/actions/workflows/ci.yml/badge.svg)](https://github.com/elpapx/enahopy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/elpapx/enahopy/branch/main/graph/badge.svg)](https://codecov.io/gh/elpapx/enahopy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Librería Python para análisis de microdatos ENAHO del INEI (Perú)**

Herramienta completa y robusta para descargar, procesar, fusionar y analizar microdatos de la Encuesta Nacional de Hogares (ENAHO). Diseñada específicamente para investigadores, analistas y profesionales que trabajan con datos sociales y económicos del Perú.

## ✨ Características Principales

- **📥 Descarga Automática**: Descarga directa desde servidores oficiales del INEI
- **📊 Multi-formato**: Compatible con DTA (Stata), SAV (SPSS), CSV, Parquet
- **✅ Validación Inteligente**: Validación automática de columnas y mapeo de variables
- **🔗 Fusión de Módulos**: Sistema avanzado para combinar módulos ENAHO (hogar, personas, ingresos)
- **🗺️ Análisis Geográfico**: Integración con datos UBIGEO (departamento/provincia/distrito)
- **🕳️ Análisis de Valores Nulos**: Detección de patrones missing y estrategias de imputación
- **⚡ Sistema de Cache**: Optimización automática de descargas repetidas
- **🚀 Alto Rendimiento**: Procesamiento eficiente con bajo uso de memoria
- **📈 Visualizaciones**: Gráficos especializados para datos de encuestas sociales

## 📦 Instalación

### Instalación básica
```bash
pip install enahopy
```

### Instalación con todas las características
```bash
pip install enahopy[all]
```

## 🚀 Inicio Rápido

### 1. Descargar datos ENAHO

```python
from enahopy.loader import ENAHODataDownloader

# Inicializar downloader
downloader = ENAHODataDownloader(verbose=True)

# Descargar múltiples módulos en paralelo
data_multi = downloader.download(
    modules=['01', '02', '05', '34'],  # Hogar, Persona, Empleo, Sumaria
    years=['2024'],
    output_dir='./data',
    decompress=True,
    load_dta=True,
    parallel=True,
    max_workers=3
)

# Extraer datasets
df_hogar = data_multi[('2024', '01')]['enaho01-2024-100']
df_persona = data_multi[('2024', '02')]['enaho01-2024-200']
df_empleo = data_multi[('2024', '05')]['enaho01a-2024-500']
df_sumaria = data_multi[('2024', '34')]['sumaria-2024']

print(f"✓ Hogares: {len(df_hogar):,} registros")
print(f"✓ Personas: {len(df_persona):,} registros")
```

### 2. Fusionar módulos

```python
from enahopy.merger import ENAHOModuleMerger
from enahopy.merger.config import ModuleMergeConfig, ModuleMergeLevel

# Configurar merger para nivel persona
config = ModuleMergeConfig(merge_level=ModuleMergeLevel.PERSONA)
merger = ENAHOModuleMerger(config)

# Fusionar módulos individuales
modules_dict = {
    '02': df_persona,
    '05': df_empleo
}

result = merger.merge_multiple_modules(
    modules_dict=modules_dict,
    base_module='02',
    merge_config=config
)

df_merged = result.merged_df
print(f"✓ Fusionado: {df_merged.shape}")
```

### 3. Análisis geográfico

```python
from enahopy.merger.geographic.merger import GeographicMerger
import pandas as pd

# Cargar tabla UBIGEO
df_ubigeo = pd.read_excel('UBIGEO_2024.xlsx')
df_ubigeo = df_ubigeo[['IDDIST', 'NOMBDEP', 'NOMBPROV', 'NOMBDIST']]
df_ubigeo = df_ubigeo.rename(columns={'IDDIST': 'ubigeo'})

# Fusionar con datos geográficos
merger_geo = GeographicMerger()
df_geo, report = merger_geo.merge(df_sumaria, df_ubigeo, columna_union='ubigeo')

print(f"✓ Registros con geografía: {report['output_rows']}")
```

### 4. Análisis de valores nulos

```python
from enahopy.null_analysis import ENAHONullAnalyzer

# Análisis completo de valores faltantes
analyzer = ENAHONullAnalyzer(complexity='advanced')
result = analyzer.analyze(df_merged)

# Generar reporte HTML
analyzer.export_report(result, 'reporte_nulos.html')
```

## 💡 Ejemplos Prácticos

### 🏆 Análisis Completo de Pobreza Monetaria y Laboral (Avanzado)

Este ejemplo muestra un **pipeline completo** de análisis fusionando 6 módulos ENAHO para estudiar la relación entre pobreza monetaria y características laborales.

**Archivos del ejemplo:**
- 📓 [`examples/investigacion/analisis_pob_mon_lab.ipynb`](examples/investigacion/analisis_pob_mon_lab.ipynb) - Notebook interactivo con análisis completo
- 🐍 [`examples/investigacion/analisis_pob_mon_lab.py`](examples/investigacion/analisis_pob_mon_lab.py) - Módulo Python con funciones de ingeniería de características

> **📝 Nota**: El módulo `.py` contiene toda la lógica de procesamiento, cálculo de variables derivadas, y transformaciones. El notebook lo importa para mantener el código organizado y reutilizable.

**Funciones principales del módulo `analisis_pob_mon_lab.py`:**
- `pipeline_completo()`: Ejecuta todo el flujo de procesamiento end-to-end
- `crear_caracteristicas_individuales()`: Calcula variables derivadas (informalidad, seguro de salud, grupos etarios, etc.)
- `extraer_jefe_hogar_completo()`: Extrae características del jefe de hogar (educación, ocupación, sector económico)
- `agregar_a_nivel_hogar()`: Agrega datos individuales a nivel hogar
- `calcular_variables_objetivo()`: Calcula pobreza monetaria y laboral
- `analisis_descriptivo_ponderado()`: Análisis con pesos muestrales (factor de expansión)

```python
from enahopy.loader import ENAHODataDownloader
from enahopy.merger import ENAHOModuleMerger
from enahopy.merger.config import ModuleMergeConfig, ModuleMergeLevel
import pandas as pd

# Importar funciones del módulo auxiliar
from analisis_pob_mon_lab import pipeline_completo, analisis_descriptivo_ponderado

# ==================================================
# PASO 1: Descargar múltiples módulos en paralelo
# ==================================================

downloader = ENAHODataDownloader(verbose=True)

# Descargar 6 módulos simultáneamente
data_multi = downloader.download(
    modules=['01', '02', '03', '04', '05', '34'],  # Hogar, Persona, Educación, Salud, Empleo, Sumaria
    years=['2024'],
    output_dir='./data',
    parallel=True,
    max_workers=3,
    load_dta=True
)

# Extraer datasets individuales
df_hogar = data_multi[('2024', '01')]['enaho01-2024-100']
df_persona = data_multi[('2024', '02')]['enaho01-2024-200']
df_educacion = data_multi[('2024', '03')]['enaho01a-2024-300']
df_salud = data_multi[('2024', '04')]['enaho01a-2024-400']
df_empleo = data_multi[('2024', '05')]['enaho01a-2024-500']
df_sumaria = data_multi[('2024', '34')]['sumaria-2024']

print(f"✓ Datasets descargados:")
print(f"  • Hogares: {df_hogar.shape[0]:,} registros")
print(f"  • Personas: {df_persona.shape[0]:,} registros")

# ==================================================
# PASO 2: Fusionar módulos a nivel PERSONA
# ==================================================

# Configurar merger para nivel persona
config_persona = ModuleMergeConfig(merge_level=ModuleMergeLevel.PERSONA)
merger_persona = ENAHOModuleMerger(config_persona)

# Fusionar módulos individuales (persona + educación + salud + empleo)
modules_dict = {
    '02': df_persona,
    '03': df_educacion,
    '04': df_salud,
    '05': df_empleo
}

merge_result = merger_persona.merge_multiple_modules(
    modules_dict=modules_dict,
    base_module='02',
    merge_config=config_persona
)

df_individuos = merge_result.merged_df
print(f"✓ Módulos individuales fusionados: {df_individuos.shape}")

# ==================================================
# PASO 3: Fusionar módulos a nivel HOGAR
# ==================================================

config_hogar = ModuleMergeConfig(merge_level=ModuleMergeLevel.HOGAR)
merger_hogar = ENAHOModuleMerger(config_hogar)

# Fusionar sumaria + características de hogar
merge_hogar = merger_hogar.merge_modules(
    left_df=df_sumaria,
    right_df=df_hogar,
    left_module='34',
    right_module='01',
    merge_config=config_hogar
)

df_hogares = merge_hogar.merged_df
print(f"✓ Módulos de hogar fusionados: {df_hogares.shape}")

# ==================================================
# PASO 4: Análisis de pobreza monetaria y laboral
# ==================================================

# Calcular informalidad laboral (ocupados sin contrato ni beneficios)
df_individuos['es_informal'] = (
    (df_individuos['ocu500'] == 1) &  # Ocupado
    (df_individuos['p507'].isin([7, 8, 9]))  # Sin contrato
).astype(int)

# Agregar a nivel hogar
df_hogar_stats = df_individuos.groupby(['conglome', 'vivienda', 'hogar']).agg({
    'codperso': 'count',  # Personas en el hogar
    'es_informal': 'sum'  # Informales en el hogar
}).rename(columns={'codperso': 'n_personas', 'es_informal': 'n_informales'})

# Fusionar con datos de hogar
df_final = df_hogares.merge(df_hogar_stats, on=['conglome', 'vivienda', 'hogar'])

# Calcular tasa de informalidad por hogar
df_final['tasa_informalidad'] = df_final['n_informales'] / df_final['n_personas']

# Clasificar pobreza monetaria (usando línea de pobreza INEI)
df_final['es_pobre_monetario'] = (df_final['pobreza'] <= 2).astype(int)  # 1=pobre extremo, 2=pobre

# Clasificar pobreza laboral (alta informalidad + bajos ingresos)
df_final['pobreza_laboral'] = (
    (df_final['tasa_informalidad'] >= 0.5) &
    (df_final['inghog2d'] < df_final['inghog2d'].quantile(0.4))
).astype(int)

# ==================================================
# PASO 5: Análisis descriptivo
# ==================================================

# Resumen general
print(f"\n{'='*60}")
print(f"RESULTADOS DEL ANÁLISIS")
print(f"{'='*60}")
print(f"Total de hogares analizados: {len(df_final):,}")
print(f"\nPobreza Monetaria:")
print(f"  • Hogares pobres: {df_final['es_pobre_monetario'].sum():,} ({df_final['es_pobre_monetario'].mean()*100:.1f}%)")
print(f"\nPobreza Laboral:")
print(f"  • Hogares con pobreza laboral: {df_final['pobreza_laboral'].sum():,} ({df_final['pobreza_laboral'].mean()*100:.1f}%)")
print(f"\nInformalidad:")
print(f"  • Tasa promedio de informalidad: {df_final['tasa_informalidad'].mean()*100:.1f}%")

# Análisis por dominio geográfico
analisis_geografico = df_final.groupby('dominio').agg({
    'es_pobre_monetario': 'mean',
    'pobreza_laboral': 'mean',
    'tasa_informalidad': 'mean',
    'inghog2d': 'median'
}).round(3) * 100

print(f"\n{'='*60}")
print("Indicadores por Dominio Geográfico (%)")
print(f"{'='*60}")
print(analisis_geografico)

# Exportar dataset final
df_final.to_csv('analisis_pobreza_completo_2024.csv', index=False)
print(f"\n✓ Dataset final guardado: {df_final.shape[0]:,} hogares × {df_final.shape[1]} variables")
```

**Output esperado:**
```
✓ Datasets descargados:
  • Hogares: 33,691 registros
  • Personas: 117,755 registros
✓ Módulos individuales fusionados: (117755, 45)
✓ Módulos de hogar fusionados: (33691, 28)

============================================================
RESULTADOS DEL ANÁLISIS
============================================================
Total de hogares analizados: 33,691

Pobreza Monetaria:
  • Hogares pobres: 6,812 (20.2%)

Pobreza Laboral:
  • Hogares con pobreza laboral: 16,352 (48.5%)

Informalidad:
  • Tasa promedio de informalidad: 72.9%

✓ Dataset final guardado: 33,691 hogares × 68 variables
```

**Código completo del ejemplo:** Ver [`examples/investigacion/analisis_pob_mon_lab.ipynb`](examples/investigacion/analisis_pob_mon_lab.ipynb) y [`examples/investigacion/analisis_pob_mon_lab.py`](examples/investigacion/analisis_pob_mon_lab.py)

---

### 🗺️ Merge Geográfico con UBIGEO

Ejemplo simple de cómo agregar información geográfica (departamento, provincia, distrito) usando la tabla UBIGEO oficial del INEI.

**Archivo:** [`examples/investigacion/merger_enahopy_geografico.py`](examples/investigacion/merger_enahopy_geografico.py)

```python
from enahopy.merger.geographic.merger import GeographicMerger
import pandas as pd

# Cargar datos finales del análisis
df = pd.read_csv('dataframe_final_2024.csv')

# Cargar tabla UBIGEO oficial (1,891 distritos)
df_ubigeo = pd.read_excel('UBIGEO_2022_1891_distritos.xlsx')
df_ubigeo = df_ubigeo[['IDDIST', 'NOMBDEP', 'NOMBPROV', 'NOMBDIST']]
df_ubigeo = df_ubigeo.rename(columns={'IDDIST': 'ubigeo'})
df_ubigeo = df_ubigeo.dropna()

# Fusionar con GeographicMerger
merger = GeographicMerger()
df_geo, report = merger.merge(df, df_ubigeo, columna_union='ubigeo')

print(f"✓ Registros con geografía: {report['output_rows']}")
print(f"✓ Columnas agregadas: NOMBDEP, NOMBPROV, NOMBDIST")

# Guardar resultado
df_geo.to_csv('dataframe_final_completo_geografico_2024.csv', index=False)
```

**Output:**
```
✓ Registros con geografía: 33,691
✓ Columnas agregadas: NOMBDEP, NOMBPROV, NOMBDIST
```

## 🏗️ Arquitectura del Paquete

```
enahopy/
├── loader/              # Descarga y lectura de datos ENAHO
│   ├── core/           # Configuración y excepciones
│   ├── io/             # Readers (DTA, SAV, CSV, Parquet) y downloaders
│   └── utils/          # Utilidades y funciones auxiliares
├── merger/             # Fusión de módulos y datos geográficos
│   ├── geographic/     # Manejo de UBIGEO y validación geográfica
│   ├── modules/        # Fusión entre módulos ENAHO (01, 02, 05, 34, sumaria)
│   └── strategies/     # Estrategias de fusión (hogar, persona, panel)
└── null_analysis/      # Análisis de valores faltantes
    ├── core/          # Motor de análisis y clasificación
    ├── patterns/      # Detección de patrones (MCAR, MAR, MNAR)
    ├── strategies/    # Estrategias de imputación (media, KNN, ML)
    └── reports/       # Generación de reportes y visualizaciones
```

## 📚 Módulos ENAHO Soportados

| Módulo | Descripción | Nivel |
|--------|-------------|-------|
| `01` | Características de la vivienda y del hogar | Hogar |
| `02` | Características de los miembros del hogar | Persona |
| `03` | Educación | Persona |
| `04` | Salud | Persona |
| `05` | Empleo e ingresos | Persona |
| `34` | Programas sociales | Hogar |
| `37` | Gastos del hogar | Hogar |
| `sumaria` | Indicadores agregados (gasto, ingreso, pobreza) | Hogar |

## 🔧 Configuración Avanzada

### Cache y Performance

```python
from enahopy.loader import ENAHOConfig, ENAHODataDownloader

# Configuración con cache habilitado
config = ENAHOConfig(
    cache_dir='.enaho_cache',       # Directorio de cache
    enable_cache=True,               # Habilitar sistema de cache
    max_workers=4,                   # Workers para descarga paralela
    chunk_size=50000,                # Tamaño de chunks para lectura
    enable_validation=True           # Validar columnas al cargar
)

downloader = ENAHODataDownloader(config=config)

# Primera descarga: ~30 segundos (descarga desde INEI)
# Segunda descarga: <1 segundo (lee desde cache local)
df = downloader.download_module(year=2023, module='sumaria', format='dta')
```

### Validación Estricta en Mergers

```python
from enahopy.merger import MergerConfig, ENAHOMerger

# Configuración con validación estricta
config = MergerConfig(
    validate_merge=True,      # Validar antes de fusionar
    strict_mode=True,         # Modo estricto (falla si hay errores)
    allow_duplicates=False,   # No permitir duplicados
    validate_ubigeo=True      # Validar códigos UBIGEO
)

merger = ENAHOMerger(config=config)
```

## 📊 Ejemplos y Tutoriales

El repositorio incluye scripts de demostración completos:

- [`01_complete_poverty_analysis.py`](examples/01_complete_poverty_analysis.py) - Análisis end-to-end de pobreza monetaria
- [`02_geographic_inequality_analysis.py`](examples/02_geographic_inequality_analysis.py) - Desigualdad territorial con UBIGEO
- [`03_multimodule_analysis.py`](examples/03_multimodule_analysis.py) - Fusión avanzada de múltiples módulos
- [`04_advanced_ml_imputation_demo.py`](examples/04_advanced_ml_imputation_demo.py) - Imputación con Machine Learning

También hay Jupyter notebooks en `examples/`:
- [`tutorial_01_loader.ipynb`](examples/tutorial_01_loader.ipynb) - Descarga y lectura básica
- [`tutorial_02_merger.ipynb`](examples/tutorial_02_merger.ipynb) - Fusión de módulos
- [`tutorial_03_null_analysis.ipynb`](examples/tutorial_03_null_analysis.ipynb) - Análisis de valores nulos

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Ver [CONTRIBUTING.md](CONTRIBUTING.md) para detalles completos.

### Setup de desarrollo:
```bash
# Clonar repositorio
git clone https://github.com/elpapx/enahopy.git
cd enahopy

# Instalar en modo desarrollo con dependencias
pip install -e .[dev]

# Instalar pre-commit hooks
pre-commit install

# Ejecutar tests
pytest tests/ -v

# Verificar estilo de código
black enahopy/ tests/
flake8 enahopy/
isort enahopy/ tests/

# Generar reporte de cobertura
pytest tests/ --cov=enahopy --cov-report=html
```

### Estado del CI/CD

Todos los PRs son validados automáticamente:
- ✅ **Quality Checks**: black, flake8, isort
- ✅ **Multi-platform Tests**: Ubuntu, Windows, macOS
- ✅ **Python Matrix**: 3.8, 3.9, 3.10, 3.11, 3.12
- ✅ **Coverage**: Cobertura mínima 40%
- ✅ **Build Validation**: Empaquetado PyPI

## 📈 Roadmap

**Próximas características:**
- [ ] Soporte para ENDES (Encuesta Demográfica y de Salud Familiar)
- [ ] Integración con ENAPRES (Encuesta Nacional de Programas Presupuestales)
- [ ] Dashboard interactivo con Streamlit
- [ ] Exportación a formatos R (RData, feather)
- [ ] Análisis longitudinal (paneles multi-año)
- [ ] API REST para servicios web

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 📞 Soporte

- 📧 Email: pcamacho447@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/elpapx/enahopy/issues)
- 💬 Discusiones: [GitHub Discussions](https://github.com/elpapx/enahopy/discussions)

## 🙏 Agradecimientos

- **INEI (Instituto Nacional de Estadística e Informática)** por la disponibilización de microdatos públicos
- Comunidad de investigadores y analistas de datos sociales en Perú
- Todos los contribuidores y usuarios del proyecto

---

**Desarrollado con ❤️ para la comunidad de investigación social y económica en Perú**

[![Made in Peru](https://img.shields.io/badge/Made%20in-Peru-red.svg)](https://en.wikipedia.org/wiki/Peru)