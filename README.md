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

### 1. Descargar y leer datos ENAHO

```python
from enahopy.loader import download_enaho_data, read_enaho_file

# Descargar módulo sumaria (información del hogar)
download_enaho_data(year=2023, modules=['sumaria'], data_dir='datos_enaho')

# Leer archivo descargado
df_hogar = read_enaho_file('datos_enaho/2023/sumaria-2023.dta')
print(f"Registros cargados: {len(df_hogar):,}")
```

### 2. Fusionar múltiples módulos

```python
from enahopy.merger import merge_enaho_modules

# Combinar módulo de hogar con personas
df_merged = merge_enaho_modules(
    modules=['01', '02'],  # 01: Características del hogar, 02: Personas
    year=2023,
    level='persona'
)

print(f"Dimensiones: {df_merged.shape}")
print(f"Columnas disponibles: {list(df_merged.columns[:10])}")
```

### 3. Análisis con información geográfica

```python
from enahopy.merger.geographic import merge_with_geography

# Agregar información geográfica (departamento, provincia, distrito)
df_geo = merge_with_geography(
    df_merged,
    nivel='departamento',
    incluir_ubigeo=True
)

# Análisis por departamento
stats_by_dept = df_geo.groupby('departamento').agg({
    'ingreso': ['mean', 'median'],
    'gasto': ['mean', 'sum']
})
```

### 4. Análisis de valores nulos

```python
from enahopy.null_analysis import ENAHONullAnalyzer

# Análisis completo de valores faltantes
analyzer = ENAHONullAnalyzer(complexity='advanced')
result = analyzer.analyze(df_merged)

# Visualizar patrones
analyzer.plot_missing_patterns(save_path='missing_analysis.png')

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

> **💡 Tip**: Este ejemplo completo está disponible como notebook interactivo en [`examples/investigacion/analisis_pob_mon_lab.ipynb`](examples/investigacion/analisis_pob_mon_lab.ipynb) con análisis adicionales, visualizaciones y modelos estadísticos.

---

### Análisis de Pobreza Monetaria (Básico)

```python
from enahopy.loader import ENAHODataDownloader, ENAHOConfig
import pandas as pd

# Configurar descarga con cache
config = ENAHOConfig(cache_dir='.enaho_cache', enable_cache=True)
downloader = ENAHODataDownloader(config=config)

# Descargar módulo sumaria (contiene gasto per cápita)
df_sumaria = downloader.download_module(year=2023, module='sumaria', format='dta')

# Líneas de pobreza INEI 2023 (soles mensuales per cápita)
POVERTY_LINE_EXTREME = 201
POVERTY_LINE_TOTAL = 378

# Clasificar hogares según línea de pobreza
df_sumaria['categoria_pobreza'] = pd.cut(
    df_sumaria['gashog2d'],  # Gasto per cápita mensual
    bins=[0, POVERTY_LINE_EXTREME, POVERTY_LINE_TOTAL, float('inf')],
    labels=['Pobre Extremo', 'Pobre', 'No Pobre']
)

# Calcular tasas de pobreza
pobreza_stats = df_sumaria['categoria_pobreza'].value_counts(normalize=True) * 100
print("\nTasas de Pobreza 2023:")
print(pobreza_stats)
```

### Análisis Geográfico de Desigualdad

```python
from enahopy.merger import ENAHOMerger, MergerConfig

# Configurar fusión geográfica
merger_config = MergerConfig(validate_merge=True, strict_mode=True)
merger = ENAHOMerger(config=merger_config)

# Fusionar sumaria con módulo 01 (información geográfica)
df_geo = merger.merge_modules(
    left_df=df_sumaria,
    right_df=df_hogar,
    left_module='sumaria',
    right_module='01',
    level='hogar'
)

# Extraer código de departamento del UBIGEO
df_geo['cod_depto'] = df_geo['ubigeo'].astype(str).str[:2]

# Mapeo de códigos a nombres
DEPARTAMENTOS = {
    '01': 'Amazonas', '02': 'Áncash', '03': 'Apurímac', '04': 'Arequipa',
    '05': 'Ayacucho', '06': 'Cajamarca', '07': 'Callao', '08': 'Cusco',
    '09': 'Huancavelica', '10': 'Huánuco', '11': 'Ica', '12': 'Junín',
    '13': 'La Libertad', '14': 'Lambayeque', '15': 'Lima', '16': 'Loreto',
    '17': 'Madre de Dios', '18': 'Moquegua', '19': 'Pasco', '20': 'Piura',
    '21': 'Puno', '22': 'San Martín', '23': 'Tacna', '24': 'Tumbes',
    '25': 'Ucayali'
}
df_geo['departamento'] = df_geo['cod_depto'].map(DEPARTAMENTOS)

# Análisis de desigualdad por departamento
desigualdad_dept = df_geo.groupby('departamento').agg({
    'gashog2d': ['mean', 'std', 'count'],
    'ingreso': 'mean'
}).round(2)

print("\nDesigualdad por Departamento (Top 5 gasto promedio):")
print(desigualdad_dept.sort_values(('gashog2d', 'mean'), ascending=False).head())
```

### Análisis de Mercado Laboral

```python
# Descargar módulo 05 (empleo e ingresos)
df_empleo = downloader.download_module(year=2023, module='05', format='dta')

# Fusionar con módulo 02 (características de personas)
df_personas = downloader.download_module(year=2023, module='02', format='dta')

merger = ENAHOMerger()
df_laboral = merger.merge_modules(
    left_df=df_personas,
    right_df=df_empleo,
    left_module='02',
    right_module='05',
    level='persona'
)

# Calcular tasa de ocupación por grupo de edad
df_laboral['grupo_edad'] = pd.cut(
    df_laboral['edad'],
    bins=[14, 24, 34, 44, 54, 64, 100],
    labels=['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
)

ocupacion_por_edad = df_laboral.groupby('grupo_edad')['ocupado'].mean() * 100
print("\nTasa de Ocupación por Grupo de Edad:")
print(ocupacion_por_edad)
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