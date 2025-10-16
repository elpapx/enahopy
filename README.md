# ENAHOPY 🇵🇪

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/elpapx/enahopy/actions/workflows/ci.yml/badge.svg)](https://github.com/elpapx/enahopy/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/elpapx/enahopy/branch/main/graph/badge.svg)](https://codecov.io/gh/elpapx/enahopy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


**Librería integral para análisis de microdatos del INEI (Perú)**

Herramienta completa y robusta para descargar, procesar, analizar y visualizar microdatos de encuestas nacionales peruanas como ENAHO. Diseñada específicamente para investigadores, instituciones públicas y profesionales del análisis social en Perú.

##  Características Principales

- **Descarga Automática**: Descarga directa desde servidores oficiales del INEI
- **Multi-formato**: Compatible con DTA (Stata), SAV (SPSS), CSV, Parquet
- **Validación Inteligente**: Validación automática de columnas y mapeo de variables
- **Fusión de Módulos**: Sistema avanzado para combinar módulos ENAHO
- **Análisis Geográfico**: Integración con datos georreferenciados y ubigeos
- **️Análisis de Valores Nulos**: Detección de patrones y estrategias de imputación
- **Sistema de Cache**: Optimización automática de descargas repetidas
- **Alto Rendimiento**: Procesamiento paralelo para archivos grandes
- **Visualizaciones**: Gráficos especializados para datos de encuestas sociales
- **Reportes Automáticos**: Generación de reportes en múltiples formatos

## 📦 Instalación

### Instalación básica
```bash
pip install enahopy
```

## 🎯 Uso Rápido

### Descarga y lectura de datos ENAHO

```python
from enahopy.loader import download_enaho_data, read_enaho_file

# Descargar módulo específico
download_enaho_data(year=2023, modules=['01'], data_dir='datos_enaho')

# Leer archivo descargado
df_hogar = read_enaho_file('datos_enaho/2023/Enaho01-2023-200.dta')
print(f"Registros cargados: {len(df_hogar):,}")
```

### Fusión de módulos

```python
from enahopy.merger import merge_enaho_modules

# Combinar módulo de hogar con personas
df_combined = merge_enaho_modules(
    modules=['01', '02'],  # Hogar + Personas
    year=2023,
    level='persona'
)
```

### Análisis geográfico

```python
from enahopy.merger import merge_with_geography

# Agregar información geográfica
df_geo = merge_with_geography(
    df_combined,
    nivel='departamento',
    incluir_ubigeo=True
)
```

### Análisis de valores nulos

```python
from enahopy.null_analysis import ENAHONullAnalyzer

# Análisis completo de missing values
analyzer = ENAHONullAnalyzer(complexity='advanced')
result = analyzer.analyze(df_combined)

# Visualizar patrones
analyzer.plot_missing_patterns(save_path='missing_analysis.png')

# Generar reporte
analyzer.export_report(result, 'reporte_nulos.html')
```

## 🏗️ Arquitectura del Paquete

```
enaho_py/
├── loader/           # Descarga y lectura de datos
│   ├── core/        # Configuración y excepciones
│   ├── io/          # Readers y downloaders
│   └── utils/       # Utilidades y shortcuts
├── merger/          # Fusión de módulos y geografía
│   ├── geographic/  # Manejo de datos geográficos
│   ├── modules/     # Fusión entre módulos ENAHO
│   └── strategies/  # Estrategias de fusión
└── null_analysis/   # Análisis de valores faltantes
    ├── core/       # Motor de análisis
    ├── patterns/   # Detección de patrones
    ├── imputation/ # Estrategias de imputación
    └── reports/    # Generación de reportes
```

## 📊 Casos de Uso Típicos

### 1. Análisis de Pobreza Nacional
```python
from enaho_analyzer import ENAHOAnalyzer

# Inicializar analizador
enaho = ENAHOAnalyzer(year=2023)

# Cargar módulos necesarios
df = enaho.load_modules(['01', '02', '34'])  # Hogar, personas, ingresos

# Calcular indicadores de pobreza
poverty_stats = enaho.calculate_poverty_indicators(df)
print(poverty_stats.summary())
```

### 2. Análisis Regional Comparativo
```python
# Análisis por departamentos
regional_analysis = enaho.analyze_by_region(
    df, 
    indicators=['poverty_rate', 'gini_coefficient'],
    level='departamento'
)

# Visualizar resultados
regional_analysis.plot_map(indicator='poverty_rate')
```

### 3. Tendencias Temporales
```python
# Análisis multi-año
trends = enaho.analyze_trends(
    years=range(2019, 2024),
    modules=['01', '02'],
    indicators=['poverty', 'education', 'health']
)

trends.plot_evolution()
```

## 🔧 Configuración Avanzada

```python
from enaho_analyzer.loader.core import ENAHOConfig

# Configuración personalizada
config = ENAHOConfig(
    cache_dir='mi_cache',
    max_workers=8,
    chunk_size=50000,
    enable_validation=True
)

# Usar configuración en análisis
enaho = ENAHOAnalyzer(config=config)
```

## 📚 Ejemplos Prácticos

El repositorio incluye notebooks con ejemplos completos:

- [📊 Análisis de Pobreza Básico](examples/analisis_pobreza.ipynb)
- [🗺️ Mapeo de Indicadores Sociales](examples/mapeo_indicadores.ipynb)
- [🔗 Fusión Avanzada de Módulos](examples/fusion_modulos.ipynb)
- [🕳️ Tratamiento de Valores Faltantes](examples/missing_values.ipynb)

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Ver [CONTRIBUTING.md](CONTRIBUTING.md) para detalles completos.

### Proceso de desarrollo:
```bash
# Clonar repositorio
git clone https://github.com/elpapx/enahopy
cd enahopy

# Instalar en modo desarrollo con dependencias
pip install -e .[dev]

# Instalar pre-commit hooks
pre-commit install

# Ejecutar tests rápidos
pytest tests/ -m "not slow"

# Verificar estilo de código
black --check enahopy/ tests/
flake8 enahopy/
isort --check-only enahopy/ tests/

# Ejecutar suite completa con cobertura
pytest tests/ --cov=enahopy --cov-report=html
```

### Estado del CI/CD

Todos los PRs son automáticamente validados por:
- ✅ **Quality Checks**: black, flake8, isort
- ✅ **Multi-platform Tests**: Ubuntu, Windows, macOS
- ✅ **Python Matrix**: 3.8, 3.9, 3.10, 3.11, 3.12
- ✅ **Coverage**: Cobertura mínima 40% (objetivo: 60%)
- ✅ **Build Validation**: Empaquetado PyPI

## 📈 Roadmap

- [ ] Soporte para ENDES (Encuesta Demográfica y de Salud Familiar)
- [ ] Integración con ENAPRES (Encuesta Nacional de Programas Presupuestales Estratégicos)
- [ ] API REST para servicios web
- [ ] Dashboard interactivo con Streamlit
- [ ] Integración con R a través de reticulate
- [ ] Soporte para análisis longitudinal (paneles)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 📞 Soporte

- 📧 Email: pcamacho447@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/tu-usuario/enaho-analyzer/issues)
- 📖 Documentación: [enaho-analyzer.readthedocs.io](https://enaho-analyzer.readthedocs.io/)
- 💬 Discusiones: [GitHub Discussions](https://github.com/tu-usuario/enaho-analyzer/discussions)

## 🙏 Agradecimientos

- Instituto Nacional de Estadística e Informática (INEI) por la disponibilización de microdatos
- Comunidad científica peruana por feedback y contribuciones
- Contribuidores del proyecto y usuarios beta

---

**Desarrollado con ❤️ para la comunidad de investigación social en Perú**

[![Made in Peru](https://img.shields.io/badge/Made%20in-Peru-red.svg)](https://en.wikipedia.org/wiki/Peru)