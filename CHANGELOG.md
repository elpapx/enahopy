# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### 🎉 Lanzamiento Inicial

Primera versión estable de ENAHOPY disponible en PyPI.

#### ✨ Características Principales

**Módulo Loader:**
- Descarga automática desde servidores oficiales del INEI
- Soporte multi-formato: DTA (Stata), SAV (SPSS), CSV, Parquet
- Sistema de cache inteligente para optimizar descargas
- Validación automática de columnas con mapeo ENAHO
- Lectura por chunks para archivos grandes
- API unificada para todos los formatos

**Módulo Merger:**
- Fusión avanzada entre módulos ENAHO (hogar, personas, ingresos)
- Integración con datos geográficos y ubigeos
- Validación de compatibilidad entre módulos
- Manejo inteligente de duplicados y conflictos
- Soporte para análisis multinivel (vivienda, hogar, persona)

**Módulo Null Analysis:**
- Detección automática de patrones de valores faltantes
- Análisis estadístico avanzado (MCAR, MAR, MNAR)
- Visualizaciones especializadas con matplotlib, seaborn y plotly
- Estrategias de imputación múltiple
- Reportes automatizados en HTML y Excel

#### 🛠️ Funcionalidades Técnicas

- **Performance**: Procesamiento paralelo con dask
- **Robustez**: Manejo de errores y logging estructurado
- **Extensibilidad**: Arquitectura modular y pluggable
- **Testing**: Cobertura completa de tests unitarios e integración
- **Documentación**: README detallado y ejemplos prácticos

#### 📦 Dependencias

**Obligatorias:**
- pandas >= 1.3.0
- numpy >= 1.20.0
- requests >= 2.25.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

**Opcionales:**
- pyreadstat >= 1.1.0 (archivos DTA/SAV)
- dask >= 2021.0.0 (big data)
- geopandas >= 0.10.0 (análisis geográfico)
- plotly >= 5.0.0 (visualizaciones interactivas)

#### 🎯 Casos de Uso Soportados

- Análisis de pobreza y desigualdad
- Estudios demográficos y socioeconómicos
- Investigación académica con microdatos INEI
- Generación de indicadores para políticas públicas
- Análisis geoespacial de condiciones de vida
- Estudios longitudinales y comparativos

#### 📊 Datos Compatibles

- **ENAHO**: Encuesta Nacional de Hogares (2007-2023)
- **ENDES**: Preparación para futura compatibilidad
- **ENAPRES**: Preparación para futura compatibilidad
- Formatos: DTA, SAV, CSV, Parquet

#### 🐛 Problemas Conocidos

- Archivos ENAHO anteriores a 2007 requieren validación manual
- Algunos módulos especiales (37) necesitan tratamiento específico
- Performance limitada en sistemas con < 4GB RAM para archivos grandes

#### 🙏 Agradecimientos

- Instituto Nacional de Estadística e Informática (INEI)
- Comunidad de investigadores sociales en Perú
- Contribuidores beta testers

---

## [Próximas Versiones]

### [1.1.0] - Planificado para Q4 2025

#### 🔮 Características Planificadas

- **Soporte ENDES**: Módulo completo para Encuesta Demográfica
- **API REST**: Servicio web para análisis remoto
- **Dashboard**: Interface web interactiva con Streamlit
- **R Integration**: Wrapper para uso desde R
- **Análisis Longitudinal**: Herramientas para paneles de datos

#### 🚀 Mejoras Planificadas

- Optimización de memoria para archivos > 1GB
- Caché distribuido para equipos de trabajo
- Exportación a formatos adicionales (HDF5, Feather)
- Integración con bases de datos (PostgreSQL, MongoDB)
- Análisis automatizado con machine learning

### [1.2.0] - Planificado para Q1 2026

#### 📈 Funcionalidades Avanzadas

- **ENAPRES Support**: Encuesta Nacional de Programas Presupuestales
- **Análisis Causal**: Herramientas de inferencia causal
- **Microsimulación**: Modelos de simulación de políticas
- **Time Series**: Análisis de series temporales para indicadores
- **Spatial Analysis**: Análisis espacial avanzado con autocorrelación

---

## Soporte y Contribuciones

- 🐛 **Reportar bugs**: [GitHub Issues](https://github.com/elpapx/enahopy/issues)
- 💡 **Solicitar features**: [GitHub Discussions](https://github.com/elpapx/enahopy/discussions)
- 🤝 **Contribuir**: Ver [CONTRIBUTING.md](CONTRIBUTING.md)
- 📧 **Contacto**: pcamacho447@gmail.com