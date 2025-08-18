# ENAHO Analyzer - Librería para Análisis de Microdatos INEI 🇵🇪

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Stable](https://img.shields.io/badge/status-stable-brightgreen.svg)](https://github.com)

Librería completa para descarga, lectura, validación y análisis de microdatos de encuestas del INEI (Perú), especialmente ENAHO (Encuesta Nacional de Hogares).

## ✨ Características Principales

- 🔄 **Descarga automática** desde servidores oficiales del INEI
- 📊 **Lectura multi-formato**: DTA (Stata), SAV (SPSS), CSV, Parquet
- 🔍 **Validación inteligente** de columnas con mapeo automático
- 💾 **Sistema de cache** para optimizar performance
- ⚡ **Lectura por chunks** para archivos grandes
- 🛠️ **Arquitectura modular** para fácil extensión
- 📈 **Funciones de análisis** especializadas para datos INEI

## 🚀 Instalación Rápida

```bash
# Clona el repositorio
git clone https://github.com/tu-usuario/enaho-analyzer.git
cd enaho-analyzer

# Instala dependencias
pip install -r requirements.txt

# Instalación opcional para formatos avanzados
pip install pyreadstat  # Para archivos DTA y SAV
pip install dask        # Para procesamiento paralelo