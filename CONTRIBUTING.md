# Guía de Contribución - ENAHOPY

¡Gracias por tu interés en contribuir a ENAHOPY! Este documento te guiará a través del proceso de contribución.

## 🚀 Formas de Contribuir

### 1. Reportar Problemas
- 🐛 **Bugs**: Reporta errores o comportamientos inesperados
- 💡 **Sugerencias**: Propón nuevas características o mejoras
- 📚 **Documentación**: Reporta errores o sugiere mejoras en la documentación

### 2. Código
- 🔧 **Bug fixes**: Corrige errores existentes
- ✨ **Nuevas características**: Implementa nuevas funcionalidades
- 📈 **Optimizaciones**: Mejora performance o eficiencia
- 🧪 **Tests**: Agrega o mejora tests existentes

### 3. Documentación
- 📖 **Guías**: Crea tutoriales o guías de uso
- 📝 **Ejemplos**: Agrega ejemplos prácticos
- 🔍 **API Docs**: Mejora documentación de funciones

## 🛠️ Configuración del Entorno de Desarrollo

### 1. Fork y Clonar

```bash
# Fork el repositorio en GitHub, luego clona tu fork
git clone https://github.com/elpapx/enahopy
cd enahopy

# Agrega el repositorio original como upstream
git remote add upstream https://github.com/tu-usuario/enahopy.git
```

### 2. Configurar Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar en modo desarrollo
pip install -e .[dev]
```

### 3. Instalar Pre-commit Hooks

```bash
# Instalar pre-commit (incluido en dev dependencies)
pre-commit install

# (Opcional) Ejecutar en todos los archivos
pre-commit run --all-files
```

### 4. Verificar Instalación

```bash
# Ejecutar tests rápidos (excluye tests lentos)
pytest tests/ -m "not slow"

# Ejecutar todos los tests
pytest tests/ -v

# Verificar estilo de código
black --check enahopy/ tests/
flake8 enahopy/
isort --check-only enahopy/ tests/

# Tests con cobertura
pytest tests/ --cov=enahopy --cov-report=html --cov-report=term-missing
```

## 📋 Proceso de Desarrollo

### 1. Crear una Rama

```bash
# Asegúrate de estar en main y actualizado
git checkout main
git pull upstream main

# Crea una nueva rama descriptiva
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/corregir-bug
# o  
git checkout -b docs/mejorar-documentacion
```

### 2. Realizar Cambios

#### Estructura del Código

```
enaho_analyzer/
├── loader/           # Descarga y lectura de datos
│   ├── core/        # Configuración, excepciones, logging
│   ├── io/          # Readers, downloaders, validators
│   └── utils/       # Utilidades y shortcuts
├── merger/          # Fusión de módulos y geografía
│   ├── geographic/  # Datos geográficos
│   ├── modules/     # Fusión entre módulos
│   └── strategies/  # Estrategias de fusión
└── null_analysis/   # Análisis de valores faltantes
    ├── core/       # Motor de análisis
    ├── patterns/   # Detección de patrones
    └── reports/    # Generación de reportes
```

#### Estándares de Código

**Estilo de Código:**
```python
# Usar black para formateo automático
black .

# Imports organizados con isort
isort .

# Líneas máximo 88 caracteres
# Usar type hints cuando sea posible
def process_data(df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
    """
    Procesa datos ENAHO.
    
    Args:
        df: DataFrame con datos ENAHO
        year: Año de la encuesta
        
    Returns:
        DataFrame procesado
        
    Raises:
        ENAHOError: Si los datos no son válidos
    """
    pass
```

**Documentación:**
```python
class ENAHOProcessor:
    """
    Procesador principal de datos ENAHO.
    
    Esta clase encapsula la lógica principal para procesar
    microdatos de la Encuesta Nacional de Hogares.
    
    Attributes:
        config: Configuración del procesador
        logger: Logger para registro de eventos
        
    Example:
        >>> processor = ENAHOProcessor()
        >>> df = processor.load_data('ruta/archivo.dta')
        >>> df_clean = processor.clean_data(df)
    """
```

### 3. Escribir Tests

```python
# tests/test_nueva_funcionalidad.py
import pytest
import pandas as pd
from enahopy.loader import ENAHODataDownloader

def test_descarga_datos():
    """Test de descarga de datos ENAHO."""
    downloader = ENAHODataDownloader()
    
    # Usar mocks para tests que no dependan de internet
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        result = downloader.download(year=2023, module='01')
        
    assert result is not None

@pytest.mark.parametrize("year,module", [
    (2023, '01'),
    (2022, '02'), 
    (2021, '34')
])
def test_descarga_parametrizada(year, module):
    """Test parametrizado para múltiples años y módulos."""
    # Test implementation
    pass

# Tests de integración (lentos)
@pytest.mark.slow
def test_integracion_completa():
    """Test de integración que puede tardar."""
    pass
```

### 4. Actualizar Documentación

Si tu cambio afecta la API pública:

1. **README.md**: Actualiza ejemplos si es necesario
2. **Docstrings**: Documenta nuevas funciones/clases
3. **CHANGELOG.md**: Agrega entrada en "Unreleased"
4. **Ejemplos**: Crea o actualiza notebooks de ejemplo

## ✅ Checklist Antes de Enviar PR

### Código
- [ ] Los tests pasan: `pytest`
- [ ] Estilo de código correcto: `black . && flake8 . && isort .`
- [ ] Cobertura de tests adecuada (>80%)
- [ ] No hay imports no utilizados
- [ ] Type hints en funciones públicas
- [ ] Docstrings en clases y funciones públicas

### Documentación
- [ ] README actualizado si es necesario
- [ ] CHANGELOG.md actualizado
- [ ] Docstrings completos y claros
- [ ] Ejemplos funcionan correctamente

### Git
- [ ] Commits con mensajes descriptivos
- [ ] Rama actualizada con main
- [ ] Sin archivos innecesarios agregados

## 📤 Enviar Pull Request

### 1. Commit y Push

```bash
# Commits atómicos con mensajes descriptivos
git add .
git commit -m "feat: agregar función de validación de ubigeos

- Implementa validación automática de códigos ubigeo
- Agrega tests para casos edge
- Actualiza documentación con ejemplos"

git push origin feature/nueva-funcionalidad
```

### 2. Crear Pull Request

**Título:** Descriptivo y conciso
```
feat: agregar validación automática de ubigeos
```

**Descripción:** Usa esta plantilla:
```markdown
## 📋 Descripción

Breve descripción de los cambios realizados.

## 🔄 Tipo de Cambio

- [ ] Bug fix (no breaking change)
- [ ] Nueva característica (no breaking change)
- [ ] Breaking change (cambio que afecta funcionalidad existente)
- [ ] Actualización de documentación

## 🧪 Tests

- [ ] Tests existentes pasan
- [ ] Agregué tests para nuevas funcionalidades
- [ ] Verifiqué cobertura de código

## 📝 Checklist

- [ ] Mi código sigue el estilo del proyecto
- [ ] He realizado auto-revisión de mi código
- [ ] He comentado código complejo
- [ ] He actualizado documentación
- [ ] Mis cambios no generan warnings
- [ ] He agregado tests que prueban mi fix/feature
- [ ] Tests nuevos y existentes pasan

## 🔗 Issues Relacionados

Closes #123
Related to #456
```

## 🏷️ Convenciones de Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Nuevas características
feat: agregar soporte para ENDES

# Bug fixes
fix: corregir error en lectura de archivos DTA

# Documentación
docs: actualizar guía de instalación

# Refactoring
refactor: reestructurar módulo de validaciones

# Tests
test: agregar tests para fusión geográfica

# Performance
perf: optimizar carga de archivos grandes

# Chores (builds, CI, etc.)
chore: actualizar dependencias
```

## 🐛 Reportar Problemas

### Información Necesaria

Al reportar un bug, incluye:

1. **Versión**: `pip show enahopy`
2. **Sistema**: SO, versión de Python
3. **Descripción**: Qué esperabas vs qué pasó
4. **Reproducción**: Pasos para reproducir el error
5. **Código**: Ejemplo mínimo que reproduce el problema
6. **Logs**: Mensajes de error completos

### Plantilla de Issue

```markdown
## 🐛 Descripción del Bug

Descripción clara y concisa del problema.

## 🔄 Pasos para Reproducir

1. Ejecutar `python script.py`
2. Llamar función `load_data()`
3. Ver error

## 💭 Comportamiento Esperado

Descripción de lo que esperabas que pasara.

## 📸 Screenshots/Logs

```
Error completo aquí
```

## 🖥️ Entorno

- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.9.7]
- enaho-analyzer: [e.g. 1.0.0]
- pandas: [e.g. 1.4.0]

## 📝 Información Adicional

Cualquier contexto adicional sobre el problema.
```

## 🎯 Áreas de Contribución Prioritarias

### 1. Alta Prioridad
- 🐛 **Bug fixes**: Errores en funcionalidad existente
- 📊 **Nuevos indicadores**: Cálculos específicos para análisis social
- 🗺️ **Mejoras geográficas**: Validación y procesamiento de ubigeos
- ⚡ **Performance**: Optimización para archivos grandes

### 2. Media Prioridad
- 📈 **Visualizaciones**: Nuevos tipos de gráficos especializados
- 🔍 **Validaciones**: Checks adicionales de calidad de datos
- 📚 **Documentación**: Tutoriales y guías específicas
- 🧪 **Tests**: Cobertura adicional y tests de integración

### 3. Proyectos Grandes
- 🌐 **Soporte ENDES**: Módulo completo para encuesta demográfica
- 🔗 **API REST**: Servicio web para análisis remoto
- 📊 **Dashboard**: Interface web con Streamlit
- 🔧 **CLI**: Herramientas de línea de comandos

## 🔄 CI/CD Pipeline

### GitHub Actions

Cada push y pull request ejecuta automáticamente:

1. **Quality Checks** - Formateo, imports, linting
2. **Tests Matrix** - Tests en Python 3.8-3.12 × Ubuntu/Windows/macOS
3. **Coverage** - Validación de cobertura mínima (16%)
4. **Integration** - Tests de integración end-to-end
5. **Build** - Verificación de empaquetado

### Verificar Estado del CI

- Ver badge en README.md
- Revisar checks en tu PR
- Los PRs deben pasar todos los checks antes de merge

### Ejecución Local del CI

Simula el CI localmente antes de push:

```bash
# Ejecutar todos los checks de calidad
pre-commit run --all-files

# Ejecutar tests como en CI
pytest tests/ -v -m "not slow" --cov=enahopy --cov-fail-under=16

# Verificar build
python -m build
twine check dist/*
```

## 🤝 Código de Conducta

### Nuestros Valores

- **Respeto**: Tratamos a todos con respeto y dignidad
- **Inclusión**: Valoramos la diversidad de perspectivas
- **Colaboración**: Trabajamos juntos hacia objetivos comunes
- **Calidad**: Nos esforzamos por código y documentación de alta calidad

### Comportamiento Esperado

- Usar lenguaje profesional e inclusivo
- Ser constructivo en críticas y sugerencias
- Enfocarse en lo que es mejor para la comunidad
- Mostrar empatía hacia otros miembros

### Comportamiento Inaceptable

- Comentarios discriminatorios o despectivos
- Ataques personales o trolling
- Spam o autopromoción no relacionada
- Violación de privacidad

## 📞 Contacto

- 💬 **Discusiones**: [GitHub Discussions](https://github.com/elpapx/enahopy/discussions)
- 📧 **Email**: pcamacho447@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/elpapx/enahopy/issues)

## 🙏 Reconocimientos

Todos los contribuidores serán reconocidos en:

- README.md (sección de contribuidores)
- CHANGELOG.md (en cada release)
- GitHub contributors page

¡Gracias por ayudar a hacer ENAHO Analyzer mejor para toda la comunidad de investigación social en Perú! 🇵🇪