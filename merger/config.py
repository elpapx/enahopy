"""
ENAHO Merger - Configuraciones y Enums
=====================================

Configuraciones, enums y dataclasses para el sistema de fusión geográfica
y merge entre módulos ENAHO.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any


# =====================================================
# ENUMS GEOGRÁFICOS
# =====================================================

class TipoManejoDuplicados(Enum):
    """Estrategias para manejar duplicados geográficos"""
    FIRST = "first"
    LAST = "last"
    ERROR = "error"
    KEEP_ALL = "keep_all"
    AGGREGATE = "aggregate"
    MOST_RECENT = "most_recent"
    BEST_QUALITY = "best_quality"


class TipoManejoErrores(Enum):
    """Estrategias para manejar errores de fusión"""
    COERCE = "coerce"
    RAISE = "raise"
    IGNORE = "ignore"
    LOG_WARNING = "log_warning"


class NivelTerritorial(Enum):
    """Niveles territoriales del Perú según INEI"""
    DEPARTAMENTO = "departamento"
    PROVINCIA = "provincia"
    DISTRITO = "distrito"
    CENTRO_POBLADO = "centro_poblado"
    CONGLOMERADO = "conglomerado"


class TipoValidacionUbigeo(Enum):
    """Tipos de validación de UBIGEO"""
    BASIC = "basic"  # Solo formato
    STRUCTURAL = "structural"  # Validar estructura jerárquica
    EXISTENCE = "existence"  # Validar existencia real
    TEMPORAL = "temporal"  # Validar vigencia temporal


# =====================================================
# ENUMS DE MÓDULOS
# =====================================================

class ModuleMergeLevel(Enum):
    """Niveles de merge entre módulos ENAHO"""
    HOGAR = "hogar"  # Merge a nivel hogar
    PERSONA = "persona"  # Merge a nivel persona
    VIVIENDA = "vivienda"  # Merge a nivel vivienda


class ModuleMergeStrategy(Enum):
    """Estrategias para manejar conflictos en merge de módulos"""
    KEEP_LEFT = "keep_left"  # Mantener valores del DataFrame izquierdo
    KEEP_RIGHT = "keep_right"  # Mantener valores del DataFrame derecho
    COALESCE = "coalesce"  # Combinar (usar derecho si izquierdo es nulo)
    AVERAGE = "average"  # Promediar valores numéricos
    CONCATENATE = "concatenate"  # Concatenar strings
    ERROR = "error"  # Error si hay conflictos


class ModuleType(Enum):
    """Tipos de módulos ENAHO según su estructura"""
    HOGAR_LEVEL = "hogar_level"  # Módulos a nivel hogar (01, 07, 08, 34)
    PERSONA_LEVEL = "persona_level"  # Módulos a nivel persona (02, 03, 04, 05)
    MIXED_LEVEL = "mixed_level"  # Módulos con ambos niveles
    SPECIAL = "special"  # Módulos especiales (37)


# =====================================================
# CONFIGURACIONES
# =====================================================

@dataclass
class GeoMergeConfiguration:
    """Configuración completa para operaciones de fusión geográfica"""
    # Configuración básica
    columna_union: str = 'ubigeo'
    manejo_duplicados: TipoManejoDuplicados = TipoManejoDuplicados.FIRST
    manejo_errores: TipoManejoErrores = TipoManejoErrores.COERCE
    valor_faltante: Union[str, float] = 'DESCONOCIDO'

    # Validaciones
    validar_formato_ubigeo: bool = True
    tipo_validacion_ubigeo: TipoValidacionUbigeo = TipoValidacionUbigeo.STRUCTURAL
    validar_consistencia_territorial: bool = True
    validar_coordenadas: bool = False

    # Reportes y logging
    generar_reporte_calidad: bool = True
    reporte_duplicados: bool = True
    mostrar_estadisticas: bool = True

    # Manejo de duplicados avanzado
    funciones_agregacion: Optional[Dict[str, str]] = None
    sufijo_duplicados: str = '_dup'
    columna_orden_duplicados: Optional[str] = None
    columna_calidad: Optional[str] = None  # Para BEST_QUALITY

    # Performance
    usar_cache: bool = True
    optimizar_memoria: bool = True
    chunk_size: int = 50000

    # Configuración territorial específica
    nivel_territorial_objetivo: NivelTerritorial = NivelTerritorial.DISTRITO
    incluir_niveles_superiores: bool = True
    prefijo_columnas: str = ''
    sufijo_columnas: str = ''


@dataclass
class ModuleMergeConfig:
    """Configuración para merge entre módulos ENAHO"""
    merge_level: ModuleMergeLevel = ModuleMergeLevel.HOGAR
    merge_strategy: ModuleMergeStrategy = ModuleMergeStrategy.COALESCE
    validate_keys: bool = True
    allow_partial_matches: bool = False
    suffix_conflicts: Tuple[str, str] = ("_x", "_y")

    # Columnas de identificación por nivel
    hogar_keys: List[str] = field(default_factory=lambda: ['conglome', 'vivienda', 'hogar'])
    persona_keys: List[str] = field(default_factory=lambda: ['conglome', 'vivienda', 'hogar', 'codperso'])
    vivienda_keys: List[str] = field(default_factory=lambda: ['conglome', 'vivienda'])

    # Configuración específica por módulo
    module_validations: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        '01': {'level': ModuleType.HOGAR_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar']},
        '02': {'level': ModuleType.PERSONA_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar', 'codperso']},
        '03': {'level': ModuleType.PERSONA_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar', 'codperso']},
        '04': {'level': ModuleType.PERSONA_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar', 'codperso']},
        '05': {'level': ModuleType.PERSONA_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar', 'codperso']},
        '07': {'level': ModuleType.HOGAR_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar']},
        '08': {'level': ModuleType.HOGAR_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar']},
        '09': {'level': ModuleType.HOGAR_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar']},
        '34': {'level': ModuleType.HOGAR_LEVEL, 'required_keys': ['conglome', 'vivienda', 'hogar']},
        '37': {'level': ModuleType.SPECIAL, 'required_keys': ['conglome', 'vivienda', 'hogar']}
    })

    # Configuración de calidad
    min_match_rate: float = 0.7  # Tasa mínima de match para considerar exitoso
    max_conflicts_allowed: int = 1000  # Máximo número de conflictos permitidos

    # Configuración de performance
    chunk_processing: bool = False
    chunk_size: int = 10000


# =====================================================
# DATACLASSES DE RESULTADOS
# =====================================================

@dataclass
class GeoValidationResult:
    """Resultado de validación geográfica"""
    is_valid: bool
    total_records: int
    valid_ubigeos: int
    invalid_ubigeos: int
    duplicate_ubigeos: int
    missing_coordinates: int
    territorial_inconsistencies: int
    coverage_percentage: float
    errors: List[str]
    warnings: List[str]
    quality_metrics: Dict[str, float]

    def get_summary_report(self) -> str:
        """Genera reporte resumido de validación"""
        status = "✅ VÁLIDO" if self.is_valid else "❌ INVÁLIDO"

        report = [
            "=== REPORTE DE VALIDACIÓN GEOGRÁFICA ===",
            f"Estado: {status}",
            f"Registros totales: {self.total_records:,}",
            f"UBIGEOs válidos: {self.valid_ubigeos:,} ({self.coverage_percentage:.1f}%)",
            f"UBIGEOs inválidos: {self.invalid_ubigeos:,}",
            f"Duplicados: {self.duplicate_ubigeos:,}",
        ]

        if self.missing_coordinates > 0:
            report.append(f"Sin coordenadas: {self.missing_coordinates:,}")

        if self.territorial_inconsistencies > 0:
            report.append(f"Inconsistencias territoriales: {self.territorial_inconsistencies:,}")

        if self.warnings:
            report.append("\n⚠️  Advertencias:")
            for warning in self.warnings[:5]:  # Máximo 5
                report.append(f"  - {warning}")

        if self.errors:
            report.append("\n❌ Errores:")
            for error in self.errors[:5]:  # Máximo 5
                report.append(f"  - {error}")

        return "\n".join(report)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            'is_valid': self.is_valid,
            'total_records': self.total_records,
            'valid_ubigeos': self.valid_ubigeos,
            'invalid_ubigeos': self.invalid_ubigeos,
            'duplicate_ubigeos': self.duplicate_ubigeos,
            'missing_coordinates': self.missing_coordinates,
            'territorial_inconsistencies': self.territorial_inconsistencies,
            'coverage_percentage': self.coverage_percentage,
            'errors': self.errors,
            'warnings': self.warnings,
            'quality_metrics': self.quality_metrics
        }


@dataclass
class ModuleMergeResult:
    """Resultado del merge entre módulos"""
    merged_df: 'pd.DataFrame'
    merge_report: Dict[str, Any]
    conflicts_resolved: int
    unmatched_left: int
    unmatched_right: int
    validation_warnings: List[str]
    quality_score: float

    def get_summary_report(self) -> str:
        """Genera reporte resumido del merge de módulos"""
        status = ("✅ EXITOSO" if self.quality_score >= 70 else
                 "⚠️ CON ADVERTENCIAS" if self.quality_score >= 50 else
                 "❌ PROBLEMÁTICO")

        report = [
            "=== REPORTE DE MERGE ENTRE MÓDULOS ===",
            f"Estado: {status}",
            f"Registros finales: {len(self.merged_df):,}",
            f"Score de calidad: {self.quality_score:.1f}%",
            f"Conflictos resueltos: {self.conflicts_resolved:,}",
            f"No coincidentes izq: {self.unmatched_left:,}",
            f"No coincidentes der: {self.unmatched_right:,}",
        ]

        if self.validation_warnings:
            report.append(f"\n⚠️  Advertencias ({len(self.validation_warnings)}):")
            for warning in self.validation_warnings[:3]:
                report.append(f"  - {warning}")
            if len(self.validation_warnings) > 3:
                report.append(f"  ... y {len(self.validation_warnings) - 3} más")

        return "\n".join(report)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            'shape': self.merged_df.shape,
            'merge_report': self.merge_report,
            'conflicts_resolved': self.conflicts_resolved,
            'unmatched_left': self.unmatched_left,
            'unmatched_right': self.unmatched_right,
            'validation_warnings': self.validation_warnings,
            'quality_score': self.quality_score
        }


# =====================================================
# CONSTANTES Y PATRONES
# =====================================================

# Rangos válidos para departamentos según INEI
DEPARTAMENTOS_VALIDOS = {
    '01': 'AMAZONAS', '02': 'ÁNCASH', '03': 'APURÍMAC', '04': 'AREQUIPA',
    '05': 'AYACUCHO', '06': 'CAJAMARCA', '07': 'CALLAO', '08': 'CUSCO',
    '09': 'HUANCAVELICA', '10': 'HUÁNUCO', '11': 'ICA', '12': 'JUNÍN',
    '13': 'LA LIBERTAD', '14': 'LAMBAYEQUE', '15': 'LIMA', '16': 'LORETO',
    '17': 'MADRE DE DIOS', '18': 'MOQUEGUA', '19': 'PASCO', '20': 'PIURA',
    '21': 'PUNO', '22': 'SAN MARTÍN', '23': 'TACNA', '24': 'TUMBES',
    '25': 'UCAYALI'
}

# Patrones extendidos de columnas geográficas
PATRONES_GEOGRAFICOS = {
    'departamento': [
        'departamento', 'dep', 'dpto', 'depto', 'department', 'region',
        'cod_dep', 'codigo_departamento', 'dpto_id', 'dep_code'
    ],
    'provincia': [
        'provincia', 'prov', 'province', 'cod_prov', 'codigo_provincia',
        'prov_id', 'prov_code', 'provincia_cod'
    ],
    'distrito': [
        'distrito', 'dist', 'district', 'cod_dist', 'codigo_distrito',
        'dist_id', 'dist_code', 'distrito_cod'
    ],
    'ubigeo': [
        'ubigeo', 'cod_ubigeo', 'codigo_ubigeo', 'ubigeo_code', 'geo_code',
        'ubigeo_inei', 'codigo_geografico', 'cod_geo'
    ],
    'centro_poblado': [
        'centro_poblado', 'ccpp', 'poblado', 'centropoblado', 'cod_ccpp',
        'codigo_ccpp', 'ccpp_code', 'centro_pob'
    ],
    'conglomerado': [
        'conglome', 'conglomerado', 'conglo', 'cod_conglome', 'cluster'
    ],
    'coordenada_x': [
        'longitud', 'longitude', 'lon', 'coord_x', 'x', 'este', 'east'
    ],
    'coordenada_y': [
        'latitud', 'latitude', 'lat', 'coord_y', 'y', 'norte', 'north'
    ]
}