"""
ENAHO Merger - Validadores Geográficos
=====================================

Validadores especializados para códigos UBIGEO y consistencia territorial.
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import DEPARTAMENTOS_VALIDOS, TipoValidacionUbigeo
from ..exceptions import TerritorialInconsistencyError, UbigeoValidationError


class UbigeoValidator:
    """Validador especializado para códigos UBIGEO del INEI"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._cache_validacion = {}

    @lru_cache(maxsize=1000)
    def validar_estructura_ubigeo(self, ubigeo: str) -> Tuple[bool, str]:
        """
        Valida la estructura jerárquica de un UBIGEO

        Args:
            ubigeo: Código UBIGEO a validar

        Returns:
            Tupla (es_valido, mensaje_error)
        """
        if not isinstance(ubigeo, str):
            return False, "UBIGEO debe ser string"

        # Normalizar a 6 dígitos
        ubigeo_norm = ubigeo.zfill(6)

        if len(ubigeo_norm) != 6:
            return False, f"Longitud inválida: {len(ubigeo_norm)}"

        if not ubigeo_norm.isdigit():
            return False, "Contiene caracteres no numéricos"

        # Validar departamento
        dep = ubigeo_norm[:2]
        if dep not in DEPARTAMENTOS_VALIDOS:
            return False, f"Departamento inválido: {dep}"

        # Validar provincia (01-99)
        prov = ubigeo_norm[2:4]
        if not ("01" <= prov <= "99"):
            return False, f"Provincia inválida: {prov}"

        # Validar distrito (01-99)
        dist = ubigeo_norm[4:6]
        if not ("01" <= dist <= "99"):
            return False, f"Distrito inválido: {dist}"

        return True, "Válido"

    def validar_serie_ubigeos(
        self, serie: pd.Series, tipo_validacion: TipoValidacionUbigeo
    ) -> Tuple[pd.Series, List[str]]:
        """
        Valida una serie completa de UBIGEOs

        Args:
            serie: Serie de pandas con códigos UBIGEO
            tipo_validacion: Tipo de validación a realizar

        Returns:
            Tupla (mask_validos, lista_errores)
        """
        errores = []

        # Convertir a string y normalizar
        serie_norm = serie.astype(str).str.zfill(6)

        if tipo_validacion == TipoValidacionUbigeo.BASIC:
            # Solo validar formato básico
            mask_valido = serie_norm.str.match(r"^\d{6}", na=False)

        elif tipo_validacion == TipoValidacionUbigeo.STRUCTURAL:
            # Validar estructura jerárquica
            mask_valido = pd.Series([False] * len(serie_norm), index=serie_norm.index)

            for idx, ubigeo in serie_norm.items():
                if pd.notna(ubigeo):
                    is_valid, error_msg = self.validar_estructura_ubigeo(ubigeo)
                    mask_valido.loc[idx] = is_valid
                    if not is_valid:
                        errores.append(f"UBIGEO {ubigeo}: {error_msg}")

        else:
            # Para EXISTENCE y TEMPORAL necesitaríamos catálogos externos
            self.logger.warning(
                f"Tipo de validación {tipo_validacion} no implementado, usando STRUCTURAL"
            )
            return self.validar_serie_ubigeos(serie, TipoValidacionUbigeo.STRUCTURAL)

        return mask_valido, errores

    def extraer_componentes_ubigeo(self, serie: pd.Series) -> pd.DataFrame:
        """
        Extrae componentes jerárquicos de UBIGEOs

        Args:
            serie: Serie con códigos UBIGEO

        Returns:
            DataFrame con componentes territoriales
        """
        serie_norm = serie.astype(str).str.zfill(6)

        componentes = pd.DataFrame(
            {
                "ubigeo": serie_norm,
                "departamento": serie_norm.str[:2],
                "provincia": serie_norm.str[:4],
                "distrito": serie_norm.str[:6],
                "nombre_departamento": serie_norm.str[:2].map(DEPARTAMENTOS_VALIDOS),
            },
            index=serie.index,
        )

        return componentes

    def get_validation_summary(
        self, serie: pd.Series, tipo_validacion: TipoValidacionUbigeo
    ) -> Dict[str, int]:
        """
        Obtiene resumen de validación de UBIGEOs

        Args:
            serie: Serie a validar
            tipo_validacion: Tipo de validación

        Returns:
            Diccionario con estadísticas de validación
        """
        mask_validos, errores = self.validar_serie_ubigeos(serie, tipo_validacion)

        return {
            "total_records": len(serie),
            "valid_ubigeos": mask_validos.sum(),
            "invalid_ubigeos": len(serie) - mask_validos.sum(),
            "null_values": serie.isnull().sum(),
            "unique_ubigeos": serie.nunique(),
            "duplicate_ubigeos": serie.duplicated().sum(),
            "error_count": len(errores),
        }

    def validate_ubigeo_consistency(self, df: pd.DataFrame, ubigeo_col: str) -> List[str]:
        """
        Valida consistencia interna de UBIGEOs en un DataFrame

        Args:
            df: DataFrame a validar
            ubigeo_col: Nombre de columna con UBIGEOs

        Returns:
            Lista de inconsistencias encontradas
        """
        inconsistencias = []

        if ubigeo_col not in df.columns:
            return [f"Columna '{ubigeo_col}' no encontrada"]

        # Extraer componentes
        componentes = self.extraer_componentes_ubigeo(df[ubigeo_col])

        # Verificar que no haya UBIGEOs con diferentes departamentos
        # para el mismo código de departamento
        dep_inconsistencies = componentes.groupby("departamento")["nombre_departamento"].nunique()

        problematic_deps = dep_inconsistencies[dep_inconsistencies > 1]

        if not problematic_deps.empty:
            inconsistencias.append(f"Inconsistencias en {len(problematic_deps)} departamentos")

        return inconsistencias


class TerritorialValidator:
    """Validador de consistencia territorial"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validar_jerarquia_territorial(
        self, df: pd.DataFrame, columnas_territoriales: Dict[str, str]
    ) -> List[str]:
        """
        Valida la consistencia jerárquica territorial

        Args:
            df: DataFrame a validar
            columnas_territoriales: Mapeo de niveles a nombres de columna

        Returns:
            Lista de inconsistencias encontradas
        """
        inconsistencias = []

        # Verificar que distrito pertenezca a provincia correcta
        if "distrito" in columnas_territoriales and "provincia" in columnas_territoriales:
            col_distrito = columnas_territoriales["distrito"]
            col_provincia = columnas_territoriales["provincia"]

            if col_distrito in df.columns and col_provincia in df.columns:
                # Extraer provincia del código de distrito
                provincia_from_distrito = df[col_distrito].astype(str).str[:4]
                provincia_declarada = df[col_provincia].astype(str).str.zfill(4)

                mask_inconsistente = (
                    (provincia_from_distrito != provincia_declarada)
                    & df[col_distrito].notna()
                    & df[col_provincia].notna()
                )

                if mask_inconsistente.any():
                    n_inconsistentes = mask_inconsistente.sum()
                    inconsistencias.append(
                        f"Inconsistencia distrito-provincia en {n_inconsistentes} registros"
                    )

        # Verificar que provincia pertenezca a departamento correcto
        if "provincia" in columnas_territoriales and "departamento" in columnas_territoriales:
            col_provincia = columnas_territoriales["provincia"]
            col_departamento = columnas_territoriales["departamento"]

            if col_provincia in df.columns and col_departamento in df.columns:
                # Extraer departamento del código de provincia
                dep_from_provincia = df[col_provincia].astype(str).str[:2]
                dep_declarado = df[col_departamento].astype(str).str.zfill(2)

                mask_inconsistente = (
                    (dep_from_provincia != dep_declarado)
                    & df[col_provincia].notna()
                    & df[col_departamento].notna()
                )

                if mask_inconsistente.any():
                    n_inconsistentes = mask_inconsistente.sum()
                    inconsistencias.append(
                        f"Inconsistencia provincia-departamento en {n_inconsistentes} registros"
                    )

        return inconsistencias

    def validate_coordinate_consistency(
        self, df: pd.DataFrame, coord_cols: Dict[str, str], ubigeo_col: Optional[str] = None
    ) -> List[str]:
        """
        Valida consistencia de coordenadas geográficas

        Args:
            df: DataFrame a validar
            coord_cols: Diccionario con columnas de coordenadas {'x': 'lon_col', 'y': 'lat_col'}
            ubigeo_col: Columna UBIGEO para validaciones territoriales

        Returns:
            Lista de problemas encontrados
        """
        problemas = []

        if "x" not in coord_cols or "y" not in coord_cols:
            return ["Especificar columnas 'x' e 'y' en coord_cols"]

        lon_col = coord_cols["x"]
        lat_col = coord_cols["y"]

        if lon_col not in df.columns:
            problemas.append(f"Columna de longitud '{lon_col}' no encontrada")

        if lat_col not in df.columns:
            problemas.append(f"Columna de latitud '{lat_col}' no encontrada")

        if problemas:
            return problemas

        # Validar rangos de coordenadas para Perú
        # Perú aproximadamente: -81.3 a -68.7 longitud, -18.4 a 0.0 latitud
        lon_min, lon_max = -81.5, -68.5
        lat_min, lat_max = -18.5, 0.2

        # Coordenadas fuera de rango
        out_of_range_lon = ((df[lon_col] < lon_min) | (df[lon_col] > lon_max)) & df[lon_col].notna()

        out_of_range_lat = ((df[lat_col] < lat_min) | (df[lat_col] > lat_max)) & df[lat_col].notna()

        if out_of_range_lon.any():
            problemas.append(
                f"{out_of_range_lon.sum()} coordenadas de longitud fuera de rango para Perú"
            )

        if out_of_range_lat.any():
            problemas.append(
                f"{out_of_range_lat.sum()} coordenadas de latitud fuera de rango para Perú"
            )

        # Coordenadas nulas
        null_coords = df[[lon_col, lat_col]].isnull().any(axis=1)
        if null_coords.any():
            problemas.append(f"{null_coords.sum()} registros con coordenadas incompletas")

        # Coordenadas duplicadas exactas (posible error)
        if len(df) > 1:
            duplicate_coords = df[[lon_col, lat_col]].duplicated()
            if duplicate_coords.any():
                problemas.append(f"{duplicate_coords.sum()} coordenadas duplicadas exactas")

        return problemas

    def check_territorial_coverage(
        self, df: pd.DataFrame, ubigeo_col: str, expected_departments: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Verifica cobertura territorial de los datos

        Args:
            df: DataFrame a analizar
            ubigeo_col: Columna con códigos UBIGEO
            expected_departments: Lista de departamentos esperados

        Returns:
            Diccionario con estadísticas de cobertura
        """
        if ubigeo_col not in df.columns:
            return {"error": "Columna UBIGEO no encontrada"}

        # Extraer departamentos de UBIGEOs
        departamentos = df[ubigeo_col].astype(str).str[:2]
        departamentos_unicos = departamentos.unique()

        # Mapear a nombres
        deps_con_nombres = {
            dep: DEPARTAMENTOS_VALIDOS.get(dep, "DESCONOCIDO")
            for dep in departamentos_unicos
            if pd.notna(dep) and dep != "na"
        }

        cobertura = {
            "departamentos_presentes": len(deps_con_nombres),
            "departamentos_total_peru": len(DEPARTAMENTOS_VALIDOS),
            "departamentos_encontrados": list(deps_con_nombres.values()),
            "cobertura_porcentual": len(deps_con_nombres) / len(DEPARTAMENTOS_VALIDOS) * 100,
        }

        # Departamentos faltantes
        deps_faltantes = [
            dep for dep in DEPARTAMENTOS_VALIDOS.keys() if dep not in departamentos_unicos
        ]

        cobertura["departamentos_faltantes"] = [
            DEPARTAMENTOS_VALIDOS[dep] for dep in deps_faltantes
        ]

        # Si se especifican departamentos esperados
        if expected_departments:
            expected_codes = [
                code for code, name in DEPARTAMENTOS_VALIDOS.items() if name in expected_departments
            ]

            missing_expected = [dep for dep in expected_codes if dep not in departamentos_unicos]

            cobertura["expected_missing"] = [DEPARTAMENTOS_VALIDOS[dep] for dep in missing_expected]
            cobertura["expected_coverage"] = (
                (len(expected_codes) - len(missing_expected)) / len(expected_codes) * 100
                if expected_codes
                else 100
            )

        return cobertura


class GeoDataQualityValidator:
    """Validador integral de calidad de datos geográficos"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ubigeo_validator = UbigeoValidator(logger)
        self.territorial_validator = TerritorialValidator(logger)

    def comprehensive_validation(
        self,
        df: pd.DataFrame,
        geo_columns: Dict[str, str],
        validation_config: Dict[str, bool] = None,
    ) -> Dict[str, any]:
        """
        Validación integral de datos geográficos

        Args:
            df: DataFrame a validar
            geo_columns: Mapeo de tipos geográficos a columnas
            validation_config: Configuración de qué validaciones ejecutar

        Returns:
            Diccionario con resultados de todas las validaciones
        """
        config = validation_config or {
            "ubigeo_structure": True,
            "territorial_hierarchy": True,
            "coordinate_consistency": True,
            "coverage_analysis": True,
        }

        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_records": len(df),
            "validation_config": config,
            "results": {},
        }

        # Validación de estructura UBIGEO
        if config.get("ubigeo_structure") and "ubigeo" in geo_columns:
            ubigeo_col = geo_columns["ubigeo"]
            ubigeo_summary = self.ubigeo_validator.get_validation_summary(
                df[ubigeo_col], TipoValidacionUbigeo.STRUCTURAL
            )
            results["results"]["ubigeo_validation"] = ubigeo_summary

        # Validación de jerarquía territorial
        if config.get("territorial_hierarchy"):
            territorial_issues = self.territorial_validator.validar_jerarquia_territorial(
                df, geo_columns
            )
            results["results"]["territorial_hierarchy"] = {
                "issues_found": len(territorial_issues),
                "issues": territorial_issues,
            }

        # Validación de coordenadas
        if config.get("coordinate_consistency"):
            coord_cols = {}
            if "coordenada_x" in geo_columns:
                coord_cols["x"] = geo_columns["coordenada_x"]
            if "coordenada_y" in geo_columns:
                coord_cols["y"] = geo_columns["coordenada_y"]

            if coord_cols:
                coord_issues = self.territorial_validator.validate_coordinate_consistency(
                    df, coord_cols, geo_columns.get("ubigeo")
                )
                results["results"]["coordinate_validation"] = {
                    "issues_found": len(coord_issues),
                    "issues": coord_issues,
                }

        # Análisis de cobertura territorial
        if config.get("coverage_analysis") and "ubigeo" in geo_columns:
            coverage = self.territorial_validator.check_territorial_coverage(
                df, geo_columns["ubigeo"]
            )
            results["results"]["territorial_coverage"] = coverage

        # Calcular score de calidad general
        results["overall_quality_score"] = self._calculate_quality_score(results["results"])

        return results

    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """Calcula score de calidad basado en resultados de validación"""
        score = 100.0

        # Penalizar por problemas de UBIGEO
        if "ubigeo_validation" in validation_results:
            ubigeo_data = validation_results["ubigeo_validation"]
            if ubigeo_data["total_records"] > 0:
                invalid_rate = ubigeo_data["invalid_ubigeos"] / ubigeo_data["total_records"]
                score -= invalid_rate * 30  # Hasta 30 puntos menos

        # Penalizar por problemas territoriales
        if "territorial_hierarchy" in validation_results:
            if validation_results["territorial_hierarchy"]["issues_found"] > 0:
                score -= 20  # 20 puntos menos por inconsistencias territoriales

        # Penalizar por problemas de coordenadas
        if "coordinate_validation" in validation_results:
            if validation_results["coordinate_validation"]["issues_found"] > 0:
                score -= 15  # 15 puntos menos por problemas de coordenadas

        # Bonificar por buena cobertura territorial
        if "territorial_coverage" in validation_results:
            coverage_data = validation_results["territorial_coverage"]
            if "cobertura_porcentual" in coverage_data:
                if coverage_data["cobertura_porcentual"] > 90:
                    score += 5  # Bonus por excelente cobertura

        return max(0, min(100, score))
