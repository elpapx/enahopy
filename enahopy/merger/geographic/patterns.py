"""

ENAHO Merger - Detector de Patrones Geográficos

==============================================



Detector automático de columnas geográficas y patrones territoriales.

"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import PATRONES_GEOGRAFICOS, NivelTerritorial


class GeoPatternDetector:
    """Detector de patrones geográficos en datasets INEI"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        self._detection_cache = {}

    def detectar_columnas_geograficas(
        self, df: pd.DataFrame, confianza_minima: float = 0.8
    ) -> Dict[str, str]:
        """

        Detecta automáticamente columnas geográficas con scoring de confianza



        Args:

            df: DataFrame a analizar

            confianza_minima: Umbral mínimo de confianza (0-1)



        Returns:

            Dict con mapeo tipo_geografico -> nombre_columna_encontrada

        """

        # Verificar cache

        cache_key = f"{id(df)}_{confianza_minima}"

        if cache_key in self._detection_cache:
            return self._detection_cache[cache_key]

        columnas_detectadas = {}

        df_columns_lower = [col.lower() for col in df.columns]

        for tipo_geo, patrones in PATRONES_GEOGRAFICOS.items():
            mejor_coincidencia = None

            mejor_score = 0

            for patron in patrones:
                for i, col_lower in enumerate(df_columns_lower):
                    score = self._calculate_similarity_score(patron, col_lower, df, df.columns[i])

                    if score > mejor_score and score >= confianza_minima:
                        mejor_score = score

                        mejor_coincidencia = df.columns[i]

            if mejor_coincidencia:
                columnas_detectadas[tipo_geo] = mejor_coincidencia

                self.logger.info(
                    f"Detectado {tipo_geo}: '{mejor_coincidencia}' (confianza: {mejor_score:.1f})"
                )

        # Guardar en cache

        self._detection_cache[cache_key] = columnas_detectadas

        return columnas_detectadas

    def _calculate_similarity_score(
        self, patron: str, columna_lower: str, df: pd.DataFrame, columna_original: str
    ) -> float:
        """

        Calcula score de similitud entre patrón y columna



        Args:

            patron: Patrón a buscar

            columna_lower: Nombre de columna en minúsculas

            df: DataFrame para análisis de contenido

            columna_original: Nombre original de la columna



        Returns:

            Score de similitud (0-1)

        """

        base_score = 0

        # Score por similitud de nombres

        if patron == columna_lower:
            base_score = 1.0  # Coincidencia exacta

        elif patron in columna_lower:
            base_score = 0.9  # Contiene el patrón

        elif columna_lower in patron:
            base_score = 0.8  # El patrón contiene la columna

        elif self._fuzzy_match(patron, columna_lower):
            base_score = 0.7  # Coincidencia aproximada

        else:
            return 0  # Sin coincidencia

        # Ajustar score basado en contenido de la columna

        content_score = self._analyze_column_content(df[columna_original], patron)

        # Score final ponderado

        final_score = (base_score * 0.7) + (content_score * 0.3)

        return min(1.0, final_score)

    def _fuzzy_match(self, patron: str, columna: str) -> bool:
        """Verifica coincidencia aproximada entre patrón y columna"""

        # Remover caracteres comunes y comparar

        def clean_string(s):
            return s.replace("_", "").replace("-", "").replace(" ", "")

        patron_clean = clean_string(patron)

        columna_clean = clean_string(columna)

        # Verificar si una está contenida en la otra

        return patron_clean in columna_clean or columna_clean in patron_clean

    def _analyze_column_content(self, serie: pd.Series, tipo_patron: str) -> float:
        """

        Analiza el contenido de la columna para validar el tipo detectado



        Args:

            serie: Serie a analizar

            tipo_patron: Tipo de patrón geográfico



        Returns:

            Score de validez del contenido (0-1)

        """

        if serie.empty or serie.isnull().all():
            return 0.0

        # Obtener muestra no nula

        muestra = serie.dropna().head(100)  # Analizar hasta 100 valores

        if muestra.empty:
            return 0.0

        score = 0.0

        if tipo_patron == "ubigeo":
            score = self._validate_ubigeo_content(muestra)

        elif tipo_patron in ["departamento", "provincia", "distrito"]:
            score = self._validate_territorial_content(muestra, tipo_patron)

        elif tipo_patron == "conglomerado":
            score = self._validate_conglomerado_content(muestra)

        elif tipo_patron in ["coordenada_x", "coordenada_y"]:
            score = self._validate_coordinate_content(muestra, tipo_patron)

        else:
            score = 0.5  # Score neutral para otros tipos

        return score

    def _validate_ubigeo_content(self, muestra: pd.Series) -> float:
        """Valida si el contenido parece ser códigos UBIGEO"""

        score = 0.0

        # Convertir a string para análisis

        str_muestra = muestra.astype(str)

        # Verificar longitud típica de UBIGEO (4-6 dígitos)

        longitudes_validas = str_muestra.str.len().between(4, 6).mean()

        score += longitudes_validas * 0.4

        # Verificar que sean numéricos

        numericos = str_muestra.str.isnumeric().mean()

        score += numericos * 0.4

        # Verificar rangos típicos de departamento (01-25)

        codigos_dep = str_muestra.str.zfill(6).str[:2]

        deps_validos = codigos_dep.isin([f"{i:02d}" for i in range(1, 26)]).mean()

        score += deps_validos * 0.2

        return min(1.0, score)

    def _validate_territorial_content(self, muestra: pd.Series, tipo: str) -> float:
        """Valida contenido territorial (departamento, provincia, distrito)"""

        score = 0.0

        # Convertir a string

        str_muestra = muestra.astype(str)

        if tipo == "departamento":
            # Verificar longitud típica (2 dígitos o nombres)

            longitudes_codigo = str_muestra.str.len().eq(2).mean()

            nombres_largos = str_muestra.str.len().between(4, 20).mean()

            score += max(longitudes_codigo, nombres_largos) * 0.6

            # Verificar si hay nombres conocidos de departamentos

            from ..config import DEPARTAMENTOS_VALIDOS

            nombres_validos = str_muestra.str.upper().isin(DEPARTAMENTOS_VALIDOS.values()).mean()

            score += nombres_validos * 0.4

        elif tipo in ["provincia", "distrito"]:
            # Verificar longitud típica

            if tipo == "provincia":
                longitudes_validas = str_muestra.str.len().between(2, 4).mean()

            else:  # distrito
                longitudes_validas = str_muestra.str.len().between(2, 6).mean()

            score += longitudes_validas * 0.5

            # Verificar que parezcan códigos o nombres

            numericos = str_muestra.str.isnumeric().mean()

            alfabeticos = str_muestra.str.isalpha().mean()

            score += max(numericos, alfabeticos) * 0.5

        return min(1.0, score)

    def _validate_conglomerado_content(self, muestra: pd.Series) -> float:
        """Valida contenido de conglomerado"""

        str_muestra = muestra.astype(str)

        # Los conglomerados en ENAHO suelen ser códigos largos

        longitudes_validas = str_muestra.str.len().between(8, 15).mean()

        numericos = str_muestra.str.isnumeric().mean()

        return min(1.0, (longitudes_validas * 0.6) + (numericos * 0.4))

    def _validate_coordinate_content(self, muestra: pd.Series, tipo: str) -> float:
        """Valida contenido de coordenadas"""

        score = 0.0

        try:
            # Intentar convertir a numérico

            numericos = pd.to_numeric(muestra, errors="coerce")

            valores_numericos = numericos.notna().mean()

            score += valores_numericos * 0.5

            if valores_numericos > 0.5:  # Si la mayoría son numéricos
                if tipo == "coordenada_x":  # Longitud
                    # Rango típico para Perú: -81.5 a -68.5

                    en_rango = numericos.between(-82, -68).mean()

                    score += en_rango * 0.5

                elif tipo == "coordenada_y":  # Latitud
                    # Rango típico para Perú: -18.5 a 0.2

                    en_rango = numericos.between(-19, 1).mean()

                    score += en_rango * 0.5

        except:
            pass

        return min(1.0, score)

    def sugerir_nivel_territorial(self, columnas_detectadas: Dict[str, str]) -> NivelTerritorial:
        """

        Sugiere el nivel territorial más apropiado basado en columnas detectadas



        Args:

            columnas_detectadas: Resultado de detectar_columnas_geograficas



        Returns:

            Nivel territorial sugerido

        """

        if "conglomerado" in columnas_detectadas:
            return NivelTerritorial.CONGLOMERADO

        elif "centro_poblado" in columnas_detectadas:
            return NivelTerritorial.CENTRO_POBLADO

        elif "distrito" in columnas_detectadas or "ubigeo" in columnas_detectadas:
            return NivelTerritorial.DISTRITO

        elif "provincia" in columnas_detectadas:
            return NivelTerritorial.PROVINCIA

        elif "departamento" in columnas_detectadas:
            return NivelTerritorial.DEPARTAMENTO

        else:
            return NivelTerritorial.DISTRITO  # Default

    def analyze_geographic_completeness(
        self, df: pd.DataFrame, columnas_detectadas: Dict[str, str]
    ) -> Dict[str, float]:
        """

        Analiza la completitud de información geográfica



        Args:

            df: DataFrame a analizar

            columnas_detectadas: Columnas geográficas detectadas



        Returns:

            Diccionario con porcentajes de completitud por nivel

        """

        completitud = {}

        for tipo_geo, columna in columnas_detectadas.items():
            if columna in df.columns:
                valores_no_nulos = df[columna].notna().sum()

                porcentaje_completo = (valores_no_nulos / len(df)) * 100

                completitud[tipo_geo] = porcentaje_completo

        return completitud

    def detect_geographic_patterns_in_data(
        self, df: pd.DataFrame, columnas_geo: Dict[str, str]
    ) -> Dict[str, any]:
        """

        Detecta patrones específicos en los datos geográficos



        Args:

            df: DataFrame a analizar

            columnas_geo: Columnas geográficas conocidas



        Returns:

            Diccionario con patrones encontrados

        """

        patrones = {
            "distribucion_territorial": {},
            "concentracion_geografica": {},
            "patrones_anomalos": [],
            "cobertura_por_nivel": {},
        }

        # Análisis de distribución territorial

        if "ubigeo" in columnas_geo and columnas_geo["ubigeo"] in df.columns:
            ubigeo_col = columnas_geo["ubigeo"]

            # Extraer departamentos

            departamentos = df[ubigeo_col].astype(str).str.zfill(6).str[:2]

            dist_departamentos = departamentos.value_counts()

            patrones["distribucion_territorial"]["departamentos"] = {
                "total_departamentos": len(dist_departamentos),
                "departamento_mas_frecuente": (
                    dist_departamentos.index[0] if not dist_departamentos.empty else None
                ),
                "concentracion_top3": (
                    dist_departamentos.head(3).sum() / len(df) * 100 if len(df) > 0 else 0
                ),
            }

            # Detectar concentración excesiva

            if not dist_departamentos.empty:
                concentracion_max = dist_departamentos.iloc[0] / len(df) * 100

                if concentracion_max > 50:
                    patrones["patrones_anomalos"].append(
                        f"Concentración excesiva en un departamento: {concentracion_max:.1f}%"
                    )

        # Análisis de cobertura por nivel

        for nivel, columna in columnas_geo.items():
            if columna in df.columns:
                valores_unicos = df[columna].nunique()

                valores_totales = len(df)

                patrones["cobertura_por_nivel"][nivel] = {
                    "valores_unicos": valores_unicos,
                    "diversidad_relativa": (
                        valores_unicos / valores_totales if valores_totales > 0 else 0
                    ),
                }

        return patrones

    def suggest_merge_strategy(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        columnas_geo1: Dict[str, str],
        columnas_geo2: Dict[str, str],
    ) -> Dict[str, any]:
        """

        Sugiere estrategia de merge basada en patrones geográficos



        Args:

            df1, df2: DataFrames a fusionar

            columnas_geo1, columnas_geo2: Columnas geográficas de cada DataFrame



        Returns:

            Diccionario con recomendaciones de merge

        """

        recomendaciones = {
            "nivel_recomendado": None,
            "columna_union_sugerida": None,
            "problemas_potenciales": [],
            "compatibilidad_score": 0.0,
            "estrategia_duplicados": "first",
        }

        # Encontrar niveles comunes

        niveles_comunes = set(columnas_geo1.keys()) & set(columnas_geo2.keys())

        if not niveles_comunes:
            recomendaciones["problemas_potenciales"].append(
                "No hay niveles geográficos comunes entre los DataFrames"
            )

            return recomendaciones

        # Evaluar cada nivel común

        mejor_nivel = None

        mejor_score = 0

        for nivel in niveles_comunes:
            col1 = columnas_geo1[nivel]

            col2 = columnas_geo2[nivel]

            if col1 in df1.columns and col2 in df2.columns:
                score = self._evaluate_merge_compatibility(df1[col1], df2[col2], nivel)

                if score > mejor_score:
                    mejor_score = score

                    mejor_nivel = nivel

        if mejor_nivel:
            recomendaciones["nivel_recomendado"] = mejor_nivel

            recomendaciones["columna_union_sugerida"] = columnas_geo1[mejor_nivel]

            recomendaciones["compatibilidad_score"] = mejor_score

            # Sugerir estrategia de duplicados basada en el análisis

            col1 = columnas_geo1[mejor_nivel]

            duplicados_df2 = df2[columnas_geo2[mejor_nivel]].duplicated().sum()

            if duplicados_df2 > len(df2) * 0.1:  # Más del 10% duplicados
                recomendaciones["estrategia_duplicados"] = "aggregate"

                recomendaciones["problemas_potenciales"].append(
                    f"Alto número de duplicados en DataFrame 2: {duplicados_df2}"
                )

        return recomendaciones

    def _evaluate_merge_compatibility(
        self, serie1: pd.Series, serie2: pd.Series, nivel: str
    ) -> float:
        """

        Evalúa compatibilidad para merge entre dos series geográficas



        Args:

            serie1, serie2: Series a evaluar

            nivel: Nivel geográfico



        Returns:

            Score de compatibilidad (0-1)

        """

        score = 0.0

        # Factor 1: Overlap de valores (40% del score)

        valores1 = set(serie1.dropna().astype(str))

        valores2 = set(serie2.dropna().astype(str))

        if valores1 and valores2:
            overlap = len(valores1 & valores2) / len(valores1 | valores2)

            score += overlap * 0.4

        # Factor 2: Consistencia de formato (30% del score)

        formato_score = self._compare_format_consistency(serie1, serie2, nivel)

        score += formato_score * 0.3

        # Factor 3: Completitud (20% del score)

        completitud1 = serie1.notna().mean()

        completitud2 = serie2.notna().mean()

        completitud_promedio = (completitud1 + completitud2) / 2

        score += completitud_promedio * 0.2

        # Factor 4: Distribución similar (10% del score)

        if len(valores1) > 1 and len(valores2) > 1:
            # Comparar diversidad relativa

            diversidad1 = len(valores1) / len(serie1)

            diversidad2 = len(valores2) / len(serie2)

            if diversidad1 > 0 and diversidad2 > 0:
                ratio_diversidad = min(diversidad1, diversidad2) / max(diversidad1, diversidad2)

                score += ratio_diversidad * 0.1

        return min(1.0, score)

    def _compare_format_consistency(
        self, serie1: pd.Series, serie2: pd.Series, nivel: str
    ) -> float:
        """Compara consistencia de formato entre dos series"""

        score = 0.0

        # Convertir a string para análisis

        str1 = serie1.dropna().astype(str)

        str2 = serie2.dropna().astype(str)

        if str1.empty or str2.empty:
            return 0.0

        # Comparar longitudes típicas

        len1_mode = str1.str.len().mode().iloc[0] if not str1.str.len().mode().empty else 0

        len2_mode = str2.str.len().mode().iloc[0] if not str2.str.len().mode().empty else 0

        if len1_mode == len2_mode and len1_mode > 0:
            score += 0.4

        elif abs(len1_mode - len2_mode) <= 1:
            score += 0.2

        # Comparar si ambos son numéricos o alfabéticos

        num1 = str1.str.isnumeric().mean()

        num2 = str2.str.isnumeric().mean()

        if (num1 > 0.8 and num2 > 0.8) or (num1 < 0.2 and num2 < 0.2):
            score += 0.3

        # Para UBIGEO, verificar estructura específica

        if nivel == "ubigeo":
            # Verificar que ambos sigan patrón de 6 dígitos

            pattern1 = str1.str.match(r"^\d{6}", na=False).mean()

            pattern2 = str2.str.match(r"^\d{6}", na=False).mean()

            if pattern1 > 0.8 and pattern2 > 0.8:
                score += 0.3

        return min(1.0, score)

    def generate_detection_report(
        self, df: pd.DataFrame, columnas_detectadas: Dict[str, str]
    ) -> str:
        """

        Genera reporte detallado de detección de columnas geográficas



        Args:

            df: DataFrame analizado

            columnas_detectadas: Resultado de detección



        Returns:

            Reporte formateado como string

        """

        if not columnas_detectadas:
            return "❌ No se detectaron columnas geográficas en el DataFrame"

        lines = [
            "🗺️  REPORTE DE DETECCIÓN GEOGRÁFICA",
            "=" * 40,
            f"DataFrame analizado: {df.shape[0]:,} filas, {df.shape[1]} columnas",
            f"Columnas geográficas detectadas: {len(columnas_detectadas)}",
            "",
        ]

        # Detalles por columna detectada

        lines.append("📍 COLUMNAS DETECTADAS:")

        for tipo_geo, columna in columnas_detectadas.items():
            completitud = df[columna].notna().sum() / len(df) * 100

            valores_unicos = df[columna].nunique()

            lines.append(f"  • {tipo_geo.upper()}: '{columna}'")

            lines.append(f"    - Completitud: {completitud:.1f}%")

            lines.append(f"    - Valores únicos: {valores_unicos:,}")

            # Muestra de valores

            muestra = df[columna].dropna().head(3).tolist()

            if muestra:
                muestra_str = ", ".join([str(v) for v in muestra])

                lines.append(f"    - Muestra: {muestra_str}")

            lines.append("")

        # Nivel territorial sugerido

        nivel_sugerido = self.sugerir_nivel_territorial(columnas_detectadas)

        lines.extend([f"🎯 NIVEL TERRITORIAL SUGERIDO: {nivel_sugerido.value.upper()}", ""])

        # Análisis de completitud

        completitud = self.analyze_geographic_completeness(df, columnas_detectadas)

        if completitud:
            lines.append("📊 COMPLETITUD POR NIVEL:")

            for nivel, porcentaje in completitud.items():
                status = "✅" if porcentaje > 90 else "⚠️" if porcentaje > 70 else "❌"

                lines.append(f"  {status} {nivel}: {porcentaje:.1f}%")

        return "\n".join(lines)
