"""
ENAHO Merger - Estrategias de Manejo de Duplicados Geográficos
=============================================================

Implementaciones específicas para diferentes estrategias de manejo
de duplicados en datos geográficos.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import GeoMergeConfiguration, TipoManejoDuplicados
from ..exceptions import DuplicateHandlingError


class DuplicateHandlingStrategy(ABC):
    """Estrategia base para manejo de duplicados geográficos"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @abstractmethod
    def handle_duplicates(
        self, df: pd.DataFrame, columna_union: str, config: GeoMergeConfiguration
    ) -> pd.DataFrame:
        """
        Maneja duplicados según la estrategia específica

        Args:
            df: DataFrame con duplicados
            columna_union: Columna que contiene duplicados
            config: Configuración de merge

        Returns:
            DataFrame con duplicados manejados
        """
        pass

    def _validate_inputs(self, df: pd.DataFrame, columna_union: str) -> None:
        """Valida inputs básicos"""
        if df.empty:
            raise DuplicateHandlingError("DataFrame vacío")

        if columna_union not in df.columns:
            raise DuplicateHandlingError(f"Columna '{columna_union}' no encontrada")

    def _get_duplicate_info(self, df: pd.DataFrame, columna_union: str) -> Dict[str, int]:
        """Obtiene información sobre duplicados"""
        duplicates_mask = df[columna_union].duplicated(keep=False)

        # Bug fix: Call .unique() on Series, not DataFrame
        duplicate_keys_series = df[duplicates_mask][columna_union]

        return {
            "total_records": len(df),
            "duplicate_records": duplicates_mask.sum(),
            "unique_keys": df[columna_union].nunique(),
            "duplicate_keys": (
                len(duplicate_keys_series.unique()) if len(duplicate_keys_series) > 0 else 0
            ),
        }


class FirstLastStrategy(DuplicateHandlingStrategy):
    """Estrategia para mantener first/last registro"""

    def handle_duplicates(
        self, df: pd.DataFrame, columna_union: str, config: GeoMergeConfiguration
    ) -> pd.DataFrame:
        """Mantiene primer o último registro por clave"""
        self._validate_inputs(df, columna_union)

        duplicate_info = self._get_duplicate_info(df, columna_union)

        if duplicate_info["duplicate_records"] == 0:
            self.logger.info("No hay duplicados para manejar")
            return df

        keep_value = "first" if config.manejo_duplicados == TipoManejoDuplicados.FIRST else "last"

        # Preparar DataFrame para procesamiento
        df_to_process = df.copy()

        # Ordenar si se especifica columna de orden
        if config.columna_orden_duplicados and config.columna_orden_duplicados in df.columns:
            ascending = keep_value == "first"
            df_to_process = df_to_process.sort_values(
                [columna_union, config.columna_orden_duplicados], ascending=[True, ascending]
            )
            self.logger.info(
                f"Datos ordenados por '{config.columna_orden_duplicados}' para estrategia {keep_value}"
            )

        # Eliminar duplicados
        result = df_to_process.drop_duplicates(subset=[columna_union], keep=keep_value)

        removed = len(df) - len(result)
        self.logger.info(f"Eliminados {removed} duplicados usando estrategia '{keep_value}'")

        # Reporte detallado si hay muchos duplicados
        if removed > len(df) * 0.1:  # Más del 10%
            self.logger.warning(
                f"Se eliminó {removed / len(df) * 100:.1f}% de registros por duplicados"
            )

        return result

    def get_duplicate_summary(self, df: pd.DataFrame, columna_union: str) -> pd.DataFrame:
        """Genera resumen de duplicados antes del procesamiento"""
        duplicates_mask = df[columna_union].duplicated(keep=False)
        duplicados = df[duplicates_mask]

        if duplicados.empty:
            return pd.DataFrame()

        # Agrupar por la columna de unión para ver detalles
        summary = (
            duplicados.groupby(columna_union)
            .agg({columna_union: "count"})
            .rename(columns={columna_union: "count_duplicates"})
        )

        summary = summary.reset_index()
        summary = summary.sort_values("count_duplicates", ascending=False)

        return summary


class AggregateStrategy(DuplicateHandlingStrategy):
    """Estrategia para agregar duplicados usando funciones"""

    def handle_duplicates(
        self, df: pd.DataFrame, columna_union: str, config: GeoMergeConfiguration
    ) -> pd.DataFrame:
        """Agrega duplicados usando funciones de agregación"""
        self._validate_inputs(df, columna_union)

        if not config.funciones_agregacion:
            raise DuplicateHandlingError(
                "Se requieren funciones_agregacion para estrategia AGGREGATE",
                duplicates_info={"strategy": "aggregate", "config_missing": "funciones_agregacion"},
            )

        duplicate_info = self._get_duplicate_info(df, columna_union)

        if duplicate_info["duplicate_records"] == 0:
            return df

        # Preparar diccionario de agregación
        agg_dict = self._prepare_aggregation_dict(df, columna_union, config.funciones_agregacion)

        try:
            result = df.groupby(columna_union).agg(agg_dict).reset_index()

            # Aplanar nombres de columnas si es necesario
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = [
                    col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in result.columns
                ]

        except Exception as e:
            raise DuplicateHandlingError(
                f"Error en agregación: {str(e)}",
                duplicates_info={"strategy": "aggregate", "agg_dict": agg_dict},
            )

        removed = len(df) - len(result)
        self.logger.info(f"Agregados {removed} duplicados usando funciones: {agg_dict}")

        return result

    def _prepare_aggregation_dict(
        self, df: pd.DataFrame, columna_union: str, funciones_usuario: Dict[str, str]
    ) -> Dict[str, str]:
        """Prepara diccionario de agregación inteligente"""
        agg_dict = {}

        for col in df.columns:
            if col == columna_union:
                continue

            if col in funciones_usuario:
                agg_dict[col] = funciones_usuario[col]
            else:
                # Elegir función automáticamente basada en tipo de datos
                agg_dict[col] = self._auto_select_aggregation(df[col])

        return agg_dict

    def _auto_select_aggregation(self, serie: pd.Series) -> str:
        """Selecciona función de agregación automáticamente"""
        dtype = serie.dtype

        if pd.api.types.is_numeric_dtype(dtype):
            # Para numéricos, preferir mean, pero usar sum si valores son conteos
            if serie.min() >= 0 and serie.max() < 1000:  # Posibles conteos
                return "sum"
            else:
                return "mean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "first"  # Para fechas, mantener la primera
        else:
            # Para categóricas/strings, preferir el más frecuente
            return "first"


class BestQualityStrategy(DuplicateHandlingStrategy):
    """Estrategia para mantener el registro de mejor calidad"""

    def handle_duplicates(
        self, df: pd.DataFrame, columna_union: str, config: GeoMergeConfiguration
    ) -> pd.DataFrame:
        """Mantiene registro con mejor score de calidad"""
        self._validate_inputs(df, columna_union)

        if not config.columna_calidad:
            raise DuplicateHandlingError(
                "Se requiere columna_calidad para estrategia BEST_QUALITY",
                duplicates_info={"strategy": "best_quality", "config_missing": "columna_calidad"},
            )

        if config.columna_calidad not in df.columns:
            raise DuplicateHandlingError(
                f"Columna de calidad '{config.columna_calidad}' no encontrada",
                duplicates_info={
                    "strategy": "best_quality",
                    "missing_column": config.columna_calidad,
                },
            )

        duplicate_info = self._get_duplicate_info(df, columna_union)

        if duplicate_info["duplicate_records"] == 0:
            return df

        # Ordenar por UBIGEO y calidad (descendente para obtener mejor calidad primero)
        df_sorted = df.sort_values([columna_union, config.columna_calidad], ascending=[True, False])

        # Mantener primer registro (mejor calidad) de cada grupo
        result = df_sorted.drop_duplicates(subset=[columna_union], keep="first")

        removed = len(df) - len(result)
        self.logger.info(f"Mantenidos registros de mejor calidad, eliminados {removed} duplicados")

        # Estadísticas de calidad
        calidad_promedio_antes = df[config.columna_calidad].mean()
        calidad_promedio_despues = result[config.columna_calidad].mean()

        self.logger.info(
            f"Calidad promedio: {calidad_promedio_antes:.3f} → {calidad_promedio_despues:.3f} "
            f"(mejora: {((calidad_promedio_despues - calidad_promedio_antes) / calidad_promedio_antes * 100):+.1f}%)"
        )

        return result

    def analyze_quality_distribution(
        self, df: pd.DataFrame, columna_union: str, columna_calidad: str
    ) -> Dict[str, float]:
        """Analiza distribución de calidad en duplicados"""
        duplicates_mask = df[columna_union].duplicated(keep=False)

        if not duplicates_mask.any():
            return {"message": "No hay duplicados para analizar"}

        duplicados = df[duplicates_mask]

        # Estadísticas por grupo de duplicados
        stats_por_grupo = duplicados.groupby(columna_union)[columna_calidad].agg(
            ["count", "mean", "std", "min", "max"]
        )

        return {
            "grupos_con_duplicados": len(stats_por_grupo),
            "calidad_promedio_duplicados": stats_por_grupo["mean"].mean(),
            "variabilidad_promedio": stats_por_grupo["std"].mean(),
            "rango_calidad_promedio": stats_por_grupo["max"].mean() - stats_por_grupo["min"].mean(),
        }


class KeepAllStrategy(DuplicateHandlingStrategy):
    """Estrategia para mantener todos los duplicados con identificadores únicos"""

    def handle_duplicates(
        self, df: pd.DataFrame, columna_union: str, config: GeoMergeConfiguration
    ) -> pd.DataFrame:
        """Mantiene todos los registros, modificando claves duplicadas"""
        self._validate_inputs(df, columna_union)

        duplicate_info = self._get_duplicate_info(df, columna_union)

        if duplicate_info["duplicate_records"] == 0:
            return df

        result = df.copy()

        # Agregar identificador de orden para duplicados
        result["_orden_duplicado"] = result.groupby(columna_union).cumcount() + 1

        # Modificar clave para duplicados (excepto el primero)
        mask_duplicados = result["_orden_duplicado"] > 1

        if mask_duplicados.any():
            nuevas_claves = (
                result.loc[mask_duplicados, columna_union].astype(str)
                + config.sufijo_duplicados
                + result.loc[mask_duplicados, "_orden_duplicado"].astype(str)
            )

            result.loc[mask_duplicados, columna_union] = nuevas_claves

        # Limpiar columna auxiliar
        result = result.drop("_orden_duplicado", axis=1)

        n_duplicados = mask_duplicados.sum()
        self.logger.info(
            f"Mantenidos todos los registros, {n_duplicados} marcados con sufijo '{config.sufijo_duplicados}'"
        )

        # Advertir si hay muchos duplicados
        if n_duplicados > len(df) * 0.2:  # Más del 20%
            self.logger.warning(
                f"Gran cantidad de duplicados ({n_duplicados / len(df) * 100:.1f}%), "
                f"verifique calidad de datos"
            )

        return result


class MostRecentStrategy(DuplicateHandlingStrategy):
    """Estrategia para mantener el registro más reciente"""

    def handle_duplicates(
        self, df: pd.DataFrame, columna_union: str, config: GeoMergeConfiguration
    ) -> pd.DataFrame:
        """Mantiene el registro más reciente basado en columna de fecha/timestamp"""
        self._validate_inputs(df, columna_union)

        # Buscar columna de fecha automáticamente o usar la especificada
        fecha_col = self._find_date_column(df, config.columna_orden_duplicados)

        if not fecha_col:
            raise DuplicateHandlingError(
                "No se encontró columna de fecha para estrategia MOST_RECENT",
                duplicates_info={"strategy": "most_recent", "available_columns": list(df.columns)},
            )

        duplicate_info = self._get_duplicate_info(df, columna_union)

        if duplicate_info["duplicate_records"] == 0:
            return df

        # Ordenar por fecha descendente (más reciente primero)
        df_sorted = df.sort_values([columna_union, fecha_col], ascending=[True, False])

        # Mantener primer registro (más reciente) de cada grupo
        result = df_sorted.drop_duplicates(subset=[columna_union], keep="first")

        removed = len(df) - len(result)
        self.logger.info(
            f"Mantenidos registros más recientes usando '{fecha_col}', eliminados {removed} duplicados"
        )

        return result

    def _find_date_column(self, df: pd.DataFrame, specified_col: Optional[str]) -> Optional[str]:
        """Encuentra columna de fecha en el DataFrame"""
        if specified_col and specified_col in df.columns:
            # Verificar que sea fecha o convertible
            if pd.api.types.is_datetime64_any_dtype(df[specified_col]):
                return specified_col
            else:
                # Intentar convertir
                try:
                    pd.to_datetime(df[specified_col].head())
                    return specified_col
                except:
                    pass

        # Buscar automáticamente
        date_patterns = ["fecha", "date", "timestamp", "time", "created", "updated", "modified"]

        for col in df.columns:
            col_lower = col.lower()

            # Verificar por nombre
            if any(pattern in col_lower for pattern in date_patterns):
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col

            # Verificar por tipo de datos
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None


# =====================================================
# FACTORY DE ESTRATEGIAS
# =====================================================


class DuplicateStrategyFactory:
    """Factory para crear estrategias de manejo de duplicados"""

    _strategies = {
        TipoManejoDuplicados.FIRST: FirstLastStrategy,
        TipoManejoDuplicados.LAST: FirstLastStrategy,
        TipoManejoDuplicados.AGGREGATE: AggregateStrategy,
        TipoManejoDuplicados.BEST_QUALITY: BestQualityStrategy,
        TipoManejoDuplicados.KEEP_ALL: KeepAllStrategy,
        TipoManejoDuplicados.MOST_RECENT: MostRecentStrategy,
    }

    @classmethod
    def create_strategy(
        cls, tipo: TipoManejoDuplicados, logger: logging.Logger
    ) -> DuplicateHandlingStrategy:
        """
        Crea instancia de estrategia según el tipo

        Args:
            tipo: Tipo de estrategia de manejo de duplicados
            logger: Logger para la estrategia

        Returns:
            Instancia de estrategia específica
        """
        if tipo not in cls._strategies:
            raise DuplicateHandlingError(
                f"Estrategia no soportada: {tipo}",
                duplicates_info={"available_strategies": list(cls._strategies.keys())},
            )

        strategy_class = cls._strategies[tipo]
        return strategy_class(logger)

    @classmethod
    def get_available_strategies(cls) -> List[TipoManejoDuplicados]:
        """Retorna lista de estrategias disponibles"""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, tipo: TipoManejoDuplicados) -> Dict[str, str]:
        """Obtiene información sobre una estrategia específica"""
        descriptions = {
            TipoManejoDuplicados.FIRST: "Mantiene el primer registro de cada grupo",
            TipoManejoDuplicados.LAST: "Mantiene el último registro de cada grupo",
            TipoManejoDuplicados.AGGREGATE: "Agrega duplicados usando funciones de agregación",
            TipoManejoDuplicados.BEST_QUALITY: "Mantiene el registro con mejor score de calidad",
            TipoManejoDuplicados.KEEP_ALL: "Mantiene todos los registros con sufijos únicos",
            TipoManejoDuplicados.MOST_RECENT: "Mantiene el registro más reciente por fecha",
        }

        requirements = {
            TipoManejoDuplicados.FIRST: "Opcional: columna_orden_duplicados",
            TipoManejoDuplicados.LAST: "Opcional: columna_orden_duplicados",
            TipoManejoDuplicados.AGGREGATE: "Requerido: funciones_agregacion",
            TipoManejoDuplicados.BEST_QUALITY: "Requerido: columna_calidad",
            TipoManejoDuplicados.KEEP_ALL: "Opcional: sufijo_duplicados",
            TipoManejoDuplicados.MOST_RECENT: "Opcional: columna_orden_duplicados (fecha)",
        }

        return {
            "description": descriptions.get(tipo, "Descripción no disponible"),
            "requirements": requirements.get(tipo, "Sin requisitos especiales"),
            "class_name": cls._strategies[tipo].__name__,
        }
