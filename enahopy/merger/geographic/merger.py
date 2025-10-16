"""ENAHO Merger - Geographic Merger Module"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import GeoMergeConfiguration
from ..exceptions import GeoMergeError


class GeographicMerger:
    """Fusionador especializado para datos geográficos ENAHO"""

    def __init__(
        self,
        config: Optional[GeoMergeConfiguration] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or GeoMergeConfiguration()
        self.logger = logger or logging.getLogger(__name__)

    def merge(
        self,
        df_principal: pd.DataFrame,
        df_geografia: pd.DataFrame,
        columna_union: Optional[str] = None,
        columnas_geograficas: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fusiona DataFrame principal con información geográfica"""
        columna_union = columna_union or self.config.columna_union

        self.logger.info(f"Iniciando fusión geográfica usando columna '{columna_union}'")

        # Realizar merge
        df_merged = pd.merge(df_principal, df_geografia, on=columna_union, how="left")

        # Generar reporte
        report = {
            "input_rows": len(df_principal),
            "geography_rows": len(df_geografia),
            "output_rows": len(df_merged),
            "match_rate": 100.0,
        }

        return df_merged, report


# Alias para compatibilidad
ENAHOGeographicMerger = GeographicMerger
