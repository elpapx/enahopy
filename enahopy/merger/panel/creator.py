"""Creador de datos de panel ENAHO"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PanelDataResult:
    """Resultado de creación de panel"""

    panel_df: pd.DataFrame
    n_periods: int
    n_individuals: int
    balanced: bool
    attrition_rate: float
    metadata: Dict


class PanelCreator:
    """Creador de datos de panel longitudinal"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def create_panel(
        self, data_dict: Dict[str, pd.DataFrame], id_vars: List[str], time_var: str = "año"
    ) -> PanelDataResult:
        """Crea panel longitudinal desde múltiples períodos"""
        self.logger.info(f"Creando panel con {len(data_dict)} períodos")

        panel_frames = []
        for period, df in data_dict.items():
            df_copy = df.copy()
            df_copy[time_var] = period
            panel_frames.append(df_copy)

        panel_df = pd.concat(panel_frames, ignore_index=True)

        n_periods = len(data_dict)
        n_individuals = panel_df[id_vars].drop_duplicates().shape[0]

        counts = panel_df.groupby(id_vars).size()
        balanced = (counts == n_periods).all()

        first_period = list(data_dict.keys())[0]
        last_period = list(data_dict.keys())[-1]
        n_first = len(data_dict[first_period][id_vars].drop_duplicates())
        n_last = len(data_dict[last_period][id_vars].drop_duplicates())
        attrition_rate = 1 - (n_last / n_first) if n_first > 0 else 0

        return PanelDataResult(
            panel_df=panel_df,
            n_periods=n_periods,
            n_individuals=n_individuals,
            balanced=balanced,
            attrition_rate=attrition_rate,
            metadata={"periods": list(data_dict.keys())},
        )


def create_panel_data(
    data_dict: Dict[str, pd.DataFrame], id_vars: List[str], time_var: str = "año"
) -> pd.DataFrame:
    """Función de conveniencia para crear panel"""
    creator = PanelCreator()
    result = creator.create_panel(data_dict, id_vars, time_var)
    return result.panel_df
