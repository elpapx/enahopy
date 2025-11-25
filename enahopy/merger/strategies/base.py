"""Estrategia base para merge"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class MergeStrategy(ABC):
    """Estrategia base abstracta para merge"""

    @abstractmethod
    def merge(
        self, left: pd.DataFrame, right: pd.DataFrame, keys: List[str], **kwargs
    ) -> pd.DataFrame:
        """Ejecuta la estrategia de merge"""
