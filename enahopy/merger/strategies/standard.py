"""Estrategia estándar de merge"""

from typing import List

import pandas as pd

from .base import MergeStrategy


class StandardMergeStrategy(MergeStrategy):
    """Estrategia estándar usando pandas merge"""

    def merge(
        self, left: pd.DataFrame, right: pd.DataFrame, keys: List[str], **kwargs
    ) -> pd.DataFrame:
        """Merge estándar usando pandas"""
        how = kwargs.pop("how", "left")
        return pd.merge(left, right, on=keys, how=how, **kwargs)
