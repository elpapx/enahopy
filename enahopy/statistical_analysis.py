"""
ENAHO Statistical Analysis Utilities
===================================

Advanced statistical analysis tools for ENAHO data including poverty indicators,
inequality measures, and econometric utilities for household survey analysis.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    pass

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class PovertyIndicators:
    """Calculator for poverty and welfare indicators from ENAHO data"""

    def __init__(self, currency_symbol: str = "S/.", logger: Optional[logging.Logger] = None):
        self.currency_symbol = currency_symbol
        self.logger = logger or logging.getLogger(__name__)

    def poverty_headcount_ratio(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate poverty headcount ratio (P0)

        Args:
            income: Series of household/individual incomes
            poverty_line: Poverty line threshold
            weights: Optional survey weights

        Returns:
            Poverty headcount ratio (proportion below poverty line)
        """
        poor_mask = income < poverty_line

        if weights is not None:
            total_weight = weights.sum()
            poor_weight = weights[poor_mask].sum()
            return poor_weight / total_weight if total_weight > 0 else 0.0
        else:
            return poor_mask.mean()

    def poverty_gap_ratio(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate poverty gap ratio (P1)

        Args:
            income: Series of household/individual incomes
            poverty_line: Poverty line threshold
            weights: Optional survey weights

        Returns:
            Poverty gap ratio (average proportional shortfall)
        """
        income < poverty_line
        gap = np.maximum(0, poverty_line - income) / poverty_line

        if weights is not None:
            total_weight = weights.sum()
            weighted_gap = (gap * weights).sum()
            return weighted_gap / total_weight if total_weight > 0 else 0.0
        else:
            return gap.mean()

    def poverty_severity_ratio(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate poverty severity ratio (P2) - squared poverty gap

        Args:
            income: Series of household/individual incomes
            poverty_line: Poverty line threshold
            weights: Optional survey weights

        Returns:
            Poverty severity ratio (squared poverty gap)
        """
        income < poverty_line
        gap = np.maximum(0, poverty_line - income) / poverty_line
        severity = gap**2

        if weights is not None:
            total_weight = weights.sum()
            weighted_severity = (severity * weights).sum()
            return weighted_severity / total_weight if total_weight > 0 else 0.0
        else:
            return severity.mean()

    def fgt_poverty_measure(
        self,
        income: pd.Series,
        poverty_line: float,
        alpha: float = 0,
        weights: Optional[pd.Series] = None,
    ) -> float:
        """
        Calculate Foster-Greer-Thorbecke (FGT) poverty measure

        Args:
            income: Series of household/individual incomes
            poverty_line: Poverty line threshold
            alpha: FGT parameter (0=headcount, 1=gap, 2=severity)
            weights: Optional survey weights

        Returns:
            FGT poverty measure
        """
        if alpha == 0:
            return self.poverty_headcount_ratio(income, poverty_line, weights)
        elif alpha == 1:
            return self.poverty_gap_ratio(income, poverty_line, weights)
        elif alpha == 2:
            return self.poverty_severity_ratio(income, poverty_line, weights)
        else:
            # General FGT formula
            gap = np.maximum(0, poverty_line - income) / poverty_line
            fgt = gap**alpha

            if weights is not None:
                total_weight = weights.sum()
                weighted_fgt = (fgt * weights).sum()
                return weighted_fgt / total_weight if total_weight > 0 else 0.0
            else:
                return fgt.mean()

    def watts_poverty_index(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Watts poverty index

        Args:
            income: Series of household/individual incomes
            poverty_line: Poverty line threshold
            weights: Optional survey weights

        Returns:
            Watts poverty index
        """
        poor_mask = income < poverty_line
        poor_income = income[poor_mask]

        if len(poor_income) == 0:
            return 0.0

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-6
        poor_income = np.maximum(poor_income, epsilon)

        log_ratio = np.log(poverty_line / poor_income)

        if weights is not None:
            poor_weights = weights[poor_mask]
            total_weight = weights.sum()
            weighted_index = (log_ratio * poor_weights).sum()
            return weighted_index / total_weight if total_weight > 0 else 0.0
        else:
            return log_ratio.mean() * (len(poor_income) / len(income))

    def poverty_profile(
        self,
        df: pd.DataFrame,
        income_col: str,
        poverty_line: float,
        group_cols: List[str],
        weight_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate poverty profile by demographic groups

        Args:
            df: DataFrame with household data
            income_col: Name of income column
            poverty_line: Poverty line threshold
            group_cols: Columns to group by
            weight_col: Optional weight column

        Returns:
            DataFrame with poverty indicators by group
        """
        df[weight_col] if weight_col else None

        # Calculate poverty indicators for each group
        profile_data = []

        for name, group in df.groupby(group_cols):
            group_income = group[income_col]
            group_weights = group[weight_col] if weight_col else None

            profile = {
                "group": str(name) if isinstance(name, tuple) else name,
                "population": len(group),
                "headcount_ratio": self.poverty_headcount_ratio(
                    group_income, poverty_line, group_weights
                ),
                "poverty_gap": self.poverty_gap_ratio(group_income, poverty_line, group_weights),
                "poverty_severity": self.poverty_severity_ratio(
                    group_income, poverty_line, group_weights
                ),
                "watts_index": self.watts_poverty_index(group_income, poverty_line, group_weights),
                "mean_income": (
                    (group_income * group_weights).sum() / group_weights.sum()
                    if group_weights is not None
                    else group_income.mean()
                ),
                "median_income": self._weighted_quantile(group_income, 0.5, group_weights),
            }

            profile_data.append(profile)

        return pd.DataFrame(profile_data)

    def _weighted_quantile(
        self, values: pd.Series, quantile: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate weighted quantile"""
        if weights is None:
            return values.quantile(quantile)

        # Sort values and weights
        sorted_indices = values.argsort()
        sorted_values = values.iloc[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices]

        # Calculate cumulative weights
        cumsum = sorted_weights.cumsum()
        cutoff = quantile * sorted_weights.sum()

        # Find quantile
        idx = (cumsum >= cutoff).argmax()
        return sorted_values.iloc[idx]


class InequalityMeasures:
    """Calculator for income inequality measures"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def gini_coefficient(self, income: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """
        Calculate Gini coefficient

        Args:
            income: Series of incomes
            weights: Optional survey weights

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if len(income) == 0:
            return 0.0

        # Remove negative incomes and sort
        income = income[income >= 0].reset_index(drop=True)
        if weights is not None:
            weights = weights[income >= 0].reset_index(drop=True)

        if len(income) == 0:
            return 0.0

        # Sort by income
        sorted_indices = income.argsort()
        sorted_income = income.iloc[sorted_indices]

        if weights is not None:
            sorted_weights = weights.iloc[sorted_indices]
            # Weighted Gini calculation
            cumsum_weights = sorted_weights.cumsum()
            cumsum_income = (sorted_income * sorted_weights).cumsum()

            total_weight = sorted_weights.sum()
            total_income = cumsum_income.iloc[-1]

            if total_income == 0:
                return 0.0

            # Calculate Gini using trapezoidal rule
            lorenz_y = cumsum_income / total_income
            lorenz_x = cumsum_weights / total_weight

            # Add origin point
            lorenz_x = pd.concat([pd.Series([0]), lorenz_x])
            lorenz_y = pd.concat([pd.Series([0]), lorenz_y])

            # Calculate area under Lorenz curve using trapezoidal rule
            area_under_curve = np.trapz(lorenz_y, lorenz_x)
            gini = 1 - 2 * area_under_curve

        else:
            # Unweighted Gini calculation
            n = len(sorted_income)
            cumsum = sorted_income.cumsum()
            total = sorted_income.sum()

            if total == 0:
                return 0.0

            gini = 1 - (2 / (n * total)) * cumsum.sum() + (1 / n)

        return max(0.0, min(1.0, gini))  # Ensure valid range

    def theil_index(self, income: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """
        Calculate Theil T index (GE(1))

        Args:
            income: Series of incomes
            weights: Optional survey weights

        Returns:
            Theil T index
        """
        if len(income) == 0:
            return 0.0

        # Remove zero and negative incomes
        positive_mask = income > 0
        income = income[positive_mask]

        if weights is not None:
            weights = weights[positive_mask]

        if len(income) == 0:
            return 0.0

        if weights is not None:
            mean_income = (income * weights).sum() / weights.sum()
            income_share = income / mean_income
            log_share = np.log(income_share)
            theil = (income_share * log_share * weights).sum() / weights.sum()
        else:
            mean_income = income.mean()
            income_share = income / mean_income
            log_share = np.log(income_share)
            theil = (income_share * log_share).mean()

        return max(0.0, theil)

    def atkinson_index(
        self, income: pd.Series, epsilon: float = 1.0, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Atkinson inequality index

        Args:
            income: Series of incomes
            epsilon: Inequality aversion parameter
            weights: Optional survey weights

        Returns:
            Atkinson index (0 = perfect equality, 1 = perfect inequality)
        """
        if len(income) == 0:
            return 0.0

        # Remove zero and negative incomes
        positive_mask = income > 0
        income = income[positive_mask]

        if weights is not None:
            weights = weights[positive_mask]

        if len(income) == 0:
            return 0.0

        if weights is not None:
            mean_income = (income * weights).sum() / weights.sum()
            total_weight = weights.sum()
        else:
            mean_income = income.mean()
            total_weight = len(income)

        if mean_income <= 0:
            return 0.0

        if epsilon == 1.0:
            # Special case: geometric mean
            if weights is not None:
                log_income = np.log(income)
                geo_mean = np.exp((log_income * weights).sum() / total_weight)
            else:
                geo_mean = stats.gmean(income)
            atkinson = 1 - geo_mean / mean_income
        else:
            # General case
            if weights is not None:
                power_mean = ((income ** (1 - epsilon)) * weights).sum() / total_weight
            else:
                power_mean = (income ** (1 - epsilon)).mean()

            power_mean = power_mean ** (1 / (1 - epsilon))
            atkinson = 1 - power_mean / mean_income

        return max(0.0, min(1.0, atkinson))

    def generalized_entropy_index(
        self, income: pd.Series, alpha: float = 1.0, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Generalized Entropy (GE) index

        Args:
            income: Series of incomes
            alpha: Sensitivity parameter (0=Theil L, 1=Theil T, 2=half CV squared)
            weights: Optional survey weights

        Returns:
            Generalized entropy index
        """
        if len(income) == 0:
            return 0.0

        # Remove zero and negative incomes
        positive_mask = income > 0
        income = income[positive_mask]

        if weights is not None:
            weights = weights[positive_mask]

        if len(income) == 0:
            return 0.0

        if weights is not None:
            mean_income = (income * weights).sum() / weights.sum()
            total_weight = weights.sum()
        else:
            mean_income = income.mean()
            total_weight = len(income)

        if mean_income <= 0:
            return 0.0

        income_ratio = income / mean_income

        if alpha == 0:
            # Theil L (mean log deviation)
            if weights is not None:
                ge = -((np.log(income_ratio) * weights).sum() / total_weight)
            else:
                ge = -(np.log(income_ratio)).mean()
        elif alpha == 1:
            # Theil T
            if weights is not None:
                ge = (income_ratio * np.log(income_ratio) * weights).sum() / total_weight
            else:
                ge = (income_ratio * np.log(income_ratio)).mean()
        else:
            # General case
            if weights is not None:
                ge = ((income_ratio**alpha - 1) * weights).sum() / (
                    alpha * (alpha - 1) * total_weight
                )
            else:
                ge = (income_ratio**alpha - 1).mean() / (alpha * (alpha - 1))

        return max(0.0, ge)

    def palma_ratio(self, income: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """
        Calculate Palma ratio (top 10% / bottom 40% income share)

        Args:
            income: Series of incomes
            weights: Optional survey weights

        Returns:
            Palma ratio
        """
        if len(income) == 0:
            return 0.0

        # Calculate income shares
        if weights is not None:
            # Weighted income shares
            df_temp = pd.DataFrame({"income": income, "weights": weights})
            df_temp = df_temp.sort_values("income")
            df_temp["cumsum_weights"] = df_temp["weights"].cumsum()
            total_weight = df_temp["weights"].sum()
            total_income = (df_temp["income"] * df_temp["weights"]).sum()

            # Find percentiles
            p40_cutoff = 0.4 * total_weight
            p90_cutoff = 0.9 * total_weight

            bottom_40 = df_temp[df_temp["cumsum_weights"] <= p40_cutoff]
            top_10 = df_temp[df_temp["cumsum_weights"] > p90_cutoff]

            bottom_40_income = (bottom_40["income"] * bottom_40["weights"]).sum()
            top_10_income = (top_10["income"] * top_10["weights"]).sum()

        else:
            # Unweighted calculation
            sorted_income = income.sort_values()
            n = len(sorted_income)

            bottom_40_end = int(0.4 * n)
            top_10_start = int(0.9 * n)

            bottom_40_income = sorted_income[:bottom_40_end].sum()
            top_10_income = sorted_income[top_10_start:].sum()
            total_income = sorted_income.sum()

        if total_income == 0 or bottom_40_income == 0:
            return np.inf

        bottom_40_share = bottom_40_income / total_income
        top_10_share = top_10_income / total_income

        return top_10_share / bottom_40_share if bottom_40_share > 0 else np.inf


class WelfareAnalysis:
    """Advanced welfare and living standards analysis"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.poverty_calc = PovertyIndicators()
        self.inequality_calc = InequalityMeasures()

    def welfare_dominance_test(
        self,
        income1: pd.Series,
        income2: pd.Series,
        poverty_lines: List[float],
        weights1: Optional[pd.Series] = None,
        weights2: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Test for first-order stochastic dominance (welfare dominance)

        Args:
            income1: First income distribution
            income2: Second income distribution
            poverty_lines: List of poverty lines to test
            weights1: Weights for first distribution
            weights2: Weights for second distribution

        Returns:
            Dictionary with dominance test results
        """
        results = {
            "first_order_dominance": None,
            "poverty_comparisons": [],
            "dominance_violations": 0,
        }

        # Test at each poverty line
        distribution1_better = 0
        distribution2_better = 0

        for poverty_line in poverty_lines:
            p1 = self.poverty_calc.poverty_headcount_ratio(income1, poverty_line, weights1)
            p2 = self.poverty_calc.poverty_headcount_ratio(income2, poverty_line, weights2)

            comparison = {
                "poverty_line": poverty_line,
                "distribution1_poverty": p1,
                "distribution2_poverty": p2,
                "distribution1_better": p1 < p2,
            }

            results["poverty_comparisons"].append(comparison)

            if p1 < p2:
                distribution1_better += 1
            elif p2 < p1:
                distribution2_better += 1

        # Determine dominance
        if distribution1_better > 0 and distribution2_better == 0:
            results["first_order_dominance"] = "distribution1"
        elif distribution2_better > 0 and distribution1_better == 0:
            results["first_order_dominance"] = "distribution2"
        else:
            results["first_order_dominance"] = "no_dominance"
            results["dominance_violations"] = min(distribution1_better, distribution2_better)

        return results

    def decompose_inequality_by_group(
        self, df: pd.DataFrame, income_col: str, group_col: str, weight_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decompose inequality by population groups (Theil decomposition)

        Args:
            df: DataFrame with data
            income_col: Income column name
            group_col: Grouping variable
            weight_col: Optional weight column

        Returns:
            Dictionary with decomposition results
        """
        weights = df[weight_col] if weight_col else None

        # Calculate overall inequality
        overall_theil = self.inequality_calc.theil_index(df[income_col], weights)

        # Calculate within-group and between-group inequality
        within_group_theil = 0
        between_group_theil = 0

        # Overall mean income
        if weights is not None:
            overall_mean = (df[income_col] * weights).sum() / weights.sum()
            total_weight = weights.sum()
        else:
            overall_mean = df[income_col].mean()
            total_weight = len(df)

        group_stats = []

        for group_name, group_data in df.groupby(group_col):
            group_income = group_data[income_col]
            group_weights = group_data[weight_col] if weight_col else None

            # Group statistics
            if group_weights is not None:
                group_mean = (group_income * group_weights).sum() / group_weights.sum()
                group_weight = group_weights.sum()
                group_share = group_weight / total_weight
            else:
                group_mean = group_income.mean()
                group_weight = len(group_income)
                group_share = group_weight / total_weight

            # Within-group Theil
            group_theil = self.inequality_calc.theil_index(group_income, group_weights)

            # Contribution to within-group inequality
            income_share = (group_mean * group_weight) / (overall_mean * total_weight)
            within_contribution = income_share * group_theil
            within_group_theil += within_contribution

            # Between-group component
            if overall_mean > 0:
                between_contribution = income_share * np.log(group_mean / overall_mean)
                between_group_theil += between_contribution

            group_stats.append(
                {
                    "group": group_name,
                    "population_share": group_share,
                    "income_share": income_share,
                    "mean_income": group_mean,
                    "theil_index": group_theil,
                    "within_contribution": within_contribution,
                }
            )

        return {
            "overall_theil": overall_theil,
            "within_group_theil": within_group_theil,
            "between_group_theil": between_group_theil,
            "within_group_share": within_group_theil / overall_theil if overall_theil > 0 else 0,
            "between_group_share": between_group_theil / overall_theil if overall_theil > 0 else 0,
            "group_statistics": group_stats,
        }

    def calculate_social_welfare(
        self, income: pd.Series, epsilon: float = 1.0, weights: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Atkinson social welfare function

        Args:
            income: Series of incomes
            epsilon: Inequality aversion parameter
            weights: Optional survey weights

        Returns:
            Social welfare level (equally distributed equivalent income)
        """
        if len(income) == 0:
            return 0.0

        # Remove zero and negative incomes
        positive_mask = income > 0
        income = income[positive_mask]

        if weights is not None:
            weights = weights[positive_mask]

        if len(income) == 0:
            return 0.0

        if epsilon == 1.0:
            # Logarithmic case
            if weights is not None:
                log_income = np.log(income)
                welfare = np.exp((log_income * weights).sum() / weights.sum())
            else:
                welfare = stats.gmean(income)
        else:
            # General case
            if weights is not None:
                power_sum = ((income ** (1 - epsilon)) * weights).sum()
                total_weight = weights.sum()
                welfare = (power_sum / total_weight) ** (1 / (1 - epsilon))
            else:
                power_mean = (income ** (1 - epsilon)).mean()
                welfare = power_mean ** (1 / (1 - epsilon))

        return welfare


def create_statistical_analyzer() -> Dict[str, Any]:
    """Factory function to create statistical analysis tools"""
    return {
        "poverty_indicators": PovertyIndicators(),
        "inequality_measures": InequalityMeasures(),
        "welfare_analysis": WelfareAnalysis(),
    }


def quick_poverty_analysis(
    df: pd.DataFrame, income_col: str, poverty_line: float, weight_col: Optional[str] = None
) -> Dict[str, Any]:
    """Quick poverty analysis with key indicators"""
    poverty_calc = PovertyIndicators()
    inequality_calc = InequalityMeasures()

    income = df[income_col]
    weights = df[weight_col] if weight_col else None

    return {
        "poverty_headcount": poverty_calc.poverty_headcount_ratio(income, poverty_line, weights),
        "poverty_gap": poverty_calc.poverty_gap_ratio(income, poverty_line, weights),
        "poverty_severity": poverty_calc.poverty_severity_ratio(income, poverty_line, weights),
        "gini_coefficient": inequality_calc.gini_coefficient(income, weights),
        "theil_index": inequality_calc.theil_index(income, weights),
        "palma_ratio": inequality_calc.palma_ratio(income, weights),
        "sample_size": len(df),
        "mean_income": income.mean(),
        "median_income": income.median(),
    }
