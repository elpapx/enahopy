"""
Advanced Poverty Analysis - ANALYZE Phase
=========================================

Comprehensive poverty analysis tools for policy research including:
- Advanced poverty measurement and decomposition
- Multidimensional poverty analysis
- Poverty trends and dynamics
- Pro-poor growth analysis
- Vulnerability assessment
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from scipy import optimize, stats
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class PovertyLine:
    """Represents different types of poverty lines"""

    absolute: float
    food_poverty: float
    extreme_poverty: float
    relative_60: float
    relative_50: float
    name: str
    currency: str = "PEN"
    year: int = 2022


@dataclass
class PovertyMeasures:
    """Standard poverty measures"""

    headcount_ratio: float  # P0
    poverty_gap: float  # P1
    poverty_severity: float  # P2
    watts_index: float
    sen_index: float
    sample_size: int
    confidence_intervals: Dict[str, Tuple[float, float]] = None


@dataclass
class MultidimensionalResult:
    """Multidimensional poverty analysis result"""

    mpi: float  # Multidimensional Poverty Index
    headcount: float  # Multidimensional headcount
    intensity: float  # Average intensity among poor
    dimensional_contributions: Dict[str, float]
    raw_headcounts: Dict[str, float]
    censored_headcounts: Dict[str, float]


class AdvancedPovertyAnalyzer:
    """
    Advanced poverty analyzer with comprehensive econometric methods
    for policy research and academic analysis
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_comprehensive_measures(
        self,
        income: pd.Series,
        poverty_line: float,
        weights: Optional[pd.Series] = None,
        alpha_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive set of poverty measures

        Args:
            income: Income series
            poverty_line: Poverty line
            weights: Survey weights
            alpha_values: Additional FGT alpha parameters

        Returns:
            Dictionary with all poverty measures
        """
        if alpha_values is None:
            alpha_values = [0, 1, 2, 3]

        # Basic FGT measures
        fgt_measures = {}
        for alpha in alpha_values:
            fgt_measures[f"fgt_{alpha}"] = self._calculate_fgt(income, poverty_line, alpha, weights)

        # Additional measures
        measures = {
            "fgt_measures": fgt_measures,
            "watts_index": self._calculate_watts(income, poverty_line, weights),
            "sen_index": self._calculate_sen_index(income, poverty_line, weights),
            "chakravarty_index": self._calculate_chakravarty(income, poverty_line, weights),
            "takayama_index": self._calculate_takayama(income, poverty_line, weights),
            "poverty_line": poverty_line,
            "sample_size": len(income),
        }

        # Calculate confidence intervals if scipy available
        if SCIPY_AVAILABLE:
            measures["confidence_intervals"] = self._calculate_confidence_intervals(
                income, poverty_line, weights
            )

        # Calculate distributional statistics
        poor_income = income[income < poverty_line]
        if len(poor_income) > 0:
            measures["poor_statistics"] = {
                "mean_income": (
                    (poor_income * weights[income < poverty_line]).sum()
                    / weights[income < poverty_line].sum()
                    if weights is not None
                    else poor_income.mean()
                ),
                "median_income": self._weighted_quantile(
                    poor_income,
                    0.5,
                    weights[income < poverty_line] if weights is not None else None,
                ),
                "income_gap_ratio": 1
                - (
                    (poor_income * weights[income < poverty_line]).sum()
                    / weights[income < poverty_line].sum()
                    if weights is not None
                    else poor_income.mean()
                )
                / poverty_line,
                "gini_among_poor": self._calculate_gini_subset(
                    poor_income, weights[income < poverty_line] if weights is not None else None
                ),
            }

        return measures

    def poverty_decomposition_by_groups(
        self,
        df: pd.DataFrame,
        income_col: str,
        group_col: str,
        poverty_line: float,
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decompose poverty by population groups

        Args:
            df: DataFrame with data
            income_col: Income column name
            group_col: Grouping variable
            poverty_line: Poverty line
            weight_col: Weight column name

        Returns:
            Detailed decomposition results
        """
        weights = df[weight_col] if weight_col else None

        # Overall poverty
        overall_poverty = self._calculate_fgt(df[income_col], poverty_line, 0, weights)
        overall_gap = self._calculate_fgt(df[income_col], poverty_line, 1, weights)
        overall_severity = self._calculate_fgt(df[income_col], poverty_line, 2, weights)

        # Group-specific analysis
        group_results = []
        total_contribution_headcount = 0
        total_contribution_gap = 0
        total_contribution_severity = 0

        total_pop = weights.sum() if weights is not None else len(df)

        for group_name, group_data in df.groupby(group_col):
            group_income = group_data[income_col]
            group_weights = group_data[weight_col] if weight_col else None

            group_pop = group_weights.sum() if group_weights is not None else len(group_data)
            pop_share = group_pop / total_pop

            # Group poverty measures
            group_headcount = self._calculate_fgt(group_income, poverty_line, 0, group_weights)
            group_gap = self._calculate_fgt(group_income, poverty_line, 1, group_weights)
            group_severity = self._calculate_fgt(group_income, poverty_line, 2, group_weights)

            # Contributions to overall poverty
            contribution_headcount = pop_share * group_headcount
            contribution_gap = pop_share * group_gap
            contribution_severity = pop_share * group_severity

            total_contribution_headcount += contribution_headcount
            total_contribution_gap += contribution_gap
            total_contribution_severity += contribution_severity

            group_results.append(
                {
                    "group": group_name,
                    "population_share": pop_share,
                    "headcount_ratio": group_headcount,
                    "poverty_gap": group_gap,
                    "poverty_severity": group_severity,
                    "contribution_to_headcount": contribution_headcount,
                    "contribution_to_gap": contribution_gap,
                    "contribution_to_severity": contribution_severity,
                    "relative_risk": (
                        group_headcount / overall_poverty if overall_poverty > 0 else 0
                    ),
                    "sample_size": len(group_data),
                }
            )

        return {
            "overall_measures": {
                "headcount_ratio": overall_poverty,
                "poverty_gap": overall_gap,
                "poverty_severity": overall_severity,
            },
            "group_results": group_results,
            "decomposition_check": {
                "headcount_sum": total_contribution_headcount,
                "gap_sum": total_contribution_gap,
                "severity_sum": total_contribution_severity,
                "headcount_residual": abs(overall_poverty - total_contribution_headcount),
                "gap_residual": abs(overall_gap - total_contribution_gap),
                "severity_residual": abs(overall_severity - total_contribution_severity),
            },
        }

    def growth_incidence_curve(
        self,
        income_baseline: pd.Series,
        income_endline: pd.Series,
        weights: Optional[pd.Series] = None,
        percentiles: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Growth Incidence Curve (GIC) for distributional analysis

        Args:
            income_baseline: Baseline period income
            income_endline: Endline period income
            weights: Survey weights
            percentiles: Percentiles to analyze

        Returns:
            Growth incidence analysis
        """
        if percentiles is None:
            percentiles = np.arange(1, 100)

        baseline_quantiles = []
        endline_quantiles = []
        growth_rates = []

        for p in percentiles:
            p_decimal = p / 100

            baseline_q = self._weighted_quantile(income_baseline, p_decimal, weights)
            endline_q = self._weighted_quantile(income_endline, p_decimal, weights)

            baseline_quantiles.append(baseline_q)
            endline_quantiles.append(endline_q)

            # Calculate growth rate
            if baseline_q > 0:
                growth_rate = ((endline_q - baseline_q) / baseline_q) * 100
            else:
                growth_rate = np.nan

            growth_rates.append(growth_rate)

        # Calculate pro-poor growth metrics
        mean_growth_baseline = (
            ((income_baseline * weights).sum() / weights.sum())
            if weights is not None
            else income_baseline.mean()
        )
        mean_growth_endline = (
            ((income_endline * weights).sum() / weights.sum())
            if weights is not None
            else income_endline.mean()
        )
        overall_growth = ((mean_growth_endline - mean_growth_baseline) / mean_growth_baseline) * 100

        # Pro-poor indices
        bottom_40_growth = np.nanmean(growth_rates[:40])  # Bottom 40%
        top_10_growth = np.nanmean(growth_rates[90:])  # Top 10%

        pro_poor_index_1 = bottom_40_growth / overall_growth if overall_growth != 0 else np.nan
        pro_poor_index_2 = bottom_40_growth - overall_growth

        return {
            "percentiles": percentiles.tolist(),
            "baseline_quantiles": baseline_quantiles,
            "endline_quantiles": endline_quantiles,
            "growth_rates": growth_rates,
            "summary_statistics": {
                "overall_growth_rate": overall_growth,
                "bottom_40_growth": bottom_40_growth,
                "top_10_growth": top_10_growth,
                "median_growth": np.nanmedian(growth_rates),
                "pro_poor_index_1": pro_poor_index_1,
                "pro_poor_index_2": pro_poor_index_2,
                "is_pro_poor": pro_poor_index_1 > 1 if not np.isnan(pro_poor_index_1) else None,
            },
        }

    def vulnerability_analysis(
        self,
        df: pd.DataFrame,
        income_col: str,
        poverty_line: float,
        vulnerability_threshold: float = 1.5,
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze vulnerability to poverty (expected poverty)

        Args:
            df: DataFrame with data
            income_col: Income column name
            poverty_line: Poverty line
            vulnerability_threshold: Vulnerability line (multiple of poverty line)
            weight_col: Weight column name

        Returns:
            Vulnerability analysis results
        """
        vulnerability_line = poverty_line * vulnerability_threshold
        income = df[income_col]
        weights = df[weight_col] if weight_col else None

        # Basic vulnerability measures
        poor_mask = income < poverty_line
        vulnerable_mask = income < vulnerability_line
        secure_mask = income >= vulnerability_line

        if weights is not None:
            total_weight = weights.sum()
            poor_rate = weights[poor_mask].sum() / total_weight
            vulnerable_rate = weights[vulnerable_mask].sum() / total_weight
            secure_rate = weights[secure_mask].sum() / total_weight
        else:
            poor_rate = poor_mask.mean()
            vulnerable_rate = vulnerable_mask.mean()
            secure_rate = secure_mask.mean()

        # Vulnerability categories
        vulnerable_non_poor = vulnerable_mask & ~poor_mask
        vulnerable_non_poor_rate = (
            (weights[vulnerable_non_poor].sum() / weights.sum())
            if weights is not None
            else vulnerable_non_poor.mean()
        )

        # Income statistics by vulnerability status
        categories = {
            "poor": income[poor_mask],
            "vulnerable_non_poor": income[vulnerable_non_poor],
            "secure": income[secure_mask],
        }

        category_stats = {}
        for category, cat_income in categories.items():
            if len(cat_income) > 0:
                cat_weights = weights[cat_income.index] if weights is not None else None
                category_stats[category] = {
                    "count": len(cat_income),
                    "share": len(cat_income) / len(income),
                    "mean_income": (
                        (cat_income * cat_weights).sum() / cat_weights.sum()
                        if cat_weights is not None
                        else cat_income.mean()
                    ),
                    "median_income": self._weighted_quantile(cat_income, 0.5, cat_weights),
                    "income_gap_from_vulnerability": vulnerability_line
                    - (
                        (cat_income * cat_weights).sum() / cat_weights.sum()
                        if cat_weights is not None
                        else cat_income.mean()
                    ),
                }

        return {
            "vulnerability_line": vulnerability_line,
            "poverty_line": poverty_line,
            "rates": {
                "poor": poor_rate,
                "vulnerable": vulnerable_rate,
                "vulnerable_non_poor": vulnerable_non_poor_rate,
                "secure": secure_rate,
            },
            "category_statistics": category_stats,
            "vulnerability_gap": self._calculate_vulnerability_gap(
                income, vulnerability_line, weights
            ),
            "sample_size": len(df),
        }

    def multidimensional_poverty_analysis(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Dict[str, Any]],
        weights: Optional[pd.Series] = None,
        k: float = 0.33,
    ) -> MultidimensionalResult:
        """
        Calculate Multidimensional Poverty Index (MPI) following Alkire-Foster method

        Args:
            df: DataFrame with indicator data
            indicators: Dictionary defining indicators, thresholds, and weights
            weights: Survey weights
            k: Poverty cutoff (proportion of weighted indicators)

        Returns:
            Multidimensional poverty results
        """
        deprivation_matrix = pd.DataFrame(index=df.index)
        indicator_weights = {}

        # Calculate deprivations for each indicator
        for indicator, config in indicators.items():
            threshold = config["threshold"]
            weight = config["weight"]
            direction = config.get("direction", "less")  # 'less' means below threshold is deprived

            if direction == "less":
                deprivation_matrix[indicator] = (df[indicator] < threshold).astype(int)
            else:
                deprivation_matrix[indicator] = (df[indicator] > threshold).astype(int)

            indicator_weights[indicator] = weight

        # Calculate deprivation scores (weighted)
        deprivation_scores = pd.Series(0.0, index=df.index)
        for indicator, weight in indicator_weights.items():
            deprivation_scores += deprivation_matrix[indicator] * weight

        # Identify multidimensionally poor (deprivation score >= k)
        poor_mask = deprivation_scores >= k

        # Calculate censored deprivation matrix (only for poor households)
        censored_deprivations = deprivation_matrix.copy()
        censored_deprivations[~poor_mask] = 0

        # Calculate MPI components
        if weights is not None:
            total_weight = weights.sum()
            # Headcount ratio (H)
            headcount = weights[poor_mask].sum() / total_weight

            # Intensity (A) - average deprivation score among poor
            if headcount > 0:
                poor_weights = weights[poor_mask]
                intensity = (
                    deprivation_scores[poor_mask] * poor_weights
                ).sum() / poor_weights.sum()
            else:
                intensity = 0

        else:
            headcount = poor_mask.mean()
            intensity = deprivation_scores[poor_mask].mean() if headcount > 0 else 0

        # MPI = H × A
        mpi = headcount * intensity

        # Dimensional contributions
        dimensional_contributions = {}
        raw_headcounts = {}
        censored_headcounts = {}

        for indicator, weight in indicator_weights.items():
            # Raw headcount (uncensored)
            raw_headcounts[indicator] = (
                (weights[deprivation_matrix[indicator] == 1].sum() / total_weight)
                if weights is not None
                else deprivation_matrix[indicator].mean()
            )

            # Censored headcount (only among MPI poor)
            censored_headcounts[indicator] = (
                (weights[censored_deprivations[indicator] == 1].sum() / total_weight)
                if weights is not None
                else censored_deprivations[indicator].mean()
            )

            # Contribution to MPI
            dimensional_contributions[indicator] = (
                (censored_headcounts[indicator] * weight) / mpi if mpi > 0 else 0
            )

        return MultidimensionalResult(
            mpi=mpi,
            headcount=headcount,
            intensity=intensity,
            dimensional_contributions=dimensional_contributions,
            raw_headcounts=raw_headcounts,
            censored_headcounts=censored_headcounts,
        )

    # Private helper methods
    def _calculate_fgt(
        self,
        income: pd.Series,
        poverty_line: float,
        alpha: float,
        weights: Optional[pd.Series] = None,
    ) -> float:
        """Calculate Foster-Greer-Thorbecke poverty measure"""
        poor_mask = income < poverty_line
        if not poor_mask.any():
            return 0.0

        gap = np.maximum(0, poverty_line - income) / poverty_line
        fgt = gap**alpha

        if weights is not None:
            return (fgt * weights).sum() / weights.sum()
        else:
            return fgt.mean()

    def _calculate_watts(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Watts poverty index"""
        poor_mask = income < poverty_line
        poor_income = income[poor_mask]

        if len(poor_income) == 0:
            return 0.0

        # Avoid log(0)
        poor_income = np.maximum(poor_income, 1e-6)
        log_ratio = np.log(poverty_line / poor_income)

        if weights is not None:
            poor_weights = weights[poor_mask]
            total_weight = weights.sum()
            return (log_ratio * poor_weights).sum() / total_weight
        else:
            return log_ratio.mean() * (len(poor_income) / len(income))

    def _calculate_sen_index(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Sen poverty index"""
        headcount = self._calculate_fgt(income, poverty_line, 0, weights)
        gap = self._calculate_fgt(income, poverty_line, 1, weights)

        poor_mask = income < poverty_line
        poor_income = income[poor_mask]

        if len(poor_income) == 0:
            return 0.0

        # Gini coefficient among poor
        gini_poor = self._calculate_gini_subset(
            poor_income, weights[poor_mask] if weights is not None else None
        )

        return headcount * (gap + (1 - gap) * gini_poor)

    def _calculate_chakravarty(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Chakravarty poverty measure"""
        poor_mask = income < poverty_line
        poor_income = income[poor_mask]

        if len(poor_income) == 0:
            return 0.0

        gap_ratio = poor_income / poverty_line
        chakravarty_values = 1 - np.sqrt(gap_ratio)

        if weights is not None:
            poor_weights = weights[poor_mask]
            return (chakravarty_values * poor_weights).sum() / weights.sum()
        else:
            return chakravarty_values.mean() * (len(poor_income) / len(income))

    def _calculate_takayama(
        self, income: pd.Series, poverty_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Takayama poverty measure"""
        poor_mask = income < poverty_line
        poor_income = income[poor_mask]

        if len(poor_income) == 0:
            return 0.0

        gap = poverty_line - poor_income
        normalized_gap = gap / poverty_line

        if weights is not None:
            poor_weights = weights[poor_mask]
            total_gap = (gap * poor_weights).sum()
            total_weight = weights.sum()
            return total_gap / (poverty_line * total_weight)
        else:
            return gap.sum() / (poverty_line * len(income))

    def _calculate_vulnerability_gap(
        self, income: pd.Series, vulnerability_line: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate vulnerability gap (similar to poverty gap but for vulnerability line)"""
        vulnerable_mask = income < vulnerability_line
        if not vulnerable_mask.any():
            return 0.0

        gap = np.maximum(0, vulnerability_line - income) / vulnerability_line

        if weights is not None:
            return (gap * weights).sum() / weights.sum()
        else:
            return gap.mean()

    def _calculate_gini_subset(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Gini coefficient for subset of population"""
        if len(income) <= 1:
            return 0.0

        sorted_indices = income.argsort()
        sorted_income = income.iloc[sorted_indices]

        if weights is not None:
            sorted_weights = weights.iloc[sorted_indices]
            cumsum_weights = sorted_weights.cumsum()
            cumsum_income = (sorted_income * sorted_weights).cumsum()

            total_weight = sorted_weights.sum()
            total_income = cumsum_income.iloc[-1]

            if total_income == 0:
                return 0.0

            # Calculate using trapezoidal rule
            lorenz_x = cumsum_weights / total_weight
            lorenz_y = cumsum_income / total_income

            # Add origin
            lorenz_x = pd.concat([pd.Series([0]), lorenz_x])
            lorenz_y = pd.concat([pd.Series([0]), lorenz_y])

            area_under_curve = np.trapz(lorenz_y, lorenz_x)
            gini = 1 - 2 * area_under_curve
        else:
            n = len(sorted_income)
            cumsum = sorted_income.cumsum()
            total = sorted_income.sum()

            if total == 0:
                return 0.0

            gini = 1 - (2 / (n * total)) * cumsum.sum() + (1 / n)

        return max(0.0, min(1.0, gini))

    def _weighted_quantile(
        self, values: pd.Series, quantile: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate weighted quantile"""
        if weights is None:
            return values.quantile(quantile)

        sorted_indices = values.argsort()
        sorted_values = values.iloc[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices]

        cumsum = sorted_weights.cumsum()
        cutoff = quantile * sorted_weights.sum()

        idx = (cumsum >= cutoff).argmax()
        return sorted_values.iloc[idx]

    def _calculate_confidence_intervals(
        self,
        income: pd.Series,
        poverty_line: float,
        weights: Optional[pd.Series] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for poverty measures using bootstrap"""
        if not SCIPY_AVAILABLE:
            return {}

        n_bootstrap = 1000
        bootstrap_results = {"headcount": [], "gap": [], "severity": []}

        np.random.seed(42)
        n_sample = len(income)

        for _ in range(n_bootstrap):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_sample, n_sample, replace=True)
            bootstrap_income = income.iloc[bootstrap_indices]
            bootstrap_weights = weights.iloc[bootstrap_indices] if weights is not None else None

            # Calculate measures
            bootstrap_results["headcount"].append(
                self._calculate_fgt(bootstrap_income, poverty_line, 0, bootstrap_weights)
            )
            bootstrap_results["gap"].append(
                self._calculate_fgt(bootstrap_income, poverty_line, 1, bootstrap_weights)
            )
            bootstrap_results["severity"].append(
                self._calculate_fgt(bootstrap_income, poverty_line, 2, bootstrap_weights)
            )

        # Calculate confidence intervals
        confidence_intervals = {}
        for measure, values in bootstrap_results.items():
            lower = np.percentile(values, (alpha / 2) * 100)
            upper = np.percentile(values, (1 - alpha / 2) * 100)
            confidence_intervals[measure] = (lower, upper)

        return confidence_intervals


class PovertyDecomposition:
    """Methods for poverty decomposition analysis"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def shapley_decomposition(
        self,
        df: pd.DataFrame,
        income_col: str,
        factor_cols: List[str],
        poverty_line: float,
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Shapley value decomposition of poverty changes

        Args:
            df: DataFrame with data
            income_col: Income column name
            factor_cols: List of factor columns
            poverty_line: Poverty line
            weight_col: Weight column name

        Returns:
            Shapley decomposition results
        """
        from itertools import combinations

        analyzer = AdvancedPovertyAnalyzer(self.logger)

        # Calculate baseline poverty (with all factors)
        baseline_poverty = analyzer._calculate_fgt(
            df[income_col], poverty_line, 0, df[weight_col] if weight_col else None
        )

        # Calculate marginal contributions for each factor
        shapley_values = {}

        for factor in factor_cols:
            marginal_contributions = []

            # Calculate marginal contribution for all possible coalitions
            other_factors = [f for f in factor_cols if f != factor]

            for r in range(len(other_factors) + 1):
                for coalition in combinations(other_factors, r):
                    # Calculate poverty without current factor
                    coalition_factors = list(coalition)

                    # This is a simplified version - in practice, you would need
                    # counterfactual income without the specific factor
                    # For demonstration, we'll use a placeholder calculation

                    # Placeholder: remove factor contribution (this needs domain-specific implementation)
                    modified_income = (
                        df[income_col] - df[factor] if factor in df.columns else df[income_col]
                    )

                    poverty_without_factor = analyzer._calculate_fgt(
                        modified_income, poverty_line, 0, df[weight_col] if weight_col else None
                    )

                    marginal_contribution = baseline_poverty - poverty_without_factor
                    marginal_contributions.append(marginal_contribution)

            # Calculate Shapley value as average marginal contribution
            shapley_values[factor] = np.mean(marginal_contributions)

        return {
            "baseline_poverty": baseline_poverty,
            "shapley_values": shapley_values,
            "total_explained": sum(shapley_values.values()),
            "explanation_rate": (
                sum(shapley_values.values()) / baseline_poverty if baseline_poverty > 0 else 0
            ),
        }

    def oaxaca_blinder_decomposition(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        income_col: str,
        covariate_cols: List[str],
        poverty_line: float,
    ) -> Dict[str, Any]:
        """
        Oaxaca-Blinder decomposition for poverty differences between groups

        Args:
            df1: First group data
            df2: Second group data
            income_col: Income column name
            covariate_cols: Covariate columns
            poverty_line: Poverty line

        Returns:
            Decomposition results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Oaxaca-Blinder decomposition")

        from sklearn.linear_model import LinearRegression

        # Calculate poverty rates
        analyzer = AdvancedPovertyAnalyzer(self.logger)
        poverty1 = analyzer._calculate_fgt(df1[income_col], poverty_line, 0)
        poverty2 = analyzer._calculate_fgt(df2[income_col], poverty_line, 0)
        poverty_diff = poverty1 - poverty2

        # Prepare data for regression
        X1 = df1[covariate_cols]
        X2 = df2[covariate_cols]
        y1 = df1[income_col]
        y2 = df2[income_col]

        # Fit regressions
        reg1 = LinearRegression().fit(X1, y1)
        reg2 = LinearRegression().fit(X2, y2)

        # Calculate means
        mean_X1 = X1.mean()
        mean_X2 = X2.mean()

        # Decomposition
        # Explained part: difference in characteristics
        explained = (mean_X1 - mean_X2) @ reg2.coef_

        # Unexplained part: difference in returns to characteristics
        unexplained = mean_X1 @ (reg1.coef_ - reg2.coef_)

        # Convert to poverty terms (approximation)
        explained_poverty_effect = explained / poverty_line * poverty_diff
        unexplained_poverty_effect = unexplained / poverty_line * poverty_diff

        return {
            "poverty_difference": poverty_diff,
            "poverty_group1": poverty1,
            "poverty_group2": poverty2,
            "explained_component": explained_poverty_effect,
            "unexplained_component": unexplained_poverty_effect,
            "explained_share": explained_poverty_effect / poverty_diff if poverty_diff != 0 else 0,
            "unexplained_share": (
                unexplained_poverty_effect / poverty_diff if poverty_diff != 0 else 0
            ),
            "detailed_contributions": dict(zip(covariate_cols, (mean_X1 - mean_X2) * reg2.coef_)),
        }


class PovertyTrends:
    """Analysis of poverty trends over time"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def analyze_poverty_trends(
        self,
        df: pd.DataFrame,
        time_col: str,
        income_col: str,
        poverty_line: float,
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze poverty trends over time

        Args:
            df: DataFrame with panel data
            time_col: Time column name
            income_col: Income column name
            poverty_line: Poverty line
            weight_col: Weight column name

        Returns:
            Trend analysis results
        """
        analyzer = AdvancedPovertyAnalyzer(self.logger)

        time_series_results = []

        for time_period in sorted(df[time_col].unique()):
            period_data = df[df[time_col] == time_period]

            if len(period_data) == 0:
                continue

            period_income = period_data[income_col]
            period_weights = period_data[weight_col] if weight_col else None

            # Calculate comprehensive poverty measures
            measures = analyzer.calculate_comprehensive_measures(
                period_income, poverty_line, period_weights
            )

            period_result = {
                "time_period": time_period,
                "sample_size": len(period_data),
                **measures["fgt_measures"],
                "watts_index": measures["watts_index"],
                "sen_index": measures["sen_index"],
            }

            time_series_results.append(period_result)

        # Calculate trend statistics
        trend_df = pd.DataFrame(time_series_results)

        if len(trend_df) > 1:
            # Calculate growth rates
            for measure in ["fgt_0", "fgt_1", "fgt_2", "watts_index"]:
                if measure in trend_df.columns:
                    trend_df[f"{measure}_growth"] = trend_df[measure].pct_change() * 100

            # Trend analysis
            if SCIPY_AVAILABLE:
                trend_analysis = {}
                for measure in ["fgt_0", "fgt_1", "fgt_2"]:
                    if measure in trend_df.columns:
                        x = np.arange(len(trend_df))
                        y = trend_df[measure].values

                        # Linear trend
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                        trend_analysis[measure] = {
                            "slope": slope,
                            "r_squared": r_value**2,
                            "p_value": p_value,
                            "trend_direction": (
                                "decreasing"
                                if slope < 0
                                else "increasing"
                                if slope > 0
                                else "stable"
                            ),
                            "annual_change": slope,
                            "total_change": trend_df[measure].iloc[-1] - trend_df[measure].iloc[0],
                            "percent_change": (
                                (trend_df[measure].iloc[-1] - trend_df[measure].iloc[0])
                                / trend_df[measure].iloc[0]
                            )
                            * 100,
                        }
            else:
                trend_analysis = {}
        else:
            trend_analysis = {}

        return {
            "time_series": time_series_results,
            "trend_analysis": trend_analysis,
            "summary": {
                "time_periods": len(time_series_results),
                "first_period": trend_df["time_period"].iloc[0] if len(trend_df) > 0 else None,
                "last_period": trend_df["time_period"].iloc[-1] if len(trend_df) > 0 else None,
                "overall_trend": (
                    self._classify_overall_trend(trend_analysis)
                    if trend_analysis
                    else "insufficient_data"
                ),
            },
        }

    def poverty_mobility_analysis(
        self, df: pd.DataFrame, id_col: str, time_col: str, income_col: str, poverty_line: float
    ) -> Dict[str, Any]:
        """
        Analyze poverty mobility (transitions in and out of poverty)

        Args:
            df: Panel DataFrame
            id_col: Individual/household ID column
            time_col: Time column
            income_col: Income column
            poverty_line: Poverty line

        Returns:
            Mobility analysis results
        """
        # Create poverty status indicator
        df = df.copy()
        df["is_poor"] = (df[income_col] < poverty_line).astype(int)

        # Pivot to have time periods as columns
        mobility_data = df.pivot(index=id_col, columns=time_col, values="is_poor")

        # Remove observations with missing data
        mobility_data = mobility_data.dropna()

        if mobility_data.shape[1] < 2:
            return {"error": "Need at least 2 time periods for mobility analysis"}

        # Calculate transition matrices for consecutive periods
        transition_matrices = {}

        time_periods = sorted(mobility_data.columns)

        for i in range(len(time_periods) - 1):
            t1, t2 = time_periods[i], time_periods[i + 1]

            transition_data = mobility_data[[t1, t2]].dropna()

            # Create transition matrix
            transition_matrix = pd.crosstab(
                transition_data[t1], transition_data[t2], normalize="index"
            )

            transition_matrices[f"{t1}_to_{t2}"] = {
                "matrix": transition_matrix,
                "non_poor_to_poor": (
                    transition_matrix.loc[0, 1]
                    if 1 in transition_matrix.columns and 0 in transition_matrix.index
                    else 0
                ),
                "poor_to_non_poor": (
                    transition_matrix.loc[1, 0]
                    if 0 in transition_matrix.columns and 1 in transition_matrix.index
                    else 0
                ),
                "poor_to_poor": (
                    transition_matrix.loc[1, 1]
                    if 1 in transition_matrix.columns and 1 in transition_matrix.index
                    else 0
                ),
                "non_poor_to_non_poor": (
                    transition_matrix.loc[0, 0]
                    if 0 in transition_matrix.columns and 0 in transition_matrix.index
                    else 0
                ),
                "sample_size": len(transition_data),
            }

        # Overall mobility statistics
        if len(time_periods) >= 2:
            first_period = time_periods[0]
            last_period = time_periods[-1]

            panel_data = mobility_data[[first_period, last_period]].dropna()

            never_poor = ((panel_data == 0).all(axis=1)).sum()
            always_poor = ((panel_data == 1).all(axis=1)).sum()
            escaped_poverty = (
                (panel_data[first_period] == 1) & (panel_data[last_period] == 0)
            ).sum()
            entered_poverty = (
                (panel_data[first_period] == 0) & (panel_data[last_period] == 1)
            ).sum()

            total_panel = len(panel_data)

            mobility_summary = {
                "never_poor_rate": never_poor / total_panel,
                "always_poor_rate": always_poor / total_panel,
                "escaped_poverty_rate": escaped_poverty / total_panel,
                "entered_poverty_rate": entered_poverty / total_panel,
                "mobility_rate": (escaped_poverty + entered_poverty) / total_panel,
                "panel_sample_size": total_panel,
            }
        else:
            mobility_summary = {}

        return {
            "transition_matrices": transition_matrices,
            "mobility_summary": mobility_summary,
            "time_periods_analyzed": len(time_periods),
            "panel_balanced": True,  # Assuming balanced panel after dropna
        }

    def _classify_overall_trend(self, trend_analysis: Dict[str, Any]) -> str:
        """Classify overall poverty trend"""
        if "fgt_0" in trend_analysis:
            headcount_trend = trend_analysis["fgt_0"]

            if headcount_trend["p_value"] < 0.05:  # Statistically significant
                if headcount_trend["slope"] < -0.01:  # Decreasing by more than 1pp per period
                    return "strongly_decreasing"
                elif headcount_trend["slope"] < 0:
                    return "decreasing"
                elif headcount_trend["slope"] > 0.01:
                    return "strongly_increasing"
                elif headcount_trend["slope"] > 0:
                    return "increasing"

            return "stable"

        return "unknown"


class MultidimensionalPoverty:
    """Advanced multidimensional poverty analysis"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def create_enaho_mpi(
        self, df: pd.DataFrame, weight_col: Optional[str] = None
    ) -> MultidimensionalResult:
        """
        Create MPI specifically designed for ENAHO data structure

        Args:
            df: ENAHO DataFrame
            weight_col: Weight column name

        Returns:
            MPI results
        """
        # Define ENAHO-specific MPI indicators
        indicators = {
            # Education dimension (1/3 weight)
            "years_education": {
                "threshold": 6,  # Less than 6 years is deprived
                "weight": 1 / 6,
                "direction": "less",
            },
            "school_attendance": {
                "threshold": 1,  # Not attending is deprived
                "weight": 1 / 6,
                "direction": "less",
            },
            # Health dimension (1/3 weight)
            "nutrition_child": {
                "threshold": 0,  # Malnutrition present
                "weight": 1 / 6,
                "direction": "greater",
            },
            "mortality_child": {
                "threshold": 0,  # Child mortality in household
                "weight": 1 / 6,
                "direction": "greater",
            },
            # Living standards (1/3 weight)
            "electricity": {
                "threshold": 1,  # No electricity access
                "weight": 1 / 18,
                "direction": "less",
            },
            "water_improved": {
                "threshold": 1,  # No improved water
                "weight": 1 / 18,
                "direction": "less",
            },
            "sanitation": {
                "threshold": 1,  # No improved sanitation
                "weight": 1 / 18,
                "direction": "less",
            },
            "floor_material": {"threshold": 1, "weight": 1 / 18, "direction": "less"},  # Dirt floor
            "cooking_fuel": {
                "threshold": 1,  # Traditional cooking fuel
                "weight": 1 / 18,
                "direction": "less",
            },
            "assets_index": {
                "threshold": 0.2,  # Low asset ownership
                "weight": 1 / 18,
                "direction": "less",
            },
        }

        # Filter indicators that exist in dataframe
        available_indicators = {k: v for k, v in indicators.items() if k in df.columns}

        if not available_indicators:
            self.logger.warning("No MPI indicators found in dataframe")
            return MultidimensionalResult(0, 0, 0, {}, {}, {})

        # Calculate MPI using Alkire-Foster method
        analyzer = AdvancedPovertyAnalyzer(self.logger)
        weights = df[weight_col] if weight_col else None

        return analyzer.multidimensional_poverty_analysis(df, available_indicators, weights, k=0.33)


def create_poverty_analyzer(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Factory function to create comprehensive poverty analysis suite

    Args:
        logger: Optional logger instance

    Returns:
        Dictionary with poverty analysis tools
    """
    return {
        "analyzer": AdvancedPovertyAnalyzer(logger),
        "decomposition": PovertyDecomposition(logger),
        "trends": PovertyTrends(logger),
        "multidimensional": MultidimensionalPoverty(logger),
    }


__all__ = [
    "PovertyLine",
    "PovertyMeasures",
    "MultidimensionalResult",
    "AdvancedPovertyAnalyzer",
    "PovertyDecomposition",
    "PovertyTrends",
    "MultidimensionalPoverty",
    "create_poverty_analyzer",
]
