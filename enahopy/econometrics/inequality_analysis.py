"""
Advanced Inequality Analysis - ANALYZE Phase
============================================

Comprehensive inequality analysis tools for policy research including:
- Advanced inequality decomposition methods
- Social mobility analysis
- Polarization measures
- Inequality of opportunity analysis
- Redistribution and tax policy analysis
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class InequalityMeasures:
    """Comprehensive inequality measures"""

    gini: float
    theil_t: float
    theil_l: float
    atkinson_05: float
    atkinson_1: float
    atkinson_2: float
    coefficient_variation: float
    percentile_ratios: Dict[str, float]
    palma_ratio: float
    top_shares: Dict[str, float]
    bottom_shares: Dict[str, float]


@dataclass
class DecompositionResult:
    """Inequality decomposition result"""

    within_group: float
    between_group: float
    overlap: float
    within_share: float
    between_share: float
    overlap_share: float
    group_contributions: Dict[str, float]


@dataclass
class MobilityMeasures:
    """Social mobility measures"""

    rank_correlation: float
    income_correlation: float
    transition_matrix: pd.DataFrame
    mobility_indices: Dict[str, float]
    directional_mobility: Dict[str, float]


class AdvancedInequalityAnalyzer:
    """
    Advanced inequality analyzer with comprehensive econometric methods
    for policy research and academic analysis
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_comprehensive_measures(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> InequalityMeasures:
        """
        Calculate comprehensive set of inequality measures

        Args:
            income: Income series
            weights: Survey weights

        Returns:
            Complete set of inequality measures
        """
        # Remove non-positive incomes
        valid_mask = income > 0
        income = income[valid_mask]
        if weights is not None:
            weights = weights[valid_mask]

        if len(income) == 0:
            return self._empty_measures()

        # Basic measures
        gini = self._calculate_gini(income, weights)
        theil_t = self._calculate_theil_t(income, weights)
        theil_l = self._calculate_theil_l(income, weights)

        # Atkinson indices
        atkinson_05 = self._calculate_atkinson(income, 0.5, weights)
        atkinson_1 = self._calculate_atkinson(income, 1.0, weights)
        atkinson_2 = self._calculate_atkinson(income, 2.0, weights)

        # Coefficient of variation
        cv = self._calculate_coefficient_variation(income, weights)

        # Percentile ratios
        percentile_ratios = self._calculate_percentile_ratios(income, weights)

        # Palma ratio
        palma = self._calculate_palma_ratio(income, weights)

        # Top and bottom shares
        top_shares = self._calculate_top_shares(income, weights)
        bottom_shares = self._calculate_bottom_shares(income, weights)

        return InequalityMeasures(
            gini=gini,
            theil_t=theil_t,
            theil_l=theil_l,
            atkinson_05=atkinson_05,
            atkinson_1=atkinson_1,
            atkinson_2=atkinson_2,
            coefficient_variation=cv,
            percentile_ratios=percentile_ratios,
            palma_ratio=palma,
            top_shares=top_shares,
            bottom_shares=bottom_shares,
        )

    def inequality_decomposition_by_groups(
        self,
        df: pd.DataFrame,
        income_col: str,
        group_col: str,
        weight_col: Optional[str] = None,
        measure: str = "theil_t",
    ) -> DecompositionResult:
        """
        Decompose inequality by population groups

        Args:
            df: DataFrame with data
            income_col: Income column name
            group_col: Grouping variable
            weight_col: Weight column name
            measure: Inequality measure to decompose ('theil_t', 'theil_l', 'gini')

        Returns:
            Decomposition results
        """
        income = df[income_col]
        weights = df[weight_col] if weight_col else None

        # Overall inequality
        if measure == "theil_t":
            overall_inequality = self._calculate_theil_t(income, weights)
        elif measure == "theil_l":
            overall_inequality = self._calculate_theil_l(income, weights)
        elif measure == "gini":
            overall_inequality = self._calculate_gini(income, weights)
        else:
            raise ValueError(f"Unsupported measure: {measure}")

        # Group-specific calculations
        within_group = 0
        between_group = 0
        overlap = 0

        total_income = (income * weights).sum() if weights is not None else income.sum()
        total_pop = weights.sum() if weights is not None else len(income)
        mean_income = total_income / total_pop

        group_contributions = {}

        for group_name, group_data in df.groupby(group_col):
            group_income = group_data[income_col]
            group_weights = group_data[weight_col] if weight_col else None

            group_pop = group_weights.sum() if group_weights is not None else len(group_data)
            group_income_total = (
                (group_income * group_weights).sum()
                if group_weights is not None
                else group_income.sum()
            )

            pop_share = group_pop / total_pop
            income_share = group_income_total / total_income
            group_mean = group_income_total / group_pop

            # Within-group contribution
            if measure == "theil_t":
                group_inequality = self._calculate_theil_t(group_income, group_weights)
                within_group += income_share * group_inequality

                # Between-group contribution
                if mean_income > 0:
                    between_group += income_share * np.log(group_mean / mean_income)

            elif measure == "theil_l":
                group_inequality = self._calculate_theil_l(group_income, group_weights)
                within_group += pop_share * group_inequality

                # Between-group contribution
                if group_mean > 0 and mean_income > 0:
                    between_group += pop_share * np.log(mean_income / group_mean)

            elif measure == "gini":
                # Gini decomposition (approximate)
                group_gini = self._calculate_gini(group_income, group_weights)
                within_group += pop_share * income_share * group_gini

                # Between-group Gini approximation
                # This is an approximation - exact Gini decomposition is more complex
                between_group += (
                    pop_share * income_share * abs(group_mean - mean_income) / mean_income
                )

            group_contributions[group_name] = {
                "population_share": pop_share,
                "income_share": income_share,
                "mean_income": group_mean,
                "within_contribution": (
                    income_share * group_inequality
                    if measure in ["theil_t"]
                    else pop_share * group_inequality
                ),
            }

        # Calculate overlap (residual for Gini)
        if measure == "gini":
            overlap = overall_inequality - within_group - between_group
        else:
            overlap = 0  # Theil measures decompose exactly

        # Calculate shares
        within_share = within_group / overall_inequality if overall_inequality > 0 else 0
        between_share = between_group / overall_inequality if overall_inequality > 0 else 0
        overlap_share = overlap / overall_inequality if overall_inequality > 0 else 0

        return DecompositionResult(
            within_group=within_group,
            between_group=between_group,
            overlap=overlap,
            within_share=within_share,
            between_share=between_share,
            overlap_share=overlap_share,
            group_contributions=group_contributions,
        )

    def polarization_analysis(
        self, income: pd.Series, weights: Optional[pd.Series] = None, alpha: float = 1.6
    ) -> Dict[str, Any]:
        """
        Calculate polarization measures (Esteban-Ray index)

        Args:
            income: Income series
            weights: Survey weights
            alpha: Polarization sensitivity parameter (1 < alpha <= 2)

        Returns:
            Polarization analysis results
        """
        if not (1 < alpha <= 2):
            raise ValueError("Alpha must be between 1 and 2")

        # Remove non-positive incomes
        valid_mask = income > 0
        income = income[valid_mask]
        if weights is not None:
            weights = weights[valid_mask]

        if len(income) == 0:
            return {"polarization_index": 0, "error": "No valid income data"}

        # Sort income and weights
        sorted_indices = income.argsort()
        sorted_income = income.iloc[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices] if weights is not None else None

        # Calculate mean income
        if weights is not None:
            mean_income = (sorted_income * sorted_weights).sum() / sorted_weights.sum()
        else:
            mean_income = sorted_income.mean()

        # Esteban-Ray polarization index
        polarization = 0
        n = len(sorted_income)

        for i in range(n):
            for j in range(n):
                yi = sorted_income.iloc[i]
                yj = sorted_income.iloc[j]

                if weights is not None:
                    pi = sorted_weights.iloc[i] / sorted_weights.sum()
                    pj = sorted_weights.iloc[j] / sorted_weights.sum()
                else:
                    pi = pj = 1 / n

                # Identification component (similar incomes)
                identification = pi**alpha

                # Alienation component (different incomes)
                alienation = abs(yi - yj) / mean_income

                polarization += identification * pj * alienation

        # Additional polarization measures
        # Duclos-Esteban-Ray (DER) polarization
        der_polarization = self._calculate_der_polarization(income, weights)

        # Wolfson polarization index
        wolfson_index = self._calculate_wolfson_polarization(income, weights)

        return {
            "esteban_ray_polarization": polarization,
            "der_polarization": der_polarization,
            "wolfson_polarization": wolfson_index,
            "alpha_parameter": alpha,
            "mean_income": mean_income,
            "sample_size": len(income),
        }

    def inequality_of_opportunity_analysis(
        self,
        df: pd.DataFrame,
        income_col: str,
        circumstance_cols: List[str],
        effort_cols: List[str],
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze inequality of opportunity using Roemer's framework

        Args:
            df: DataFrame with data
            income_col: Income column name
            circumstance_cols: Circumstance variables (beyond individual control)
            effort_cols: Effort variables (within individual control)
            weight_col: Weight column name

        Returns:
            Inequality of opportunity results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for inequality of opportunity analysis")

        # Prepare data
        X_circumstances = df[circumstance_cols].fillna(0)
        X_effort = df[effort_cols].fillna(0)
        y_income = df[income_col]
        weights = df[weight_col] if weight_col else None

        # Remove missing income observations
        valid_mask = y_income.notna() & (y_income > 0)
        X_circumstances = X_circumstances[valid_mask]
        X_effort = X_effort[valid_mask]
        y_income = y_income[valid_mask]
        if weights is not None:
            weights = weights[valid_mask]

        # Standardize variables
        scaler_circ = StandardScaler()
        scaler_effort = StandardScaler()

        X_circumstances_scaled = scaler_circ.fit_transform(X_circumstances)
        X_effort_scaled = scaler_effort.fit_transform(X_effort)

        # Method 1: Ex-ante approach (predict income from circumstances only)
        rf_circumstances = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_circumstances.fit(X_circumstances_scaled, y_income, sample_weight=weights)

        predicted_income_circumstances = rf_circumstances.predict(X_circumstances_scaled)

        # Method 2: Ex-post approach (predict income from circumstances and effort)
        X_full = np.hstack([X_circumstances_scaled, X_effort_scaled])
        rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_full.fit(X_full, y_income, sample_weight=weights)

        rf_full.predict(X_full)

        # Calculate inequality measures
        actual_inequality = self._calculate_gini(y_income, weights)

        # Ex-ante inequality of opportunity
        opportunity_inequality_ex_ante = self._calculate_gini(
            pd.Series(predicted_income_circumstances), weights
        )

        # Ex-post approach: residual inequality after accounting for circumstances
        residual_income = y_income - predicted_income_circumstances
        residual_inequality = self._calculate_gini(
            residual_income + residual_income.mean(), weights
        )

        opportunity_inequality_ex_post = actual_inequality - residual_inequality

        # Relative inequality of opportunity
        relative_iop_ex_ante = opportunity_inequality_ex_ante / actual_inequality
        relative_iop_ex_post = opportunity_inequality_ex_post / actual_inequality

        # Variable importance for circumstances
        circumstance_importance = dict(
            zip(circumstance_cols, rf_circumstances.feature_importances_)
        )

        return {
            "total_inequality": actual_inequality,
            "inequality_of_opportunity": {
                "ex_ante": opportunity_inequality_ex_ante,
                "ex_post": opportunity_inequality_ex_post,
                "relative_ex_ante": relative_iop_ex_ante,
                "relative_ex_post": relative_iop_ex_post,
            },
            "inequality_of_effort": {
                "absolute": residual_inequality,
                "relative": 1 - relative_iop_ex_ante,
            },
            "circumstance_importance": circumstance_importance,
            "model_performance": {
                "circumstances_r2": rf_circumstances.score(X_circumstances_scaled, y_income),
                "full_model_r2": rf_full.score(X_full, y_income),
            },
            "sample_size": len(y_income),
        }

    def redistribution_analysis(
        self, gross_income: pd.Series, net_income: pd.Series, weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze redistributive effects of taxes and transfers

        Args:
            gross_income: Pre-tax/transfer income
            net_income: Post-tax/transfer income
            weights: Survey weights

        Returns:
            Redistribution analysis results
        """
        # Remove invalid observations
        valid_mask = (gross_income > 0) & (net_income > 0)
        gross_income = gross_income[valid_mask]
        net_income = net_income[valid_mask]
        if weights is not None:
            weights = weights[valid_mask]

        if len(gross_income) == 0:
            return {"error": "No valid income data"}

        # Calculate inequality measures
        gross_gini = self._calculate_gini(gross_income, weights)
        net_gini = self._calculate_gini(net_income, weights)

        # Reynolds-Smolensky redistributive effect
        rs_effect = gross_gini - net_gini

        # Relative redistributive effect
        relative_effect = rs_effect / gross_gini if gross_gini > 0 else 0

        # Kakwani progressivity index
        # First, calculate concentration index of net income with respect to gross income rank
        gross_ranks = gross_income.rank(method="min")
        concentration_index = self._calculate_concentration_index(net_income, gross_ranks, weights)
        kakwani_index = concentration_index - gross_gini

        # Average tax rate
        total_gross = (gross_income * weights).sum() if weights is not None else gross_income.sum()
        total_net = (net_income * weights).sum() if weights is not None else net_income.sum()
        avg_tax_rate = 1 - (total_net / total_gross)

        # Decomposition of redistributive effect
        # RE = Progressivity × Average Tax Rate
        theoretical_re = kakwani_index * avg_tax_rate

        # Reranking effect (difference between actual and theoretical RE)
        reranking_effect = theoretical_re - rs_effect

        # Marginal tax rates by income deciles
        decile_analysis = self._analyze_by_deciles(gross_income, net_income, weights)

        return {
            "inequality_measures": {
                "gross_gini": gross_gini,
                "net_gini": net_gini,
                "inequality_reduction": rs_effect,
                "relative_reduction": relative_effect,
            },
            "progressivity_measures": {
                "kakwani_index": kakwani_index,
                "concentration_index": concentration_index,
                "average_tax_rate": avg_tax_rate,
            },
            "decomposition": {
                "reynolds_smolensky_effect": rs_effect,
                "theoretical_effect": theoretical_re,
                "reranking_effect": reranking_effect,
                "progressivity_component": kakwani_index,
                "tax_rate_component": avg_tax_rate,
            },
            "decile_analysis": decile_analysis,
            "sample_size": len(gross_income),
        }

    # Private helper methods
    def _calculate_gini(self, income: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """Calculate Gini coefficient"""
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

    def _calculate_theil_t(self, income: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """Calculate Theil T index"""
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

        income_share = income / mean_income
        log_share = np.log(income_share)

        if weights is not None:
            theil = (income_share * log_share * weights).sum() / total_weight
        else:
            theil = (income_share * log_share).mean()

        return max(0.0, theil)

    def _calculate_theil_l(self, income: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """Calculate Theil L index (mean log deviation)"""
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

        log_ratio = np.log(mean_income / income)

        if weights is not None:
            theil_l = (log_ratio * weights).sum() / total_weight
        else:
            theil_l = log_ratio.mean()

        return max(0.0, theil_l)

    def _calculate_atkinson(
        self, income: pd.Series, epsilon: float, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Atkinson index"""
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
            # Logarithmic case
            log_income = np.log(income)
            if weights is not None:
                geo_mean = np.exp((log_income * weights).sum() / total_weight)
            else:
                geo_mean = np.exp(log_income.mean())
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

    def _calculate_coefficient_variation(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate coefficient of variation"""
        if weights is not None:
            mean_income = (income * weights).sum() / weights.sum()
            variance = ((income - mean_income) ** 2 * weights).sum() / weights.sum()
        else:
            mean_income = income.mean()
            variance = income.var()

        if mean_income == 0:
            return 0.0

        return np.sqrt(variance) / mean_income

    def _calculate_percentile_ratios(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate various percentile ratios"""

        def weighted_quantile(values, quantile, weights=None):
            if weights is None:
                return values.quantile(quantile)

            sorted_indices = values.argsort()
            sorted_values = values.iloc[sorted_indices]
            sorted_weights = weights.iloc[sorted_indices]

            cumsum = sorted_weights.cumsum()
            cutoff = quantile * sorted_weights.sum()

            idx = (cumsum >= cutoff).argmax()
            return sorted_values.iloc[idx]

        percentiles = {}
        for p in [10, 50, 90, 95, 99]:
            percentiles[f"p{p}"] = weighted_quantile(income, p / 100, weights)

        ratios = {
            "p90_p10": (
                percentiles["p90"] / percentiles["p10"] if percentiles["p10"] > 0 else np.inf
            ),
            "p95_p5": weighted_quantile(income, 0.95, weights)
            / weighted_quantile(income, 0.05, weights),
            "p99_p1": percentiles["p99"] / weighted_quantile(income, 0.01, weights),
            "p90_p50": (
                percentiles["p90"] / percentiles["p50"] if percentiles["p50"] > 0 else np.inf
            ),
            "p50_p10": (
                percentiles["p50"] / percentiles["p10"] if percentiles["p10"] > 0 else np.inf
            ),
        }

        return ratios

    def _calculate_palma_ratio(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Palma ratio"""

        def weighted_quantile(values, quantile, weights=None):
            if weights is None:
                return values.quantile(quantile)

            sorted_indices = values.argsort()
            sorted_values = values.iloc[sorted_indices]
            sorted_weights = weights.iloc[sorted_indices]

            cumsum = sorted_weights.cumsum()
            cutoff = quantile * sorted_weights.sum()

            idx = (cumsum >= cutoff).argmax()
            return sorted_values.iloc[idx]

        # Calculate income shares
        if weights is not None:
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

    def _calculate_top_shares(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate top income shares"""

        def calculate_share(threshold):
            if weights is not None:
                df_temp = pd.DataFrame({"income": income, "weights": weights})
                df_temp = df_temp.sort_values("income")
                df_temp["cumsum_weights"] = df_temp["weights"].cumsum()
                total_weight = df_temp["weights"].sum()
                total_income = (df_temp["income"] * df_temp["weights"]).sum()

                cutoff_weight = (1 - threshold) * total_weight
                top_group = df_temp[df_temp["cumsum_weights"] > cutoff_weight]
                top_income = (top_group["income"] * top_group["weights"]).sum()
            else:
                sorted_income = income.sort_values(ascending=False)
                n = len(sorted_income)
                top_n = int(threshold * n)
                top_income = sorted_income[:top_n].sum()
                total_income = sorted_income.sum()

            return top_income / total_income if total_income > 0 else 0

        return {
            "top_1_percent": calculate_share(0.01),
            "top_5_percent": calculate_share(0.05),
            "top_10_percent": calculate_share(0.10),
            "top_20_percent": calculate_share(0.20),
        }

    def _calculate_bottom_shares(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate bottom income shares"""

        def calculate_share(threshold):
            if weights is not None:
                df_temp = pd.DataFrame({"income": income, "weights": weights})
                df_temp = df_temp.sort_values("income")
                df_temp["cumsum_weights"] = df_temp["weights"].cumsum()
                total_weight = df_temp["weights"].sum()
                total_income = (df_temp["income"] * df_temp["weights"]).sum()

                cutoff_weight = threshold * total_weight
                bottom_group = df_temp[df_temp["cumsum_weights"] <= cutoff_weight]
                bottom_income = (bottom_group["income"] * bottom_group["weights"]).sum()
            else:
                sorted_income = income.sort_values()
                n = len(sorted_income)
                bottom_n = int(threshold * n)
                bottom_income = sorted_income[:bottom_n].sum()
                total_income = sorted_income.sum()

            return bottom_income / total_income if total_income > 0 else 0

        return {
            "bottom_10_percent": calculate_share(0.10),
            "bottom_20_percent": calculate_share(0.20),
            "bottom_40_percent": calculate_share(0.40),
            "bottom_50_percent": calculate_share(0.50),
        }

    def _calculate_der_polarization(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Duclos-Esteban-Ray polarization"""
        # Simplified version - full implementation requires numerical integration
        gini = self._calculate_gini(income, weights)
        cv = self._calculate_coefficient_variation(income, weights)

        # Approximate DER polarization
        return 2 * gini * cv

    def _calculate_wolfson_polarization(
        self, income: pd.Series, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate Wolfson polarization index"""

        def weighted_quantile(values, quantile, weights=None):
            if weights is None:
                return values.quantile(quantile)

            sorted_indices = values.argsort()
            sorted_values = values.iloc[sorted_indices]
            sorted_weights = weights.iloc[sorted_indices]

            cumsum = sorted_weights.cumsum()
            cutoff = quantile * sorted_weights.sum()

            idx = (cumsum >= cutoff).argmax()
            return sorted_values.iloc[idx]

        median = weighted_quantile(income, 0.5, weights)
        mean = (income * weights).sum() / weights.sum() if weights is not None else income.mean()
        gini = self._calculate_gini(income, weights)

        # Wolfson polarization
        return (mean / median - 1) * (0.5 - gini)

    def _calculate_concentration_index(
        self, income: pd.Series, ranks: pd.Series, weights: Optional[pd.Series] = None
    ) -> float:
        """Calculate concentration index"""
        if weights is not None:
            # Weighted concentration index calculation
            df_temp = pd.DataFrame({"income": income, "ranks": ranks, "weights": weights})
            df_temp = df_temp.sort_values("ranks")

            total_weight = df_temp["weights"].sum()
            total_income = (df_temp["income"] * df_temp["weights"]).sum()

            df_temp["cumsum_weights"] = df_temp["weights"].cumsum()
            df_temp["cumsum_income"] = (df_temp["income"] * df_temp["weights"]).cumsum()

            # Concentration curve
            lorenz_x = df_temp["cumsum_weights"] / total_weight
            lorenz_y = df_temp["cumsum_income"] / total_income

            # Add origin
            lorenz_x = pd.concat([pd.Series([0]), lorenz_x])
            lorenz_y = pd.concat([pd.Series([0]), lorenz_y])

            area_under_curve = np.trapz(lorenz_y, lorenz_x)
            concentration = 1 - 2 * area_under_curve
        else:
            # Unweighted concentration index
            sorted_indices = ranks.argsort()
            sorted_income = income.iloc[sorted_indices]

            n = len(sorted_income)
            cumsum = sorted_income.cumsum()
            total = sorted_income.sum()

            if total == 0:
                return 0.0

            concentration = 1 - (2 / (n * total)) * cumsum.sum() + (1 / n)

        return concentration

    def _analyze_by_deciles(
        self, gross_income: pd.Series, net_income: pd.Series, weights: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze tax rates and redistribution by income deciles"""
        df_temp = pd.DataFrame({"gross": gross_income, "net": net_income})

        if weights is not None:
            df_temp["weights"] = weights

        df_temp["decile"] = pd.qcut(df_temp["gross"], 10, labels=False) + 1

        decile_stats = []

        for decile in range(1, 11):
            decile_data = df_temp[df_temp["decile"] == decile]

            if weights is not None:
                avg_gross = (decile_data["gross"] * decile_data["weights"]).sum() / decile_data[
                    "weights"
                ].sum()
                avg_net = (decile_data["net"] * decile_data["weights"]).sum() / decile_data[
                    "weights"
                ].sum()
            else:
                avg_gross = decile_data["gross"].mean()
                avg_net = decile_data["net"].mean()

            marginal_tax_rate = 1 - (avg_net / avg_gross) if avg_gross > 0 else 0

            decile_stats.append(
                {
                    "decile": decile,
                    "avg_gross_income": avg_gross,
                    "avg_net_income": avg_net,
                    "marginal_tax_rate": marginal_tax_rate,
                    "sample_size": len(decile_data),
                }
            )

        return decile_stats

    def _empty_measures(self) -> InequalityMeasures:
        """Return empty inequality measures for edge cases"""
        return InequalityMeasures(
            gini=0.0,
            theil_t=0.0,
            theil_l=0.0,
            atkinson_05=0.0,
            atkinson_1=0.0,
            atkinson_2=0.0,
            coefficient_variation=0.0,
            percentile_ratios={},
            palma_ratio=0.0,
            top_shares={},
            bottom_shares={},
        )


class InequalityDecomposition:
    """Specialized inequality decomposition methods"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def fields_ok_decomposition(
        self,
        df: pd.DataFrame,
        income_col: str,
        income_sources: List[str],
        weight_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fields-Ok decomposition of inequality by income sources

        Args:
            df: DataFrame with data
            income_col: Total income column
            income_sources: List of income source columns
            weight_col: Weight column name

        Returns:
            Source decomposition results
        """
        analyzer = AdvancedInequalityAnalyzer(self.logger)

        total_income = df[income_col]
        weights = df[weight_col] if weight_col else None

        # Overall inequality
        total_gini = analyzer._calculate_gini(total_income, weights)

        source_contributions = {}

        for source in income_sources:
            if source not in df.columns:
                continue

            source_income = df[source].fillna(0)

            # Calculate source Gini
            source_gini = (
                analyzer._calculate_gini(source_income, weights) if source_income.sum() > 0 else 0
            )

            # Calculate correlation with total income ranks
            total_ranks = total_income.rank()
            source_ranks = source_income.rank()

            if SCIPY_AVAILABLE:
                correlation = stats.spearmanr(total_ranks, source_ranks)[0]
            else:
                correlation = total_ranks.corr(source_ranks)

            # Income share
            total_income_sum = (
                (total_income * weights).sum() if weights is not None else total_income.sum()
            )
            source_income_sum = (
                (source_income * weights).sum() if weights is not None else source_income.sum()
            )

            income_share = source_income_sum / total_income_sum if total_income_sum > 0 else 0

            # Contribution to inequality
            contribution = income_share * source_gini * correlation

            source_contributions[source] = {
                "income_share": income_share,
                "gini": source_gini,
                "correlation": correlation,
                "contribution": contribution,
                "relative_contribution": contribution / total_gini if total_gini > 0 else 0,
            }

        return {
            "total_gini": total_gini,
            "source_contributions": source_contributions,
            "sum_of_contributions": sum(s["contribution"] for s in source_contributions.values()),
        }


class SocialMobility:
    """Social mobility analysis methods"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def intergenerational_mobility(
        self,
        df: pd.DataFrame,
        parent_income_col: str,
        child_income_col: str,
        weight_col: Optional[str] = None,
    ) -> MobilityMeasures:
        """
        Analyze intergenerational income mobility

        Args:
            df: DataFrame with data
            parent_income_col: Parent income column
            child_income_col: Child income column
            weight_col: Weight column name

        Returns:
            Mobility measures
        """
        # Remove missing observations
        mobility_data = df[[parent_income_col, child_income_col]].dropna()

        if len(mobility_data) < 10:
            raise ValueError("Insufficient data for mobility analysis")

        parent_income = mobility_data[parent_income_col]
        child_income = mobility_data[child_income_col]

        # Calculate ranks
        parent_ranks = parent_income.rank() / len(parent_income)
        child_ranks = child_income.rank() / len(child_income)

        # Rank correlation (Spearman)
        if SCIPY_AVAILABLE:
            rank_correlation, rank_p_value = stats.spearmanr(parent_ranks, child_ranks)
        else:
            rank_correlation = parent_ranks.corr(child_ranks)

        # Income correlation (log-log)
        log_parent = np.log(parent_income + 1)
        log_child = np.log(child_income + 1)

        if SCIPY_AVAILABLE:
            income_correlation, income_p_value = stats.pearsonr(log_parent, log_child)
        else:
            income_correlation = log_parent.corr(log_child)

        # Create transition matrix (quintiles)
        parent_quintiles = pd.qcut(parent_income, 5, labels=[1, 2, 3, 4, 5])
        child_quintiles = pd.qcut(child_income, 5, labels=[1, 2, 3, 4, 5])

        transition_matrix = pd.crosstab(parent_quintiles, child_quintiles, normalize="index")

        # Mobility indices
        mobility_indices = {
            "rank_mobility": 1 - abs(rank_correlation),
            "income_mobility": 1 - abs(income_correlation),
            "diagonal_mobility": 1
            - np.trace(transition_matrix) / 5,  # 1 - average diagonal element
            "upward_mobility": self._calculate_upward_mobility(transition_matrix),
            "downward_mobility": self._calculate_downward_mobility(transition_matrix),
        }

        # Directional mobility
        directional_mobility = {
            "stayed_same_quintile": np.trace(transition_matrix),
            "moved_up": transition_matrix.values[np.triu_indices(5, k=1)].sum(),
            "moved_down": transition_matrix.values[np.tril_indices(5, k=-1)].sum(),
            "bottom_to_top_quintile": transition_matrix.iloc[0, 4],
            "top_to_bottom_quintile": transition_matrix.iloc[4, 0],
        }

        return MobilityMeasures(
            rank_correlation=rank_correlation,
            income_correlation=income_correlation,
            transition_matrix=transition_matrix,
            mobility_indices=mobility_indices,
            directional_mobility=directional_mobility,
        )

    def _calculate_upward_mobility(self, transition_matrix: pd.DataFrame) -> float:
        """Calculate upward mobility from transition matrix"""
        upward = 0
        n = len(transition_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                upward += transition_matrix.iloc[i, j]

        return upward

    def _calculate_downward_mobility(self, transition_matrix: pd.DataFrame) -> float:
        """Calculate downward mobility from transition matrix"""
        downward = 0
        n = len(transition_matrix)

        for i in range(n):
            for j in range(i):
                downward += transition_matrix.iloc[i, j]

        return downward


def create_inequality_analyzer(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Factory function to create comprehensive inequality analysis suite

    Args:
        logger: Optional logger instance

    Returns:
        Dictionary with inequality analysis tools
    """
    return {
        "analyzer": AdvancedInequalityAnalyzer(logger),
        "decomposition": InequalityDecomposition(logger),
        "mobility": SocialMobility(logger),
    }


__all__ = [
    "InequalityMeasures",
    "DecompositionResult",
    "MobilityMeasures",
    "AdvancedInequalityAnalyzer",
    "InequalityDecomposition",
    "SocialMobility",
    "create_inequality_analyzer",
]
