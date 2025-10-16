"""
ENAHO Advanced Survey Methods - ANALYZE Phase
=============================================

Comprehensive survey methodology toolkit for complex sample analysis
with ENAHO data. Includes survey-weighted regression, design effects,
variance estimation, and complex sample analysis methods.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class SurveyDesign:
    """Container for survey design specifications"""

    strata_col: Optional[str] = None
    cluster_col: Optional[str] = None
    weight_col: Optional[str] = None
    fpc_col: Optional[str] = None  # Finite population correction
    replicate_weights: Optional[List[str]] = None
    survey_type: str = "complex"  # "complex", "stratified", "cluster", "simple"


@dataclass
class SurveyResult:
    """Container for survey analysis results"""

    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    design_effects: Dict[str, float]
    effective_sample_sizes: Dict[str, int]
    survey_design: SurveyDesign
    method: str


class SurveyWeights:
    """
    Advanced survey weight management and calibration
    Handles post-stratification, raking, and weight trimming
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.calibrated_weights = None
        self.weight_diagnostics = {}

    def post_stratify_weights(
        self,
        df: pd.DataFrame,
        weight_col: str,
        post_strata_cols: List[str],
        population_totals: Dict[str, float],
    ) -> pd.Series:
        """
        Post-stratify survey weights to known population totals

        Args:
            df: Survey DataFrame
            weight_col: Current weight column name
            post_strata_cols: Post-stratification variables
            population_totals: Known population totals by strata

        Returns:
            Post-stratified weights
        """
        df_work = df.copy()

        # Create post-strata groups
        if len(post_strata_cols) == 1:
            df_work["post_stratum"] = df_work[post_strata_cols[0]]
        else:
            df_work["post_stratum"] = df_work[post_strata_cols].apply(
                lambda x: "_".join(x.astype(str)), axis=1
            )

        # Calculate current weighted totals by post-stratum
        current_totals = df_work.groupby("post_stratum")[weight_col].sum()

        # Calculate post-stratification factors
        ps_factors = {}
        for stratum in current_totals.index:
            if stratum in population_totals:
                ps_factors[stratum] = population_totals[stratum] / current_totals[stratum]
            else:
                ps_factors[stratum] = 1.0
                self.logger.warning(f"No population total for stratum: {stratum}")

        # Apply post-stratification factors
        df_work["ps_factor"] = df_work["post_stratum"].map(ps_factors)
        calibrated_weights = df_work[weight_col] * df_work["ps_factor"]

        self.calibrated_weights = calibrated_weights
        return calibrated_weights

    def rake_weights(
        self,
        df: pd.DataFrame,
        weight_col: str,
        raking_variables: List[str],
        population_margins: Dict[str, Dict[Any, float]],
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> pd.Series:
        """
        Perform iterative proportional fitting (raking) for weight calibration

        Args:
            df: Survey DataFrame
            weight_col: Current weight column name
            raking_variables: Variables to rake on
            population_margins: Known population margins for each variable
            max_iterations: Maximum raking iterations
            tolerance: Convergence tolerance

        Returns:
            Raked weights
        """
        df_work = df.copy()
        current_weights = df_work[weight_col].copy()

        for iteration in range(max_iterations):
            old_weights = current_weights.copy()

            # Rake on each variable sequentially
            for var in raking_variables:
                if var not in population_margins:
                    continue

                # Calculate current weighted totals
                current_totals = df_work.groupby(var).apply(
                    lambda x: (x[weight_col] * current_weights[x.index]).sum()
                )

                # Calculate raking factors
                raking_factors = {}
                for category, pop_total in population_margins[var].items():
                    if category in current_totals.index:
                        raking_factors[category] = pop_total / current_totals[category]
                    else:
                        raking_factors[category] = 1.0

                # Apply raking factors
                adjustment = df_work[var].map(raking_factors).fillna(1.0)
                current_weights *= adjustment

            # Check convergence
            weight_change = np.abs(current_weights - old_weights).max()
            if weight_change < tolerance:
                self.logger.info(f"Raking converged after {iteration + 1} iterations")
                break
        else:
            self.logger.warning(f"Raking did not converge after {max_iterations} iterations")

        self.calibrated_weights = current_weights
        return current_weights

    def trim_weights(
        self, weights: pd.Series, lower_percentile: float = 1, upper_percentile: float = 99
    ) -> pd.Series:
        """
        Trim extreme weights to improve stability

        Args:
            weights: Survey weights
            lower_percentile: Lower trimming percentile
            upper_percentile: Upper trimming percentile

        Returns:
            Trimmed weights
        """
        lower_bound = np.percentile(weights, lower_percentile)
        upper_bound = np.percentile(weights, upper_percentile)

        trimmed_weights = weights.clip(lower=lower_bound, upper=upper_bound)

        # Adjust to maintain total weight
        adjustment_factor = weights.sum() / trimmed_weights.sum()
        trimmed_weights *= adjustment_factor

        # Calculate diagnostics
        n_trimmed_low = (weights < lower_bound).sum()
        n_trimmed_high = (weights > upper_bound).sum()

        self.weight_diagnostics = {
            "n_trimmed_low": n_trimmed_low,
            "n_trimmed_high": n_trimmed_high,
            "trim_rate": (n_trimmed_low + n_trimmed_high) / len(weights),
            "weight_range_before": (weights.min(), weights.max()),
            "weight_range_after": (trimmed_weights.min(), trimmed_weights.max()),
        }

        return trimmed_weights

    def calculate_weight_summary(self, weights: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive weight diagnostics"""
        return {
            "n_observations": len(weights),
            "mean_weight": weights.mean(),
            "median_weight": weights.median(),
            "min_weight": weights.min(),
            "max_weight": weights.max(),
            "weight_cv": weights.std() / weights.mean(),
            "effective_sample_size": (weights.sum() ** 2) / (weights**2).sum(),
            "design_effect": len(weights) / ((weights.sum() ** 2) / (weights**2).sum()),
        }


class DesignEffects:
    """
    Calculate design effects and effective sample sizes
    for complex survey designs
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def calculate_design_effect(
        self,
        df: pd.DataFrame,
        variable: str,
        weight_col: str,
        strata_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
    ) -> float:
        """
        Calculate design effect for a variable

        Args:
            df: Survey DataFrame
            variable: Variable to calculate design effect for
            weight_col: Survey weight column
            strata_col: Stratification variable
            cluster_col: Cluster variable

        Returns:
            Design effect (DEFF)
        """
        # Simple random sample variance
        y = df[variable].dropna()
        weights = df.loc[y.index, weight_col]

        # Weighted mean
        weighted_mean = (y * weights).sum() / weights.sum()

        # Design-based variance (simplified Horvitz-Thompson estimator)
        design_variance = self._calculate_design_variance(
            df, variable, weight_col, strata_col, cluster_col
        )

        # Simple random sample variance
        srs_variance = np.var(y) / len(y)

        # Design effect
        design_effect = design_variance / srs_variance if srs_variance > 0 else 1.0

        return design_effect

    def _calculate_design_variance(
        self,
        df: pd.DataFrame,
        variable: str,
        weight_col: str,
        strata_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
    ) -> float:
        """Calculate design-based variance estimate"""

        # Simple case: no clustering or stratification
        if not strata_col and not cluster_col:
            y = df[variable].dropna()
            weights = df.loc[y.index, weight_col]
            weighted_mean = (y * weights).sum() / weights.sum()
            variance = ((y - weighted_mean) ** 2 * weights).sum() / weights.sum()
            return variance / len(y)

        # Stratified design
        if strata_col and not cluster_col:
            return self._stratified_variance(df, variable, weight_col, strata_col)

        # Cluster design
        if cluster_col and not strata_col:
            return self._cluster_variance(df, variable, weight_col, cluster_col)

        # Stratified cluster design
        if strata_col and cluster_col:
            return self._stratified_cluster_variance(
                df, variable, weight_col, strata_col, cluster_col
            )

        return 0.0

    def _stratified_variance(
        self, df: pd.DataFrame, variable: str, weight_col: str, strata_col: str
    ) -> float:
        """Calculate variance for stratified design"""
        total_variance = 0.0

        for stratum in df[strata_col].unique():
            stratum_data = df[df[strata_col] == stratum]
            y_h = stratum_data[variable].dropna()
            w_h = stratum_data.loc[y_h.index, weight_col]

            if len(y_h) > 1:
                # Stratum variance
                y_h_mean = (y_h * w_h).sum() / w_h.sum()
                s_h_sq = ((y_h - y_h_mean) ** 2 * w_h).sum() / (w_h.sum() - 1)
                n_h = len(y_h)

                # Add to total variance
                total_variance += (w_h.sum() ** 2) * s_h_sq / n_h

        return total_variance / (df[weight_col].sum() ** 2)

    def _cluster_variance(
        self, df: pd.DataFrame, variable: str, weight_col: str, cluster_col: str
    ) -> float:
        """Calculate variance for cluster design"""
        # Calculate cluster totals
        cluster_totals = df.groupby(cluster_col).apply(
            lambda x: (x[variable] * x[weight_col]).sum()
        )
        cluster_weights = df.groupby(cluster_col)[weight_col].sum()

        # Overall mean
        overall_total = (df[variable] * df[weight_col]).sum()
        overall_weight = df[weight_col].sum()
        overall_mean = overall_total / overall_weight

        # Cluster means
        cluster_means = cluster_totals / cluster_weights

        # Between-cluster variance
        n_clusters = len(cluster_totals)
        if n_clusters > 1:
            between_variance = ((cluster_means - overall_mean) ** 2 * cluster_weights).sum() / (
                n_clusters - 1
            )
            return between_variance / n_clusters

        return 0.0

    def _stratified_cluster_variance(
        self, df: pd.DataFrame, variable: str, weight_col: str, strata_col: str, cluster_col: str
    ) -> float:
        """Calculate variance for stratified cluster design"""
        total_variance = 0.0

        for stratum in df[strata_col].unique():
            stratum_data = df[df[strata_col] == stratum]
            stratum_variance = self._cluster_variance(
                stratum_data, variable, weight_col, cluster_col
            )
            stratum_weight = stratum_data[weight_col].sum()

            total_variance += (stratum_weight**2) * stratum_variance

        return total_variance / (df[weight_col].sum() ** 2)

    def effective_sample_size(
        self,
        df: pd.DataFrame,
        variable: str,
        weight_col: str,
        design_effect: Optional[float] = None,
        **kwargs,
    ) -> int:
        """
        Calculate effective sample size

        Args:
            df: Survey DataFrame
            variable: Variable name
            weight_col: Weight column
            design_effect: Pre-calculated design effect
            **kwargs: Additional arguments for design effect calculation

        Returns:
            Effective sample size
        """
        if design_effect is None:
            design_effect = self.calculate_design_effect(df, variable, weight_col, **kwargs)

        actual_sample_size = df[variable].notna().sum()
        effective_n = int(actual_sample_size / design_effect)

        return effective_n


class SurveyRegression:
    """
    Survey-weighted regression analysis with proper variance estimation
    Supports linear and non-linear models with complex survey designs
    """

    def __init__(self, survey_design: SurveyDesign, logger: Optional[logging.Logger] = None):
        self.survey_design = survey_design
        self.logger = logger or logging.getLogger(__name__)
        self.model_results = {}

    def weighted_ols(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: List[str],
        model_name: str = "weighted_ols",
    ) -> SurveyResult:
        """
        Survey-weighted Ordinary Least Squares regression

        Args:
            df: Survey DataFrame
            outcome_col: Outcome variable column name
            predictor_cols: Predictor variable column names
            model_name: Name for the model

        Returns:
            SurveyResult with regression results
        """
        # Prepare data
        all_cols = [outcome_col] + predictor_cols
        if self.survey_design.weight_col:
            all_cols.append(self.survey_design.weight_col)

        analysis_df = df[all_cols].dropna()

        X = analysis_df[predictor_cols].values
        y = analysis_df[outcome_col].values

        # Add intercept
        X = np.column_stack([np.ones(X.shape[0]), X])
        predictor_cols_with_intercept = ["intercept"] + predictor_cols

        # Get weights
        if self.survey_design.weight_col:
            weights = analysis_df[self.survey_design.weight_col].values
        else:
            weights = np.ones(len(y))

        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y

        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        # Residuals
        y_pred = X @ beta
        residuals = y - y_pred

        # Design-based variance estimation
        variance_matrix = self._calculate_design_variance_matrix(analysis_df, X, residuals, weights)

        standard_errors = np.sqrt(np.diag(variance_matrix))

        # Results
        estimates = {name: coef for name, coef in zip(predictor_cols_with_intercept, beta)}
        se_dict = {name: se for name, se in zip(predictor_cols_with_intercept, standard_errors)}

        # Confidence intervals
        df_resid = len(y) - len(beta)
        t_crit = stats.t.ppf(0.975, df_resid)
        ci_dict = {
            name: (
                estimates[name] - t_crit * se_dict[name],
                estimates[name] + t_crit * se_dict[name],
            )
            for name in predictor_cols_with_intercept
        }

        # Design effects (simplified)
        design_effects = {name: 1.5 for name in predictor_cols_with_intercept}  # Placeholder
        effective_sample_sizes = {name: int(len(y) / 1.5) for name in predictor_cols_with_intercept}

        result = SurveyResult(
            estimates=estimates,
            standard_errors=se_dict,
            confidence_intervals=ci_dict,
            design_effects=design_effects,
            effective_sample_sizes=effective_sample_sizes,
            survey_design=self.survey_design,
            method="Weighted_OLS",
        )

        self.model_results[model_name] = result
        return result

    def survey_logistic(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: List[str],
        model_name: str = "survey_logistic",
    ) -> SurveyResult:
        """
        Survey-weighted logistic regression

        Args:
            df: Survey DataFrame
            outcome_col: Binary outcome variable
            predictor_cols: Predictor variables
            model_name: Model name

        Returns:
            SurveyResult with logistic regression results
        """
        # Prepare data
        all_cols = [outcome_col] + predictor_cols
        if self.survey_design.weight_col:
            all_cols.append(self.survey_design.weight_col)

        analysis_df = df[all_cols].dropna()

        X = analysis_df[predictor_cols].values
        y = analysis_df[outcome_col].values

        # Add intercept
        X = np.column_stack([np.ones(X.shape[0]), X])
        predictor_cols_with_intercept = ["intercept"] + predictor_cols

        # Get weights
        if self.survey_design.weight_col:
            weights = analysis_df[self.survey_design.weight_col].values
        else:
            weights = np.ones(len(y))

        # Weighted logistic regression (simplified Newton-Raphson)
        beta = self._weighted_logistic_regression(X, y, weights)

        # Calculate standard errors using survey design
        variance_matrix = self._logistic_design_variance(analysis_df, X, beta, weights)
        standard_errors = np.sqrt(np.diag(variance_matrix))

        # Results
        estimates = {name: coef for name, coef in zip(predictor_cols_with_intercept, beta)}
        se_dict = {name: se for name, se in zip(predictor_cols_with_intercept, standard_errors)}

        # Confidence intervals (using normal approximation for large samples)
        z_crit = stats.norm.ppf(0.975)
        ci_dict = {
            name: (
                estimates[name] - z_crit * se_dict[name],
                estimates[name] + z_crit * se_dict[name],
            )
            for name in predictor_cols_with_intercept
        }

        # Design effects
        design_effects = {name: 1.3 for name in predictor_cols_with_intercept}  # Placeholder
        effective_sample_sizes = {name: int(len(y) / 1.3) for name in predictor_cols_with_intercept}

        result = SurveyResult(
            estimates=estimates,
            standard_errors=se_dict,
            confidence_intervals=ci_dict,
            design_effects=design_effects,
            effective_sample_sizes=effective_sample_sizes,
            survey_design=self.survey_design,
            method="Weighted_Logistic",
        )

        self.model_results[model_name] = result
        return result

    def _calculate_design_variance_matrix(
        self, df: pd.DataFrame, X: np.ndarray, residuals: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Calculate design-based variance matrix for regression coefficients"""

        # Simplified variance calculation - in practice would be more complex
        # and account for stratification and clustering

        n = len(residuals)
        p = X.shape[1]

        # Residual sum of squares
        rss = np.sum(weights * residuals**2)
        mse = rss / (n - p)

        # Design-based adjustment (simplified)
        design_adjustment = 1.5  # Typical design effect for household surveys

        # Variance matrix
        XtWX = X.T @ np.diag(weights) @ X
        try:
            var_matrix = mse * design_adjustment * np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            var_matrix = mse * design_adjustment * np.linalg.pinv(XtWX)

        return var_matrix

    def _weighted_logistic_regression(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, max_iter: int = 100
    ) -> np.ndarray:
        """Weighted logistic regression using Newton-Raphson"""

        n, p = X.shape
        beta = np.zeros(p)
        W_diag = np.diag(weights)

        for iteration in range(max_iter):
            # Linear predictor
            eta = X @ beta

            # Predicted probabilities
            mu = 1 / (1 + np.exp(-eta))

            # Gradient
            gradient = X.T @ W_diag @ (y - mu)

            # Hessian (Fisher information)
            V = np.diag(weights * mu * (1 - mu))
            hessian = X.T @ V @ X

            # Newton-Raphson update
            try:
                delta = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            beta_new = beta + delta

            # Check convergence
            if np.abs(beta_new - beta).max() < 1e-6:
                break

            beta = beta_new

        return beta

    def _logistic_design_variance(
        self, df: pd.DataFrame, X: np.ndarray, beta: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Calculate design-based variance for logistic regression"""

        # Predicted probabilities
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))

        # Fisher information matrix
        V = np.diag(weights * mu * (1 - mu))
        fisher_info = X.T @ V @ X

        # Design effect adjustment
        design_adjustment = 1.3  # Typical for logistic regression

        try:
            var_matrix = design_adjustment * np.linalg.inv(fisher_info)
        except np.linalg.LinAlgError:
            var_matrix = design_adjustment * np.linalg.pinv(fisher_info)

        return var_matrix


class ComplexSampleAnalysis:
    """
    Comprehensive complex sample analysis toolkit
    Integrates all survey methodology components
    """

    def __init__(self, survey_design: SurveyDesign, logger: Optional[logging.Logger] = None):
        self.survey_design = survey_design
        self.logger = logger or logging.getLogger(__name__)
        self.survey_weights = SurveyWeights(logger)
        self.design_effects = DesignEffects(logger)
        self.survey_regression = SurveyRegression(survey_design, logger)
        self.analysis_results = {}

    def descriptive_analysis(
        self, df: pd.DataFrame, variables: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive descriptive analysis with design-based inference

        Args:
            df: Survey DataFrame
            variables: Variables to analyze

        Returns:
            Dictionary with descriptive statistics for each variable
        """
        results = {}

        for var in variables:
            if var not in df.columns:
                continue

            var_results = {}

            # Basic weighted statistics
            if self.survey_design.weight_col and self.survey_design.weight_col in df.columns:
                weights = df[self.survey_design.weight_col]
                valid_data = df[var].dropna()
                valid_weights = weights.loc[valid_data.index]

                # Weighted mean and total
                weighted_mean = (valid_data * valid_weights).sum() / valid_weights.sum()
                weighted_total = (valid_data * valid_weights).sum()

                # Weighted variance
                weighted_var = (
                    (valid_data - weighted_mean) ** 2 * valid_weights
                ).sum() / valid_weights.sum()

                var_results["weighted_mean"] = weighted_mean
                var_results["weighted_total"] = weighted_total
                var_results["weighted_std"] = np.sqrt(weighted_var)

            else:
                var_results["mean"] = df[var].mean()
                var_results["std"] = df[var].std()

            # Design effects
            if self.survey_design.weight_col:
                try:
                    design_effect = self.design_effects.calculate_design_effect(
                        df,
                        var,
                        self.survey_design.weight_col,
                        self.survey_design.strata_col,
                        self.survey_design.cluster_col,
                    )
                    var_results["design_effect"] = design_effect

                    effective_n = self.design_effects.effective_sample_size(
                        df, var, self.survey_design.weight_col, design_effect
                    )
                    var_results["effective_sample_size"] = effective_n

                except Exception as e:
                    self.logger.warning(f"Could not calculate design effect for {var}: {e}")
                    var_results["design_effect"] = 1.0
                    var_results["effective_sample_size"] = df[var].notna().sum()

            # Sample size
            var_results["sample_size"] = df[var].notna().sum()

            # Confidence intervals
            if "weighted_mean" in var_results and "design_effect" in var_results:
                se = var_results["weighted_std"] / np.sqrt(var_results["effective_sample_size"])
                t_crit = stats.t.ppf(0.975, var_results["effective_sample_size"] - 1)
                ci_lower = var_results["weighted_mean"] - t_crit * se
                ci_upper = var_results["weighted_mean"] + t_crit * se
                var_results["confidence_interval"] = (ci_lower, ci_upper)

            results[var] = var_results

        self.analysis_results["descriptive"] = results
        return results

    def crosstabulation_analysis(
        self, df: pd.DataFrame, row_var: str, col_var: str, test_independence: bool = True
    ) -> Dict[str, Any]:
        """
        Survey-weighted crosstabulation with design-based tests

        Args:
            df: Survey DataFrame
            row_var: Row variable
            col_var: Column variable
            test_independence: Whether to test independence

        Returns:
            Dictionary with crosstabulation results
        """
        # Create crosstab with weights
        if self.survey_design.weight_col and self.survey_design.weight_col in df.columns:
            weights = df[self.survey_design.weight_col]
        else:
            weights = pd.Series(1.0, index=df.index)

        # Weighted crosstab
        crosstab_data = df[[row_var, col_var]].dropna()
        crosstab_weights = weights.loc[crosstab_data.index]

        # Calculate weighted frequencies
        grouped = crosstab_data.groupby([row_var, col_var])
        weighted_freq = grouped.apply(lambda x: crosstab_weights.loc[x.index].sum())

        # Convert to DataFrame
        crosstab_df = weighted_freq.unstack(fill_value=0)

        # Row and column totals
        row_totals = crosstab_df.sum(axis=1)
        col_totals = crosstab_df.sum(axis=0)
        grand_total = crosstab_df.sum().sum()

        # Percentages
        row_percentages = crosstab_df.div(row_totals, axis=0) * 100
        col_percentages = crosstab_df.div(col_totals, axis=1) * 100

        results = {
            "crosstab": crosstab_df,
            "row_totals": row_totals,
            "col_totals": col_totals,
            "grand_total": grand_total,
            "row_percentages": row_percentages,
            "col_percentages": col_percentages,
        }

        # Design-based chi-square test (simplified)
        if test_independence:
            try:
                expected = np.outer(row_totals, col_totals) / grand_total
                chi_square = ((crosstab_df - expected) ** 2 / expected).sum().sum()

                # Adjust for design effects
                design_effect = 1.5  # Simplified assumption
                adjusted_chi_square = chi_square / design_effect

                df_chi = (len(row_totals) - 1) * (len(col_totals) - 1)
                p_value = 1 - stats.chi2.cdf(adjusted_chi_square, df_chi)

                results["chi_square_test"] = {
                    "chi_square": adjusted_chi_square,
                    "df": df_chi,
                    "p_value": p_value,
                }
            except Exception as e:
                self.logger.warning(f"Could not perform chi-square test: {e}")

        return results

    def domain_analysis(
        self,
        df: pd.DataFrame,
        outcome_var: str,
        domain_vars: List[str],
        analysis_type: str = "mean",
    ) -> Dict[str, Any]:
        """
        Domain (subpopulation) analysis with proper variance estimation

        Args:
            df: Survey DataFrame
            outcome_var: Outcome variable
            domain_vars: Domain definition variables
            analysis_type: Type of analysis ("mean", "total", "proportion")

        Returns:
            Domain analysis results
        """
        if len(domain_vars) == 1:
            domains = df[domain_vars[0]].unique()
            domain_col = domain_vars[0]
        else:
            # Create combined domain variable
            df["combined_domain"] = df[domain_vars].apply(lambda x: "_".join(x.astype(str)), axis=1)
            domains = df["combined_domain"].unique()
            domain_col = "combined_domain"

        results = {}

        for domain in domains:
            domain_data = df[df[domain_col] == domain]

            if len(domain_data) == 0:
                continue

            domain_result = {}

            if analysis_type == "mean":
                if self.survey_design.weight_col:
                    weights = domain_data[self.survey_design.weight_col]
                    outcome_data = domain_data[outcome_var].dropna()
                    domain_weights = weights.loc[outcome_data.index]

                    if len(outcome_data) > 0 and domain_weights.sum() > 0:
                        weighted_mean = (outcome_data * domain_weights).sum() / domain_weights.sum()
                        domain_result["estimate"] = weighted_mean
                        domain_result["sample_size"] = len(outcome_data)

                        # Standard error (simplified)
                        variance = (
                            (outcome_data - weighted_mean) ** 2 * domain_weights
                        ).sum() / domain_weights.sum()
                        design_effect = 1.5  # Simplified
                        se = np.sqrt(variance * design_effect / len(outcome_data))
                        domain_result["standard_error"] = se

                        # Confidence interval
                        t_crit = stats.t.ppf(0.975, len(outcome_data) - 1)
                        ci = (weighted_mean - t_crit * se, weighted_mean + t_crit * se)
                        domain_result["confidence_interval"] = ci

            elif analysis_type == "proportion":
                # Binary variable analysis
                if self.survey_design.weight_col:
                    weights = domain_data[self.survey_design.weight_col]
                    outcome_data = domain_data[outcome_var].dropna()
                    domain_weights = weights.loc[outcome_data.index]

                    if len(outcome_data) > 0:
                        weighted_prop = (outcome_data * domain_weights).sum() / domain_weights.sum()
                        domain_result["estimate"] = weighted_prop
                        domain_result["sample_size"] = len(outcome_data)

                        # Standard error for proportion
                        variance = weighted_prop * (1 - weighted_prop) / len(outcome_data)
                        design_effect = 1.3  # Typical for proportions
                        se = np.sqrt(variance * design_effect)
                        domain_result["standard_error"] = se

                        # Confidence interval
                        z_crit = stats.norm.ppf(0.975)
                        ci = (
                            max(0, weighted_prop - z_crit * se),
                            min(1, weighted_prop + z_crit * se),
                        )
                        domain_result["confidence_interval"] = ci

            results[str(domain)] = domain_result

        return results


def create_survey_analyzer(
    survey_design: SurveyDesign, logger: Optional[logging.Logger] = None
) -> ComplexSampleAnalysis:
    """Factory function to create ComplexSampleAnalysis instance"""
    return ComplexSampleAnalysis(survey_design, logger)


# Export classes and functions
__all__ = [
    "SurveyRegression",
    "ComplexSampleAnalysis",
    "SurveyWeights",
    "DesignEffects",
    "SurveyDesign",
    "SurveyResult",
    "create_survey_analyzer",
]
