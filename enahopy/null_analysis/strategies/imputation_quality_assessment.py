"""
Imputation Quality Assessment - ANALYZE Phase
==============================================

Comprehensive quality assessment and validation framework for
missing data imputation methods in ENAHO survey data.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Statistical imports
try:
    from scipy import stats
    from scipy.spatial.distance import wasserstein_distance

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import cross_val_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class QualityMetricType(Enum):
    """Types of quality metrics for imputation assessment"""

    STATISTICAL = "statistical"  # Distribution preservation
    PREDICTIVE = "predictive"  # Predictive accuracy
    STRUCTURAL = "structural"  # Correlation/covariance preservation
    DOMAIN_SPECIFIC = "domain_specific"  # ENAHO-specific constraints
    OUTLIER_DETECTION = "outlier"  # Imputed outlier detection
    UNCERTAINTY = "uncertainty"  # Imputation uncertainty measures


@dataclass
class QualityAssessmentConfig:
    """Configuration for quality assessment"""

    metrics_to_compute: List[str] = field(
        default_factory=lambda: [
            "distribution_preservation",
            "correlation_preservation",
            "predictive_accuracy",
            "outlier_detection",
            "uncertainty_quantification",
        ]
    )

    # Statistical testing parameters
    significance_level: float = 0.05
    bootstrap_samples: int = 1000

    # Cross-validation parameters
    cv_folds: int = 5
    test_size: float = 0.2

    # Outlier detection parameters
    outlier_contamination: float = 0.1

    # Uncertainty quantification
    uncertainty_method: str = "bootstrap"  # "bootstrap", "multiple_imputation"
    n_imputations: int = 5


@dataclass
class QualityAssessmentResult:
    """Container for quality assessment results"""

    overall_score: float
    metric_scores: Dict[str, float]
    detailed_metrics: Dict[str, Dict[str, Any]]
    statistical_tests: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    validation_plots: Dict[str, Any] = field(default_factory=dict)


class ImputationQualityAssessor:
    """Comprehensive quality assessment for imputation methods"""

    def __init__(self, config: QualityAssessmentConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def assess_quality(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: Optional[pd.DataFrame] = None,
        categorical_cols: Optional[List[str]] = None,
        survey_weights: Optional[pd.Series] = None,
    ) -> QualityAssessmentResult:
        """
        Comprehensive quality assessment of imputation results

        Args:
            original_df: Original data with missing values
            imputed_df: Data after imputation
            missing_mask: Boolean mask of originally missing values
            categorical_cols: List of categorical column names
            survey_weights: Survey weights for weighted analysis

        Returns:
            QualityAssessmentResult with comprehensive metrics
        """
        self.logger.info("Starting comprehensive imputation quality assessment...")

        if missing_mask is None:
            missing_mask = original_df.isnull()

        categorical_cols = categorical_cols or []

        # Initialize results containers
        metric_scores = {}
        detailed_metrics = {}
        statistical_tests = {}
        validation_plots = {}
        recommendations = []

        # 1. Distribution Preservation Assessment
        if "distribution_preservation" in self.config.metrics_to_compute:
            self.logger.info("Assessing distribution preservation...")
            dist_metrics = self._assess_distribution_preservation(
                original_df, imputed_df, missing_mask, categorical_cols, survey_weights
            )
            metric_scores["distribution_preservation"] = dist_metrics["overall_score"]
            detailed_metrics["distribution_preservation"] = dist_metrics

        # 2. Correlation Structure Preservation
        if "correlation_preservation" in self.config.metrics_to_compute:
            self.logger.info("Assessing correlation preservation...")
            corr_metrics = self._assess_correlation_preservation(
                original_df, imputed_df, missing_mask, categorical_cols
            )
            metric_scores["correlation_preservation"] = corr_metrics["overall_score"]
            detailed_metrics["correlation_preservation"] = corr_metrics

        # 3. Predictive Accuracy Assessment
        if "predictive_accuracy" in self.config.metrics_to_compute:
            self.logger.info("Assessing predictive accuracy...")
            pred_metrics = self._assess_predictive_accuracy(
                original_df, imputed_df, missing_mask, categorical_cols
            )
            metric_scores["predictive_accuracy"] = pred_metrics["overall_score"]
            detailed_metrics["predictive_accuracy"] = pred_metrics

        # 4. Outlier Detection in Imputations
        if "outlier_detection" in self.config.metrics_to_compute:
            self.logger.info("Detecting imputation outliers...")
            outlier_metrics = self._detect_imputation_outliers(
                original_df, imputed_df, missing_mask, categorical_cols
            )
            metric_scores["outlier_detection"] = outlier_metrics["overall_score"]
            detailed_metrics["outlier_detection"] = outlier_metrics

        # 5. Uncertainty Quantification
        if "uncertainty_quantification" in self.config.metrics_to_compute:
            self.logger.info("Quantifying imputation uncertainty...")
            uncertainty_metrics = self._quantify_imputation_uncertainty(
                original_df, imputed_df, missing_mask
            )
            metric_scores["uncertainty_quantification"] = uncertainty_metrics["overall_score"]
            detailed_metrics["uncertainty_quantification"] = uncertainty_metrics

        # 6. Domain-specific validation (ENAHO)
        domain_metrics = self._validate_domain_constraints(
            original_df, imputed_df, missing_mask, categorical_cols
        )
        metric_scores["domain_validation"] = domain_metrics["overall_score"]
        detailed_metrics["domain_validation"] = domain_metrics

        # 7. Statistical significance tests
        statistical_tests = self._perform_statistical_tests(
            original_df, imputed_df, missing_mask, categorical_cols
        )

        # 8. Generate validation plots
        if PLOTTING_AVAILABLE:
            validation_plots = self._generate_validation_plots(
                original_df, imputed_df, missing_mask, categorical_cols
            )

        # 9. Generate recommendations
        recommendations = self._generate_recommendations(
            metric_scores, detailed_metrics, statistical_tests
        )

        # Calculate overall quality score
        overall_score = self._calculate_overall_score(metric_scores)

        self.logger.info(f"Quality assessment completed. Overall score: {overall_score:.2f}")

        return QualityAssessmentResult(
            overall_score=overall_score,
            metric_scores=metric_scores,
            detailed_metrics=detailed_metrics,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            validation_plots=validation_plots,
        )

    def _assess_distribution_preservation(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
        survey_weights: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """Assess how well the imputation preserves original distributions"""

        results = {"column_scores": {}, "overall_score": 0.0}
        column_scores = []

        for col in original_df.columns:
            if not missing_mask[col].any():
                continue  # No missing values to assess

            # Get observed and imputed values
            observed_values = original_df[col].dropna()
            imputed_values = imputed_df.loc[missing_mask[col], col]

            if len(observed_values) == 0 or len(imputed_values) == 0:
                continue

            col_metrics = {}

            # Auto-detect categorical columns if not specified
            is_categorical = (
                col in categorical_cols
                or original_df[col].dtype == "object"
                or original_df[col].dtype.name == "category"
                or (original_df[col].dtype == "string")
            )

            if is_categorical:
                # Categorical distribution assessment
                col_metrics = self._assess_categorical_distribution(
                    observed_values, imputed_values, col
                )
            else:
                # Numerical distribution assessment
                col_metrics = self._assess_numerical_distribution(
                    observed_values, imputed_values, col, survey_weights
                )

            results["column_scores"][col] = col_metrics
            column_scores.append(col_metrics.get("distribution_score", 0))

        # Calculate overall distribution preservation score
        results["overall_score"] = np.mean(column_scores) if column_scores else 0.0

        return results

    def _assess_categorical_distribution(
        self, observed: pd.Series, imputed: pd.Series, col_name: str
    ) -> Dict[str, Any]:
        """Assess distribution preservation for categorical variables"""

        metrics = {}

        # Calculate proportions
        observed_counts = observed.value_counts(normalize=True)
        imputed_counts = imputed.value_counts(normalize=True)

        # Align categories
        all_categories = set(observed_counts.index) | set(imputed_counts.index)
        observed_props = pd.Series(index=all_categories, dtype=float).fillna(0)
        imputed_props = pd.Series(index=all_categories, dtype=float).fillna(0)

        for cat in all_categories:
            observed_props[cat] = observed_counts.get(cat, 0)
            imputed_props[cat] = imputed_counts.get(cat, 0)

        # Chi-square test for independence
        if SCIPY_AVAILABLE and len(all_categories) > 1:
            try:
                chi2_stat, p_value = stats.chisquare(imputed_props + 1e-8, observed_props + 1e-8)
                metrics["chi2_statistic"] = chi2_stat
                metrics["chi2_p_value"] = p_value
            except Exception:
                metrics["chi2_statistic"] = np.nan
                metrics["chi2_p_value"] = np.nan

        # Total Variation Distance
        tvd = 0.5 * np.sum(np.abs(observed_props - imputed_props))
        metrics["total_variation_distance"] = tvd

        # Category coverage (proportion of observed categories in imputed)
        coverage = len(set(imputed.unique()) & set(observed.unique())) / len(observed.unique())
        metrics["category_coverage"] = coverage

        # Distribution score (inverse of TVD, higher is better)
        metrics["distribution_score"] = 1 - tvd

        return metrics

    def _assess_numerical_distribution(
        self,
        observed: pd.Series,
        imputed: pd.Series,
        col_name: str,
        survey_weights: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """Assess distribution preservation for numerical variables"""

        metrics = {}

        # Check if data is actually numerical
        try:
            observed_numeric = pd.to_numeric(observed, errors="coerce")
            imputed_numeric = pd.to_numeric(imputed, errors="coerce")

            # If too many values can't be converted, skip numerical assessment
            if observed_numeric.isna().mean() > 0.5 or imputed_numeric.isna().mean() > 0.5:
                return {"distribution_score": 0.5, "message": "Column appears non-numerical"}

            observed = observed_numeric.dropna()
            imputed = imputed_numeric.dropna()

            if len(observed) == 0 or len(imputed) == 0:
                return {"distribution_score": 0.5, "message": "Insufficient numerical data"}
        except Exception:
            return {"distribution_score": 0.5, "message": "Error converting to numeric"}

        # Basic moments comparison
        metrics["mean_observed"] = observed.mean()
        metrics["mean_imputed"] = imputed.mean()
        metrics["mean_difference"] = abs(metrics["mean_imputed"] - metrics["mean_observed"])

        metrics["std_observed"] = observed.std()
        metrics["std_imputed"] = imputed.std()
        metrics["std_ratio"] = metrics["std_imputed"] / (metrics["std_observed"] + 1e-8)

        # Skewness and kurtosis
        if SCIPY_AVAILABLE:
            metrics["skewness_observed"] = stats.skew(observed.dropna())
            metrics["skewness_imputed"] = stats.skew(imputed.dropna())
            metrics["kurtosis_observed"] = stats.kurtosis(observed.dropna())
            metrics["kurtosis_imputed"] = stats.kurtosis(imputed.dropna())

        # Kolmogorov-Smirnov test
        if SCIPY_AVAILABLE and len(observed) > 10 and len(imputed) > 10:
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(observed, imputed)
                metrics["ks_statistic"] = ks_stat
                metrics["ks_p_value"] = ks_pvalue
            except Exception:
                metrics["ks_statistic"] = np.nan
                metrics["ks_p_value"] = np.nan

        # Wasserstein distance (Earth Mover's Distance)
        if SCIPY_AVAILABLE:
            try:
                wasserstein_dist = wasserstein_distance(observed, imputed)
                metrics["wasserstein_distance"] = wasserstein_dist
            except Exception:
                metrics["wasserstein_distance"] = np.nan

        # Distribution score based on KS test and moment preservation
        moment_score = 1 - min(metrics["mean_difference"] / (metrics["mean_observed"] + 1e-8), 1.0)
        spread_score = 1 - min(abs(metrics["std_ratio"] - 1.0), 1.0)

        if SCIPY_AVAILABLE and not pd.isna(metrics.get("ks_p_value")):
            # Higher p-value indicates better distribution preservation
            ks_score = metrics["ks_p_value"]
        else:
            ks_score = 0.5  # Neutral score if KS test unavailable

        metrics["distribution_score"] = np.mean([moment_score, spread_score, ks_score])

        return metrics

    def _assess_correlation_preservation(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Assess preservation of correlation structure"""

        results = {}

        # Select numerical columns for correlation analysis
        numerical_cols = [
            col
            for col in original_df.columns
            if col not in categorical_cols and pd.api.types.is_numeric_dtype(original_df[col])
        ]

        if len(numerical_cols) < 2:
            return {
                "overall_score": 1.0,
                "message": "Insufficient numerical columns for correlation analysis",
            }

        # Calculate correlation matrices
        # Use complete cases from original data
        original_complete = original_df[numerical_cols].dropna()
        imputed_complete = imputed_df[numerical_cols]

        if len(original_complete) < 10:
            return {
                "overall_score": 0.5,
                "message": "Insufficient complete cases for correlation analysis",
            }

        try:
            original_corr = original_complete.corr()
            imputed_corr = imputed_complete.corr()

            # Calculate correlation matrix differences
            corr_diff = np.abs(original_corr - imputed_corr)

            # Mean absolute correlation difference
            mean_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
            results["mean_correlation_difference"] = mean_corr_diff

            # Maximum correlation difference
            max_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].max()
            results["max_correlation_difference"] = max_corr_diff

            # Correlation of correlations (meta-correlation)
            original_corr_vec = original_corr.values[
                np.triu_indices_from(original_corr.values, k=1)
            ]
            imputed_corr_vec = imputed_corr.values[np.triu_indices_from(imputed_corr.values, k=1)]

            if len(original_corr_vec) > 1:
                meta_correlation = np.corrcoef(original_corr_vec, imputed_corr_vec)[0, 1]
                results["meta_correlation"] = (
                    meta_correlation if not np.isnan(meta_correlation) else 0
                )
            else:
                results["meta_correlation"] = 1.0

            # Overall correlation preservation score
            correlation_score = max(0, 1 - mean_corr_diff)
            meta_score = max(0, results["meta_correlation"])
            results["overall_score"] = 0.7 * correlation_score + 0.3 * meta_score

        except Exception as e:
            self.logger.warning(f"Error in correlation assessment: {e}")
            results["overall_score"] = 0.5
            results["error"] = str(e)

        return results

    def _assess_predictive_accuracy(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Assess predictive accuracy of imputations using cross-validation"""

        if not SKLEARN_AVAILABLE:
            return {
                "overall_score": 0.5,
                "message": "scikit-learn not available for predictive accuracy assessment",
            }

        results = {"column_scores": {}, "overall_score": 0.0}
        column_scores = []

        for col in original_df.columns:
            if not missing_mask[col].any():
                continue

            try:
                # Get complete cases for this column
                complete_mask = original_df[col].notna()
                if complete_mask.sum() < 50:  # Need sufficient data
                    continue

                # Prepare feature matrix (exclude the target column)
                feature_cols = [c for c in original_df.columns if c != col]
                X = imputed_df[feature_cols].loc[complete_mask]
                y = original_df[col].loc[complete_mask]

                # Handle categorical features
                X_processed = X.copy()
                for cat_col in categorical_cols:
                    if cat_col in X_processed.columns:
                        # Simple label encoding for categorical features
                        X_processed[cat_col] = pd.Categorical(X_processed[cat_col]).codes

                # Fill any remaining NaNs
                X_processed = X_processed.fillna(X_processed.mean())

                # Choose appropriate model and metric
                if col in categorical_cols:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import accuracy_score, make_scorer

                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    scorer = make_scorer(accuracy_score)

                else:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import make_scorer, mean_squared_error

                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    scorer = make_scorer(mean_squared_error, greater_is_better=False)

                # Perform cross-validation
                cv_scores = cross_val_score(
                    model,
                    X_processed,
                    y,
                    cv=min(self.config.cv_folds, len(y) // 10),
                    scoring=scorer,
                )

                if col in categorical_cols:
                    # For classification, higher accuracy is better
                    pred_score = np.mean(cv_scores)
                else:
                    # For regression, convert negative MSE to positive score
                    pred_score = 1.0 / (1.0 + abs(np.mean(cv_scores)))

                results["column_scores"][col] = {
                    "predictive_score": pred_score,
                    "cv_mean": np.mean(cv_scores),
                    "cv_std": np.std(cv_scores),
                }

                column_scores.append(pred_score)

            except Exception as e:
                self.logger.warning(f"Error in predictive accuracy assessment for {col}: {e}")
                continue

        results["overall_score"] = np.mean(column_scores) if column_scores else 0.5
        return results

    def _detect_imputation_outliers(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Detect outliers in imputed values"""

        if not SKLEARN_AVAILABLE:
            return {
                "overall_score": 0.5,
                "message": "scikit-learn not available for outlier detection",
            }

        results = {"column_outliers": {}, "overall_score": 0.0}
        outlier_rates = []

        for col in original_df.columns:
            if col in categorical_cols or not missing_mask[col].any():
                continue

            try:
                # Get observed and imputed values
                observed_values = original_df[col].dropna().values.reshape(-1, 1)
                imputed_values = imputed_df.loc[missing_mask[col], col].values.reshape(-1, 1)

                if len(observed_values) < 10 or len(imputed_values) == 0:
                    continue

                # Fit Isolation Forest on observed values
                isolation_forest = IsolationForest(
                    contamination=self.config.outlier_contamination, random_state=42
                )
                isolation_forest.fit(observed_values)

                # Predict outliers in imputed values
                outlier_predictions = isolation_forest.predict(imputed_values)
                outlier_scores = isolation_forest.score_samples(imputed_values)

                # Calculate outlier rate
                outlier_rate = (outlier_predictions == -1).mean()

                results["column_outliers"][col] = {
                    "outlier_rate": outlier_rate,
                    "mean_outlier_score": outlier_scores.mean(),
                    "min_outlier_score": outlier_scores.min(),
                    "outlier_indices": np.where(outlier_predictions == -1)[0].tolist(),
                }

                # Score: lower outlier rate is better (up to contamination threshold)
                outlier_score = max(0, 1 - outlier_rate / self.config.outlier_contamination)
                outlier_rates.append(outlier_score)

            except Exception as e:
                self.logger.warning(f"Error in outlier detection for {col}: {e}")
                continue

        results["overall_score"] = np.mean(outlier_rates) if outlier_rates else 0.5
        return results

    def _quantify_imputation_uncertainty(
        self, original_df: pd.DataFrame, imputed_df: pd.DataFrame, missing_mask: pd.DataFrame
    ) -> Dict[str, Any]:
        """Quantify uncertainty in imputed values using bootstrap sampling"""

        results = {"column_uncertainties": {}, "overall_score": 0.0}
        uncertainty_scores = []

        for col in original_df.columns:
            if not missing_mask[col].any():
                continue

            try:
                # Get complete cases for bootstrap sampling
                complete_cases = original_df[col].dropna()
                if len(complete_cases) < 20:
                    continue

                # Bootstrap sampling to estimate uncertainty
                bootstrap_means = []
                bootstrap_stds = []

                for _ in range(self.config.bootstrap_samples):
                    bootstrap_sample = np.random.choice(
                        complete_cases, size=min(len(complete_cases), 100), replace=True
                    )
                    bootstrap_means.append(np.mean(bootstrap_sample))
                    bootstrap_stds.append(np.std(bootstrap_sample))

                # Calculate uncertainty metrics
                uncertainty_mean = np.std(bootstrap_means)
                uncertainty_std = np.std(bootstrap_stds)

                # Compare with imputed value variability
                imputed_values = imputed_df.loc[missing_mask[col], col]
                imputed_variability = imputed_values.std() if len(imputed_values) > 1 else 0

                # Coefficient of variation as uncertainty measure
                cv_observed = np.mean(bootstrap_stds) / (np.mean(bootstrap_means) + 1e-8)
                cv_imputed = imputed_variability / (imputed_values.mean() + 1e-8)

                results["column_uncertainties"][col] = {
                    "bootstrap_mean_uncertainty": uncertainty_mean,
                    "bootstrap_std_uncertainty": uncertainty_std,
                    "cv_observed": cv_observed,
                    "cv_imputed": cv_imputed,
                    "uncertainty_ratio": cv_imputed / (cv_observed + 1e-8),
                }

                # Uncertainty score: lower uncertainty ratio is better
                uncertainty_score = 1.0 / (
                    1.0
                    + abs(np.log(results["column_uncertainties"][col]["uncertainty_ratio"] + 1e-8))
                )
                uncertainty_scores.append(uncertainty_score)

            except Exception as e:
                self.logger.warning(f"Error in uncertainty quantification for {col}: {e}")
                continue

        results["overall_score"] = np.mean(uncertainty_scores) if uncertainty_scores else 0.5
        return results

    def _validate_domain_constraints(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Validate ENAHO-specific domain constraints"""

        results = {"constraint_violations": {}, "overall_score": 0.0}
        violation_scores = []

        # Age constraints (0-120 years)
        if "edad" in imputed_df.columns:
            age_violations = (imputed_df["edad"] < 0) | (imputed_df["edad"] > 120)
            age_violation_rate = age_violations.mean()
            results["constraint_violations"]["age_range"] = {
                "violation_rate": age_violation_rate,
                "violations_count": age_violations.sum(),
            }
            violation_scores.append(1 - age_violation_rate)

        # Income constraints (non-negative for most income variables)
        income_cols = [col for col in imputed_df.columns if "ing" in col.lower()]
        for income_col in income_cols:
            if income_col in imputed_df.columns:
                negative_income = imputed_df[income_col] < 0
                negative_rate = negative_income.mean()
                results["constraint_violations"][f"{income_col}_negative"] = {
                    "violation_rate": negative_rate,
                    "violations_count": negative_income.sum(),
                }
                violation_scores.append(1 - negative_rate)

        # Education consistency (years should be consistent with level)
        if "nivel_educ" in imputed_df.columns and "anios_educ" in imputed_df.columns:
            # Simplified consistency check
            education_violations = 0
            total_education_records = 0

            for idx, row in imputed_df.iterrows():
                if not pd.isna(row["nivel_educ"]) and not pd.isna(row["anios_educ"]):
                    total_education_records += 1
                    nivel = row["nivel_educ"]
                    anios = row["anios_educ"]

                    # Basic consistency rules (simplified)
                    if nivel == 1 and anios > 5:  # Sin nivel but many years
                        education_violations += 1
                    elif nivel == 3 and anios < 6:  # Primary but too few years
                        education_violations += 1
                    elif nivel >= 5 and anios < 12:  # Higher education but too few years
                        education_violations += 1

            education_violation_rate = education_violations / max(total_education_records, 1)
            results["constraint_violations"]["education_consistency"] = {
                "violation_rate": education_violation_rate,
                "violations_count": education_violations,
            }
            violation_scores.append(1 - education_violation_rate)

        # Calculate overall domain validation score
        results["overall_score"] = np.mean(violation_scores) if violation_scores else 1.0

        return results

    def _perform_statistical_tests(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""

        if not SCIPY_AVAILABLE:
            return {"message": "SciPy not available for statistical tests"}

        tests = {}

        for col in original_df.columns:
            if not missing_mask[col].any():
                continue

            try:
                observed_values = original_df[col].dropna()
                imputed_values = imputed_df.loc[missing_mask[col], col].dropna()

                if len(observed_values) < 10 or len(imputed_values) < 10:
                    continue

                col_tests = {}

                if col in categorical_cols:
                    # Chi-square test for categorical variables
                    observed_counts = observed_values.value_counts()
                    imputed_counts = imputed_values.value_counts()

                    # Align categories
                    all_categories = set(observed_counts.index) | set(imputed_counts.index)
                    obs_aligned = [observed_counts.get(cat, 0) for cat in all_categories]
                    imp_aligned = [imputed_counts.get(cat, 0) for cat in all_categories]

                    if len(all_categories) > 1:
                        chi2_stat, p_val = stats.chisquare(
                            np.array(imp_aligned) + 1e-8, np.array(obs_aligned) + 1e-8
                        )
                        col_tests["chi_square"] = {
                            "statistic": chi2_stat,
                            "p_value": p_val,
                            "significant": p_val < self.config.significance_level,
                        }

                else:
                    # Numerical tests
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(observed_values, imputed_values)
                    col_tests["kolmogorov_smirnov"] = {
                        "statistic": ks_stat,
                        "p_value": ks_p,
                        "significant": ks_p < self.config.significance_level,
                    }

                    # Mann-Whitney U test (non-parametric)
                    mw_stat, mw_p = stats.mannwhitneyu(
                        observed_values, imputed_values, alternative="two-sided"
                    )
                    col_tests["mann_whitney"] = {
                        "statistic": mw_stat,
                        "p_value": mw_p,
                        "significant": mw_p < self.config.significance_level,
                    }

                    # Levene's test for equality of variances
                    levene_stat, levene_p = stats.levene(observed_values, imputed_values)
                    col_tests["levene"] = {
                        "statistic": levene_stat,
                        "p_value": levene_p,
                        "significant": levene_p < self.config.significance_level,
                    }

                tests[col] = col_tests

            except Exception as e:
                self.logger.warning(f"Error in statistical tests for {col}: {e}")
                continue

        return tests

    def _generate_validation_plots(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        missing_mask: pd.DataFrame,
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Generate validation plots for visual assessment"""

        if not PLOTTING_AVAILABLE:
            return {"message": "Matplotlib/Seaborn not available for plotting"}

        plots = {}

        try:
            # Select a few key columns for plotting
            plot_cols = [col for col in original_df.columns if missing_mask[col].any()][
                :6
            ]  # Limit to 6 columns

            # Distribution comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten() if len(plot_cols) > 1 else [axes]

            for i, col in enumerate(plot_cols):
                if i >= len(axes):
                    break

                ax = axes[i]

                observed = original_df[col].dropna()
                imputed = imputed_df.loc[missing_mask[col], col].dropna()

                if col in categorical_cols:
                    # Bar plot for categorical
                    obs_counts = observed.value_counts(normalize=True)
                    imp_counts = imputed.value_counts(normalize=True)

                    x = range(len(obs_counts))
                    ax.bar(
                        [i - 0.2 for i in x],
                        obs_counts.values,
                        width=0.4,
                        label="Observed",
                        alpha=0.7,
                    )
                    ax.bar(
                        [i + 0.2 for i in x],
                        [imp_counts.get(cat, 0) for cat in obs_counts.index],
                        width=0.4,
                        label="Imputed",
                        alpha=0.7,
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(obs_counts.index, rotation=45)

                else:
                    # Histogram for numerical
                    ax.hist(observed, bins=20, alpha=0.7, label="Observed", density=True)
                    ax.hist(imputed, bins=20, alpha=0.7, label="Imputed", density=True)

                ax.set_title(f"{col}")
                ax.legend()

            plt.tight_layout()
            plots["distribution_comparison"] = fig

            # Correlation heatmap comparison
            numerical_cols = [
                col
                for col in original_df.columns
                if col not in categorical_cols and pd.api.types.is_numeric_dtype(original_df[col])
            ]

            if len(numerical_cols) >= 2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Original correlation
                original_corr = original_df[numerical_cols].corr()
                sns.heatmap(original_corr, ax=ax1, annot=True, cmap="coolwarm", center=0)
                ax1.set_title("Original Correlations")

                # Imputed correlation
                imputed_corr = imputed_df[numerical_cols].corr()
                sns.heatmap(imputed_corr, ax=ax2, annot=True, cmap="coolwarm", center=0)
                ax2.set_title("Imputed Correlations")

                plt.tight_layout()
                plots["correlation_comparison"] = fig

        except Exception as e:
            self.logger.warning(f"Error generating validation plots: {e}")
            plots["error"] = str(e)

        return plots

    def _generate_recommendations(
        self,
        metric_scores: Dict[str, float],
        detailed_metrics: Dict[str, Any],
        statistical_tests: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on assessment results"""

        recommendations = []

        # Distribution preservation recommendations
        if "distribution_preservation" in metric_scores:
            dist_score = metric_scores["distribution_preservation"]
            if dist_score < 0.7:
                recommendations.append(
                    "Distribution preservation is below acceptable threshold. "
                    "Consider using methods that better preserve distributional properties "
                    "(e.g., MICE with proper model selection, or distribution-matching imputation)."
                )

        # Correlation preservation recommendations
        if "correlation_preservation" in metric_scores:
            corr_score = metric_scores["correlation_preservation"]
            if corr_score < 0.8:
                recommendations.append(
                    "Correlation structure is not well preserved. "
                    "Consider multivariate imputation methods like MICE or missForest "
                    "that account for inter-variable relationships."
                )

        # Outlier detection recommendations
        if "outlier_detection" in metric_scores:
            outlier_score = metric_scores["outlier_detection"]
            if outlier_score < 0.7:
                recommendations.append(
                    "High rate of outliers detected in imputed values. "
                    "Consider post-imputation outlier detection and replacement, "
                    "or use robust imputation methods."
                )

        # Domain validation recommendations
        if "domain_validation" in metric_scores:
            domain_score = metric_scores["domain_validation"]
            if domain_score < 0.9:
                recommendations.append(
                    "Domain-specific constraints are violated. "
                    "Implement post-imputation constraint enforcement or "
                    "use domain-aware imputation methods."
                )

        # Statistical significance recommendations
        significant_tests = 0
        total_tests = 0
        for col_tests in statistical_tests.values():
            # Skip if col_tests is not a dict (e.g., error message string)
            if not isinstance(col_tests, dict):
                continue
            for test_name, test_result in col_tests.items():
                if isinstance(test_result, dict) and "significant" in test_result:
                    total_tests += 1
                    if test_result["significant"]:
                        significant_tests += 1

        if total_tests > 0 and significant_tests / total_tests > 0.5:
            recommendations.append(
                "Many statistical tests indicate significant differences between "
                "observed and imputed distributions. Consider alternative imputation "
                "methods or parameter tuning."
            )

        # Overall quality recommendation
        overall_score = self._calculate_overall_score(metric_scores)
        if overall_score < 0.6:
            recommendations.append(
                "Overall imputation quality is below acceptable standards. "
                "Consider trying multiple imputation methods and selecting the best performer, "
                "or implementing ensemble imputation approaches."
            )
        elif overall_score > 0.85:
            recommendations.append(
                "Imputation quality is excellent. The current method performs well "
                "across all quality metrics."
            )

        return recommendations

    def _calculate_overall_score(self, metric_scores: Dict[str, float]) -> float:
        """Calculate overall quality score as weighted average of metric scores"""

        # Define weights for different metrics
        weights = {
            "distribution_preservation": 0.25,
            "correlation_preservation": 0.20,
            "predictive_accuracy": 0.20,
            "domain_validation": 0.20,
            "outlier_detection": 0.10,
            "uncertainty_quantification": 0.05,
        }

        weighted_scores = []
        total_weight = 0

        for metric, score in metric_scores.items():
            if metric in weights:
                weighted_scores.append(score * weights[metric])
                total_weight += weights[metric]

        if total_weight > 0:
            return sum(weighted_scores) / total_weight
        else:
            return np.mean(list(metric_scores.values())) if metric_scores else 0.0


def assess_imputation_quality(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    missing_mask: Optional[pd.DataFrame] = None,
    categorical_cols: Optional[List[str]] = None,
    config: Optional[QualityAssessmentConfig] = None,
) -> QualityAssessmentResult:
    """
    Convenience function for comprehensive imputation quality assessment

    Args:
        original_df: Original data with missing values
        imputed_df: Data after imputation
        missing_mask: Boolean mask of originally missing values
        categorical_cols: List of categorical column names
        config: Assessment configuration

    Returns:
        QualityAssessmentResult with comprehensive metrics
    """
    if config is None:
        config = QualityAssessmentConfig()

    assessor = ImputationQualityAssessor(config)
    return assessor.assess_quality(original_df, imputed_df, missing_mask, categorical_cols)


# Export main classes and functions
__all__ = [
    "QualityMetricType",
    "QualityAssessmentConfig",
    "QualityAssessmentResult",
    "ImputationQualityAssessor",
    "assess_imputation_quality",
]
