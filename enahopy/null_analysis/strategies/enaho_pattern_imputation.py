"""
ENAHO Pattern-Aware Imputation - ANALYZE Phase
===============================================

Specialized imputation strategies that understand ENAHO survey data patterns
and apply domain-specific knowledge for better missing value handling.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import existing advanced imputation
from .advanced_ml_imputation import ImputationConfig, ImputationResult, MICEImputer


class ENAHOMissingPattern(Enum):
    """ENAHO-specific missing data patterns"""

    INCOME_REPORTING = "income_reporting"  # Income variables often missing together
    HOUSEHOLD_COMPOSITION = "household_comp"  # Missing household member data
    EMPLOYMENT_SEQUENCE = "employment_seq"  # Employment variables have logical dependencies
    EDUCATION_CASCADE = "education_cascade"  # Education levels have hierarchical structure
    GEOGRAPHIC_CLUSTER = "geographic_cluster"  # Geographic variables missing by region
    DEMOGRAPHIC_CONSISTENCY = "demo_consistency"  # Age, gender, relationship consistency
    SEASONAL_WORK = "seasonal_work"  # Work-related variables seasonal patterns
    URBAN_RURAL_DIVIDE = "urban_rural"  # Different patterns by urban/rural


@dataclass
class ENAHOImputationConfig(ImputationConfig):
    """Extended configuration for ENAHO-specific imputation"""

    household_id_col: str = "vivienda"
    person_id_col: str = "persona"
    weight_col: str = "factor07"
    geographic_cols: List[str] = field(default_factory=lambda: ["ubigeo", "region", "provincia"])
    income_cols: List[str] = field(default_factory=lambda: ["ing"])
    employment_cols: List[str] = field(default_factory=lambda: ["ocupacion", "rama"])
    education_cols: List[str] = field(default_factory=lambda: ["nivel_educ", "anios_educ"])
    demographic_cols: List[str] = field(default_factory=lambda: ["edad", "sexo", "parentesco"])

    # Pattern-specific parameters
    respect_household_structure: bool = True
    preserve_income_ratios: bool = True
    maintain_education_hierarchy: bool = True
    use_geographic_proximity: bool = True
    seasonal_adjustment: bool = False


class ENAHOPatternDetector:
    """Detects ENAHO-specific missing data patterns"""

    def __init__(self, config: ENAHOImputationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect missing patterns specific to ENAHO survey data"""

        patterns = {}
        missing_matrix = df.isnull()

        # 1. Income Reporting Patterns
        if any(col in df.columns for col in self.config.income_cols):
            income_missing = self._analyze_income_patterns(df, missing_matrix)
            patterns[ENAHOMissingPattern.INCOME_REPORTING.value] = income_missing

        # 2. Household Composition Patterns
        if self.config.household_id_col in df.columns:
            household_patterns = self._analyze_household_patterns(df, missing_matrix)
            patterns[ENAHOMissingPattern.HOUSEHOLD_COMPOSITION.value] = household_patterns

        # 3. Employment Sequence Patterns
        employment_patterns = self._analyze_employment_patterns(df, missing_matrix)
        patterns[ENAHOMissingPattern.EMPLOYMENT_SEQUENCE.value] = employment_patterns

        # 4. Education Cascade Patterns
        education_patterns = self._analyze_education_patterns(df, missing_matrix)
        patterns[ENAHOMissingPattern.EDUCATION_CASCADE.value] = education_patterns

        # 5. Geographic Clustering
        if any(col in df.columns for col in self.config.geographic_cols):
            geo_patterns = self._analyze_geographic_patterns(df, missing_matrix)
            patterns[ENAHOMissingPattern.GEOGRAPHIC_CLUSTER.value] = geo_patterns

        # 6. Demographic Consistency
        demo_patterns = self._analyze_demographic_patterns(df, missing_matrix)
        patterns[ENAHOMissingPattern.DEMOGRAPHIC_CONSISTENCY.value] = demo_patterns

        return patterns

    def _analyze_income_patterns(
        self, df: pd.DataFrame, missing_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze income-related missing patterns"""
        income_cols_present = [col for col in self.config.income_cols if col in df.columns]

        if not income_cols_present:
            return {"pattern": "no_income_columns"}

        # Check for systematic income non-reporting
        income_missing = missing_matrix[income_cols_present]

        # Pattern: All income variables missing together (refusal to report)
        all_income_missing = income_missing.all(axis=1)

        # Pattern: Partial income reporting
        partial_income_missing = income_missing.any(axis=1) & ~all_income_missing

        return {
            "total_income_refusal_rate": all_income_missing.mean(),
            "partial_income_missing_rate": partial_income_missing.mean(),
            "income_correlation_matrix": income_missing.corr().to_dict(),
            "high_correlation_pairs": self._find_high_correlation_pairs(income_missing),
        }

    def _analyze_household_patterns(
        self, df: pd.DataFrame, missing_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze household-level missing patterns"""
        if self.config.household_id_col not in df.columns:
            return {"pattern": "no_household_id"}

        # Group by household
        household_missing_rates = missing_matrix.groupby(df[self.config.household_id_col]).mean()

        # Households with systematic missing data
        high_missing_households = (household_missing_rates > 0.5).any(axis=1)

        return {
            "households_with_high_missing": high_missing_households.sum(),
            "avg_household_missing_rate": household_missing_rates.mean().mean(),
            "household_missing_variance": household_missing_rates.var().mean(),
            "systematic_household_nonresponse": (household_missing_rates > 0.8).any(axis=1).mean(),
        }

    def _analyze_employment_patterns(
        self, df: pd.DataFrame, missing_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze employment sequence patterns"""
        employment_cols_present = [col for col in self.config.employment_cols if col in df.columns]

        if not employment_cols_present:
            return {"pattern": "no_employment_columns"}

        # Logical dependencies in employment data
        employment_missing = missing_matrix[employment_cols_present]

        # Pattern: Employment variables should follow logical sequence
        sequence_violations = 0
        dependency_patterns = {}

        # Example: If occupation is present, economic sector should be present
        if "ocupacion" in employment_cols_present and "rama" in employment_cols_present:
            occupation_present = ~missing_matrix["ocupacion"]
            sector_missing = missing_matrix["rama"]
            sequence_violations += (occupation_present & sector_missing).sum()
            dependency_patterns["occupation_without_sector"] = (
                occupation_present & sector_missing
            ).mean()

        return {
            "employment_sequence_violations": sequence_violations,
            "dependency_patterns": dependency_patterns,
            "employment_missing_correlation": employment_missing.corr().to_dict(),
        }

    def _analyze_education_patterns(
        self, df: pd.DataFrame, missing_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze education hierarchy patterns"""
        education_cols_present = [col for col in self.config.education_cols if col in df.columns]

        if not education_cols_present:
            return {"pattern": "no_education_columns"}

        # Education should follow hierarchical pattern
        hierarchy_violations = 0

        # Example: Years of education should be consistent with education level
        if "nivel_educ" in education_cols_present and "anios_educ" in education_cols_present:
            level_present = ~missing_matrix["nivel_educ"]
            years_missing = missing_matrix["anios_educ"]
            hierarchy_violations += (level_present & years_missing).sum()

        return {
            "education_hierarchy_violations": hierarchy_violations,
            "education_missing_rates": {
                col: missing_matrix[col].mean() for col in education_cols_present
            },
        }

    def _analyze_geographic_patterns(
        self, df: pd.DataFrame, missing_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze geographic clustering patterns"""
        geo_cols_present = [col for col in self.config.geographic_cols if col in df.columns]

        if not geo_cols_present:
            return {"pattern": "no_geographic_columns"}

        # Geographic variables often missing together
        geo_missing = missing_matrix[geo_cols_present]

        # Spatial clustering of missing data
        regional_missing = {}
        if "region" in geo_cols_present and not df["region"].isna().all():
            regional_missing = missing_matrix.groupby(df["region"]).mean().to_dict()

        return {
            "geographic_missing_correlation": geo_missing.corr().to_dict(),
            "regional_missing_patterns": regional_missing,
            "geographic_completeness": (~geo_missing.any(axis=1)).mean(),
        }

    def _analyze_demographic_patterns(
        self, df: pd.DataFrame, missing_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze demographic consistency patterns"""
        demo_cols_present = [col for col in self.config.demographic_cols if col in df.columns]

        if not demo_cols_present:
            return {"pattern": "no_demographic_columns"}

        # Demographic variables should be consistent
        demo_missing = missing_matrix[demo_cols_present]

        # Age should rarely be missing in survey data
        age_missing_rate = (
            demo_missing.get("edad", pd.Series([0])).mean() if "edad" in demo_cols_present else 0
        )

        # Gender should rarely be missing
        gender_missing_rate = (
            demo_missing.get("sexo", pd.Series([0])).mean() if "sexo" in demo_cols_present else 0
        )

        return {
            "critical_demographic_missing": {
                "age_missing_rate": age_missing_rate,
                "gender_missing_rate": gender_missing_rate,
            },
            "demographic_completeness": (~demo_missing.any(axis=1)).mean(),
            "demographic_correlation": demo_missing.corr().to_dict(),
        }

    def _find_high_correlation_pairs(
        self, missing_df: pd.DataFrame, threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of variables with high missing correlation"""
        corr_matrix = missing_df.corr()
        high_corr_pairs = []

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        high_corr_pairs.append((col1, col2, corr_value))

        return high_corr_pairs


class ENAHOPatternImputer:
    """Pattern-aware imputer for ENAHO survey data"""

    def __init__(self, config: ENAHOImputationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pattern_detector = ENAHOPatternDetector(config, logger)
        self.base_imputers = {}

    def fit_transform(self, df: pd.DataFrame) -> ImputationResult:
        """
        Perform pattern-aware imputation for ENAHO data

        Args:
            df: ENAHO DataFrame with missing values

        Returns:
            ImputationResult with pattern-aware imputed data
        """
        import time

        start_time = time.time()

        self.logger.info("Starting ENAHO pattern-aware imputation...")

        # 1. Detect patterns
        missing_patterns = self.pattern_detector.detect_patterns(df)

        # 2. Prepare working dataframe
        working_df = df.copy()

        # 3. Apply pattern-specific imputation strategies

        # Strategy 1: Household-level imputation for household composition
        if self.config.respect_household_structure and self.config.household_id_col in df.columns:
            working_df = self._impute_household_structure(working_df)

        # Strategy 2: Geographic proximity imputation
        if self.config.use_geographic_proximity:
            working_df = self._impute_geographic_proximity(working_df)

        # Strategy 3: Income ratio preservation
        if self.config.preserve_income_ratios:
            working_df = self._impute_preserve_income_ratios(working_df)

        # Strategy 4: Education hierarchy maintenance
        if self.config.maintain_education_hierarchy:
            working_df = self._impute_education_hierarchy(working_df)

        # Strategy 5: Employment sequence logic
        working_df = self._impute_employment_sequence(working_df)

        # 4. Final cleanup with advanced ML for remaining missing values
        remaining_missing = working_df.isnull().sum().sum()

        if remaining_missing > 0:
            self.logger.info(f"Applying MICE for {remaining_missing} remaining missing values...")
            mice_config = ImputationConfig(method="mice", max_iter=5)
            mice_imputer = MICEImputer(mice_config, self.logger)

            # Identify categorical columns for MICE
            categorical_cols = self._identify_categorical_columns(working_df)

            mice_result = mice_imputer.fit_transform(working_df, categorical_cols=categorical_cols)
            working_df = mice_result.imputed_data

        # 5. Quality validation
        quality_metrics = self._validate_enaho_constraints(df, working_df)

        # 6. Prepare final result
        total_time = time.time() - start_time

        diagnostics = {
            "missing_patterns": missing_patterns,
            "pattern_strategies_applied": [
                "household_structure" if self.config.respect_household_structure else None,
                "geographic_proximity" if self.config.use_geographic_proximity else None,
                "income_ratios" if self.config.preserve_income_ratios else None,
                "education_hierarchy" if self.config.maintain_education_hierarchy else None,
                "employment_sequence",
            ],
            "remaining_missing_after_patterns": remaining_missing,
            "constraint_violations": quality_metrics.get("constraint_violations", {}),
        }

        return ImputationResult(
            imputed_data=working_df,
            method="ENAHO_Pattern_Aware",
            quality_metrics=quality_metrics,
            imputation_diagnostics=diagnostics,
            missing_patterns=missing_patterns,
            computational_time=total_time,
        )

    def _impute_household_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using household structure information"""
        if self.config.household_id_col not in df.columns:
            return df

        working_df = df.copy()

        # For each household, use other household members to impute missing values
        for household_id in df[self.config.household_id_col].unique():
            if pd.isna(household_id):
                continue

            household_mask = df[self.config.household_id_col] == household_id
            household_data = working_df[household_mask].copy()

            if len(household_data) <= 1:
                continue

            # Impute within household using household head information
            for col in self.config.demographic_cols + self.config.geographic_cols:
                if col in household_data.columns:
                    # Use most common value within household
                    household_mode = household_data[col].mode()
                    if len(household_mode) > 0 and not pd.isna(household_mode.iloc[0]):
                        missing_mask = household_data[col].isnull()
                        working_df.loc[household_mask & missing_mask, col] = household_mode.iloc[0]

        self.logger.info("Applied household structure imputation")
        return working_df

    def _impute_geographic_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using geographic proximity"""
        working_df = df.copy()

        for geo_col in self.config.geographic_cols:
            if geo_col not in df.columns:
                continue

            # Group by higher-level geographic unit
            if geo_col == "provincia" and "region" in df.columns:
                # Impute province using region mode
                for region in df["region"].unique():
                    if pd.isna(region):
                        continue
                    region_mask = df["region"] == region
                    region_data = working_df[region_mask]

                    province_mode = region_data["provincia"].mode()
                    if len(province_mode) > 0:
                        missing_mask = region_data["provincia"].isnull()
                        working_df.loc[
                            region_mask & missing_mask, "provincia"
                        ] = province_mode.iloc[0]

        self.logger.info("Applied geographic proximity imputation")
        return working_df

    def _impute_preserve_income_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute income while preserving household income ratios"""
        working_df = df.copy()

        # If household income is available, use it to impute individual income
        income_cols_present = [col for col in self.config.income_cols if col in df.columns]

        if not income_cols_present or self.config.household_id_col not in df.columns:
            return working_df

        for income_col in income_cols_present:
            # Calculate household totals for complete cases
            household_income = working_df.groupby(self.config.household_id_col)[income_col].sum()
            household_counts = working_df.groupby(self.config.household_id_col)[income_col].count()

            # Average individual income per household member
            avg_individual_income = household_income / household_counts
            avg_individual_income = avg_individual_income.replace([np.inf, -np.inf], np.nan)

            # Fill missing individual incomes with household average
            for household_id in working_df[self.config.household_id_col].unique():
                if pd.isna(household_id):
                    continue

                household_mask = working_df[self.config.household_id_col] == household_id
                missing_income_mask = working_df[income_col].isnull()

                if household_id in avg_individual_income.index and not pd.isna(
                    avg_individual_income[household_id]
                ):
                    working_df.loc[
                        household_mask & missing_income_mask, income_col
                    ] = avg_individual_income[household_id]

        self.logger.info("Applied income ratio preservation imputation")
        return working_df

    def _impute_education_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute education maintaining hierarchical consistency"""
        working_df = df.copy()

        # Define education hierarchy (ENAHO specific)
        education_hierarchy = {
            1: (0, 5),  # Sin nivel: 0-5 años
            2: (6, 6),  # Inicial: 6 años
            3: (6, 11),  # Primaria: 6-11 años
            4: (12, 16),  # Secundaria: 12-16 años
            5: (17, 22),  # Superior: 17+ años
            6: (15, 25),  # Superior universitaria: 15+ años
        }

        if "nivel_educ" in df.columns and "anios_educ" in df.columns:
            for idx, row in working_df.iterrows():
                nivel = row["nivel_educ"]
                anios = row["anios_educ"]

                # If education level is present but years is missing
                if not pd.isna(nivel) and pd.isna(anios):
                    if nivel in education_hierarchy:
                        min_years, max_years = education_hierarchy[nivel]
                        # Use midpoint of range
                        working_df.loc[idx, "anios_educ"] = (min_years + max_years) / 2

                # If years is present but level is missing
                elif pd.isna(nivel) and not pd.isna(anios):
                    for level, (min_y, max_y) in education_hierarchy.items():
                        if min_y <= anios <= max_y:
                            working_df.loc[idx, "nivel_educ"] = level
                            break

        self.logger.info("Applied education hierarchy imputation")
        return working_df

    def _impute_employment_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute employment variables maintaining logical sequence"""
        working_df = df.copy()

        employment_cols_present = [col for col in self.config.employment_cols if col in df.columns]

        # If occupation is present, try to impute economic sector
        if "ocupacion" in employment_cols_present and "rama" in employment_cols_present:
            # Create occupation-sector mapping from complete cases
            complete_cases = working_df[["ocupacion", "rama"]].dropna()
            if len(complete_cases) > 0:
                occupation_sector_map = (
                    complete_cases.groupby("ocupacion")["rama"]
                    .apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
                    .to_dict()
                )

                # Apply mapping to missing sectors
                missing_sector = working_df["rama"].isnull() & working_df["ocupacion"].notna()
                for idx in working_df[missing_sector].index:
                    occupation = working_df.loc[idx, "ocupacion"]
                    if occupation in occupation_sector_map and not pd.isna(
                        occupation_sector_map[occupation]
                    ):
                        working_df.loc[idx, "rama"] = occupation_sector_map[occupation]

        self.logger.info("Applied employment sequence imputation")
        return working_df

    def _identify_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify categorical columns for final MICE imputation"""
        categorical_cols = []

        for col in df.columns:
            if df[col].dtype == "object":
                categorical_cols.append(col)
            elif df[col].dtype in ["int64", "float64"] and df[col].nunique() <= 20:
                # Likely categorical if few unique values
                categorical_cols.append(col)

        return categorical_cols

    def _validate_enaho_constraints(
        self, original_df: pd.DataFrame, imputed_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate ENAHO-specific constraints after imputation"""
        violations = {}

        # 1. Age constraints (should be between 0-120)
        if "edad" in imputed_df.columns:
            age_violations = ((imputed_df["edad"] < 0) | (imputed_df["edad"] > 120)).sum()
            violations["age_constraint_violations"] = age_violations

        # 2. Education hierarchy violations
        if "nivel_educ" in imputed_df.columns and "anios_educ" in imputed_df.columns:
            # Check if years of education is consistent with level
            hierarchy_violations = 0
            for idx, row in imputed_df.iterrows():
                if not pd.isna(row["nivel_educ"]) and not pd.isna(row["anios_educ"]):
                    nivel = row["nivel_educ"]
                    anios = row["anios_educ"]

                    # Basic consistency check (simplified)
                    if nivel == 1 and anios > 5:  # Sin nivel but many years
                        hierarchy_violations += 1
                    elif nivel == 3 and anios > 11:  # Primary but too many years
                        hierarchy_violations += 1

            violations["education_hierarchy_violations"] = hierarchy_violations

        # 3. Income reasonableness (positive values)
        for income_col in self.config.income_cols:
            if income_col in imputed_df.columns:
                negative_income = (imputed_df[income_col] < 0).sum()
                violations[f"{income_col}_negative_values"] = negative_income

        # 4. Distribution preservation
        distribution_metrics = {}
        for col in original_df.columns:
            if original_df[col].notna().sum() > 10:  # Enough data for comparison
                original_mean = original_df[col].mean()
                imputed_mean = imputed_df[col].mean()

                if not pd.isna(original_mean) and not pd.isna(imputed_mean):
                    mean_change_pct = abs(imputed_mean - original_mean) / abs(original_mean) * 100
                    distribution_metrics[f"{col}_mean_change_pct"] = mean_change_pct

        return {
            "constraint_violations": violations,
            "distribution_preservation": distribution_metrics,
            "overall_quality_score": self._calculate_quality_score(
                violations, distribution_metrics
            ),
        }

    def _calculate_quality_score(self, violations: Dict, distribution_metrics: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        score = 100.0

        # Penalize constraint violations
        for violation_count in violations.values():
            if isinstance(violation_count, (int, float)):
                score -= min(violation_count * 0.1, 20)  # Max 20 points penalty per violation type

        # Penalize large distribution changes
        for change_pct in distribution_metrics.values():
            if isinstance(change_pct, (int, float)):
                if change_pct > 10:  # More than 10% change
                    score -= min(change_pct - 10, 15)  # Max 15 points penalty

        return max(score, 0.0)


def create_enaho_pattern_imputer(
    config: Optional[ENAHOImputationConfig] = None,
) -> ENAHOPatternImputer:
    """Factory function to create ENAHO pattern-aware imputer"""
    if config is None:
        config = ENAHOImputationConfig(method="enaho_pattern_aware")

    return ENAHOPatternImputer(config)


# Export main classes
__all__ = [
    "ENAHOMissingPattern",
    "ENAHOImputationConfig",
    "ENAHOPatternDetector",
    "ENAHOPatternImputer",
    "create_enaho_pattern_imputer",
]
