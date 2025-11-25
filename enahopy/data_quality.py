"""
ENAHO Data Quality Assessment System
==================================

Comprehensive data quality scoring and assessment tools for ENAHO survey data.
Evaluates completeness, consistency, accuracy, and validity of household survey data.
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
    pass

    REGEX_AVAILABLE = True
except ImportError:
    REGEX_AVAILABLE = False


@dataclass
class QualityDimension:
    """Represents a data quality dimension with score and details"""

    name: str
    score: float
    max_score: float
    weight: float
    details: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]


@dataclass
class DataQualityReport:
    """Complete data quality assessment report"""

    overall_score: float
    max_score: float
    grade: str
    dimensions: Dict[str, QualityDimension]
    sample_info: Dict[str, Any]
    critical_issues: List[str]
    recommendations: List[str]
    timestamp: pd.Timestamp


class CompletenessAssessor:
    """Assesses data completeness across different dimensions"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def assess_completeness(self, df: pd.DataFrame) -> QualityDimension:
        """
        Assess overall data completeness

        Args:
            df: DataFrame to assess

        Returns:
            QualityDimension with completeness assessment
        """
        details = {}
        issues = []
        recommendations = []

        # Overall completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_rate = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        details["total_cells"] = total_cells
        details["missing_cells"] = missing_cells
        details["completeness_rate"] = completeness_rate

        # Column-level completeness
        column_completeness = {}
        critical_columns = []

        for col in df.columns:
            col_completeness = 1 - (df[col].isnull().sum() / len(df))
            column_completeness[col] = col_completeness

            if col_completeness < 0.5:
                critical_columns.append(col)
                issues.append(f"Column '{col}' has low completeness: {col_completeness:.2%}")

        details["column_completeness"] = column_completeness
        details["critical_columns"] = critical_columns

        # Row-level completeness
        row_completeness = df.notna().mean(axis=1)
        complete_rows = (row_completeness == 1).sum()
        incomplete_rows = len(df) - complete_rows

        details["complete_rows"] = complete_rows
        details["incomplete_rows"] = incomplete_rows
        details["complete_rows_rate"] = complete_rows / len(df) if len(df) > 0 else 0

        # Pattern analysis
        missing_patterns = self._analyze_missing_patterns(df)
        details["missing_patterns"] = missing_patterns

        # Scoring (0-100)
        score = completeness_rate * 100

        # Adjust score based on critical issues
        if len(critical_columns) > 0:
            penalty = min(30, len(critical_columns) * 10)
            score = max(0, score - penalty)

        # Generate recommendations
        if completeness_rate < 0.95:
            recommendations.append("Investigate sources of missing data")
        if len(critical_columns) > 0:
            recommendations.append(f"Review data collection for columns: {critical_columns[:3]}")
        if details["complete_rows_rate"] < 0.8:
            recommendations.append("Consider record-level data quality controls")

        return QualityDimension(
            name="Completeness",
            score=score,
            max_score=100,
            weight=0.3,
            details=details,
            issues=issues,
            recommendations=recommendations,
        )

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        missing_matrix = df.isnull()

        # Common missing patterns
        pattern_counts = {}
        for _, row in missing_matrix.iterrows():
            pattern = "".join(["1" if x else "0" for x in row])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Most common patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        top_patterns = sorted_patterns[:5]

        return {
            "total_patterns": len(pattern_counts),
            "top_patterns": top_patterns,
            "pattern_diversity": len(pattern_counts) / len(df) if len(df) > 0 else 0,
        }


class ConsistencyAssessor:
    """Assesses data consistency and logical coherence"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def assess_consistency(
        self, df: pd.DataFrame, consistency_rules: Optional[Dict[str, Any]] = None
    ) -> QualityDimension:
        """
        Assess data consistency

        Args:
            df: DataFrame to assess
            consistency_rules: Optional custom consistency rules

        Returns:
            QualityDimension with consistency assessment
        """
        details = {}
        issues = []
        recommendations = []

        # Default consistency checks for ENAHO data
        default_rules = self._get_default_consistency_rules()
        consistency_rules or default_rules

        # Track consistency violations
        violations = {}

        # Age consistency checks
        age_violations = self._check_age_consistency(df)
        if age_violations["count"] > 0:
            violations["age_consistency"] = age_violations
            issues.extend(age_violations["issues"])

        # Income consistency checks
        income_violations = self._check_income_consistency(df)
        if income_violations["count"] > 0:
            violations["income_consistency"] = income_violations
            issues.extend(income_violations["issues"])

        # Household composition checks
        household_violations = self._check_household_consistency(df)
        if household_violations["count"] > 0:
            violations["household_consistency"] = household_violations
            issues.extend(household_violations["issues"])

        # Education consistency checks
        education_violations = self._check_education_consistency(df)
        if education_violations["count"] > 0:
            violations["education_consistency"] = education_violations
            issues.extend(education_violations["issues"])

        # Calculate consistency score
        total_violations = sum(v["count"] for v in violations.values())
        total_records = len(df)

        consistency_rate = 1 - (total_violations / total_records) if total_records > 0 else 1
        score = consistency_rate * 100

        details["consistency_rate"] = consistency_rate
        details["total_violations"] = total_violations
        details["violations_by_type"] = violations
        details["violation_rate"] = total_violations / total_records if total_records > 0 else 0

        # Generate recommendations
        if consistency_rate < 0.95:
            recommendations.append("Implement data validation rules during collection")
        if total_violations > 0:
            recommendations.append("Review and correct logical inconsistencies")

        return QualityDimension(
            name="Consistency",
            score=score,
            max_score=100,
            weight=0.25,
            details=details,
            issues=issues,
            recommendations=recommendations,
        )

    def _get_default_consistency_rules(self) -> Dict[str, Any]:
        """Get default consistency rules for ENAHO data"""
        return {
            "age_rules": {"min_age": 0, "max_age": 120, "working_age_min": 14},
            "income_rules": {"min_income": 0, "max_income_multiplier": 100},  # times median
            "education_rules": {"min_education_age": 3, "max_years_education": 25},
        }

    def _check_age_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check age-related consistency"""
        violations = {"count": 0, "issues": [], "details": {}}

        age_columns = [col for col in df.columns if "edad" in col.lower() or "age" in col.lower()]

        for age_col in age_columns:
            if age_col in df.columns:
                # Check for reasonable age ranges
                invalid_ages = (df[age_col] < 0) | (df[age_col] > 120)
                invalid_count = invalid_ages.sum()

                if invalid_count > 0:
                    violations["count"] += invalid_count
                    violations["issues"].append(
                        f"Found {invalid_count} records with invalid ages in column '{age_col}'"
                    )

                violations["details"][age_col] = {
                    "invalid_ages": invalid_count,
                    "min_age": df[age_col].min(),
                    "max_age": df[age_col].max(),
                }

        return violations

    def _check_income_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check income-related consistency"""
        violations = {"count": 0, "issues": [], "details": {}}

        income_columns = [
            col
            for col in df.columns
            if any(keyword in col.lower() for keyword in ["ingreso", "income", "sueldo", "salario"])
        ]

        for income_col in income_columns:
            if income_col in df.columns and pd.api.types.is_numeric_dtype(df[income_col]):
                # Check for negative incomes
                negative_income = df[income_col] < 0
                negative_count = negative_income.sum()

                if negative_count > 0:
                    violations["count"] += negative_count
                    violations["issues"].append(
                        f"Found {negative_count} records with negative income in column '{income_col}'"
                    )

                # Check for extremely high incomes (outliers)
                if len(df[income_col].dropna()) > 0:
                    median_income = df[income_col].median()
                    if median_income > 0:
                        extreme_threshold = median_income * 100
                        extreme_income = df[income_col] > extreme_threshold
                        extreme_count = extreme_income.sum()

                        if extreme_count > 0:
                            violations["count"] += extreme_count
                            violations["issues"].append(
                                f"Found {extreme_count} records with extremely high income in column '{income_col}'"
                            )

                violations["details"][income_col] = {
                    "negative_income": negative_count,
                    "min_income": df[income_col].min(),
                    "max_income": df[income_col].max(),
                    "median_income": df[income_col].median(),
                }

        return violations

    def _check_household_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check household composition consistency"""
        violations = {"count": 0, "issues": [], "details": {}}

        # Check for household size consistency
        household_cols = [
            col for col in df.columns if "hogar" in col.lower() or "household" in col.lower()
        ]

        if household_cols:
            # Basic household size checks
            for col in household_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    invalid_size = (df[col] < 0) | (
                        df[col] > 20
                    )  # Reasonable household size limits
                    invalid_count = invalid_size.sum()

                    if invalid_count > 0:
                        violations["count"] += invalid_count
                        violations["issues"].append(
                            f"Found {invalid_count} records with invalid household size in column '{col}'"
                        )

        return violations

    def _check_education_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check education-related consistency"""
        violations = {"count": 0, "issues": [], "details": {}}

        education_cols = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["educacion", "education", "escolaridad", "años_estudio"]
            )
        ]

        for edu_col in education_cols:
            if edu_col in df.columns and pd.api.types.is_numeric_dtype(df[edu_col]):
                # Check for reasonable education years
                invalid_education = (df[edu_col] < 0) | (df[edu_col] > 25)
                invalid_count = invalid_education.sum()

                if invalid_count > 0:
                    violations["count"] += invalid_count
                    violations["issues"].append(
                        f"Found {invalid_count} records with invalid education years in column '{edu_col}'"
                    )

                violations["details"][edu_col] = {
                    "invalid_education": invalid_count,
                    "min_education": df[edu_col].min(),
                    "max_education": df[edu_col].max(),
                }

        return violations


class AccuracyAssessor:
    """Assesses data accuracy through statistical analysis and outlier detection"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def assess_accuracy(self, df: pd.DataFrame) -> QualityDimension:
        """
        Assess data accuracy through statistical analysis

        Args:
            df: DataFrame to assess

        Returns:
            QualityDimension with accuracy assessment
        """
        details = {}
        issues = []
        recommendations = []

        # Outlier detection
        outlier_analysis = self._detect_outliers(df)
        details["outlier_analysis"] = outlier_analysis

        # Statistical distribution analysis
        distribution_analysis = self._analyze_distributions(df)
        details["distribution_analysis"] = distribution_analysis

        # Duplicate detection
        duplicate_analysis = self._detect_duplicates(df)
        details["duplicate_analysis"] = duplicate_analysis

        # Calculate accuracy score
        total_records = len(df)
        accuracy_issues = outlier_analysis["total_outliers"] + duplicate_analysis["duplicate_count"]

        accuracy_rate = 1 - (accuracy_issues / total_records) if total_records > 0 else 1
        score = accuracy_rate * 100

        # Adjust score based on distribution quality
        distribution_penalty = 0
        for col, dist_info in distribution_analysis.items():
            if dist_info.get("normality_p_value", 1) < 0.01 and dist_info.get("skewness", 0) > 3:
                distribution_penalty += 5

        score = max(0, score - min(distribution_penalty, 20))

        # Generate issues and recommendations
        if outlier_analysis["total_outliers"] > 0:
            issues.append(f"Detected {outlier_analysis['total_outliers']} potential outliers")
            recommendations.append("Review and validate outlying values")

        if duplicate_analysis["duplicate_count"] > 0:
            issues.append(f"Found {duplicate_analysis['duplicate_count']} duplicate records")
            recommendations.append("Remove or consolidate duplicate entries")

        if distribution_penalty > 10:
            recommendations.append("Review data distributions for potential data entry errors")

        return QualityDimension(
            name="Accuracy",
            score=score,
            max_score=100,
            weight=0.25,
            details=details,
            issues=issues,
            recommendations=recommendations,
        )

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using statistical methods"""
        outlier_info = {"by_column": {}, "total_outliers": 0}

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) < 10:  # Need sufficient data
                continue

            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            # Z-score method
            if SCIPY_AVAILABLE:
                z_scores = np.abs(stats.zscore(col_data))
                z_outliers = (z_scores > 3).sum()
            else:
                z_outliers = 0

            outlier_info["by_column"][col] = {
                "iqr_outliers": iqr_outliers,
                "z_score_outliers": z_outliers,
                "outlier_rate": iqr_outliers / len(col_data),
                "bounds": {"lower": lower_bound, "upper": upper_bound},
            }

            outlier_info["total_outliers"] += iqr_outliers

        return outlier_info

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze statistical distributions of numeric columns"""
        distribution_info = {}

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue

            info = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
            }

            # Normality test
            if SCIPY_AVAILABLE and len(col_data) > 8:
                try:
                    _, p_value = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                    info["normality_p_value"] = p_value
                except Exception:
                    info["normality_p_value"] = None

            distribution_info[col] = info

        return distribution_info

    def _detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate records"""
        duplicate_info = {
            "total_duplicates": df.duplicated().sum(),
            "duplicate_count": df.duplicated().sum(),
            "unique_records": len(df) - df.duplicated().sum(),
            "duplicate_rate": df.duplicated().sum() / len(df) if len(df) > 0 else 0,
        }

        # Check for partial duplicates (subset of columns)
        key_columns = df.select_dtypes(include=[object, "category"]).columns[
            :5
        ]  # First 5 categorical columns
        if len(key_columns) > 0:
            partial_duplicates = df.duplicated(subset=key_columns).sum()
            duplicate_info["partial_duplicates"] = partial_duplicates
            duplicate_info["partial_duplicate_rate"] = (
                partial_duplicates / len(df) if len(df) > 0 else 0
            )

        return duplicate_info


class ValidityAssessor:
    """Assesses data validity through format and domain validation"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def assess_validity(
        self, df: pd.DataFrame, validation_rules: Optional[Dict[str, Any]] = None
    ) -> QualityDimension:
        """
        Assess data validity

        Args:
            df: DataFrame to assess
            validation_rules: Optional custom validation rules

        Returns:
            QualityDimension with validity assessment
        """
        details = {}
        issues = []
        recommendations = []

        # Format validation
        format_validation = self._validate_formats(df)
        details["format_validation"] = format_validation

        # Domain validation
        domain_validation = self._validate_domains(df, validation_rules)
        details["domain_validation"] = domain_validation

        # Data type validation
        type_validation = self._validate_data_types(df)
        details["type_validation"] = type_validation

        # Calculate validity score
        total_records = len(df)
        validity_issues = (
            format_validation.get("total_invalid", 0)
            + domain_validation.get("total_invalid", 0)
            + type_validation.get("total_invalid", 0)
        )

        validity_rate = (
            1 - (validity_issues / (total_records * len(df.columns))) if total_records > 0 else 1
        )
        score = validity_rate * 100

        # Generate issues and recommendations
        if format_validation.get("total_invalid", 0) > 0:
            issues.append(f"Found {format_validation['total_invalid']} format validation errors")
            recommendations.append("Standardize data formats during collection")

        if domain_validation.get("total_invalid", 0) > 0:
            issues.append(f"Found {domain_validation['total_invalid']} domain validation errors")
            recommendations.append("Implement domain validation rules")

        return QualityDimension(
            name="Validity",
            score=score,
            max_score=100,
            weight=0.2,
            details=details,
            issues=issues,
            recommendations=recommendations,
        )

    def _validate_formats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data formats"""
        format_info = {"by_column": {}, "total_invalid": 0}

        if not REGEX_AVAILABLE:
            return format_info

        # Common format patterns
        patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?[\d\s\-\(\)]{7,15}$",
            "date": r"^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$",
            "numeric": r"^-?\d+\.?\d*$",
        }

        for col in df.select_dtypes(include=[object]).columns:
            col_data = df[col].dropna().astype(str)

            # Detect likely format based on column name
            likely_format = None
            col_lower = col.lower()

            if any(keyword in col_lower for keyword in ["email", "correo"]):
                likely_format = "email"
            elif any(keyword in col_lower for keyword in ["telefono", "phone", "celular"]):
                likely_format = "phone"
            elif any(keyword in col_lower for keyword in ["fecha", "date"]):
                likely_format = "date"

            if likely_format and likely_format in patterns:
                pattern = patterns[likely_format]
                valid_mask = col_data.str.match(pattern)
                invalid_count = (~valid_mask).sum()

                format_info["by_column"][col] = {
                    "format_type": likely_format,
                    "invalid_count": invalid_count,
                    "validity_rate": valid_mask.mean(),
                }

                format_info["total_invalid"] += invalid_count

        return format_info

    def _validate_domains(
        self, df: pd.DataFrame, validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate values against expected domains"""
        domain_info = {"by_column": {}, "total_invalid": 0}

        # Default domain rules for ENAHO
        default_rules = {
            "gender": ["M", "F", "Masculino", "Femenino", "1", "2"],
            "yes_no": ["Si", "No", "Sí", "YES", "NO", "1", "0"],
            "urban_rural": ["Urbano", "Rural", "U", "R", "1", "2"],
        }

        rules = validation_rules or {}

        for col in df.columns:
            col_lower = col.lower()

            # Detect likely domain based on column name
            likely_domain = None
            if any(keyword in col_lower for keyword in ["sexo", "gender", "genero"]):
                likely_domain = "gender"
            elif any(keyword in col_lower for keyword in ["si_no", "yes_no", "respuesta"]):
                likely_domain = "yes_no"
            elif any(keyword in col_lower for keyword in ["urbano", "rural", "area"]):
                likely_domain = "urban_rural"

            # Apply validation if domain is detected or specified
            domain_values = rules.get(col) or (
                default_rules.get(likely_domain) if likely_domain else None
            )

            if domain_values:
                col_data = df[col].dropna()
                valid_mask = col_data.isin(domain_values)
                invalid_count = (~valid_mask).sum()

                domain_info["by_column"][col] = {
                    "expected_values": domain_values,
                    "invalid_count": invalid_count,
                    "validity_rate": valid_mask.mean(),
                    "unique_values": col_data.unique().tolist()[:10],  # First 10 unique values
                }

                domain_info["total_invalid"] += invalid_count

        return domain_info

    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types consistency"""
        type_info = {"by_column": {}, "total_invalid": 0}

        for col in df.columns:
            col_data = df[col].dropna()

            # Check for mixed types in supposedly numeric columns
            if df[col].dtype == "object":
                # Try to identify if column should be numeric
                numeric_count = 0
                non_numeric_count = 0

                for value in col_data.sample(min(100, len(col_data))):
                    try:
                        float(str(value))
                        numeric_count += 1
                    except (ValueError, TypeError):
                        non_numeric_count += 1

                # If mostly numeric, flag non-numeric as invalid
                if numeric_count > non_numeric_count and numeric_count > 10:
                    # Count actual non-numeric values
                    invalid_count = 0
                    for value in col_data:
                        try:
                            float(str(value))
                        except (ValueError, TypeError):
                            invalid_count += 1

                    if invalid_count > 0:
                        type_info["by_column"][col] = {
                            "expected_type": "numeric",
                            "actual_type": "mixed",
                            "invalid_count": invalid_count,
                            "type_consistency": 1 - (invalid_count / len(col_data)),
                        }

                        type_info["total_invalid"] += invalid_count

        return type_info


class DataQualityAssessment:
    """Main data quality assessment system"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        self.completeness_assessor = CompletenessAssessor(logger)
        self.consistency_assessor = ConsistencyAssessor(logger)
        self.accuracy_assessor = AccuracyAssessor(logger)
        self.validity_assessor = ValidityAssessor(logger)

    def assess_data_quality(
        self, df: pd.DataFrame, custom_rules: Optional[Dict[str, Any]] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality assessment

        Args:
            df: DataFrame to assess
            custom_rules: Optional custom validation rules

        Returns:
            Complete data quality report
        """
        self.logger.info(f"Starting data quality assessment for DataFrame with shape {df.shape}")

        # Assess each dimension
        dimensions = {}

        dimensions["completeness"] = self.completeness_assessor.assess_completeness(df)
        dimensions["consistency"] = self.consistency_assessor.assess_consistency(df, custom_rules)
        dimensions["accuracy"] = self.accuracy_assessor.assess_accuracy(df)
        dimensions["validity"] = self.validity_assessor.assess_validity(df, custom_rules)

        # Calculate overall score
        weighted_score = sum(dim.score * dim.weight for dim in dimensions.values())
        max_weighted_score = sum(dim.max_score * dim.weight for dim in dimensions.values())
        overall_score = weighted_score / max_weighted_score * 100 if max_weighted_score > 0 else 0

        # Determine grade
        grade = self._calculate_grade(overall_score)

        # Collect all issues and recommendations
        all_issues = []
        all_recommendations = []
        critical_issues = []

        for dim in dimensions.values():
            all_issues.extend(dim.issues)
            all_recommendations.extend(dim.recommendations)

            # Identify critical issues (score < 50)
            if dim.score < 50:
                critical_issues.append(f"Critical issue in {dim.name}: Score {dim.score:.1f}/100")

        # Sample information
        sample_info = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=[object, "category"]).columns),
            "datetime_columns": len(df.select_dtypes(include=["datetime64"]).columns),
        }

        return DataQualityReport(
            overall_score=overall_score,
            max_score=100,
            grade=grade,
            dimensions=dimensions,
            sample_info=sample_info,
            critical_issues=critical_issues,
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            timestamp=pd.Timestamp.now(),
        )

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on score"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def generate_summary_report(self, report: DataQualityReport) -> str:
        """Generate human-readable summary report"""
        summary = []
        summary.append("=" * 60)
        summary.append("ENAHO DATA QUALITY ASSESSMENT REPORT")
        summary.append("=" * 60)
        summary.append(f"Assessment Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Overall Score: {report.overall_score:.1f}/100 (Grade: {report.grade})")
        summary.append("")

        # Sample information
        summary.append("DATASET INFORMATION:")
        summary.append(f"  • Total Records: {report.sample_info['total_records']:,}")
        summary.append(f"  • Total Columns: {report.sample_info['total_columns']}")
        summary.append(f"  • Memory Usage: {report.sample_info['memory_usage_mb']:.1f} MB")
        summary.append("")

        # Dimension scores
        summary.append("QUALITY DIMENSIONS:")
        for name, dimension in report.dimensions.items():
            summary.append(
                f"  • {dimension.name}: {dimension.score:.1f}/100 (Weight: {dimension.weight:.0%})"
            )
        summary.append("")

        # Critical issues
        if report.critical_issues:
            summary.append("CRITICAL ISSUES:")
            for issue in report.critical_issues:
                summary.append(f"  ⚠️  {issue}")
            summary.append("")

        # Top recommendations
        if report.recommendations:
            summary.append("KEY RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                summary.append(f"  {i}. {rec}")
            summary.append("")

        summary.append("=" * 60)

        return "\n".join(summary)


def assess_data_quality(
    df: pd.DataFrame, custom_rules: Optional[Dict[str, Any]] = None
) -> DataQualityReport:
    """
    Convenience function for quick data quality assessment

    Args:
        df: DataFrame to assess
        custom_rules: Optional custom validation rules

    Returns:
        Complete data quality report
    """
    assessor = DataQualityAssessment()
    return assessor.assess_data_quality(df, custom_rules)


def quick_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick data quality check with key metrics

    Args:
        df: DataFrame to assess

    Returns:
        Dictionary with key quality metrics
    """
    report = assess_data_quality(df)

    return {
        "overall_score": report.overall_score,
        "grade": report.grade,
        "completeness_score": report.dimensions["completeness"].score,
        "consistency_score": report.dimensions["consistency"].score,
        "accuracy_score": report.dimensions["accuracy"].score,
        "validity_score": report.dimensions["validity"].score,
        "critical_issues_count": len(report.critical_issues),
        "total_records": report.sample_info["total_records"],
        "recommendations_count": len(report.recommendations),
    }
