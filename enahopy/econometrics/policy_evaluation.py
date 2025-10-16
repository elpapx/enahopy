"""
ENAHO Advanced Policy Evaluation - ANALYZE Phase
===============================================

Comprehensive policy evaluation toolkit for impact assessment and microsimulation
using ENAHO survey data. Includes policy impact evaluators, microsimulation models,
and scenario analysis for evidence-based policy making.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


@dataclass
class PolicyScenario:
    """Container for policy scenario definition"""

    name: str
    description: str
    parameters: Dict[str, Any]
    target_groups: Optional[List[str]] = None
    implementation_date: Optional[str] = None
    duration_months: Optional[int] = None


@dataclass
class PolicyImpact:
    """Container for policy impact results"""

    scenario_name: str
    direct_effects: Dict[str, float]
    indirect_effects: Dict[str, float]
    distributional_effects: Dict[str, Any]
    welfare_effects: Dict[str, float]
    fiscal_effects: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sample_size: int
    method: str


class PolicyImpactEvaluator:
    """
    Advanced policy impact evaluation using multiple methodologies
    Supports ex-ante and ex-post policy evaluation
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.baseline_data = None
        self.policy_scenarios = {}
        self.impact_results = {}

    def set_baseline_data(self, df: pd.DataFrame, weight_col: Optional[str] = None):
        """Set baseline data for policy evaluation"""
        self.baseline_data = df.copy()
        self.weight_col = weight_col

        if weight_col and weight_col in df.columns:
            self.baseline_weights = df[weight_col]
        else:
            self.baseline_weights = pd.Series(1.0, index=df.index)

    def define_policy_scenario(self, scenario: PolicyScenario):
        """Define a policy scenario for evaluation"""
        self.policy_scenarios[scenario.name] = scenario
        self.logger.info(f"Defined policy scenario: {scenario.name}")

    def evaluate_cash_transfer_program(
        self,
        transfer_amount: float,
        targeting_criteria: Dict[str, Any],
        scenario_name: str = "cash_transfer",
        income_col: str = "income_pc",
        household_size_col: str = "household_size",
    ) -> PolicyImpact:
        """
        Evaluate impact of targeted cash transfer program

        Args:
            transfer_amount: Monthly transfer amount per eligible household
            targeting_criteria: Dictionary defining eligibility criteria
            scenario_name: Name for this scenario
            income_col: Column name for per capita income
            household_size_col: Column name for household size

        Returns:
            PolicyImpact with evaluation results
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Use set_baseline_data() first.")

        df = self.baseline_data.copy()

        # Apply targeting criteria
        eligible_mask = self._apply_targeting_criteria(df, targeting_criteria)

        # Calculate direct transfer effects
        df["transfer_received"] = 0.0
        df.loc[eligible_mask, "transfer_received"] = transfer_amount

        # Update income measures
        df["new_income_pc"] = df[income_col] + df["transfer_received"]
        df["income_change"] = df["new_income_pc"] - df[income_col]

        # Calculate poverty impacts (assuming poverty line exists)
        poverty_line = df[income_col].median() * 0.6  # 60% median income

        baseline_poverty = (df[income_col] < poverty_line).astype(int)
        new_poverty = (df["new_income_pc"] < poverty_line).astype(int)

        # Direct effects
        direct_effects = {
            "eligible_households": eligible_mask.sum(),
            "coverage_rate": eligible_mask.mean(),
            "average_transfer": (
                df.loc[eligible_mask, "transfer_received"].mean() if eligible_mask.any() else 0
            ),
            "poverty_reduction": (baseline_poverty.sum() - new_poverty.sum()),
            "poverty_rate_change": baseline_poverty.mean() - new_poverty.mean(),
        }

        # Distributional effects
        distributional_effects = self._calculate_distributional_effects(
            df, income_col, "new_income_pc"
        )

        # Welfare effects (using equivalent variation approximation)
        welfare_effects = self._calculate_welfare_effects(
            df, income_col, "new_income_pc", transfer_amount
        )

        # Fiscal effects
        total_cost = df["transfer_received"].sum()
        fiscal_effects = {
            "total_program_cost": total_cost,
            "cost_per_beneficiary": total_cost / eligible_mask.sum() if eligible_mask.any() else 0,
            "cost_per_person_lifted_out_of_poverty": (
                total_cost / direct_effects["poverty_reduction"]
                if direct_effects["poverty_reduction"] > 0
                else np.inf
            ),
        }

        # Bootstrap confidence intervals
        confidence_intervals = self._bootstrap_confidence_intervals(
            df, eligible_mask, income_col, "new_income_pc", poverty_line
        )

        return PolicyImpact(
            scenario_name=scenario_name,
            direct_effects=direct_effects,
            indirect_effects={},  # Simplified for now
            distributional_effects=distributional_effects,
            welfare_effects=welfare_effects,
            fiscal_effects=fiscal_effects,
            confidence_intervals=confidence_intervals,
            sample_size=len(df),
            method="Direct_Simulation",
        )

    def evaluate_tax_reform(
        self,
        tax_brackets: List[Tuple[float, float]],
        scenario_name: str = "tax_reform",
        income_col: str = "income_pc",
        current_tax_col: Optional[str] = None,
    ) -> PolicyImpact:
        """
        Evaluate impact of progressive tax reform

        Args:
            tax_brackets: List of (income_threshold, tax_rate) tuples
            scenario_name: Name for this scenario
            income_col: Column name for income
            current_tax_col: Column name for current tax burden

        Returns:
            PolicyImpact with tax reform evaluation
        """
        df = self.baseline_data.copy()

        # Calculate new tax burden
        df["new_tax"] = self._calculate_progressive_tax(df[income_col], tax_brackets)

        # Calculate current tax if not provided
        if current_tax_col and current_tax_col in df.columns:
            df["current_tax"] = df[current_tax_col]
        else:
            # Estimate current tax (simplified flat rate)
            current_rate = 0.15  # 15% flat tax assumption
            df["current_tax"] = df[income_col] * current_rate

        # Calculate tax change and disposable income
        df["tax_change"] = df["new_tax"] - df["current_tax"]
        df["new_disposable_income"] = df[income_col] - df["new_tax"]
        df["current_disposable_income"] = df[income_col] - df["current_tax"]

        # Direct effects
        direct_effects = {
            "average_tax_change": df["tax_change"].mean(),
            "median_tax_change": df["tax_change"].median(),
            "revenue_change": df["tax_change"].sum(),
            "households_tax_increase": (df["tax_change"] > 0).sum(),
            "households_tax_decrease": (df["tax_change"] < 0).sum(),
        }

        # Distributional effects by income quintiles
        df["income_quintile"] = pd.qcut(df[income_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

        quintile_effects = {}
        for quintile in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = df["income_quintile"] == quintile
            quintile_effects[f"{quintile}_tax_change"] = df.loc[mask, "tax_change"].mean()
            quintile_effects[f"{quintile}_effective_rate"] = (
                df.loc[mask, "new_tax"].sum() / df.loc[mask, income_col].sum()
            )

        distributional_effects = {
            "quintile_effects": quintile_effects,
            "gini_before": self._calculate_gini(df["current_disposable_income"]),
            "gini_after": self._calculate_gini(df["new_disposable_income"]),
            "progressivity_index": self._calculate_progressivity_index(df, income_col, "new_tax"),
        }

        # Welfare effects
        welfare_effects = self._calculate_welfare_effects(
            df, "current_disposable_income", "new_disposable_income", 0
        )

        # Fiscal effects
        fiscal_effects = {
            "total_revenue_change": df["tax_change"].sum(),
            "revenue_elasticity": self._calculate_revenue_elasticity(df, income_col, "tax_change"),
        }

        return PolicyImpact(
            scenario_name=scenario_name,
            direct_effects=direct_effects,
            indirect_effects={},
            distributional_effects=distributional_effects,
            welfare_effects=welfare_effects,
            fiscal_effects=fiscal_effects,
            confidence_intervals={},
            sample_size=len(df),
            method="Tax_Microsimulation",
        )

    def evaluate_education_subsidy(
        self,
        subsidy_amount: float,
        eligibility_criteria: Dict[str, Any],
        education_return_rate: float = 0.08,
        scenario_name: str = "education_subsidy",
    ) -> PolicyImpact:
        """
        Evaluate impact of education subsidy program

        Args:
            subsidy_amount: Annual subsidy per eligible student
            eligibility_criteria: Dictionary defining eligibility
            education_return_rate: Annual return to education (default 8%)
            scenario_name: Name for this scenario

        Returns:
            PolicyImpact with education program evaluation
        """
        df = self.baseline_data.copy()

        # Identify eligible households
        eligible_mask = self._apply_targeting_criteria(df, eligibility_criteria)

        # Simulate education investment response
        # Assumption: subsidy increases education investment probability
        baseline_education_prob = 0.6  # 60% baseline enrollment
        subsidy_effect = min(subsidy_amount / 1000, 0.3)  # Max 30% increase
        new_education_prob = baseline_education_prob + subsidy_effect

        # Calculate long-term income effects
        years_to_impact = 5  # Education impact materializes after 5 years
        discount_rate = 0.05

        # Estimate future income gains
        df["education_investment"] = 0
        df.loc[eligible_mask, "education_investment"] = np.random.binomial(
            1, new_education_prob, size=eligible_mask.sum()
        )

        # Calculate present value of future income gains
        annual_income_gain = df["income_pc"] * education_return_rate
        present_value_gain = annual_income_gain / ((1 + discount_rate) ** years_to_impact)

        df["future_income_gain"] = 0
        df.loc[df["education_investment"] == 1, "future_income_gain"] = present_value_gain.loc[
            df["education_investment"] == 1
        ]

        # Direct effects
        direct_effects = {
            "eligible_students": eligible_mask.sum(),
            "students_receiving_subsidy": df["education_investment"].sum(),
            "participation_rate": df.loc[eligible_mask, "education_investment"].mean(),
            "average_subsidy_amount": subsidy_amount,
            "total_future_income_gains": df["future_income_gain"].sum(),
        }

        # Distributional effects
        distributional_effects = {
            "beneficiaries_by_income_quintile": self._analyze_beneficiaries_by_quintile(
                df, "education_investment", "income_pc"
            ),
            "targeting_effectiveness": self._calculate_targeting_effectiveness(
                df, "education_investment", "income_pc"
            ),
        }

        # Welfare effects
        welfare_effects = {
            "total_welfare_gain": df["future_income_gain"].sum(),
            "welfare_gain_per_beneficiary": (
                df["future_income_gain"].sum() / df["education_investment"].sum()
                if df["education_investment"].sum() > 0
                else 0
            ),
        }

        # Fiscal effects
        total_subsidy_cost = df["education_investment"].sum() * subsidy_amount
        fiscal_effects = {
            "total_program_cost": total_subsidy_cost,
            "cost_benefit_ratio": (
                df["future_income_gain"].sum() / total_subsidy_cost if total_subsidy_cost > 0 else 0
            ),
            "fiscal_return_period": years_to_impact,
        }

        return PolicyImpact(
            scenario_name=scenario_name,
            direct_effects=direct_effects,
            indirect_effects={},
            distributional_effects=distributional_effects,
            welfare_effects=welfare_effects,
            fiscal_effects=fiscal_effects,
            confidence_intervals={},
            sample_size=len(df),
            method="Education_Microsimulation",
        )

    def _apply_targeting_criteria(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.Series:
        """Apply targeting criteria to identify eligible households"""
        mask = pd.Series(True, index=df.index)

        for column, condition in criteria.items():
            if column not in df.columns:
                continue

            if isinstance(condition, dict):
                if "min" in condition:
                    mask &= df[column] >= condition["min"]
                if "max" in condition:
                    mask &= df[column] <= condition["max"]
                if "values" in condition:
                    mask &= df[column].isin(condition["values"])
            else:
                mask &= df[column] == condition

        return mask

    def _calculate_progressive_tax(
        self, income: pd.Series, brackets: List[Tuple[float, float]]
    ) -> pd.Series:
        """Calculate progressive tax based on bracket structure"""
        tax = pd.Series(0.0, index=income.index)

        for i, (threshold, rate) in enumerate(brackets):
            if i == 0:
                # First bracket: 0 to threshold
                taxable = np.minimum(income, threshold)
                tax += taxable * rate
            else:
                # Subsequent brackets
                prev_threshold = brackets[i - 1][0]
                taxable = np.maximum(
                    0, np.minimum(income - prev_threshold, threshold - prev_threshold)
                )
                tax += taxable * rate

        return tax

    def _calculate_distributional_effects(
        self, df: pd.DataFrame, baseline_col: str, new_col: str
    ) -> Dict[str, Any]:
        """Calculate distributional effects of policy change"""
        # Income quintiles
        df["quintile"] = pd.qcut(df[baseline_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

        quintile_effects = {}
        for quintile in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = df["quintile"] == quintile
            baseline_mean = df.loc[mask, baseline_col].mean()
            new_mean = df.loc[mask, new_col].mean()
            quintile_effects[quintile] = {
                "baseline_mean": baseline_mean,
                "new_mean": new_mean,
                "absolute_change": new_mean - baseline_mean,
                "percent_change": (new_mean - baseline_mean) / baseline_mean * 100,
            }

        return {
            "quintile_effects": quintile_effects,
            "gini_before": self._calculate_gini(df[baseline_col]),
            "gini_after": self._calculate_gini(df[new_col]),
        }

    def _calculate_welfare_effects(
        self, df: pd.DataFrame, baseline_col: str, new_col: str, transfer_amount: float
    ) -> Dict[str, float]:
        """Calculate welfare effects using equivalent variation"""
        # Simplified welfare calculation
        income_change = df[new_col] - df[baseline_col]

        # Social welfare using Atkinson index (inequality aversion = 1)
        baseline_welfare = self._calculate_social_welfare(df[baseline_col])
        new_welfare = self._calculate_social_welfare(df[new_col])

        return {
            "total_welfare_change": new_welfare - baseline_welfare,
            "average_welfare_gain": income_change.mean(),
            "welfare_gain_per_dollar": (
                (new_welfare - baseline_welfare) / transfer_amount if transfer_amount > 0 else 0
            ),
        }

    def _calculate_social_welfare(self, income: pd.Series, epsilon: float = 1.0) -> float:
        """Calculate social welfare using Atkinson social welfare function"""
        if epsilon == 1.0:
            # Logarithmic case
            return np.log(income).mean()
        else:
            return ((income ** (1 - epsilon)).mean()) ** (1 / (1 - epsilon))

    def _calculate_gini(self, income: pd.Series) -> float:
        """Calculate Gini coefficient"""
        income_sorted = np.sort(income.dropna())
        n = len(income_sorted)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * income_sorted)) / (n * np.sum(income_sorted))

    def _calculate_progressivity_index(
        self, df: pd.DataFrame, income_col: str, tax_col: str
    ) -> float:
        """Calculate tax progressivity index (Kakwani index)"""
        # Kakwani index = Gini(gross income) - Concentration index(taxes)
        gini_gross = self._calculate_gini(df[income_col])

        # Calculate concentration index for taxes
        df_sorted = df.sort_values(income_col)
        n = len(df_sorted)
        cumulative_income = df_sorted[income_col].cumsum()
        cumulative_tax = df_sorted[tax_col].cumsum()

        # Concentration curve for taxes
        concentration_index = 1 - 2 * np.trapz(
            cumulative_tax / cumulative_tax.iloc[-1], cumulative_income / cumulative_income.iloc[-1]
        )

        return gini_gross - concentration_index

    def _calculate_revenue_elasticity(
        self, df: pd.DataFrame, income_col: str, tax_change_col: str
    ) -> float:
        """Calculate tax revenue elasticity"""
        # Simplified elasticity calculation
        income_change_pct = (df[income_col] / df[income_col].mean() - 1).mean()
        tax_change_pct = (df[tax_change_col] / df[tax_change_col].mean()).mean()

        if abs(income_change_pct) > 0.001:
            return tax_change_pct / income_change_pct
        return 0.0

    def _bootstrap_confidence_intervals(
        self,
        df: pd.DataFrame,
        eligible_mask: pd.Series,
        baseline_col: str,
        new_col: str,
        poverty_line: float,
        n_bootstrap: int = 100,
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for key metrics"""
        poverty_reduction_samples = []
        welfare_gain_samples = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_idx = np.random.choice(df.index, size=len(df), replace=True)
            sample_df = df.loc[sample_idx]

            # Calculate metrics for sample
            baseline_poverty = (sample_df[baseline_col] < poverty_line).sum()
            new_poverty = (sample_df[new_col] < poverty_line).sum()
            poverty_reduction = baseline_poverty - new_poverty

            welfare_gain = (sample_df[new_col] - sample_df[baseline_col]).mean()

            poverty_reduction_samples.append(poverty_reduction)
            welfare_gain_samples.append(welfare_gain)

        # Calculate confidence intervals
        poverty_ci = (
            np.percentile(poverty_reduction_samples, 2.5),
            np.percentile(poverty_reduction_samples, 97.5),
        )
        welfare_ci = (
            np.percentile(welfare_gain_samples, 2.5),
            np.percentile(welfare_gain_samples, 97.5),
        )

        return {"poverty_reduction_ci": poverty_ci, "welfare_gain_ci": welfare_ci}

    def _analyze_beneficiaries_by_quintile(
        self, df: pd.DataFrame, treatment_col: str, income_col: str
    ) -> Dict[str, float]:
        """Analyze distribution of beneficiaries by income quintile"""
        df["quintile"] = pd.qcut(df[income_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

        quintile_analysis = {}
        for quintile in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = df["quintile"] == quintile
            coverage_rate = df.loc[mask, treatment_col].mean()
            share_of_beneficiaries = (
                df.loc[mask, treatment_col].sum() / df[treatment_col].sum()
                if df[treatment_col].sum() > 0
                else 0
            )
            quintile_analysis[quintile] = {
                "coverage_rate": coverage_rate,
                "share_of_beneficiaries": share_of_beneficiaries,
            }

        return quintile_analysis

    def _calculate_targeting_effectiveness(
        self, df: pd.DataFrame, treatment_col: str, income_col: str
    ) -> float:
        """Calculate targeting effectiveness (share going to bottom 40%)"""
        df["bottom_40"] = df[income_col] <= df[income_col].quantile(0.4)

        total_transfers = df[treatment_col].sum()
        transfers_to_bottom_40 = df[df["bottom_40"]][treatment_col].sum()

        return transfers_to_bottom_40 / total_transfers if total_transfers > 0 else 0


class Microsimulation:
    """
    Advanced microsimulation model for policy analysis
    Supports behavioral responses and dynamic effects
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.behavioral_models = {}
        self.simulation_results = {}

    def add_behavioral_model(self, outcome: str, model: Any, features: List[str]):
        """Add behavioral response model"""
        self.behavioral_models[outcome] = {"model": model, "features": features}

    def train_labor_supply_model(
        self, df: pd.DataFrame, work_col: str, income_col: str, demographic_cols: List[str]
    ):
        """Train labor supply response model"""
        # Prepare features
        features = [income_col] + demographic_cols
        X = df[features].fillna(df[features].mean())
        y = df[work_col]

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Store model
        self.behavioral_models["labor_supply"] = {"model": model, "features": features}

        # Model performance
        cv_scores = cross_val_score(model, X, y, cv=5)
        self.logger.info(f"Labor supply model R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    def simulate_policy_with_behavior(
        self, df: pd.DataFrame, policy_scenario: PolicyScenario, periods: int = 12
    ) -> pd.DataFrame:
        """
        Simulate policy with behavioral responses over multiple periods

        Args:
            df: Baseline data
            policy_scenario: Policy scenario to simulate
            periods: Number of periods to simulate

        Returns:
            DataFrame with simulation results
        """
        simulation_df = df.copy()
        results = []

        for period in range(periods):
            period_result = simulation_df.copy()
            period_result["period"] = period

            # Apply policy intervention
            if policy_scenario.name == "cash_transfer":
                period_result = self._apply_cash_transfer_with_behavior(
                    period_result, policy_scenario
                )
            elif policy_scenario.name == "tax_reform":
                period_result = self._apply_tax_reform_with_behavior(period_result, policy_scenario)

            # Update for next period (behavioral adaptation)
            simulation_df = self._update_behavioral_responses(simulation_df, period_result)

            results.append(period_result)

        return pd.concat(results, ignore_index=True)

    def _apply_cash_transfer_with_behavior(
        self, df: pd.DataFrame, scenario: PolicyScenario
    ) -> pd.DataFrame:
        """Apply cash transfer with behavioral responses"""
        # Get policy parameters
        transfer_amount = scenario.parameters.get("transfer_amount", 0)
        targeting = scenario.parameters.get("targeting_criteria", {})

        # Apply targeting
        eligible_mask = self._apply_targeting_criteria(df, targeting)
        df["transfer"] = 0
        df.loc[eligible_mask, "transfer"] = transfer_amount

        # Behavioral responses
        if "labor_supply" in self.behavioral_models:
            df = self._predict_labor_response(df, "transfer")

        # Update income
        df["new_income"] = df["income_pc"] + df["transfer"]

        return df

    def _apply_tax_reform_with_behavior(
        self, df: pd.DataFrame, scenario: PolicyScenario
    ) -> pd.DataFrame:
        """Apply tax reform with behavioral responses"""
        tax_brackets = scenario.parameters.get("tax_brackets", [])

        # Calculate new taxes
        df["new_tax"] = self._calculate_progressive_tax(df["income_pc"], tax_brackets)
        df["tax_change"] = df["new_tax"] - df.get("current_tax", df["income_pc"] * 0.15)

        # Behavioral responses to tax changes
        if "labor_supply" in self.behavioral_models:
            df = self._predict_labor_response(df, "tax_change")

        # Update disposable income
        df["new_disposable_income"] = df["income_pc"] - df["new_tax"]

        return df

    def _predict_labor_response(self, df: pd.DataFrame, policy_var: str) -> pd.DataFrame:
        """Predict labor supply response to policy change"""
        model_info = self.behavioral_models["labor_supply"]
        model = model_info["model"]
        features = model_info["features"]

        # Create modified feature matrix (policy affects income)
        X = df[features].copy()
        if "income_pc" in features:
            X["income_pc"] = X["income_pc"] + df[policy_var]

        # Predict new labor supply
        X_filled = X.fillna(X.mean())
        df["predicted_labor_change"] = model.predict(X_filled) - df.get("work_hours", 0)

        return df

    def _update_behavioral_responses(
        self, current_df: pd.DataFrame, period_result: pd.DataFrame
    ) -> pd.DataFrame:
        """Update behavioral variables for next period"""
        updated_df = current_df.copy()

        # Update labor supply if there were responses
        if "predicted_labor_change" in period_result.columns:
            updated_df["work_hours"] = (
                updated_df.get("work_hours", 40)
                + period_result["predicted_labor_change"] * 0.1  # Gradual adjustment
            )

        # Update income based on labor supply changes
        if "work_hours" in updated_df.columns:
            wage_rate = updated_df.get("wage_rate", updated_df["income_pc"] / 40)
            updated_df["income_pc"] = updated_df["work_hours"] * wage_rate

        return updated_df

    def _apply_targeting_criteria(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.Series:
        """Apply targeting criteria (same as PolicyImpactEvaluator)"""
        mask = pd.Series(True, index=df.index)

        for column, condition in criteria.items():
            if column not in df.columns:
                continue

            if isinstance(condition, dict):
                if "min" in condition:
                    mask &= df[column] >= condition["min"]
                if "max" in condition:
                    mask &= df[column] <= condition["max"]
                if "values" in condition:
                    mask &= df[column].isin(condition["values"])
            else:
                mask &= df[column] == condition

        return mask

    def _calculate_progressive_tax(
        self, income: pd.Series, brackets: List[Tuple[float, float]]
    ) -> pd.Series:
        """Calculate progressive tax (same as PolicyImpactEvaluator)"""
        tax = pd.Series(0.0, index=income.index)

        for i, (threshold, rate) in enumerate(brackets):
            if i == 0:
                taxable = np.minimum(income, threshold)
                tax += taxable * rate
            else:
                prev_threshold = brackets[i - 1][0]
                taxable = np.maximum(
                    0, np.minimum(income - prev_threshold, threshold - prev_threshold)
                )
                tax += taxable * rate

        return tax


class PolicyScenarios:
    """
    Policy scenario management and comparison
    Supports multiple policy alternatives and cost-benefit analysis
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.scenarios = {}
        self.comparison_results = {}

    def add_scenario(self, scenario: PolicyScenario):
        """Add policy scenario"""
        self.scenarios[scenario.name] = scenario

    def compare_scenarios(self, evaluation_results: Dict[str, PolicyImpact]) -> Dict[str, Any]:
        """
        Compare multiple policy scenarios

        Args:
            evaluation_results: Dictionary of PolicyImpact results by scenario name

        Returns:
            Comprehensive comparison analysis
        """
        comparison = {
            "scenario_summary": {},
            "cost_effectiveness": {},
            "distributional_comparison": {},
            "ranking": {},
        }

        # Summarize each scenario
        for scenario_name, impact in evaluation_results.items():
            comparison["scenario_summary"][scenario_name] = {
                "poverty_reduction": impact.direct_effects.get("poverty_reduction", 0),
                "total_cost": impact.fiscal_effects.get("total_program_cost", 0),
                "beneficiaries": impact.direct_effects.get("eligible_households", 0),
                "welfare_gain": impact.welfare_effects.get("total_welfare_change", 0),
            }

        # Cost-effectiveness analysis
        for scenario_name, impact in evaluation_results.items():
            cost = impact.fiscal_effects.get("total_program_cost", 1)
            poverty_reduction = impact.direct_effects.get("poverty_reduction", 0)
            welfare_gain = impact.welfare_effects.get("total_welfare_change", 0)

            comparison["cost_effectiveness"][scenario_name] = {
                "cost_per_person_out_of_poverty": cost / max(poverty_reduction, 1),
                "welfare_gain_per_dollar": welfare_gain / max(cost, 1),
                "benefit_cost_ratio": welfare_gain / max(cost, 1),
            }

        # Distributional comparison
        for scenario_name, impact in evaluation_results.items():
            dist_effects = impact.distributional_effects
            comparison["distributional_comparison"][scenario_name] = {
                "gini_change": (
                    dist_effects.get("gini_after", 0) - dist_effects.get("gini_before", 0)
                ),
                "targeting_effectiveness": (
                    dist_effects.get("targeting_effectiveness", 0)
                    if "targeting_effectiveness" in dist_effects
                    else 0
                ),
            }

        # Scenario ranking
        scenarios = list(evaluation_results.keys())

        # Rank by different criteria
        cost_effectiveness_ranking = sorted(
            scenarios,
            key=lambda x: comparison["cost_effectiveness"][x]["welfare_gain_per_dollar"],
            reverse=True,
        )

        poverty_impact_ranking = sorted(
            scenarios,
            key=lambda x: comparison["scenario_summary"][x]["poverty_reduction"],
            reverse=True,
        )

        comparison["ranking"] = {
            "by_cost_effectiveness": cost_effectiveness_ranking,
            "by_poverty_impact": poverty_impact_ranking,
        }

        self.comparison_results = comparison
        return comparison

    def generate_policy_recommendations(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate policy recommendations based on comparison"""
        recommendations = []

        # Top scenario by cost-effectiveness
        top_cost_effective = comparison_results["ranking"]["by_cost_effectiveness"][0]
        recommendations.append(
            f"For maximum cost-effectiveness, implement '{top_cost_effective}' "
            f"(welfare gain per dollar: "
            f"{comparison_results['cost_effectiveness'][top_cost_effective]['welfare_gain_per_dollar']:.2f})"
        )

        # Top scenario by poverty reduction
        top_poverty_impact = comparison_results["ranking"]["by_poverty_impact"][0]
        if top_poverty_impact != top_cost_effective:
            recommendations.append(
                f"For maximum poverty reduction, consider '{top_poverty_impact}' "
                f"(reduces poverty for "
                f"{comparison_results['scenario_summary'][top_poverty_impact]['poverty_reduction']} households)"
            )

        # Budget constraints
        total_costs = {
            name: summary["total_cost"]
            for name, summary in comparison_results["scenario_summary"].items()
        }
        lowest_cost_scenario = min(total_costs.keys(), key=lambda x: total_costs[x])

        recommendations.append(
            f"For budget-constrained implementation, start with '{lowest_cost_scenario}' "
            f"(total cost: {total_costs[lowest_cost_scenario]:,.0f})"
        )

        return recommendations


def create_policy_evaluator(logger: Optional[logging.Logger] = None) -> PolicyImpactEvaluator:
    """Factory function to create PolicyImpactEvaluator instance"""
    return PolicyImpactEvaluator(logger)


# Export classes and functions
__all__ = [
    "PolicyImpactEvaluator",
    "Microsimulation",
    "PolicyScenarios",
    "PolicyScenario",
    "PolicyImpact",
    "create_policy_evaluator",
]
