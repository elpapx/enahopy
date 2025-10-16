"""
ENAHO Automated Report Generation System
=======================================

Advanced reporting system with visualizations for ENAHO data analysis.
Generates comprehensive reports for data quality, statistical analysis, and survey insights.
"""

import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader, Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Import our custom modules
try:
    from .data_quality import DataQualityReport, assess_data_quality
    from .null_analysis.strategies.ml_imputation import create_ml_imputation_manager
    from .statistical_analysis import create_statistical_analyzer, quick_poverty_analysis

    DATA_QUALITY_AVAILABLE = True
except ImportError:
    DATA_QUALITY_AVAILABLE = False

# Configure plotting style
if MATPLOTLIB_AVAILABLE:
    plt.style.use("seaborn-v0_8" if hasattr(plt.style, "seaborn-v0_8") else "default")
    sns.set_palette("husl")

if PLOTLY_AVAILABLE:
    pio.templates.default = "plotly_white"


@dataclass
class ReportSection:
    """Represents a section of a report"""

    title: str
    content: str
    visualizations: List[str]
    data: Dict[str, Any]
    priority: int = 1


@dataclass
class ENAHOReport:
    """Complete ENAHO analysis report"""

    title: str
    summary: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    created_at: datetime
    version: str = "1.0"


class VisualizationEngine:
    """Creates visualizations for ENAHO data analysis"""

    def __init__(self, style: str = "matplotlib", logger: Optional[logging.Logger] = None):
        self.style = style  # "matplotlib" or "plotly"
        self.logger = logger or logging.getLogger(__name__)
        self.figures = {}

        # Set up style
        if style == "matplotlib" and MATPLOTLIB_AVAILABLE:
            self.backend = "matplotlib"
        elif style == "plotly" and PLOTLY_AVAILABLE:
            self.backend = "plotly"
        else:
            self.backend = None
            self.logger.warning(f"Visualization backend '{style}' not available")

    def create_data_quality_dashboard(
        self, quality_report: "DataQualityReport", save_path: Optional[str] = None
    ) -> Optional[str]:
        """Create comprehensive data quality dashboard"""
        if not self.backend:
            return None

        if self.backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_quality_dashboard(quality_report, save_path)
        elif self.backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_quality_dashboard(quality_report, save_path)

        return None

    def _create_plotly_quality_dashboard(
        self, quality_report: "DataQualityReport", save_path: Optional[str] = None
    ) -> str:
        """Create interactive Plotly dashboard for data quality"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Quality Dimensions",
                "Completeness by Column",
                "Issue Distribution",
                "Quality Timeline",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "pie"}, {"type": "scatter"}]],
        )

        # Quality dimensions bar chart
        dimensions = list(quality_report.dimensions.keys())
        scores = [quality_report.dimensions[dim].score for dim in dimensions]

        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=scores,
                name="Quality Scores",
                marker_color=["red" if s < 70 else "yellow" if s < 85 else "green" for s in scores],
            ),
            row=1,
            col=1,
        )

        # Completeness by column (if available)
        if "completeness" in quality_report.dimensions:
            completeness_data = quality_report.dimensions["completeness"].details.get(
                "column_completeness", {}
            )
            if completeness_data:
                cols = list(completeness_data.keys())[:15]  # Top 15 columns
                completeness = [completeness_data[col] * 100 for col in cols]

                fig.add_trace(
                    go.Bar(x=cols, y=completeness, name="Completeness %", marker_color="lightblue"),
                    row=1,
                    col=2,
                )

        # Issue distribution pie chart
        issue_counts = {}
        for dim_name, dim in quality_report.dimensions.items():
            if dim.issues:
                issue_counts[dim_name] = len(dim.issues)

        if issue_counts:
            fig.add_trace(
                go.Pie(
                    labels=list(issue_counts.keys()),
                    values=list(issue_counts.values()),
                    name="Issues by Dimension",
                ),
                row=2,
                col=1,
            )

        # Quality timeline (mock data for demonstration)
        dates = pd.date_range(start="2024-01-01", end="2024-12-01", freq="M")
        quality_scores = [quality_report.overall_score + np.random.normal(0, 5) for _ in dates]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=quality_scores,
                mode="lines+markers",
                name="Quality Trend",
                line=dict(color="blue"),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"Data Quality Dashboard - Overall Score: {quality_report.overall_score:.1f}/100 ({quality_report.grade})",
            height=800,
            showlegend=True,
        )

        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Quality dashboard saved to {save_path}")
            return save_path

        return fig.to_html()

    def _create_matplotlib_quality_dashboard(
        self, quality_report: "DataQualityReport", save_path: Optional[str] = None
    ) -> str:
        """Create matplotlib dashboard for data quality"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Data Quality Dashboard - Score: {quality_report.overall_score:.1f}/100 ({quality_report.grade})",
            fontsize=16,
        )

        # Quality dimensions
        dimensions = list(quality_report.dimensions.keys())
        scores = [quality_report.dimensions[dim].score for dim in dimensions]
        colors = ["red" if s < 70 else "orange" if s < 85 else "green" for s in scores]

        axes[0, 0].bar(dimensions, scores, color=colors)
        axes[0, 0].set_title("Quality Dimensions")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Completeness by column
        if "completeness" in quality_report.dimensions:
            completeness_data = quality_report.dimensions["completeness"].details.get(
                "column_completeness", {}
            )
            if completeness_data:
                cols = list(completeness_data.keys())[:10]  # Top 10 columns
                completeness = [completeness_data[col] * 100 for col in cols]

                axes[0, 1].barh(cols, completeness, color="lightblue")
                axes[0, 1].set_title("Column Completeness (%)")
                axes[0, 1].set_xlabel("Completeness %")

        # Issue distribution
        issue_counts = {}
        for dim_name, dim in quality_report.dimensions.items():
            if dim.issues:
                issue_counts[dim_name] = len(dim.issues)

        if issue_counts:
            axes[1, 0].pie(
                list(issue_counts.values()),
                labels=list(issue_counts.keys()),
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[1, 0].set_title("Issues by Dimension")

        # Sample info
        sample_info = quality_report.sample_info
        info_text = f"""Dataset Information:
        Records: {sample_info['total_records']:,}
        Columns: {sample_info['total_columns']}
        Memory: {sample_info['memory_usage_mb']:.1f} MB
        Numeric Cols: {sample_info['numeric_columns']}
        Categorical Cols: {sample_info['categorical_columns']}"""

        axes[1, 1].text(
            0.1,
            0.5,
            info_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="center",
        )
        axes[1, 1].set_title("Dataset Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Quality dashboard saved to {save_path}")
            return save_path

        return "matplotlib_figure"

    def create_poverty_analysis_charts(
        self,
        df: pd.DataFrame,
        income_col: str,
        poverty_line: float,
        group_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Create poverty analysis visualizations"""
        if not self.backend:
            return None

        # Get poverty analysis data
        from .statistical_analysis import InequalityMeasures, PovertyIndicators

        poverty_calc = PovertyIndicators()
        inequality_calc = InequalityMeasures()

        if self.backend == "plotly" and PLOTLY_AVAILABLE:
            return self._create_plotly_poverty_charts(
                df, income_col, poverty_line, group_col, save_path, poverty_calc, inequality_calc
            )
        elif self.backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_poverty_charts(
                df, income_col, poverty_line, group_col, save_path, poverty_calc, inequality_calc
            )

        return None

    def _create_plotly_poverty_charts(
        self,
        df: pd.DataFrame,
        income_col: str,
        poverty_line: float,
        group_col: Optional[str],
        save_path: Optional[str],
        poverty_calc,
        inequality_calc,
    ) -> str:
        """Create Plotly poverty analysis charts"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Income Distribution",
                "Poverty by Groups",
                "Inequality Measures",
                "Lorenz Curve",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Income distribution
        fig.add_trace(
            go.Histogram(x=df[income_col], nbinsx=50, name="Income Distribution", opacity=0.7),
            row=1,
            col=1,
        )

        # Add poverty line
        fig.add_vline(
            x=poverty_line,
            line_dash="dash",
            line_color="red",
            annotation_text="Poverty Line",
            row=1,
            col=1,
        )

        # Poverty by groups (if group column provided)
        if group_col and group_col in df.columns:
            profile = poverty_calc.poverty_profile(df, income_col, poverty_line, [group_col])

            fig.add_trace(
                go.Bar(
                    x=profile["group"],
                    y=profile["headcount_ratio"] * 100,
                    name="Poverty Rate %",
                    marker_color="red",
                ),
                row=1,
                col=2,
            )

        # Inequality measures
        income_data = df[income_col].dropna()
        measures = {
            "Gini Coefficient": inequality_calc.gini_coefficient(income_data),
            "Theil Index": inequality_calc.theil_index(income_data),
            "Palma Ratio": inequality_calc.palma_ratio(income_data),
        }

        fig.add_trace(
            go.Bar(
                x=list(measures.keys()),
                y=list(measures.values()),
                name="Inequality Measures",
                marker_color="orange",
            ),
            row=2,
            col=1,
        )

        # Lorenz curve
        sorted_income = income_data.sort_values()
        cumsum_income = sorted_income.cumsum()
        total_income = sorted_income.sum()

        lorenz_x = np.arange(1, len(sorted_income) + 1) / len(sorted_income)
        lorenz_y = cumsum_income / total_income

        fig.add_trace(
            go.Scatter(
                x=lorenz_x, y=lorenz_y, mode="lines", name="Lorenz Curve", line=dict(color="blue")
            ),
            row=2,
            col=2,
        )

        # Add perfect equality line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Equality",
                line=dict(dash="dash", color="gray"),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(title="Poverty and Inequality Analysis", height=800, showlegend=True)

        if save_path:
            fig.write_html(save_path)
            return save_path

        return fig.to_html()

    def _create_matplotlib_poverty_charts(
        self,
        df: pd.DataFrame,
        income_col: str,
        poverty_line: float,
        group_col: Optional[str],
        save_path: Optional[str],
        poverty_calc,
        inequality_calc,
    ) -> str:
        """Create matplotlib poverty analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Poverty and Inequality Analysis", fontsize=16)

        # Income distribution
        axes[0, 0].hist(df[income_col].dropna(), bins=50, alpha=0.7, color="skyblue")
        axes[0, 0].axvline(poverty_line, color="red", linestyle="--", label="Poverty Line")
        axes[0, 0].set_title("Income Distribution")
        axes[0, 0].set_xlabel("Income")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        # Poverty by groups
        if group_col and group_col in df.columns:
            profile = poverty_calc.poverty_profile(df, income_col, poverty_line, [group_col])

            axes[0, 1].bar(profile["group"], profile["headcount_ratio"] * 100, color="red")
            axes[0, 1].set_title("Poverty Rate by Group")
            axes[0, 1].set_ylabel("Poverty Rate (%)")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Inequality measures
        income_data = df[income_col].dropna()
        measures = {
            "Gini": inequality_calc.gini_coefficient(income_data),
            "Theil": inequality_calc.theil_index(income_data),
            "Palma": inequality_calc.palma_ratio(income_data),
        }

        axes[1, 0].bar(list(measures.keys()), list(measures.values()), color="orange")
        axes[1, 0].set_title("Inequality Measures")
        axes[1, 0].set_ylabel("Index Value")

        # Lorenz curve
        sorted_income = income_data.sort_values()
        cumsum_income = sorted_income.cumsum()
        total_income = sorted_income.sum()

        lorenz_x = np.arange(1, len(sorted_income) + 1) / len(sorted_income)
        lorenz_y = cumsum_income / total_income

        axes[1, 1].plot(lorenz_x, lorenz_y, "b-", label="Lorenz Curve")
        axes[1, 1].plot([0, 1], [0, 1], "k--", label="Perfect Equality")
        axes[1, 1].set_title("Lorenz Curve")
        axes[1, 1].set_xlabel("Cumulative Population Share")
        axes[1, 1].set_ylabel("Cumulative Income Share")
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path

        return "matplotlib_figure"


class ReportGenerator:
    """Main report generator for ENAHO analysis"""

    def __init__(
        self,
        output_dir: str = "./reports",
        visualization_style: str = "plotly",
        logger: Optional[logging.Logger] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

        self.viz_engine = VisualizationEngine(visualization_style, logger)

        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        income_col: Optional[str] = None,
        poverty_line: Optional[float] = None,
        group_col: Optional[str] = None,
        custom_analysis: Optional[Dict[str, Any]] = None,
    ) -> ENAHOReport:
        """
        Generate comprehensive ENAHO analysis report

        Args:
            df: Input DataFrame
            income_col: Name of income column for poverty analysis
            poverty_line: Poverty line threshold
            group_col: Column for group analysis
            custom_analysis: Custom analysis parameters

        Returns:
            Complete ENAHO report object
        """
        self.logger.info("Starting comprehensive ENAHO report generation")

        report_sections = []

        # 1. Data Quality Section
        if DATA_QUALITY_AVAILABLE:
            quality_section = self._generate_data_quality_section(df)
            report_sections.append(quality_section)

        # 2. Descriptive Statistics Section
        stats_section = self._generate_descriptive_stats_section(df)
        report_sections.append(stats_section)

        # 3. Poverty Analysis Section (if income data provided)
        if income_col and poverty_line and income_col in df.columns:
            poverty_section = self._generate_poverty_analysis_section(
                df, income_col, poverty_line, group_col
            )
            report_sections.append(poverty_section)

        # 4. Missing Data Analysis Section
        missing_section = self._generate_missing_data_section(df)
        report_sections.append(missing_section)

        # 5. Custom Analysis Section
        if custom_analysis:
            custom_section = self._generate_custom_analysis_section(df, custom_analysis)
            report_sections.append(custom_section)

        # Generate summary
        summary = self._generate_executive_summary(df, report_sections)

        # Create report metadata
        metadata = {
            "dataset_shape": df.shape,
            "generation_time": datetime.now(),
            "columns": df.columns.tolist(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "has_income_analysis": income_col is not None,
            "visualization_style": self.viz_engine.backend,
        }

        report = ENAHOReport(
            title="ENAHO Data Analysis Report",
            summary=summary,
            sections=report_sections,
            metadata=metadata,
            created_at=datetime.now(),
        )

        self.logger.info("Comprehensive report generation completed")
        return report

    def _generate_data_quality_section(self, df: pd.DataFrame) -> ReportSection:
        """Generate data quality analysis section"""
        quality_report = assess_data_quality(df)

        # Create visualization
        viz_path = self.output_dir / "visualizations" / "data_quality_dashboard.html"
        self.viz_engine.create_data_quality_dashboard(quality_report, str(viz_path))

        content = f"""
        ## Data Quality Assessment
        
        **Overall Quality Score: {quality_report.overall_score:.1f}/100 (Grade: {quality_report.grade})**
        
        ### Quality Dimensions:
        """

        for name, dimension in quality_report.dimensions.items():
            content += f"\n- **{dimension.name}**: {dimension.score:.1f}/100"
            if dimension.issues:
                content += f" ({len(dimension.issues)} issues detected)"

        if quality_report.critical_issues:
            content += "\n\n### Critical Issues:\n"
            for issue in quality_report.critical_issues:
                content += f"- ⚠️ {issue}\n"

        if quality_report.recommendations:
            content += "\n\n### Recommendations:\n"
            for i, rec in enumerate(quality_report.recommendations[:5], 1):
                content += f"{i}. {rec}\n"

        return ReportSection(
            title="Data Quality Assessment",
            content=content,
            visualizations=["data_quality_dashboard.html"] if viz_path.exists() else [],
            data=asdict(quality_report),
            priority=1,
        )

    def _generate_descriptive_stats_section(self, df: pd.DataFrame) -> ReportSection:
        """Generate descriptive statistics section"""
        content = f"""
        ## Descriptive Statistics
        
        ### Dataset Overview:
        - **Total Records**: {len(df):,}
        - **Total Columns**: {len(df.columns)}
        - **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
        - **Numeric Columns**: {len(df.select_dtypes(include=[np.number]).columns)}
        - **Categorical Columns**: {len(df.select_dtypes(include=[object, 'category']).columns)}
        
        ### Summary Statistics for Numeric Variables:
        """

        # Numeric summary
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary_stats = numeric_df.describe()
            content += "\n" + summary_stats.to_string() + "\n"

        # Categorical summary
        categorical_df = df.select_dtypes(include=[object, "category"])
        if not categorical_df.empty and len(categorical_df.columns) > 0:
            content += "\n### Categorical Variables Summary:\n"
            for col in categorical_df.columns[:5]:  # Top 5 categorical columns
                value_counts = df[col].value_counts().head()
                content += f"\n**{col}**:\n"
                content += value_counts.to_string() + "\n"

        # Missing data summary
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]

        if not missing_summary.empty:
            content += "\n### Missing Data Summary:\n"
            for col, missing_count in missing_summary.items():
                missing_pct = (missing_count / len(df)) * 100
                content += f"- **{col}**: {missing_count} ({missing_pct:.1f}%)\n"

        return ReportSection(
            title="Descriptive Statistics",
            content=content,
            visualizations=[],
            data={
                "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else {},
                "categorical_summary": {
                    col: df[col].value_counts().to_dict() for col in categorical_df.columns[:5]
                },
                "missing_summary": missing_summary.to_dict(),
            },
            priority=2,
        )

    def _generate_poverty_analysis_section(
        self, df: pd.DataFrame, income_col: str, poverty_line: float, group_col: Optional[str]
    ) -> ReportSection:
        """Generate poverty analysis section"""
        analysis = quick_poverty_analysis(df, income_col, poverty_line)

        # Create visualizations
        viz_path = self.output_dir / "visualizations" / "poverty_analysis.html"
        self.viz_engine.create_poverty_analysis_charts(
            df, income_col, poverty_line, group_col, str(viz_path)
        )

        content = f"""
        ## Poverty and Inequality Analysis
        
        ### Key Indicators:
        - **Poverty Headcount Ratio**: {analysis['poverty_headcount']:.1%}
        - **Poverty Gap Ratio**: {analysis['poverty_gap']:.1%}
        - **Poverty Severity Ratio**: {analysis['poverty_severity']:.1%}
        
        ### Inequality Measures:
        - **Gini Coefficient**: {analysis['gini_coefficient']:.3f}
        - **Theil Index**: {analysis['theil_index']:.3f}
        - **Palma Ratio**: {analysis['palma_ratio']:.2f}
        
        ### Income Distribution:
        - **Mean Income**: {analysis['mean_income']:,.2f}
        - **Median Income**: {analysis['median_income']:,.2f}
        - **Sample Size**: {analysis['sample_size']:,}
        
        ### Analysis Notes:
        - Poverty line used: {poverty_line:,.2f}
        - {analysis['poverty_headcount']:.1%} of the population lives below the poverty line
        """

        # Add interpretation
        if analysis["gini_coefficient"] < 0.3:
            inequality_level = "low"
        elif analysis["gini_coefficient"] < 0.4:
            inequality_level = "moderate"
        elif analysis["gini_coefficient"] < 0.5:
            inequality_level = "high"
        else:
            inequality_level = "very high"

        content += f"\n- Income inequality is **{inequality_level}** (Gini = {analysis['gini_coefficient']:.3f})"

        return ReportSection(
            title="Poverty and Inequality Analysis",
            content=content,
            visualizations=["poverty_analysis.html"] if viz_path.exists() else [],
            data=analysis,
            priority=3,
        )

    def _generate_missing_data_section(self, df: pd.DataFrame) -> ReportSection:
        """Generate missing data analysis section"""
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        total_cells = df.size
        missing_pct = (total_missing / total_cells) * 100

        content = f"""
        ## Missing Data Analysis
        
        ### Overall Missing Data:
        - **Total Missing Values**: {total_missing:,}
        - **Percentage Missing**: {missing_pct:.2f}%
        - **Complete Records**: {(~df.isnull().any(axis=1)).sum():,}
        - **Records with Missing Data**: {df.isnull().any(axis=1).sum():,}
        
        ### Missing Data by Column:
        """

        # Columns with missing data
        missing_cols = missing_summary[missing_summary > 0]
        if not missing_cols.empty:
            for col, missing_count in missing_cols.items():
                col_missing_pct = (missing_count / len(df)) * 100
                content += f"\n- **{col}**: {missing_count} ({col_missing_pct:.1f}%)"
        else:
            content += "\n✅ No missing data detected in any column"

        # Missing data patterns
        if total_missing > 0:
            missing_matrix = df.isnull()
            pattern_counts = {}

            for _, row in missing_matrix.iterrows():
                pattern = "".join(["1" if x else "0" for x in row])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            content += f"\n\n### Missing Data Patterns:\n"
            content += f"- **Unique Patterns**: {len(pattern_counts)}\n"

            # Most common patterns
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
            top_patterns = sorted_patterns[:3]

            content += "- **Most Common Patterns**:\n"
            for i, (pattern, count) in enumerate(top_patterns, 1):
                content += f"  {i}. Pattern '{pattern}': {count} records\n"

        return ReportSection(
            title="Missing Data Analysis",
            content=content,
            visualizations=[],
            data={
                "total_missing": int(total_missing),
                "missing_percentage": missing_pct,
                "missing_by_column": missing_cols.to_dict(),
                "complete_records": int((~df.isnull().any(axis=1)).sum()),
            },
            priority=4,
        )

    def _generate_custom_analysis_section(
        self, df: pd.DataFrame, custom_analysis: Dict[str, Any]
    ) -> ReportSection:
        """Generate custom analysis section based on user parameters"""
        content = "## Custom Analysis\n\n"

        # Placeholder for custom analysis
        # This can be extended based on specific requirements
        content += "Custom analysis results will be displayed here based on user specifications.\n"

        return ReportSection(
            title="Custom Analysis",
            content=content,
            visualizations=[],
            data=custom_analysis,
            priority=5,
        )

    def _generate_executive_summary(self, df: pd.DataFrame, sections: List[ReportSection]) -> str:
        """Generate executive summary"""
        summary = f"""
        # Executive Summary
        
        This report presents a comprehensive analysis of ENAHO survey data containing {len(df):,} records 
        and {len(df.columns)} variables. The analysis covers data quality assessment, descriptive statistics, 
        and specialized econometric indicators relevant to household survey analysis.
        
        ## Key Findings:
        """

        # Extract key findings from sections
        for section in sections:
            if section.title == "Data Quality Assessment" and section.data:
                overall_score = section.data.get("overall_score", 0)
                grade = section.data.get("grade", "N/A")
                summary += f"\n- **Data Quality**: {overall_score:.1f}/100 (Grade: {grade})"

            elif section.title == "Poverty and Inequality Analysis" and section.data:
                poverty_rate = section.data.get("poverty_headcount", 0)
                gini = section.data.get("gini_coefficient", 0)
                summary += f"\n- **Poverty Rate**: {poverty_rate:.1%}"
                summary += f"\n- **Income Inequality (Gini)**: {gini:.3f}"

        summary += (
            "\n\nDetailed analysis and recommendations are provided in the following sections."
        )

        return summary

    def export_report(
        self, report: ENAHOReport, format: str = "html", filename: Optional[str] = None
    ) -> str:
        """
        Export report to specified format

        Args:
            report: ENAHOReport object to export
            format: Export format ('html', 'markdown', 'json')
            filename: Optional custom filename

        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enaho_report_{timestamp}.{format}"

        output_path = self.output_dir / filename

        if format == "html":
            html_content = self._generate_html_report(report)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        elif format == "markdown":
            md_content = self._generate_markdown_report(report)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)

        elif format == "json":
            json_content = self._serialize_report_to_json(report)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_content)

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Report exported to {output_path}")
        return str(output_path)

    def _generate_html_report(self, report: ENAHOReport) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ report.title }}</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1, h2, h3 { color: #2c5aa0; }
                .summary { background: #f4f4f4; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .section { margin: 30px 0; }
                .metadata { font-size: 0.9em; color: #666; border-top: 1px solid #ddd; padding-top: 20px; margin-top: 40px; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .visualization { margin: 20px 0; text-align: center; }
            </style>
        </head>
        <body>
            <h1>{{ report.title }}</h1>
            <p><strong>Generated:</strong> {{ report.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            
            <div class="summary">
                {{ report.summary | markdown }}
            </div>
            
            {% for section in report.sections %}
            <div class="section">
                <h2>{{ section.title }}</h2>
                {{ section.content | markdown }}
                
                {% if section.visualizations %}
                <div class="visualization">
                    {% for viz in section.visualizations %}
                    <p><a href="visualizations/{{ viz }}" target="_blank">View {{ viz }}</a></p>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
            
            <div class="metadata">
                <h3>Report Metadata</h3>
                <p><strong>Version:</strong> {{ report.version }}</p>
                <p><strong>Dataset Shape:</strong> {{ report.metadata.dataset_shape }}</p>
                <p><strong>Memory Usage:</strong> {{ "%.1f"|format(report.metadata.memory_usage_mb) }} MB</p>
            </div>
        </body>
        </html>
        """

        if JINJA2_AVAILABLE:
            template = Template(html_template)
            return template.render(report=report)
        else:
            # Simple string replacement if Jinja2 not available
            html = html_template.replace("{{ report.title }}", report.title)
            html = html.replace(
                "{{ report.created_at.strftime('%Y-%m-%d %H:%M:%S') }}",
                report.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            )
            # Add more replacements as needed
            return html

    def _generate_markdown_report(self, report: ENAHOReport) -> str:
        """Generate Markdown report"""
        md_content = []

        md_content.append(f"# {report.title}\n")
        md_content.append(f"**Generated:** {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_content.append(f"{report.summary}\n")

        for section in report.sections:
            md_content.append(f"\n{section.content}\n")

            if section.visualizations:
                md_content.append("\n### Visualizations:\n")
                for viz in section.visualizations:
                    md_content.append(f"- [View {viz}](visualizations/{viz})\n")

        md_content.append(f"\n---\n**Report Version:** {report.version}\n")
        md_content.append(f"**Dataset Shape:** {report.metadata['dataset_shape']}\n")

        return "\n".join(md_content)

    def _serialize_report_to_json(self, report: ENAHOReport) -> str:
        """Serialize report to JSON"""
        import json

        report_dict = asdict(report)
        # Convert datetime to string
        report_dict["created_at"] = report.created_at.isoformat()
        report_dict["metadata"]["generation_time"] = report.metadata["generation_time"].isoformat()

        return json.dumps(report_dict, indent=2, ensure_ascii=False)


def generate_enaho_report(
    df: pd.DataFrame,
    output_dir: str = "./reports",
    income_col: Optional[str] = None,
    poverty_line: Optional[float] = None,
    group_col: Optional[str] = None,
    export_format: str = "html",
) -> str:
    """
    Convenience function to generate comprehensive ENAHO report

    Args:
        df: Input DataFrame
        output_dir: Output directory for reports
        income_col: Income column for poverty analysis
        poverty_line: Poverty line threshold
        group_col: Grouping column
        export_format: Export format ('html', 'markdown', 'json')

    Returns:
        Path to generated report file
    """
    generator = ReportGenerator(output_dir)

    report = generator.generate_comprehensive_report(df, income_col, poverty_line, group_col)

    return generator.export_report(report, export_format)


def create_quick_dashboard(df: pd.DataFrame, output_path: str = "dashboard.html") -> str:
    """
    Create quick interactive dashboard for ENAHO data

    Args:
        df: Input DataFrame
        output_path: Path for dashboard file

    Returns:
        Path to dashboard file
    """
    if DATA_QUALITY_AVAILABLE:
        quality_report = assess_data_quality(df)
        viz_engine = VisualizationEngine("plotly")
        return viz_engine.create_data_quality_dashboard(quality_report, output_path)
    else:
        raise ImportError("Data quality module not available")
