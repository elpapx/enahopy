"""
ML-Based Imputation Strategies for ENAHO Data
=============================================

Advanced machine learning imputation methods for handling missing values
in ENAHO survey data with economic and demographic variables.
"""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer, KNNImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    pass

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..config import NullAnalysisConfig


class ImputationStrategy:
    """Base class for imputation strategies"""

    def __init__(self, name: str):
        self.name = name
        self.fitted = False
        self.feature_columns = []
        self.target_column = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the imputation strategy"""
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted strategy"""
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class KNNImputationStrategy(ImputationStrategy):
    """K-Nearest Neighbors imputation for numerical and categorical variables"""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        super().__init__("KNN_Imputation")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = None
        self.encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit KNN imputer"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for KNN imputation")

        # Separate numerical and categorical columns
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Prepare data for fitting
        X_processed = X.copy()

        # Encode categorical variables
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            # Handle missing values in categorical columns
            non_null_mask = X_processed[col].notna()
            if non_null_mask.sum() > 0:
                encoder.fit(X_processed[col][non_null_mask].astype(str))
                X_processed.loc[non_null_mask, col] = encoder.transform(
                    X_processed[col][non_null_mask].astype(str)
                )
                self.encoders[col] = encoder

        # Fit KNN imputer
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        self.imputer.fit(X_processed[self.numerical_cols + self.categorical_cols])
        self.fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted KNN imputer"""
        if not self.fitted:
            raise ValueError("Strategy must be fitted before transform")

        # Check if there's anything to impute
        if X.isnull().sum().sum() == 0:
            # No missing values - return copy with original dtypes preserved
            return X.copy()

        # Handle edge case: single row
        if len(X) == 1:
            # KNN imputation requires multiple rows
            # Return copy with simple mean/mode imputation or original data
            result = X.copy()
            for col in self.numerical_cols:
                if result[col].isnull().any():
                    # Can't impute single row - leave as NaN or fill with 0
                    result[col].fillna(0, inplace=True)
            return result

        X_processed = X.copy()

        # Encode categorical variables using fitted encoders
        for col in self.categorical_cols:
            if col in self.encoders:
                non_null_mask = X_processed[col].notna()
                if non_null_mask.sum() > 0:
                    # Handle unseen categories
                    known_categories = set(self.encoders[col].classes_)
                    X_processed.loc[non_null_mask, col] = (
                        X_processed[col][non_null_mask]
                        .astype(str)
                        .apply(lambda x: x if x in known_categories else "unknown")
                    )
                    # Add 'unknown' to encoder if needed
                    if "unknown" not in self.encoders[col].classes_:
                        self.encoders[col].classes_ = np.append(
                            self.encoders[col].classes_, "unknown"
                        )
                    X_processed.loc[non_null_mask, col] = self.encoders[col].transform(
                        X_processed[col][non_null_mask]
                    )

        # Select columns for imputation
        imputation_cols = self.numerical_cols + self.categorical_cols

        # Apply imputation
        if imputation_cols:
            imputed_data = self.imputer.transform(X_processed[imputation_cols])

            # Create result DataFrame
            result = X.copy()

            # Handle both 1D and 2D output from imputer
            if imputed_data.ndim == 1:
                # Single column case
                result[imputation_cols[0]] = imputed_data
            else:
                # Multiple columns case
                for i, col in enumerate(imputation_cols):
                    result[col] = imputed_data[:, i]
        else:
            result = X.copy()

        # Decode categorical variables
        for col in self.categorical_cols:
            if col in self.encoders:
                result[col] = self.encoders[col].inverse_transform(result[col].astype(int))

        return result


class IterativeImputationStrategy(ImputationStrategy):
    """Iterative imputation using multivariate models"""

    def __init__(self, estimator=None, max_iter: int = 10, random_state: int = 42):
        super().__init__("Iterative_Imputation")
        self.estimator = estimator or BayesianRidge()
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer = None
        self.encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit iterative imputer"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for iterative imputation")

        # Separate column types
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        X_processed = X.copy()

        # Encode categorical variables
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            non_null_mask = X_processed[col].notna()
            if non_null_mask.sum() > 0:
                encoder.fit(X_processed[col][non_null_mask].astype(str))
                X_processed.loc[non_null_mask, col] = encoder.transform(
                    X_processed[col][non_null_mask].astype(str)
                )
                self.encoders[col] = encoder

        # Fit iterative imputer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.imputer = IterativeImputer(
                estimator=self.estimator, max_iter=self.max_iter, random_state=self.random_state
            )
            self.imputer.fit(X_processed[self.numerical_cols + self.categorical_cols])

        self.fitted = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted iterative imputer"""
        if not self.fitted:
            raise ValueError("Strategy must be fitted before transform")

        X_processed = X.copy()

        # Encode categorical variables
        for col in self.categorical_cols:
            if col in self.encoders:
                non_null_mask = X_processed[col].notna()
                if non_null_mask.sum() > 0:
                    known_categories = set(self.encoders[col].classes_)
                    X_processed.loc[non_null_mask, col] = (
                        X_processed[col][non_null_mask]
                        .astype(str)
                        .apply(lambda x: x if x in known_categories else "unknown")
                    )
                    if "unknown" not in self.encoders[col].classes_:
                        self.encoders[col].classes_ = np.append(
                            self.encoders[col].classes_, "unknown"
                        )
                    X_processed.loc[non_null_mask, col] = self.encoders[col].transform(
                        X_processed[col][non_null_mask]
                    )

        # Apply imputation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed_data = self.imputer.transform(
                X_processed[self.numerical_cols + self.categorical_cols]
            )

        # Create result DataFrame
        result = X.copy()
        for i, col in enumerate(self.numerical_cols + self.categorical_cols):
            result[col] = imputed_data[:, i]

        # Decode categorical variables
        for col in self.categorical_cols:
            if col in self.encoders:
                result[col] = self.encoders[col].inverse_transform(result[col].astype(int))

        return result


class RandomForestImputationStrategy(ImputationStrategy):
    """Random Forest-based imputation for mixed data types"""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__("RandomForest_Imputation")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_importance_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit Random Forest models for each column with missing values"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest imputation")

        # Find columns with missing values
        missing_cols = X.columns[X.isnull().any()].tolist()

        for target_col in missing_cols:
            # Get feature columns (excluding target)
            feature_cols = [col for col in X.columns if col != target_col]

            # Get complete cases for this target column
            complete_mask = X[target_col].notna() & X[feature_cols].notna().all(axis=1)

            if complete_mask.sum() < 10:  # Need minimum samples
                continue

            X_train = X[complete_mask][feature_cols]
            y_train = X[complete_mask][target_col]

            # Prepare features
            X_processed = self._prepare_features(X_train, target_col, fit=True)

            # Choose model based on target type
            if X[target_col].dtype == "object" or X[target_col].nunique() < 10:
                # Classification for categorical or low-cardinality variables
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators, random_state=self.random_state
                )
                # Encode target if necessary
                if X[target_col].dtype == "object":
                    encoder = LabelEncoder()
                    y_encoded = encoder.fit_transform(y_train.astype(str))
                    self.encoders[f"{target_col}_target"] = encoder
                    model.fit(X_processed, y_encoded)
                else:
                    model.fit(X_processed, y_train)
            else:
                # Regression for numerical variables
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators, random_state=self.random_state
                )
                model.fit(X_processed, y_train)

            self.models[target_col] = model
            self.feature_importance_[target_col] = dict(
                zip(feature_cols, model.feature_importances_)
            )

        self.fitted = True

    def _prepare_features(
        self, X: pd.DataFrame, target_col: str, fit: bool = False
    ) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        X_processed = X.copy()

        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            key = f"{target_col}_{col}"
            if fit:
                encoder = LabelEncoder()
                non_null_mask = X_processed[col].notna()
                if non_null_mask.sum() > 0:
                    encoder.fit(X_processed[col][non_null_mask].astype(str))
                    X_processed.loc[non_null_mask, col] = encoder.transform(
                        X_processed[col][non_null_mask].astype(str)
                    )
                    self.encoders[key] = encoder
            else:
                if key in self.encoders:
                    non_null_mask = X_processed[col].notna()
                    if non_null_mask.sum() > 0:
                        known_categories = set(self.encoders[key].classes_)
                        X_processed.loc[non_null_mask, col] = (
                            X_processed[col][non_null_mask]
                            .astype(str)
                            .apply(lambda x: x if x in known_categories else "unknown")
                        )
                        if "unknown" not in self.encoders[key].classes_:
                            self.encoders[key].classes_ = np.append(
                                self.encoders[key].classes_, "unknown"
                            )
                        X_processed.loc[non_null_mask, col] = self.encoders[key].transform(
                            X_processed[col][non_null_mask]
                        )

        # Handle numerical variables
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            key = f"{target_col}_scaler"
            if fit:
                scaler = StandardScaler()
                X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
                self.scalers[key] = scaler
            else:
                if key in self.scalers:
                    X_processed[numerical_cols] = self.scalers[key].transform(
                        X_processed[numerical_cols]
                    )

        return X_processed.fillna(0)  # Fill remaining NaNs with 0 for model

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted Random Forest models"""
        if not self.fitted:
            raise ValueError("Strategy must be fitted before transform")

        result = X.copy()

        for target_col, model in self.models.items():
            # Find rows with missing values in target column
            missing_mask = result[target_col].isnull()
            if not missing_mask.any():
                continue

            # Get feature columns
            feature_cols = [col for col in X.columns if col != target_col]

            # Get rows with missing target but complete features
            predict_mask = missing_mask & result[feature_cols].notna().all(axis=1)

            if not predict_mask.any():
                continue

            # Prepare features for prediction
            X_predict = self._prepare_features(
                result[predict_mask][feature_cols], target_col, fit=False
            )

            # Make predictions
            if f"{target_col}_target" in self.encoders:
                # Classification with encoding
                predictions = model.predict(X_predict)
                predictions = self.encoders[f"{target_col}_target"].inverse_transform(predictions)
            else:
                # Regression or classification without encoding
                predictions = model.predict(X_predict)

            # Update result
            result.loc[predict_mask, target_col] = predictions

        return result

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each imputed variable"""
        return self.feature_importance_


class MLImputationManager:
    """Manager for ML-based imputation strategies"""

    def __init__(self, config: Optional[NullAnalysisConfig] = None):
        self.config = config or NullAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        self.strategies = {}
        self.evaluation_results = {}

    def register_strategy(self, name: str, strategy: ImputationStrategy):
        """Register a new imputation strategy"""
        self.strategies[name] = strategy
        self.logger.info(f"Registered imputation strategy: {name}")

    def fit_strategy(self, strategy_name: str, X: pd.DataFrame, y: pd.Series = None):
        """Fit a specific strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        strategy = self.strategies[strategy_name]
        strategy.fit(X, y)
        self.logger.info(f"Fitted strategy: {strategy_name}")

    def impute(self, strategy_name: str, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation using specified strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        strategy = self.strategies[strategy_name]
        return strategy.transform(X)

    def compare_strategies(self, X: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Compare multiple imputation strategies using cross-validation"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for strategy comparison")

        results = {}

        # Create artificial missing data for evaluation
        X_complete = X.dropna()
        if len(X_complete) < 50:
            self.logger.warning("Insufficient complete cases for strategy comparison")
            return results

        # Split data
        X_train, X_test = train_test_split(X_complete, test_size=test_size, random_state=42)

        # Introduce artificial missingness in test set
        X_test_missing = self._introduce_missingness(X_test, missing_rate=0.2)

        for name, strategy in self.strategies.items():
            try:
                # Fit on training data
                strategy.fit(X_train)

                # Impute test data
                X_imputed = strategy.transform(X_test_missing)

                # Evaluate imputation quality
                evaluation = self._evaluate_imputation(X_test, X_imputed, X_test_missing)
                results[name] = evaluation

            except Exception as e:
                self.logger.warning(f"Error evaluating strategy {name}: {str(e)}")
                results[name] = {"error": str(e)}

        return results

    def _introduce_missingness(self, X: pd.DataFrame, missing_rate: float = 0.2) -> pd.DataFrame:
        """Introduce artificial missingness for evaluation"""
        X_missing = X.copy()
        np.random.seed(42)

        for col in X.columns:
            # Randomly select cells to make missing
            n_missing = int(len(X) * missing_rate)
            missing_indices = np.random.choice(len(X), n_missing, replace=False)
            X_missing.iloc[missing_indices, X_missing.columns.get_loc(col)] = np.nan

        return X_missing

    def _evaluate_imputation(
        self, X_true: pd.DataFrame, X_imputed: pd.DataFrame, X_missing: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate imputation quality"""
        evaluation = {}

        for col in X_true.columns:
            # Find artificially missing values
            missing_mask = X_missing[col].isnull() & X_true[col].notna()

            if not missing_mask.any():
                continue

            true_values = X_true[col][missing_mask]
            imputed_values = X_imputed[col][missing_mask]

            if X_true[col].dtype in ["object", "category"]:
                # Categorical evaluation
                accuracy = accuracy_score(true_values, imputed_values)
                evaluation[f"{col}_accuracy"] = accuracy
            else:
                # Numerical evaluation
                mse = mean_squared_error(true_values, imputed_values)
                rmse = np.sqrt(mse)

                # Normalized RMSE
                range_val = X_true[col].max() - X_true[col].min()
                nrmse = rmse / range_val if range_val > 0 else 0

                evaluation[f"{col}_rmse"] = rmse
                evaluation[f"{col}_nrmse"] = nrmse

        return evaluation

    def get_best_strategy(self, evaluation_results: Dict[str, Any]) -> str:
        """Determine best strategy based on evaluation results"""
        if not evaluation_results:
            return None

        # Calculate average performance for each strategy
        strategy_scores = {}

        for strategy, results in evaluation_results.items():
            if "error" in results:
                continue

            scores = []
            for metric, value in results.items():
                if "accuracy" in metric:
                    scores.append(value)  # Higher is better
                elif "nrmse" in metric:
                    scores.append(1 - value)  # Lower is better, so invert

            if scores:
                strategy_scores[strategy] = np.mean(scores)

        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            return best_strategy

        return None

    def create_default_strategies(self) -> None:
        """Create and register default ML imputation strategies"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available, cannot create ML strategies")
            return

        # KNN Imputation
        knn_strategy = KNNImputationStrategy(n_neighbors=5)
        self.register_strategy("knn", knn_strategy)

        # Iterative Imputation with Bayesian Ridge
        iterative_strategy = IterativeImputationStrategy(estimator=BayesianRidge(), max_iter=10)
        self.register_strategy("iterative", iterative_strategy)

        # Random Forest Imputation
        rf_strategy = RandomForestImputationStrategy(n_estimators=50)
        self.register_strategy("random_forest", rf_strategy)

        self.logger.info("Created default ML imputation strategies")


def create_ml_imputation_manager(
    config: Optional[NullAnalysisConfig] = None,
) -> MLImputationManager:
    """Factory function to create ML imputation manager with default strategies"""
    manager = MLImputationManager(config)
    manager.create_default_strategies()
    return manager


def quick_ml_imputation(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """Quick ML imputation with automatic strategy selection"""
    manager = create_ml_imputation_manager()

    if strategy == "auto":
        # Compare strategies and select best
        comparison = manager.compare_strategies(df)
        best_strategy = manager.get_best_strategy(comparison)
        strategy = best_strategy or "knn"  # Fallback to KNN

    # Fit and apply chosen strategy
    if strategy in manager.strategies:
        manager.fit_strategy(strategy, df)
        return manager.impute(strategy, df)
    else:
        raise ValueError(f"Strategy '{strategy}' not available")
