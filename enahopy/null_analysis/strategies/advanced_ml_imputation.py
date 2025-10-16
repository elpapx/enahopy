"""
ENAHO Advanced ML-Based Imputation - ANALYZE Phase
==================================================

State-of-the-art machine learning imputation algorithms for ENAHO survey data.
Includes MICE, missForest, Autoencoders, and pattern-aware imputation methods
with comprehensive quality assessment and validation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None


@dataclass
class ImputationConfig:
    """Configuration for imputation methods"""

    method: str
    random_state: int = 42
    max_iter: int = 10
    n_estimators: int = 100
    early_stopping: bool = True
    convergence_threshold: float = 1e-3
    categorical_encoding: str = "label"  # "label", "onehot"
    validation_split: float = 0.2
    quality_metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "accuracy"])


@dataclass
class ImputationResult:
    """Container for imputation results"""

    imputed_data: pd.DataFrame
    method: str
    quality_metrics: Dict[str, float]
    imputation_diagnostics: Dict[str, Any]
    missing_patterns: Dict[str, Any]
    computational_time: float
    convergence_info: Optional[Dict[str, Any]] = None


class MICEImputer:
    """
    Multiple Imputation by Chained Equations (MICE)
    Advanced implementation with survey-specific adaptations
    """

    def __init__(self, config: ImputationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.imputers = {}
        self.convergence_history = []
        self.feature_importance = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        ordinal_cols: Optional[Dict[str, List]] = None,
        survey_weights: Optional[pd.Series] = None,
    ) -> ImputationResult:
        """
        Perform MICE imputation with survey-specific enhancements

        Args:
            df: DataFrame with missing values
            categorical_cols: List of categorical column names
            ordinal_cols: Dictionary mapping ordinal columns to their ordered categories
            survey_weights: Survey weights for weighted imputation

        Returns:
            ImputationResult with imputed data and diagnostics
        """
        import time

        start_time = time.time()

        self.logger.info("Starting MICE imputation...")

        # Prepare data
        working_df = df.copy()
        categorical_cols = categorical_cols or []
        ordinal_cols = ordinal_cols or {}

        # Identify missing patterns
        missing_patterns = self._analyze_missing_patterns(working_df)

        # Encode categorical variables
        encoders = {}
        for col in categorical_cols:
            if col in working_df.columns:
                le = LabelEncoder()
                # Fit on non-missing values
                non_missing = working_df[col].dropna()
                if len(non_missing) > 0:
                    le.fit(non_missing)
                    # Transform non-missing values
                    mask = working_df[col].notna()
                    working_df.loc[mask, col] = le.transform(working_df.loc[mask, col])
                    encoders[col] = le

        # Initialize imputation order based on missing rates
        missing_rates = working_df.isnull().mean()
        imputation_order = missing_rates[missing_rates > 0].sort_values().index.tolist()

        self.logger.info(f"Imputation order: {imputation_order}")

        # Perform chained equations
        convergence_metrics = []

        for iteration in range(self.config.max_iter):
            iteration_start = time.time()
            old_values = working_df.copy()

            for col in imputation_order:
                if col not in working_df.columns:
                    continue

                # Get complete cases for this variable
                predictor_cols = [c for c in working_df.columns if c != col]
                complete_mask = (
                    working_df[predictor_cols].notna().all(axis=1) & working_df[col].notna()
                )
                missing_mask = (
                    working_df[predictor_cols].notna().all(axis=1) & working_df[col].isna()
                )

                if complete_mask.sum() < 10 or missing_mask.sum() == 0:
                    continue

                # Prepare training data
                X_train = working_df.loc[complete_mask, predictor_cols]
                y_train = working_df.loc[complete_mask, col]
                X_predict = working_df.loc[missing_mask, predictor_cols]

                # Handle survey weights
                sample_weights = None
                if survey_weights is not None:
                    sample_weights = survey_weights.loc[complete_mask]

                # Choose model based on variable type
                if col in categorical_cols:
                    model = self._fit_categorical_model(X_train, y_train, sample_weights)
                else:
                    model = self._fit_continuous_model(X_train, y_train, sample_weights)

                # Predict missing values
                if len(X_predict) > 0:
                    predictions = model.predict(X_predict)

                    # Add random noise for continuous variables
                    if col not in categorical_cols:
                        residuals = y_train - model.predict(X_train)
                        residual_std = np.std(residuals)
                        noise = np.random.normal(0, residual_std, len(predictions))
                        predictions += noise

                    # Update working dataframe
                    working_df.loc[missing_mask, col] = predictions

                # Store model for diagnostics
                self.imputers[col] = model

            # Check convergence
            if iteration > 0:
                convergence_metric = self._calculate_convergence_metric(
                    old_values, working_df, imputation_order
                )
                convergence_metrics.append(convergence_metric)

                self.logger.info(
                    f"Iteration {iteration + 1}: Convergence metric = {convergence_metric:.6f}"
                )

                if convergence_metric < self.config.convergence_threshold:
                    self.logger.info(f"Converged after {iteration + 1} iterations")
                    break

            iteration_time = time.time() - iteration_start
            self.logger.debug(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")

        # Decode categorical variables
        for col, encoder in encoders.items():
            if col in working_df.columns:
                # Round to nearest integer for categorical
                working_df[col] = np.round(working_df[col]).astype(int)
                # Clip to valid range
                valid_range = range(len(encoder.classes_))
                working_df[col] = working_df[col].clip(min(valid_range), max(valid_range))
                # Inverse transform
                working_df[col] = encoder.inverse_transform(working_df[col])

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(df, working_df, categorical_cols)

        # Prepare diagnostics
        diagnostics = {
            "convergence_history": convergence_metrics,
            "imputation_order": imputation_order,
            "iterations_performed": len(convergence_metrics) + 1,
            "models_trained": len(self.imputers),
            "feature_importance": self._extract_feature_importance(),
        }

        total_time = time.time() - start_time
        self.logger.info(f"MICE imputation completed in {total_time:.2f}s")

        return ImputationResult(
            imputed_data=working_df,
            method="MICE",
            quality_metrics=quality_metrics,
            imputation_diagnostics=diagnostics,
            missing_patterns=missing_patterns,
            computational_time=total_time,
            convergence_info={
                "converged": len(convergence_metrics) > 0
                and convergence_metrics[-1] < self.config.convergence_threshold
            },
        )

    def _fit_continuous_model(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ):
        """Fit model for continuous variables"""
        if self.config.method.lower() == "rf":
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            model = LinearRegression()

        if (
            sample_weights is not None
            and hasattr(model, "fit")
            and "sample_weight" in model.fit.__code__.co_varnames
        ):
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)

        return model

    def _fit_categorical_model(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ):
        """Fit model for categorical variables"""
        if self.config.method.lower() == "rf":
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            model = LogisticRegression(random_state=self.config.random_state, max_iter=1000)

        if (
            sample_weights is not None
            and hasattr(model, "fit")
            and "sample_weight" in model.fit.__code__.co_varnames
        ):
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)

        return model

    def _calculate_convergence_metric(
        self, old_df: pd.DataFrame, new_df: pd.DataFrame, cols: List[str]
    ) -> float:
        """Calculate convergence metric between iterations"""
        differences = []
        for col in cols:
            if col in old_df.columns and col in new_df.columns:
                old_vals = old_df[col].dropna()
                new_vals = new_df.loc[old_vals.index, col]
                if len(old_vals) > 0:
                    diff = np.mean(np.abs(new_vals - old_vals)) / (np.std(old_vals) + 1e-8)
                    differences.append(diff)

        return np.mean(differences) if differences else 1.0

    def _extract_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Extract feature importance from trained models"""
        importance_dict = {}

        for col, model in self.imputers.items():
            if hasattr(model, "feature_importances_"):
                importance_dict[col] = {
                    f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)
                }
            elif hasattr(model, "coef_"):
                importance_dict[col] = {
                    f"feature_{i}": abs(coef) for i, coef in enumerate(model.coef_.flatten())
                }

        return importance_dict

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_matrix = df.isnull()

        # Pattern frequencies
        pattern_counts = missing_matrix.value_counts()

        # Missing rates by variable
        missing_rates = missing_matrix.mean()

        # Pairwise missing patterns
        pairwise_patterns = {}
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    both_missing = (missing_matrix[col1] & missing_matrix[col2]).sum()
                    pairwise_patterns[f"{col1}_{col2}"] = both_missing / len(df)

        return {
            "pattern_frequencies": pattern_counts.to_dict(),
            "missing_rates": missing_rates.to_dict(),
            "pairwise_patterns": pairwise_patterns,
            "total_patterns": len(pattern_counts),
        }

    def _calculate_quality_metrics(
        self, original_df: pd.DataFrame, imputed_df: pd.DataFrame, categorical_cols: List[str]
    ) -> Dict[str, float]:
        """Calculate imputation quality metrics"""
        metrics = {}

        # For each column with missing values
        for col in original_df.columns:
            if original_df[col].isnull().any():
                # Split into observed and imputed
                observed_mask = original_df[col].notna()

                if observed_mask.sum() == 0:
                    continue

                observed_vals = original_df.loc[observed_mask, col]
                imputed_vals = imputed_df.loc[~observed_mask, col]

                if len(imputed_vals) == 0:
                    continue

                if col in categorical_cols:
                    # For categorical: check if imputed values are in observed range
                    unique_observed = set(observed_vals.unique())
                    unique_imputed = set(imputed_vals.unique())
                    coverage = len(unique_imputed.intersection(unique_observed)) / len(
                        unique_observed
                    )
                    metrics[f"{col}_category_coverage"] = coverage
                else:
                    # For continuous: statistical comparisons
                    metrics[f"{col}_mean_diff"] = abs(imputed_vals.mean() - observed_vals.mean())
                    metrics[f"{col}_std_ratio"] = imputed_vals.std() / (observed_vals.std() + 1e-8)

                    # Distribution comparison (simplified Kolmogorov-Smirnov)
                    try:
                        ks_stat, ks_pvalue = stats.ks_2samp(observed_vals, imputed_vals)
                        metrics[f"{col}_ks_statistic"] = ks_stat
                        metrics[f"{col}_ks_pvalue"] = ks_pvalue
                    except:
                        pass

        return metrics


class MissForestImputer:
    """
    MissForest imputation using Random Forests
    Handles mixed-type data and non-linear relationships
    """

    def __init__(self, config: ImputationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.rf_models = {}
        self.oob_errors = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        survey_weights: Optional[pd.Series] = None,
    ) -> ImputationResult:
        """
        Perform missForest imputation

        Args:
            df: DataFrame with missing values
            categorical_cols: List of categorical column names
            survey_weights: Survey weights (not directly used in RF but stored)

        Returns:
            ImputationResult with imputed data and diagnostics
        """
        import time

        start_time = time.time()

        self.logger.info("Starting missForest imputation...")

        working_df = df.copy()
        categorical_cols = categorical_cols or []

        # Analyze missing patterns
        missing_patterns = self._analyze_missing_patterns(working_df)

        # Encode categorical variables
        encoders = {}
        for col in categorical_cols:
            if col in working_df.columns and working_df[col].notna().any():
                le = LabelEncoder()
                non_missing = working_df[col].dropna()
                le.fit(non_missing)
                mask = working_df[col].notna()
                working_df.loc[mask, col] = le.transform(working_df.loc[mask, col])
                encoders[col] = le

        # Initial imputation with mean/mode
        for col in working_df.columns:
            if working_df[col].isnull().any():
                if col in categorical_cols:
                    # Mode for categorical
                    mode_val = working_df[col].mode()
                    if len(mode_val) > 0:
                        working_df[col].fillna(mode_val[0], inplace=True)
                else:
                    # Mean for continuous
                    working_df[col].fillna(working_df[col].mean(), inplace=True)

        # Iterative imputation
        convergence_history = []
        columns_with_missing = df.columns[df.isnull().any()].tolist()

        for iteration in range(self.config.max_iter):
            old_values = working_df.copy()

            # Sort by missing rate (ascending)
            missing_rates = df[columns_with_missing].isnull().mean()
            sorted_cols = missing_rates.sort_values().index.tolist()

            for col in sorted_cols:
                if col not in working_df.columns:
                    continue

                # Identify originally missing values
                originally_missing = df[col].isnull()

                if not originally_missing.any():
                    continue

                # Prepare features and target
                feature_cols = [c for c in working_df.columns if c != col]
                X = working_df[feature_cols]
                y_observed = working_df.loc[~originally_missing, col]
                X_observed = X.loc[~originally_missing]
                X_missing = X.loc[originally_missing]

                if len(X_observed) < 10:
                    continue

                # Train Random Forest
                if col in categorical_cols:
                    rf = RandomForestClassifier(
                        n_estimators=self.config.n_estimators,
                        random_state=self.config.random_state,
                        oob_score=True,
                        n_jobs=-1,
                    )
                else:
                    rf = RandomForestRegressor(
                        n_estimators=self.config.n_estimators,
                        random_state=self.config.random_state,
                        oob_score=True,
                        n_jobs=-1,
                    )

                rf.fit(X_observed, y_observed)

                # Store model and OOB error
                self.rf_models[col] = rf
                self.oob_errors[col] = 1 - rf.oob_score_ if hasattr(rf, "oob_score_") else None

                # Predict missing values
                if len(X_missing) > 0:
                    predictions = rf.predict(X_missing)
                    working_df.loc[originally_missing, col] = predictions

            # Check convergence
            if iteration > 0:
                convergence_metric = self._calculate_convergence_metric(
                    old_values, working_df, sorted_cols
                )
                convergence_history.append(convergence_metric)

                self.logger.info(
                    f"MissForest iteration {iteration + 1}: Convergence = {convergence_metric:.6f}"
                )

                if convergence_metric < self.config.convergence_threshold:
                    self.logger.info(f"MissForest converged after {iteration + 1} iterations")
                    break

        # Decode categorical variables
        for col, encoder in encoders.items():
            if col in working_df.columns:
                working_df[col] = working_df[col].round().astype(int)
                # Ensure values are in valid range
                valid_range = range(len(encoder.classes_))
                working_df[col] = working_df[col].clip(min(valid_range), max(valid_range))
                working_df[col] = encoder.inverse_transform(working_df[col])

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(df, working_df, categorical_cols)

        # Prepare diagnostics
        diagnostics = {
            "convergence_history": convergence_history,
            "oob_errors": self.oob_errors,
            "feature_importances": self._get_feature_importances(),
            "iterations_performed": len(convergence_history) + 1,
        }

        total_time = time.time() - start_time
        self.logger.info(f"MissForest imputation completed in {total_time:.2f}s")

        return ImputationResult(
            imputed_data=working_df,
            method="missForest",
            quality_metrics=quality_metrics,
            imputation_diagnostics=diagnostics,
            missing_patterns=missing_patterns,
            computational_time=total_time,
        )

    def _calculate_convergence_metric(
        self, old_df: pd.DataFrame, new_df: pd.DataFrame, cols: List[str]
    ) -> float:
        """Calculate convergence metric for missForest"""
        sum_diff = 0
        sum_total = 0

        for col in cols:
            if col in old_df.columns and col in new_df.columns:
                diff = ((new_df[col] - old_df[col]) ** 2).sum()
                total = (new_df[col] ** 2).sum()
                sum_diff += diff
                sum_total += total

        return np.sqrt(sum_diff / (sum_total + 1e-8))

    def _get_feature_importances(self) -> Dict[str, np.ndarray]:
        """Get feature importances from trained RF models"""
        importances = {}
        for col, model in self.rf_models.items():
            if hasattr(model, "feature_importances_"):
                importances[col] = model.feature_importances_
        return importances

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing patterns (same as MICE)"""
        missing_matrix = df.isnull()

        pattern_counts = missing_matrix.value_counts()
        missing_rates = missing_matrix.mean()

        return {
            "pattern_frequencies": pattern_counts.to_dict(),
            "missing_rates": missing_rates.to_dict(),
            "total_patterns": len(pattern_counts),
        }

    def _calculate_quality_metrics(
        self, original_df: pd.DataFrame, imputed_df: pd.DataFrame, categorical_cols: List[str]
    ) -> Dict[str, float]:
        """Calculate quality metrics (same as MICE)"""
        metrics = {}

        for col in original_df.columns:
            if original_df[col].isnull().any():
                observed_mask = original_df[col].notna()

                if observed_mask.sum() == 0:
                    continue

                observed_vals = original_df.loc[observed_mask, col]
                imputed_vals = imputed_df.loc[~observed_mask, col]

                if len(imputed_vals) == 0:
                    continue

                if col in categorical_cols:
                    unique_observed = set(observed_vals.unique())
                    unique_imputed = set(imputed_vals.unique())
                    if len(unique_observed) > 0:
                        coverage = len(unique_imputed.intersection(unique_observed)) / len(
                            unique_observed
                        )
                        metrics[f"{col}_category_coverage"] = coverage
                else:
                    metrics[f"{col}_mean_diff"] = abs(imputed_vals.mean() - observed_vals.mean())
                    metrics[f"{col}_std_ratio"] = imputed_vals.std() / (observed_vals.std() + 1e-8)

        return metrics


class AutoencoderImputer:
    """
    Deep learning-based imputation using Autoencoders
    Captures complex non-linear relationships in data
    """

    def __init__(self, config: ImputationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.encoders = {}

        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Please install: pip install tensorflow")

    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        hidden_dims: List[int] = [64, 32, 16, 32, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
    ) -> ImputationResult:
        """
        Perform autoencoder-based imputation

        Args:
            df: DataFrame with missing values
            categorical_cols: List of categorical columns
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Training epochs

        Returns:
            ImputationResult with imputed data
        """
        import time

        start_time = time.time()

        self.logger.info("Starting Autoencoder imputation...")

        working_df = df.copy()
        categorical_cols = categorical_cols or []

        # Analyze missing patterns
        missing_patterns = self._analyze_missing_patterns(working_df)

        # Encode categorical variables
        for col in categorical_cols:
            if col in working_df.columns and working_df[col].notna().any():
                le = LabelEncoder()
                non_missing = working_df[col].dropna()
                le.fit(non_missing)
                mask = working_df[col].notna()
                working_df.loc[mask, col] = le.transform(working_df.loc[mask, col])
                self.encoders[col] = le

        # Initial imputation with mean/median
        for col in working_df.columns:
            if working_df[col].isnull().any():
                if col in categorical_cols:
                    mode_val = working_df[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                else:
                    fill_val = working_df[col].median()
                working_df[col].fillna(fill_val, inplace=True)

        # Scale features
        X_scaled = self.scaler.fit_transform(working_df)

        # Create missing mask for loss function
        missing_mask = df.isnull().values

        # Build autoencoder
        input_dim = X_scaled.shape[1]
        self.autoencoder = self._build_autoencoder(input_dim, hidden_dims, dropout_rate)

        # Compile model
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._masked_mse_loss(missing_mask),
            metrics=["mae"],
        )

        # Train autoencoder
        history = self.autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=self.config.validation_split,
            verbose=0,
            callbacks=[
                (
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                    if self.config.early_stopping
                    else None
                )
            ],
        )

        # Generate imputations
        X_imputed = self.autoencoder.predict(X_scaled, verbose=0)

        # Scale back
        X_imputed_scaled = self.scaler.inverse_transform(X_imputed)

        # Create imputed dataframe
        imputed_df = pd.DataFrame(X_imputed_scaled, columns=df.columns, index=df.index)

        # Only replace originally missing values
        for col in df.columns:
            originally_missing = df[col].isnull()
            if originally_missing.any():
                imputed_df.loc[~originally_missing, col] = df.loc[~originally_missing, col]

        # Decode categorical variables
        for col, encoder in self.encoders.items():
            if col in imputed_df.columns:
                imputed_df[col] = imputed_df[col].round().astype(int)
                # Clip to valid range
                valid_range = range(len(encoder.classes_))
                imputed_df[col] = imputed_df[col].clip(min(valid_range), max(valid_range))
                imputed_df[col] = encoder.inverse_transform(imputed_df[col])

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(df, imputed_df, categorical_cols)

        # Training diagnostics
        diagnostics = {
            "training_history": {
                "loss": history.history.get("loss", []),
                "val_loss": history.history.get("val_loss", []),
                "mae": history.history.get("mae", []),
                "val_mae": history.history.get("val_mae", []),
            },
            "epochs_trained": len(history.history.get("loss", [])),
            "final_loss": (
                history.history.get("loss", [])[-1] if history.history.get("loss") else None
            ),
            "architecture": hidden_dims,
        }

        total_time = time.time() - start_time
        self.logger.info(f"Autoencoder imputation completed in {total_time:.2f}s")

        return ImputationResult(
            imputed_data=imputed_df,
            method="Autoencoder",
            quality_metrics=quality_metrics,
            imputation_diagnostics=diagnostics,
            missing_patterns=missing_patterns,
            computational_time=total_time,
        )

    def _build_autoencoder(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        """Build autoencoder architecture"""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = input_layer

        for dim in hidden_dims[: len(hidden_dims) // 2]:
            encoded = layers.Dense(dim, activation="relu")(encoded)
            encoded = layers.Dropout(dropout_rate)(encoded)

        # Bottleneck
        bottleneck_dim = hidden_dims[len(hidden_dims) // 2]
        bottleneck = layers.Dense(bottleneck_dim, activation="relu")(encoded)

        # Decoder
        decoded = bottleneck
        for dim in hidden_dims[len(hidden_dims) // 2 + 1 :]:
            decoded = layers.Dense(dim, activation="relu")(decoded)
            decoded = layers.Dropout(dropout_rate)(decoded)

        # Output layer
        output = layers.Dense(input_dim, activation="linear")(decoded)

        autoencoder = keras.Model(input_layer, output)
        return autoencoder

    def _masked_mse_loss(self, missing_mask: np.ndarray):
        """Custom loss function that only considers observed values"""

        def loss_fn(y_true, y_pred):
            # Create mask for observed values (inverse of missing_mask)
            observed_mask = tf.constant(~missing_mask, dtype=tf.float32)

            # Calculate squared error only for observed values
            squared_error = tf.square(y_true - y_pred) * observed_mask

            # Return mean over observed values
            return tf.reduce_sum(squared_error) / (tf.reduce_sum(observed_mask) + 1e-8)

        return loss_fn

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing patterns"""
        missing_matrix = df.isnull()
        return {
            "missing_rates": missing_matrix.mean().to_dict(),
            "total_missing": missing_matrix.sum().sum(),
            "complete_cases": (~missing_matrix.any(axis=1)).sum(),
        }

    def _calculate_quality_metrics(
        self, original_df: pd.DataFrame, imputed_df: pd.DataFrame, categorical_cols: List[str]
    ) -> Dict[str, float]:
        """Calculate quality metrics"""
        metrics = {}

        for col in original_df.columns:
            if original_df[col].isnull().any():
                observed_mask = original_df[col].notna()
                imputed_vals = imputed_df.loc[~observed_mask, col]

                if len(imputed_vals) == 0:
                    continue

                if col in categorical_cols:
                    # For categorical: check uniqueness and coverage
                    observed_unique = set(original_df.loc[observed_mask, col].unique())
                    imputed_unique = set(imputed_vals.unique())
                    coverage = (
                        len(imputed_unique.intersection(observed_unique)) / len(observed_unique)
                        if observed_unique
                        else 0
                    )
                    metrics[f"{col}_category_coverage"] = coverage
                else:
                    # For continuous: distribution similarity
                    observed_vals = original_df.loc[observed_mask, col]
                    metrics[f"{col}_mean_diff"] = abs(imputed_vals.mean() - observed_vals.mean())
                    metrics[f"{col}_std_ratio"] = imputed_vals.std() / (observed_vals.std() + 1e-8)

        return metrics


# Factory functions and utilities
def create_advanced_imputer(
    method: str, config: Optional[ImputationConfig] = None, logger: Optional[logging.Logger] = None
):
    """Factory function to create imputation models"""
    if config is None:
        config = ImputationConfig(method=method)

    if method.lower() == "mice":
        return MICEImputer(config, logger)
    elif method.lower() == "missforest":
        return MissForestImputer(config, logger)
    elif method.lower() == "autoencoder":
        return AutoencoderImputer(config, logger)
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def compare_imputation_methods(
    df: pd.DataFrame,
    methods: List[str],
    categorical_cols: Optional[List[str]] = None,
    test_fraction: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple imputation methods using artificial missing data

    Args:
        df: Complete DataFrame for testing
        methods: List of methods to compare ('mice', 'missforest', 'autoencoder')
        categorical_cols: List of categorical columns
        test_fraction: Fraction of data to artificially make missing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with comparison metrics for each method
    """
    np.random.seed(random_state)
    logger = logging.getLogger(__name__)

    # Create artificial missing data
    complete_df = df.dropna()
    test_df = complete_df.copy()

    # Randomly remove values
    for col in complete_df.columns:
        n_missing = int(len(complete_df) * test_fraction)
        missing_indices = np.random.choice(complete_df.index, n_missing, replace=False)
        test_df.loc[missing_indices, col] = np.nan

    results = {}

    for method in methods:
        logger.info(f"Testing {method} imputation method...")

        try:
            config = ImputationConfig(method=method, max_iter=5)  # Reduced for comparison
            imputer = create_advanced_imputer(method, config, logger)

            # Perform imputation
            imputation_result = imputer.fit_transform(test_df, categorical_cols)
            imputed_df = imputation_result.imputed_data

            # Calculate accuracy metrics
            method_metrics = {}

            for col in complete_df.columns:
                if col in categorical_cols or []:
                    # Accuracy for categorical
                    true_vals = complete_df[col]
                    pred_vals = imputed_df[col]
                    missing_mask = test_df[col].isnull()

                    if missing_mask.any():
                        accuracy = (true_vals[missing_mask] == pred_vals[missing_mask]).mean()
                        method_metrics[f"{col}_accuracy"] = accuracy
                else:
                    # RMSE and MAE for continuous
                    true_vals = complete_df[col]
                    pred_vals = imputed_df[col]
                    missing_mask = test_df[col].isnull()

                    if missing_mask.any():
                        rmse = np.sqrt(
                            mean_squared_error(true_vals[missing_mask], pred_vals[missing_mask])
                        )
                        mae = mean_absolute_error(true_vals[missing_mask], pred_vals[missing_mask])
                        method_metrics[f"{col}_rmse"] = rmse
                        method_metrics[f"{col}_mae"] = mae

            # Overall metrics
            method_metrics["computation_time"] = imputation_result.computational_time
            method_metrics["method"] = method

            results[method] = method_metrics

        except Exception as e:
            logger.error(f"Error testing {method}: {str(e)}")
            results[method] = {"error": str(e)}

    return results


# Export main classes and functions
__all__ = [
    "MICEImputer",
    "MissForestImputer",
    "AutoencoderImputer",
    "ImputationConfig",
    "ImputationResult",
    "create_advanced_imputer",
    "compare_imputation_methods",
]
