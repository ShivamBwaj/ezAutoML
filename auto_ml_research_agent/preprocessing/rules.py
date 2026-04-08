"""
Preprocessing Engine: Automatically builds sklearn preprocessing pipelines.
"""
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def convert_bool_to_object(X):
    """
    Convert boolean array/DataFrame to object dtype for SimpleImputer compatibility.
    SimpleImputer doesn't support bool dtype, so we convert True/False to strings.
    Works with numpy arrays, pandas Series, and pandas DataFrames.
    """
    # pandas objects have 'dtypes' (plural) for DataFrame/Series
    if hasattr(X, 'dtypes'):
        # DataFrame or Series - check any boolean columns and convert them
        return X.astype(object)
    elif hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.bool_):
        return X.astype(object)
    return X


class BoolToObjectConverter(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapper for convert_bool_to_object"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return convert_bool_to_object(X)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer for frequency encoding categorical variables"""

    def __init__(self):
        self.freq_maps_ = {}  # Trained attribute must end with _

    def fit(self, X, y=None):
        """Fit frequency encoder"""
        # X is expected to be a 2D array (DataFrame column(s))
        self.freq_maps_ = {}
        for col_idx in range(X.shape[1]):
            col_series = pd.Series(X[:, col_idx])
            freq_map = col_series.value_counts(normalize=True).to_dict()
            self.freq_maps_[col_idx] = freq_map
        return self

    def transform(self, X):
        """Transform using frequency encoding"""
        transformed = X.copy()
        for col_idx in range(X.shape[1]):
            col_series = pd.Series(X[:, col_idx])
            freq_map = self.freq_maps_.get(col_idx, {})
            transformed[:, col_idx] = col_series.map(freq_map).fillna(0.0).values
        return transformed


class PreprocessingEngine:
    """
    Automatically constructs preprocessing pipelines based on dataset characteristics.
    """

    def __init__(
        self,
        numeric_imputation_strategy: str = "median",
        numeric_scaling: bool = True,
        categorical_encoding_threshold: int = 10,
        high_cardinality_encoding: str = "frequency",
        id_cardinality_threshold: float = 0.8,
        high_missing_threshold: float = 0.7,
        low_variance_threshold: float = 0.01
    ):
        """
        Initialize preprocessing engine.

        Args:
            numeric_imputation_strategy: "mean", "median", or "most_frequent"
            numeric_scaling: Whether to scale numeric features
            categorical_encoding_threshold: Max unique values for one-hot encoding
            high_cardinality_encoding: "frequency" or "drop" for high-cardinality cats
        """
        self.numeric_imputation_strategy = numeric_imputation_strategy
        self.numeric_scaling = numeric_scaling
        self.categorical_encoding_threshold = categorical_encoding_threshold
        self.high_cardinality_encoding = high_cardinality_encoding
        self.id_cardinality_threshold = id_cardinality_threshold
        self.high_missing_threshold = high_missing_threshold
        self.low_variance_threshold = low_variance_threshold

    def detect_id_columns(
        self,
        df: pd.DataFrame,
        cardinality_threshold: Optional[float] = None,
        name_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """Detect likely identifier columns that should be dropped."""
        if df.empty:
            return []
        if name_patterns is None:
            name_patterns = [
                "id", "uuid", "guid", "key", "index", "unique",
                "student", "user", "employee", "member", "patient",
                "order", "transaction", "invoice", "account",
                "ssn", "social", "passport", "license"
            ]
        threshold = cardinality_threshold if cardinality_threshold is not None else self.id_cardinality_threshold
        n_rows = len(df)
        candidates = set()
        for col in df.columns:
            unique_count = df[col].nunique(dropna=False)
            unique_ratio = unique_count / max(1, n_rows)
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in name_patterns) and unique_ratio > threshold:
                candidates.add(col)
                continue
            if unique_ratio > 0.95 or unique_count == n_rows:
                # Keep known geographic-like fields that can still carry signal.
                if "postal" in col_lower or "zip" in col_lower:
                    continue
                candidates.add(col)
        return sorted(candidates)

    def drop_high_missing(self, X: pd.DataFrame, threshold: Optional[float] = None) -> List[str]:
        """Columns with missingness above threshold."""
        th = threshold if threshold is not None else self.high_missing_threshold
        missing_pct = X.isnull().mean()
        return missing_pct[missing_pct > th].index.tolist()

    def drop_constant(self, X: pd.DataFrame) -> List[str]:
        """Columns with a single unique value."""
        nunique = X.nunique(dropna=False)
        return nunique[nunique <= 1].index.tolist()

    def drop_low_variance(self, X: pd.DataFrame, threshold: Optional[float] = None) -> List[str]:
        """Numeric columns with very low variance."""
        th = threshold if threshold is not None else self.low_variance_threshold
        variances = X.select_dtypes(include=["number"]).var(numeric_only=True)
        return variances[variances < th].index.tolist()

    def build_preprocessor(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[ColumnTransformer, Dict[str, Any]]:
        """
        Build a sklearn ColumnTransformer for the given DataFrame.

        Args:
            df: Input DataFrame (includes target)
            target_column: Name of target column (will be excluded)

        Returns:
            Tuple of (configured ColumnTransformer, metadata dictionary with transformation details)
        """
        # Separate features and target
        X = df.drop(columns=[target_column])

        # Initialize metadata log
        metadata = {
            'input_features': {},
            'transformers': [],
            'summary': {},
            'dropped_columns': {
                'id_columns': [],
                'high_missing': [],
                'constant': [],
                'low_variance': []
            }
        }

        # Drop known-bad feature columns before typing/transformers.
        id_cols = self.detect_id_columns(X)
        high_missing_cols = self.drop_high_missing(X)
        constant_cols = self.drop_constant(X)
        low_variance_cols = self.drop_low_variance(X)
        drop_cols = sorted(set(id_cols + high_missing_cols + constant_cols + low_variance_cols))
        if drop_cols:
            X = X.drop(columns=drop_cols, errors="ignore")

        metadata['dropped_columns']['id_columns'] = id_cols
        metadata['dropped_columns']['high_missing'] = high_missing_cols
        metadata['dropped_columns']['constant'] = constant_cols
        metadata['dropped_columns']['low_variance'] = low_variance_cols

        # Record input feature info
        for col in X.columns:
            col_info = {
                'dtype': str(X[col].dtype),
                'missing_count': int(X[col].isnull().sum()),
                'missing_pct': float(round(X[col].isnull().sum() / len(X) * 100, 2)),
                'unique_count': int(X[col].nunique()),
                'sample_values': [str(v) for v in X[col].dropna().head(3).tolist()]
            }
            metadata['input_features'][col] = col_info

        # Identify column types
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        transformers = []

        # Build numeric pipeline
        if numeric_cols:
            numeric_steps = []

            # Imputation
            numeric_steps.append(('imputer', SimpleImputer(strategy=self.numeric_imputation_strategy)))

            # Scaling
            if self.numeric_scaling:
                numeric_steps.append(('scaler', StandardScaler()))

            numeric_pipeline = Pipeline(numeric_steps)
            transformers.append(('numeric', numeric_pipeline, numeric_cols))

            # Log numeric transformer
            numeric_steps_log = [{'name': 'imputer', 'strategy': self.numeric_imputation_strategy}]
            if self.numeric_scaling:
                numeric_steps_log.append({'name': 'scaler', 'type': 'StandardScaler'})

            metadata['transformers'].append({
                'name': 'numeric',
                'columns': numeric_cols,
                'steps': numeric_steps_log
            })

        # Build categorical pipelines (one per column for flexibility)
        for col in categorical_cols:
            cardinality = X[col].nunique()
            col_dtype = X[col].dtype

            # Build pipeline steps
            steps = []

            # Step 1: If boolean, convert to object dtype (SimpleImputer doesn't support bool)
            # Use pandas API to safely check for bool dtype (handles numpy bool and pandas BooleanDtype)
            if pd.api.types.is_bool_dtype(col_dtype):
                steps.append(('bool_to_obj', BoolToObjectConverter()))

            # Step 2: Imputation (most_frequent)
            steps.append(('imputer', SimpleImputer(strategy='most_frequent')))

            # Step 3: Encoding based on cardinality
            encoding_method = None
            if cardinality <= self.categorical_encoding_threshold:
                # One-hot encoding for low cardinality
                steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                encoding_method = 'onehot'
                transformers.append((f'cat_{col}', Pipeline(steps), [col]))
            else:
                # Frequency encoding for high cardinality
                if self.high_cardinality_encoding == "frequency":
                    steps.append(('freq', FrequencyEncoder()))
                    encoding_method = 'frequency'
                    transformers.append((f'cat_{col}', Pipeline(steps), [col]))
                else:
                    # Drop high-cardinality categoricals
                    continue

            # Log categorical transformer (store string representations for JSON serialization)
            metadata['transformers'].append({
                'name': f'categorical_{col}',
                'column': col,
                'dtype': str(col_dtype),
                'cardinality': int(cardinality),
                'steps': [
                    {'name': step[0], 'type': step[1].__class__.__name__ if hasattr(step[1], '__class__') else str(step[1])}
                    for step in steps
                ],
                'encoding_method': encoding_method
            })

        # Create ColumnTransformer
        if len(transformers) == 0:
            # No valid transformers - this will cause issues
            raise ValueError("No valid transformers found. Check your data columns.")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop any columns not explicitly handled
            verbose_feature_names_out=False  # Cleaner feature names
        )

        # Add summary
        metadata['summary'] = {
            'total_input_features': len(X.columns),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'transformers_created': len(transformers)
        }

        return preprocessor, metadata

    def get_feature_names_out(self, preprocessor: ColumnTransformer) -> List[str]:
        """
        Get output feature names after preprocessing.

        Args:
            preprocessor: Fitted ColumnTransformer

        Returns:
            List of feature names after transformation
        """
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception as e:
            # Fallback for older sklearn versions
            return [f"feature_{i}" for i in range(1000)]  # Placeholder
