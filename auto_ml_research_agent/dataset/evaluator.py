"""
Dataset Evaluator: Quick baseline evaluation of dataset quality.
"""
from typing import Dict, Any
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from auto_ml_research_agent.exceptions import DatasetError


class DatasetEvaluator:
    """
    Evaluates dataset suitability for ML tasks using quick baselines.
    """

    def __init__(
        self,
        target_column: str,
        task: str,
        metric: str,
        cv_folds: int = 3
    ):
        """
        Initialize evaluator.

        Args:
            target_column: Name of the target column
            task: "classification" or "regression"
            metric: Primary metric to optimize (accuracy, f1, rmse, r2)
            cv_folds: Number of cross-validation folds for small datasets
        """
        self.target_column = target_column
        self.task = task
        self.metric = metric
        self.cv_folds = cv_folds

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run quick evaluation of dataset quality.

        Args:
            df: Input DataFrame

        Returns:
            Evaluation result with suitability flag and baseline score
        """
        n_rows, n_cols = df.shape

        # Check 1: Minimum size
        if n_rows < 100:
            return {
                "suitable": False,
                "reason": f"Too few rows ({n_rows} < 100 minimum)",
                "n_rows": n_rows,
                "n_cols": n_cols
            }

        # Check 2: Target column exists
        if self.target_column not in df.columns:
            # Try case-insensitive match
            cols_lower = {c.lower(): c for c in df.columns}
            if self.target_column.lower() in cols_lower:
                # Update to actual column name
                actual_target = cols_lower[self.target_column.lower()]
                # We'll note this but continue with original for now
                return {
                    "suitable": False,
                    "reason": f"Target column '{self.target_column}' not found (did you mean '{actual_target}'?)",
                    "n_rows": n_rows,
                    "n_cols": n_cols
                }
            return {
                "suitable": False,
                "reason": f"Target column '{self.target_column}' not found in dataset",
                "n_rows": n_rows,
                "n_cols": n_cols,
                "available_columns": list(df.columns)
            }

        # Check 3: Target has variance
        target_series = df[self.target_column]
        if target_series.nunique() < 2:
            return {
                "suitable": False,
                "reason": f"Target column has only {target_series.nunique()} unique value(s)",
                "n_rows": n_rows,
                "n_cols": n_cols
            }

        # Check 4: Missing data
        total_cells = n_rows * n_cols
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / total_cells * 100) if total_cells > 0 else 0.0

        if missing_pct > 70:
            return {
                "suitable": False,
                "reason": f"Too much missing data: {missing_pct:.1f}%",
                "n_rows": n_rows,
                "n_cols": n_cols,
                "missing_pct": missing_pct
            }

        # Check 5: Try to train a quick baseline model
        try:
            # Prepare features: only numeric for quick baseline
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            # Select only numeric features, impute median
            X = X.select_dtypes(include=['number'])
            if X.shape[1] == 0:
                return {
                    "suitable": False,
                    "reason": "No numeric features available for baseline",
                    "n_rows": n_rows,
                    "n_cols": n_cols
                }

            # Simple imputation
            X = X.fillna(X.median(numeric_only=True))

            # Check for any remaining NaNs
            if X.isnull().any().any():
                X = X.fillna(0)  # Last resort

            # Choose model
            if self.task == "classification":
                model = LogisticRegression(max_iter=1000, random_state=42)
                scoring = "accuracy"
            else:
                model = LinearRegression()
                scoring = "neg_root_mean_squared_error"

            # Use cross-validation for quick estimate
            cv_folds = min(self.cv_folds, n_rows // 10)  # Ensure enough samples per fold
            if cv_folds < 2:
                # Not enough for CV, use single train-test split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if self.task == "classification" else None
                )
                model.fit(X_train, y_train)
                if self.task == "classification":
                    baseline_score = model.score(X_val, y_val)
                else:
                    from sklearn.metrics import mean_squared_error
                    y_pred = model.predict(X_val)
                    baseline_score = -np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
                baseline_score = float(np.mean(scores))
                if scoring == "neg_root_mean_squared_error":
                    baseline_score = -baseline_score  # Make positive RMSE

            return {
                "suitable": True,
                "baseline_score": float(baseline_score),
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_features": X.shape[1],
                "missing_pct": float(missing_pct),
                "target_unique_values": int(target_series.nunique()),
                "validation_method": "cv" if cv_folds >= 2 else "holdout"
            }

        except Exception as e:
            return {
                "suitable": False,
                "reason": f"Baseline training failed: {str(e)}",
                "n_rows": n_rows,
                "n_cols": n_cols
            }
