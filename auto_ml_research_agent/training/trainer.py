"""
Trainer: Trains sklearn pipelines with adaptive CV/holdout validation.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from auto_ml_research_agent.exceptions import AutoMLError


class Trainer:
    """
    Trains machine learning pipelines using appropriate validation strategy:
    - Small datasets (< cv_threshold): 3-fold cross-validation
    - Large datasets: single holdout split
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_threshold: int = 500
    ):
        """
        Initialize trainer.

        Args:
            test_size: Fraction for holdout validation (0.0-1.0)
            random_state: Random seed for reproducibility
            cv_threshold: Use CV if n_rows < threshold, else holdout
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_threshold = cv_threshold

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        pipeline: Pipeline,
        task: str
    ) -> Dict[str, Any]:
        """
        Train pipeline and compute metrics.

        Args:
            df: Training DataFrame
            target_column: Name of target column
            pipeline: Sklearn Pipeline (preprocessor + model)
            task: "classification" or "regression"

        Returns:
            Dictionary with trained pipeline, metrics, and split info
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        n_rows = len(df)

        # Choose validation strategy based on dataset size
        if n_rows < self.cv_threshold:
            result = self._train_with_cv(X, y, pipeline, task)
        else:
            result = self._train_with_holdout(X, y, pipeline, task, target_column)

        result['n_samples'] = n_rows
        return result

    def _train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        task: str
    ) -> Dict[str, Any]:
        """Train using cross-validation (for small datasets)"""
        cv_folds = min(3, len(X) // 20)  # At least 20 samples per fold
        if cv_folds < 2:
            # Too small for CV, fall back to simple split
            return self._train_with_simple_split(X, y, pipeline, task)

        # Determine scoring
        if task == "classification":
            scoring = "accuracy"
            # Also compute F1
            f1_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='f1_weighted', error_score='raise')
            f1_score = float(np.mean(f1_scores))
        else:
            scoring = "neg_root_mean_squared_error"
            f1_score = None

        # Main CV score
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=scoring, error_score='raise')

        # Convert to positive metrics
        if scoring == "neg_root_mean_squared_error":
            primary_score = float(np.mean(-cv_scores))
            metric_name = "rmse"
        else:
            primary_score = float(np.mean(cv_scores))
            metric_name = "accuracy"

        # Fit on full data for deployment
        pipeline.fit(X, y)

        metrics = {'rmse' if task == 'regression' else 'accuracy': primary_score}
        if f1_score is not None:
            metrics['f1'] = f1_score

        return {
            'pipeline': pipeline,
            'X': X,
            'y': y,
            'metrics': metrics,
            'validation_method': 'cv',
            'cv_folds': cv_folds,
            'metric_name': metric_name,
            'score': primary_score  # For easy access by controller
        }

    def _train_with_holdout(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        task: str,
        target_column: str
    ) -> Dict[str, Any]:
        """Train with holdout validation (for large datasets)"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if task == "classification" else None
        )

        print(f"    [Trainer] Fitting pipeline with {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)
        print(f"    [Trainer] Pipeline fitted, predicting on validation...")
        y_pred = pipeline.predict(X_val)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

        metrics = {}
        if task == "classification":
            metrics['accuracy'] = float(accuracy_score(y_val, y_pred))
            try:
                # Weighted F1 for multi-class, binary for binary
                if len(np.unique(y)) == 2:
                    metrics['f1'] = float(f1_score(y_val, y_pred))
                else:
                    metrics['f1'] = float(f1_score(y_val, y_pred, average='weighted'))
            except:
                metrics['f1'] = None
            primary_score = metrics['accuracy']
            metric_name = 'accuracy'
        else:
            mse = mean_squared_error(y_val, y_pred)
            metrics['rmse'] = float(np.sqrt(mse))
            metrics['r2'] = float(r2_score(y_val, y_pred))
            primary_score = metrics['rmse']  # Will be negated by evaluator
            metric_name = 'rmse'

        return {
            'pipeline': pipeline,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'y_pred': y_pred,
            'metrics': metrics,
            'validation_method': 'holdout',
            'metric_name': metric_name,
            'score': primary_score
        }

    def _train_with_simple_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
        task: str
    ) -> Dict[str, Any]:
        """Fallback: simple 50-50 split for tiny datasets"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.5, random_state=self.random_state
        )
        pipeline.fit(X_train, y_train)

        if task == "classification":
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_val, pipeline.predict(X_val))
            metrics = {'accuracy': score}
            metric_name = 'accuracy'
        else:
            from sklearn.metrics import mean_squared_error
            import numpy as np
            mse = mean_squared_error(y_val, pipeline.predict(X_val))
            score = np.sqrt(mse)
            metrics = {'rmse': score}
            metric_name = 'rmse'

        return {
            'pipeline': pipeline,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'metrics': metrics,
            'validation_method': 'simple_split',
            'metric_name': metric_name,
            'score': score
        }
