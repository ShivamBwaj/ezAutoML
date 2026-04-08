"""
Pipeline Generator: Creates sklearn pipeline configurations with multiple variants.
"""
from typing import List, Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from auto_ml_research_agent.exceptions import AutoMLError


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier wrapper that supports non-numeric class labels.
    It encodes labels during fit and decodes them during predict.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_ = None
        self._label_encoder = LabelEncoder()

    def fit(self, X, y):
        import pandas as pd
        from xgboost import XGBClassifier

        y_series = pd.Series(y).astype(str).str.strip()
        y_encoded = self._label_encoder.fit_transform(y_series)
        self.model_ = XGBClassifier(**self.kwargs)
        self.model_.fit(X, y_encoded)
        self.classes_ = self._label_encoder.classes_
        return self

    def predict(self, X):
        y_encoded = self.model_.predict(X)
        return self._label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def get_params(self, deep=True):
        return dict(self.kwargs)

    def set_params(self, **params):
        self.kwargs.update(params)
        return self


class PipelineGenerator:
    """
    Generates multiple pipeline configurations with diverse models.
    Supports dynamic instantiation of any sklearn model by name.
    """

    # Curated top-10 model mapping for classification
    CLASSIFICATION_MODELS = {
        'logistic': ('LogisticRegression', {'max_iter': 1000}),
        'logisticregression': ('LogisticRegression', {'max_iter': 1000}),
        'randomforest': ('RandomForestClassifier', {}),
        'rf': ('RandomForestClassifier', {}),
        'gradientboosting': ('GradientBoostingClassifier', {}),
        'gb': ('GradientBoostingClassifier', {}),
        'extratrees': ('ExtraTreesClassifier', {}),
        'et': ('ExtraTreesClassifier', {}),
        'adaboost': ('AdaBoostClassifier', {}),
        'ada': ('AdaBoostClassifier', {}),
        'svc': ('SVC', {'probability': True}),
        'svm': ('SVC', {'probability': True}),
        'kneighbors': ('KNeighborsClassifier', {}),
        'knn': ('KNeighborsClassifier', {}),
        'decisiontree': ('DecisionTreeClassifier', {}),
        'dt': ('DecisionTreeClassifier', {}),
        'gaussiannb': ('GaussianNB', {}),
        'histgradientboosting': ('HistGradientBoostingClassifier', {'random_state': 42}),
        'xgboost': ('XGBClassifier', {}),
        'xgb': ('XGBClassifier', {}),
    }

    # Curated top-10 model mapping for regression
    REGRESSION_MODELS = {
        'linear': ('LinearRegression', {}),
        'linearregression': ('LinearRegression', {}),
        'ridge': ('Ridge', {}),
        'lasso': ('Lasso', {}),
        'randomforest': ('RandomForestRegressor', {}),
        'rf': ('RandomForestRegressor', {}),
        'gradientboosting': ('GradientBoostingRegressor', {}),
        'gb': ('GradientBoostingRegressor', {}),
        'extratrees': ('ExtraTreesRegressor', {}),
        'et': ('ExtraTreesRegressor', {}),
        'adaboost': ('AdaBoostRegressor', {}),
        'ada': ('AdaBoostRegressor', {}),
        'svr': ('SVR', {}),
        'knr': ('KNeighborsRegressor', {}),
        'kneighborsregressor': ('KNeighborsRegressor', {}),
        'decisiontree': ('DecisionTreeRegressor', {}),
        'dt': ('DecisionTreeRegressor', {}),
        'histgradientboosting': ('HistGradientBoostingRegressor', {}),
        'xgboost': ('XGBRegressor', {}),
        'xgb': ('XGBRegressor', {}),
    }

    def __init__(self, task: str, random_state: int = 42):
        """
        Initialize generator.

        Args:
            task: "classification" or "regression"
            random_state: Random seed for reproducibility
        """
        if task not in {"classification", "regression"}:
            raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'regression'")

        self.task = task
        self.random_state = random_state

    def _instantiate_model(self, model_name: str, params: Optional[Dict] = None) -> Optional[BaseEstimator]:
        """
        Dynamically instantiate sklearn model by name.

        Args:
            model_name: Model identifier (e.g., "rf", "logistic", "SVC")
            params: Additional parameters to pass to model constructor

        Returns:
            Sklearn estimator instance or None if not found
        """
        params = params or {}
        model_map = self.CLASSIFICATION_MODELS if self.task == "classification" else self.REGRESSION_MODELS
        name_lower = model_name.lower().replace(' ', '').replace('-', '')

        if name_lower not in model_map:
            # Try fuzzy matching (substring)
            for key in model_map:
                if key in name_lower or name_lower in key:
                    model_name = key
                    break
            else:
                return None

        class_name, default_params = model_map[model_name]

        # Merge defaults with provided params
        merged_params = {**default_params, **params}

        # Add random_state if model supports it
        try:
            import inspect
            sig = inspect.signature(getattr(self._get_sklearn_module(), class_name).__init__)
            if 'random_state' in sig.parameters:
                merged_params['random_state'] = self.random_state
        except:
            pass

        # Instantiate
        try:
            cls = self._get_class(class_name)
            if cls is None:
                return None
            return cls(**merged_params)
        except Exception as e:
            print(f"Failed to instantiate {class_name} with params {merged_params}: {e}")
            return None

    def _get_sklearn_module(self):
        """Get appropriate sklearn module based on task"""
        if self.task == "classification":
            from sklearn import linear_model, ensemble, svm, neighbors, tree, naive_bayes, discriminant_analysis, neural_network, linear_model as sk_linear
            return {
                # Classic sklearn
                'LogisticRegression': linear_model.LogisticRegression,
                'RandomForestClassifier': ensemble.RandomForestClassifier,
                'GradientBoostingClassifier': ensemble.GradientBoostingClassifier,
                'ExtraTreesClassifier': ensemble.ExtraTreesClassifier,
                'AdaBoostClassifier': ensemble.AdaBoostClassifier,
                'SVC': svm.SVC,
                'KNeighborsClassifier': neighbors.KNeighborsClassifier,
                'DecisionTreeClassifier': tree.DecisionTreeClassifier,
                'GaussianNB': naive_bayes.GaussianNB,
                'LinearDiscriminantAnalysis': discriminant_analysis.LinearDiscriminantAnalysis,
                'QuadraticDiscriminantAnalysis': discriminant_analysis.QuadraticDiscriminantAnalysis,
                # Additional models
                'MLPClassifier': neural_network.MLPClassifier,
                'Perceptron': sk_linear.Perceptron,
                'PassiveAggressiveClassifier': sk_linear.PassiveAggressiveClassifier,
                'RidgeClassifier': sk_linear.RidgeClassifier,
                'SGDClassifier': sk_linear.SGDClassifier,
                'HistGradientBoostingClassifier': ensemble.HistGradientBoostingClassifier,
            }
        else:
            from sklearn import linear_model, ensemble, svm, neighbors, tree, neural_network
            return {
                # Classic sklearn
                'LinearRegression': linear_model.LinearRegression,
                'Ridge': linear_model.Ridge,
                'Lasso': linear_model.Lasso,
                'ElasticNet': linear_model.ElasticNet,
                'RandomForestRegressor': ensemble.RandomForestRegressor,
                'GradientBoostingRegressor': ensemble.GradientBoostingRegressor,
                'ExtraTreesRegressor': ensemble.ExtraTreesRegressor,
                'AdaBoostRegressor': ensemble.AdaBoostRegressor,
                'SVR': svm.SVR,
                'KNeighborsRegressor': neighbors.KNeighborsRegressor,
                'DecisionTreeRegressor': tree.DecisionTreeRegressor,
                # Additional models
                'MLPRegressor': neural_network.MLPRegressor,
                'PassiveAggressiveRegressor': linear_model.PassiveAggressiveRegressor,
                'SGDRegressor': linear_model.SGDRegressor,
                'TheilSenRegressor': linear_model.TheilSenRegressor,
                'RANSACRegressor': linear_model.RANSACRegressor,
                'HuberRegressor': linear_model.HuberRegressor,
                'PoissonRegressor': linear_model.PoissonRegressor,
                'GammaRegressor': linear_model.GammaRegressor,
                'TweedieRegressor': linear_model.TweedieRegressor,
                'HistGradientBoostingRegressor': ensemble.HistGradientBoostingRegressor,
            }

    def _get_class(self, class_name: str):
        """Get class from sklearn or external libraries (xgboost, lightgbm, catboost)"""
        # First try sklearn
        modules = self._get_sklearn_module()
        cls = modules.get(class_name)
        if cls is not None:
            return cls

        # Try external libraries
        external_map = {
            'XGBClassifier': ('xgboost', 'xgb.XGBClassifier'),
            'XGBRegressor': ('xgboost', 'xgb.XGBRegressor'),
            'LGBMClassifier': ('lightgbm', 'lgb.LGBMClassifier'),
            'LGBMRegressor': ('lightgbm', 'lgb.LGBMRegressor'),
            'CatBoostClassifier': ('catboost', 'cb.CatBoostClassifier'),
            'CatBoostRegressor': ('catboost', 'cb.CatBoostRegressor'),
        }

        if class_name == 'XGBClassifier':
            try:
                import xgboost  # noqa: F401
                return LabelEncodedXGBClassifier
            except ImportError:
                print("[WARN]  xgboost not installed. Install with: pip install xgboost")
                return None

        if class_name in external_map:
            package, import_path = external_map[class_name]
            try:
                # Dynamic import
                parts = import_path.split('.')
                module = __import__(package, fromlist=[parts[1]])
                return getattr(module, parts[1])
            except ImportError:
                print(f"[WARN]  {package} not installed. Install with: pip install {package}")
                return None
            except AttributeError:
                print(f"[WARN]  Could not find {class_name} in {package}")
                return None

        return None

    def generate_variants(
        self,
        preprocessor,
        n_variants: int = 5,
        custom_models: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse pipeline configurations.

        Args:
            preprocessor: Fitted or unfitted ColumnTransformer
            n_variants: Number of variants to generate
            custom_models: Optional list of specific model names to use

        Returns:
            List of variant configs with 'name', 'pipeline', 'params', 'model_name'
        """
        variants = []

        # Determine which models to use
        if custom_models:
            model_names = custom_models[:n_variants]
        else:
            # Use default diverse set with expanded model options
            if self.task == "classification":
                model_names = ['logistic', 'randomforest', 'gradientboosting', 'svc', 'kneighbors', 'extratrees', 'adaboost', 'decisiontree', 'xgboost', 'histgradientboosting']
            else:
                model_names = ['linear', 'ridge', 'lasso', 'randomforest', 'gradientboosting', 'svr', 'knr', 'extratrees', 'xgboost', 'histgradientboosting']

        # Generate one variant per model type
        for model_name in model_names[:n_variants]:
            model = self._instantiate_model(model_name)
            if model is None:
                continue

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            variants.append({
                'name': f"{model_name}_default",
                'pipeline': pipeline,
                'params': {},
                'model_name': model_name
            })

        # If we need more variants, create parameter variations
        if len(variants) < n_variants:
            for base_variant in variants:
                if len(variants) >= n_variants:
                    break
                param_variants = self._create_param_variants(base_variant)
                for pv in param_variants:
                    if len(variants) >= n_variants:
                        break
                    variants.append(pv)

        return variants[:n_variants]

    def _create_param_variants(self, base_variant: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create parameter-tuned variants of a base pipeline.

        Args:
            base_variant: Base variant configuration

        Returns:
            List of variant configs with adjusted parameters
        """
        model_name = base_variant['model_name']
        preprocessor = base_variant['pipeline'].named_steps['preprocessor']
        variants = []

        # Define parameter grids for common models
        param_configs = {
            # Classic sklearn
            'logistic': [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
            'rf': [{'n_estimators': 50}, {'n_estimators': 100}, {'n_estimators': 200}],
            'gb': [{'learning_rate': 0.01}, {'learning_rate': 0.1}, {'n_estimators': 200}],
            'et': [{'n_estimators': 50}, {'n_estimators': 100}, {'n_estimators': 200}],
            'ada': [{'n_estimators': 50}, {'n_estimators': 100}, {'learning_rate': 0.5}],
            'ridge': [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}],
            'lasso': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1.0}],
            'elasticnet': [{'alpha': 0.1}, {'alpha': 1.0}, {'l1_ratio': 0.5}],
            'svc': [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
            'svr': [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
            'knn': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 10}],
            # Additional sklearn models
            'mlp': [
                {'hidden_layer_sizes': (50,)},
                {'hidden_layer_sizes': (100,)},
                {'hidden_layer_sizes': (100, 50), 'alpha': 0.001}
            ],
            'perceptron': [{'penalty': 'l2', 'alpha': 0.0001}, {'penalty': 'l1', 'alpha': 0.0001}, {'max_iter': 2000}],
            'passiveaggressive': [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
            'ridgeclassifier': [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}],
            'sgdclassifier': [
                {'loss': 'log_loss', 'learning_rate': 'optimal'},
                {'loss': 'hinge', 'alpha': 0.0001},
                {'loss': 'log_loss', 'learning_rate': 'constant', 'eta0': 0.01}
            ],
            'histgradientboosting': [
                {'learning_rate': 0.01, 'max_iter': 100},
                {'learning_rate': 0.1, 'max_iter': 100},
                {'max_depth': 7, 'l2_regularization': 1.0}
            ],
            # External libraries
            'xgb': [
                {'n_estimators': 50, 'learning_rate': 0.05},
                {'n_estimators': 100, 'learning_rate': 0.1},
                {'max_depth': 3, 'n_estimators': 100}
            ],
            'lgbm': [
                {'n_estimators': 50, 'learning_rate': 0.05},
                {'n_estimators': 100, 'learning_rate': 0.1},
                {'num_leaves': 31, 'n_estimators': 100}
            ],
            'cat': [
                {'learning_rate': 0.05, 'depth': 4},
                {'learning_rate': 0.1, 'depth': 6},
                {'l2_leaf_reg': 3, 'depth': 6}
            ],
        }

        key = model_name.lower()
        param_list = param_configs.get(key, [{}])

        for idx, params in enumerate(param_list[:3]):  # Max 3 param variants
            model = self._instantiate_model(model_name, params)
            if model is None:
                continue

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            variants.append({
                'name': f"{model_name}_params_{idx}",
                'pipeline': pipeline,
                'params': params,
                'model_name': model_name
            })

        return variants

    def create_pipeline_from_config(
        self,
        preprocessor,
        model_name: str,
        model_params: Optional[Dict] = None
    ) -> Pipeline:
        """
        Create a single pipeline from model name and params.

        Args:
            preprocessor: ColumnTransformer
            model_name: Model identifier
            model_params: Parameters for model

        Returns:
            sklearn Pipeline
        """
        model = self._instantiate_model(model_name, model_params)
        if model is None:
            raise AutoMLError(f"Failed to instantiate model: {model_name}")

        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
