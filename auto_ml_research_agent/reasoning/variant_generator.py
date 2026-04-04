"""
Variant Generator: Converts LLM suggestions into concrete pipeline configurations.
"""
from typing import List, Dict, Any, Optional
import re
from sklearn.pipeline import Pipeline
from auto_ml_research_agent.pipeline.generator import PipelineGenerator
from auto_ml_research_agent.exceptions import AutoMLError


class VariantGenerator:
    """
    Takes LLM suggestions and generates new pipeline variants.
    Parses natural language suggestions to extract model names and parameters.
    """

    def __init__(self, task: str):
        """
        Initialize variant generator.

        Args:
            task: "classification" or "regression"
        """
        self.task = task
        self.pipeline_gen = PipelineGenerator(task=task)

    def generate(
        self,
        base_config: Dict[str, Any],
        llm_suggestions: List[str],
        preprocessor,
        n_variants: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate new pipeline variants from LLM suggestions.

        Args:
            base_config: Current best config with 'model_name', 'model_params', 'preprocessor'
            llm_suggestions: List of suggestion strings from LLM
            preprocessor: ColumnTransformer to reuse (will be cloned for each variant)
            n_variants: Number of variants to generate

        Returns:
            List of variant configs (each has 'name', 'pipeline', 'params', 'model_name')
        """
        import copy
        variants = []
        base_model = base_config.get('model_name', '')

        # Variant 1: Slight parameter adjustment of current best
        if base_model:
            # Clone preprocessor to avoid state sharing
            preprocessor_clone = copy.deepcopy(preprocessor)
            variant = self._create_param_variant_with_preprocessor(base_config, factor=1.5, preprocessor=preprocessor_clone)
            if variant:
                variants.append(variant)

        # Variants 2-4: Models mentioned in LLM suggestions
        model_suggestions = self._extract_model_names(llm_suggestions)
        for model_name in model_suggestions[:3]:
            try:
                # Clone preprocessor to avoid state sharing
                preprocessor_clone = copy.deepcopy(preprocessor)
                pipeline = self.pipeline_gen.create_pipeline_from_config(
                    preprocessor=preprocessor_clone,
                    model_name=model_name,
                    model_params={}
                )
                variants.append({
                    'name': f"{model_name}_from_llm",
                    'pipeline': pipeline,
                    'params': {},
                    'model_name': model_name
                })
            except AutoMLError as e:
                print(f"Could not create model {model_name}: {e}")
                continue

        # Variant 5+: Parameter variations of base model
        if len(variants) < n_variants:
            for factor in [0.5, 0.75, 2.0]:
                if len(variants) >= n_variants:
                    break
                # Clone preprocessor to avoid state sharing
                preprocessor_clone = copy.deepcopy(preprocessor)
                variant = self._create_param_variant_with_preprocessor(base_config, factor=factor, preprocessor=preprocessor_clone)
                if variant:
                    variants.append(variant)

        return variants[:n_variants]

    def _extract_model_names(self, suggestions: List[str]) -> List[str]:
        """
        Extract sklearn model identifiers from natural language suggestions.

        Args:
            suggestions: List of suggestion strings like "try XGBoost", "increase max_depth"

        Returns:
            List of model names (e.g., ['xgboost', 'randomforest'])
        """
        model_keywords = {
            # Classic sklearn
            'logistic': 'logistic',
            'logisticregression': 'logistic',
            'randomforest': 'randomforest',
            'rf': 'randomforest',
            'gradientboosting': 'gradientboosting',
            'gb': 'gradientboosting',
            'extratrees': 'extratrees',
            'et': 'extratrees',
            'adaboost': 'adaboost',
            'ada': 'adaboost',
            'svc': 'svc',
            'svm': 'svc',
            'supportvector': 'svc',
            'kneighbors': 'kneighbors',
            'knn': 'kneighbors',
            'decisiontree': 'decisiontree',
            'dt': 'decisiontree',
            'gaussiannb': 'gaussiannb',
            'lda': 'lda',
            'lineardiscriminant': 'lda',
            'qda': 'qda',
            'quadraticdiscriminant': 'qda',
            'ridge': 'ridge',
            'lasso': 'lasso',
            'elasticnet': 'elasticnet',
            'linearregression': 'linear',
            'linear': 'linear',
            'svr': 'svr',
            'knr': 'knr',
            'kneighborsregressor': 'knr',
            'decisiontreeregressor': 'decisiontree',
            # Additional sklearn models
            'mlp': 'mlp',
            'mlpclassifier': 'mlp',
            'neuralnetwork': 'mlp',
            'neuralnet': 'mlp',
            'perceptron': 'perceptron',
            'passiveaggressive': 'passiveaggressive',
            'pa': 'passiveaggressive',
            'ridgeclassifier': 'ridgeclassifier',
            'sgdclassifier': 'sgdclassifier',
            'sgd': 'sgdclassifier',
            'stochasticgradient': 'sgdclassifier',
            'histgradientboosting': 'histgradientboosting',
            'hgb': 'histgradientboosting',
            'histgb': 'histgradientboosting',
            'mlpregressor': 'mlp',
            'passiveaggressiveregressor': 'passiveaggressive',
            'sgdregressor': 'svr',  # fallback to SVR-like behavior
            'theilsen': 'theilsen',
            'theilsenregressor': 'theilsen',
            'ransac': 'ransac',
            'ransacregressor': 'ransac',
            'huber': 'huber',
            'huberregressor': 'huber',
            'poisson': 'poisson',
            'poissonregressor': 'poisson',
            'gamma': 'gamma',
            'gammaregressor': 'gamma',
            'tweedie': 'tweedie',
            'tweedieregressor': 'tweedie',
            # External libraries
            'xgboost': 'xgboost',
            'xgb': 'xgboost',
            'xgbclassifier': 'xgboost',
            'xgbregressor': 'xgboost',
            'lightgbm': 'lightgbm',
            'lgbm': 'lightgbm',
            'lgb': 'lightgbm',
            'lgbmclassifier': 'lightgbm',
            'lgbmregressor': 'lightgbm',
            'catboost': 'catboost',
            'cat': 'catboost',
            'catboostclassifier': 'catboost',
            'catboostregressor': 'catboost',
        }

        found_models = []
        for suggestion in suggestions:
            sugg_lower = suggestion.lower()
            # Check each keyword
            for keyword, model_name in model_keywords.items():
                if keyword in sugg_lower and model_name not in found_models:
                    found_models.append(model_name)
                    break  # One model per suggestion

        return found_models

    def _create_param_variant_with_preprocessor(
        self,
        base_config: Dict[str, Any],
        factor: float,
        preprocessor
    ) -> Optional[Dict[str, Any]]:
        """
        Create variant with adjusted parameters using provided preprocessor clone.

        Args:
            base_config: Base configuration
            factor: Multiplier for numeric parameters
            preprocessor: Preprocessor instance to use (cloned)

        Returns:
            New variant config or None if failed
        """
        model_name = base_config.get('model_name')
        base_params = base_config.get('model_params', {}).copy()

        if not model_name or not preprocessor:
            return None

        # Apply factor to numeric parameters
        new_params = {}
        for key, value in base_params.items():
            if isinstance(value, (int, float)) and key not in ['max_iter', 'random_state']:
                new_value = value * factor
                # Round integers if original was integer
                if isinstance(value, int):
                    new_value = int(round(new_value))
                new_params[key] = new_value
            else:
                new_params[key] = value

        # If no params to adjust, add some defaults based on model type
        if not new_params:
            new_params = self._get_default_params(model_name, factor)

        try:
            pipeline = self.pipeline_gen.create_pipeline_from_config(
                preprocessor=preprocessor,
                model_name=model_name,
                model_params=new_params
            )
            return {
                'name': f"{model_name}_factor_{factor:.2f}",
                'pipeline': pipeline,
                'params': new_params,
                'model_name': model_name
            }
        except AutoMLError:
            return None

    def _get_default_params(self, model_name: str, factor: float) -> Dict[str, Any]:
        """Get reasonable default parameters for a model based on factor"""
        model_lower = model_name.lower()

        if 'rf' in model_lower or 'randomforest' in model_lower:
            return {'n_estimators': int(100 * factor), 'max_depth': int(10 * factor) if factor > 1 else 10}
        elif 'et' in model_lower or 'extratrees' in model_lower:
            return {'n_estimators': int(100 * factor), 'max_depth': int(10 * factor) if factor > 1 else 10}
        elif 'ada' in model_lower or 'adaboost' in model_lower:
            return {'n_estimators': int(100 * factor), 'learning_rate': 0.1 * factor}
        elif 'gb' in model_lower or 'gradientboosting' in model_lower:
            return {'learning_rate': 0.1 * factor, 'n_estimators': int(100 * factor)}
        elif 'xgb' in model_lower or 'xgboost' in model_lower:
            return {'n_estimators': int(100 * factor), 'learning_rate': 0.1 * factor, 'max_depth': int(6 * factor) if factor > 1 else 6}
        elif 'lgb' in model_lower or 'lightgbm' in model_lower:
            return {'n_estimators': int(100 * factor), 'learning_rate': 0.1 * factor, 'num_leaves': int(31 * factor)}
        elif 'cat' in model_lower or 'catboost' in model_lower:
            return {'iterations': int(100 * factor), 'learning_rate': 0.1 * factor, 'depth': int(6 * factor) if factor > 1 else 6}
        elif 'logistic' in model_lower or 'svc' in model_lower or 'svm' in model_lower:
            return {'C': 1.0 * factor}
        elif 'ridge' in model_lower or 'lasso' in model_lower or 'elasticnet' in model_lower:
            return {'alpha': 1.0 * factor}
        elif 'knn' in model_lower or 'neighbor' in model_lower:
            n = int(5 / factor) if factor < 1 else int(5 * factor)
            return {'n_neighbors': max(1, n)}
        elif 'mlp' in model_lower:
            # MLP: vary hidden layers and regularization
            if factor < 1:
                return {'hidden_layer_sizes': (50,), 'alpha': 0.001}
            elif factor > 1:
                return {'hidden_layer_sizes': (100, 50), 'alpha': 0.0001}
            else:
                return {'hidden_layer_sizes': (100,)}
        elif 'perceptron' in model_lower:
            return {'penalty': 'l2', 'alpha': 0.0001 * factor}
        elif 'passiveaggressive' in model_lower:
            return {'C': 1.0 * factor, 'max_iter': 1000}
        elif 'ridgeclassifier' in model_lower:
            return {'alpha': 1.0 * factor}
        elif 'sgd' in model_lower:
            return {'loss': 'log_loss' if self.task == 'classification' else 'squared_error', 'alpha': 0.0001 * factor, 'learning_rate': 'optimal'}
        elif 'histgradientboosting' in model_lower:
            return {'learning_rate': 0.1 * factor, 'max_iter': int(100 * factor), 'max_depth': int(10 * factor) if factor > 1 else None}
        elif 'theilsen' in model_lower:
            return {'max_subpopulation': max(10, int(100 / factor))}
        elif 'ransac' in model_lower:
            return {'min_samples': max(0.1, 0.5 * factor), 'residual_threshold': None}
        elif 'huber' in model_lower:
            return {'epsilon': 1.35, 'alpha': 1.0 * factor}
        elif 'poisson' in model_lower or 'gamma' in model_lower or 'tweedie' in model_lower:
            return {'alpha': 1.0 * factor, 'max_iter': 1000}
        else:
            return {}
