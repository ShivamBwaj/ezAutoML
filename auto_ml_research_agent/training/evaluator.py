"""
Training Evaluator: Extracts normalized scores from trainer results for comparison.
"""
from typing import Dict, Any, Optional


class TrainingEvaluator:
    """
    Evaluates training results and extracts a normalized score for comparison.
    Higher is always better (negative RMSE for regression).
    """

    @staticmethod
    def evaluate_variant(trainer_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract primary metric score from trainer result.

        Args:
            trainer_result: Output from Trainer.train()

        Returns:
            Dictionary with normalized 'score' (higher is better), 'metric_name',
            'all_metrics', and dataset size info
        """
        metrics = trainer_result.get('metrics', {})
        method = trainer_result.get('validation_method', 'holdout')

        # Determine primary metric and score (higher is better)
        if 'accuracy' in metrics:
            score = float(metrics['accuracy'])
            metric_name = 'accuracy'
        elif 'f1' in metrics and metrics['f1'] is not None:
            score = float(metrics['f1'])
            metric_name = 'f1'
        elif 'rmse' in metrics:
            # RMSE: lower is better, so use negative for maximization
            score = -float(metrics['rmse'])
            metric_name = 'neg_rmse'
        elif 'r2' in metrics:
            score = float(metrics['r2'])
            metric_name = 'r2'
        else:
            score = float('-inf')
            metric_name = None

        result = {
            'score': score,
            'metric_name': metric_name,
            'all_metrics': metrics,
            'validation_method': method
        }

        # Add size info
        if method == 'holdout':
            result['train_size'] = len(trainer_result.get('y_train', []))
            result['val_size'] = len(trainer_result.get('y_val', []))
        else:  # CV or simple_split
            result['train_size'] = trainer_result.get('n_samples', 0)
            result['val_size'] = None

        return result
