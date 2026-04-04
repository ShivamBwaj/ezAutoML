"""
Experiment Tracker: Logs all experimental runs to JSON file.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class ExperimentTracker:
    """
    Simple JSON-based experiment tracking.
    Stores all runs with configs, scores, and metadata.
    """

    def __init__(self, db_path: str = "experiments.json"):
        """
        Initialize tracker.

        Args:
            db_path: Path to JSON log file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing experiments from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    self.experiments = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load experiment database: {e}")
                self.experiments = []
        else:
            self.experiments = []

    def _save(self) -> None:
        """Save experiments to disk"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.experiments, f, indent=2)
        except IOError as e:
            print(f"Error saving experiments: {e}")

    def log(
        self,
        iteration: int,
        config: Dict[str, Any],
        score: float,
        metric_name: str,
        model_path: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an experimental run.

        Args:
            iteration: Iteration number
            config: Pipeline configuration (model, params, etc.)
            score: Primary metric score
            metric_name: Name of metric ('accuracy', 'f1', 'neg_rmse', 'r2')
            model_path: Path to saved model file
            extra: Additional metadata to store
        """
        entry = {
            'iteration': iteration,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'config': config,
            'score': float(score),
            'metric_name': metric_name,
            'model_path': model_path
        }
        if extra:
            entry.update(extra)

        self.experiments.append(entry)
        self._save()

    def get_best(
        self,
        metric_name: Optional[str] = None,
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get best experiment result.

        Args:
            metric_name: Filter by metric name (None = any)
            maximize: True if higher score is better, False for lower

        Returns:
            Best experiment entry or None if no experiments
        """
        if not self.experiments:
            return None

        if metric_name:
            filtered = [e for e in self.experiments if e['metric_name'] == metric_name]
        else:
            filtered = self.experiments

        if not filtered:
            return None

        best = max(filtered, key=lambda x: x['score']) if maximize else min(filtered, key=lambda x: x['score'])
        return best

    def get_history(self, n_recent: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get experiment history.

        Args:
            n_recent: If provided, return only N most recent entries

        Returns:
            List of experiment entries
        """
        if n_recent:
            return self.experiments[-n_recent:].copy()
        return self.experiments.copy()

    def get_scores(self, metric_name: Optional[str] = None) -> List[float]:
        """
        Get list of all scores (useful for plotting trends).

        Args:
            metric_name: Filter by metric name

        Returns:
            List of scores in chronological order
        """
        if metric_name:
            return [e['score'] for e in self.experiments if e['metric_name'] == metric_name]
        return [e['score'] for e in self.experiments]

    def recent_scores(self, n: int = 5) -> List[float]:
        """Get N most recent scores"""
        return self.get_scores()[-n:]

    def clear(self) -> None:
        """Clear all experiments (use with caution)"""
        self.experiments = []
        self._save()
