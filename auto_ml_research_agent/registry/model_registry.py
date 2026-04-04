"""
Model Registry: Saves and manages best models.
"""
import joblib
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from auto_ml_research_agent.exceptions import AutoMLError


class ModelRegistry:
    """
    Manages saved models and their metadata.
    """

    def __init__(self, registry_dir: str = "models"):
        """
        Initialize registry.

        Args:
            registry_dir: Directory to store models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load registry metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                if 'models' not in self.metadata:
                    self.metadata['models'] = []
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load registry metadata: {e}")
                self.metadata = {'models': []}
        else:
            self.metadata = {'models': []}

    def _save_metadata(self) -> None:
        """Save registry metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            print(f"Error saving registry metadata: {e}")

    def save_best(
        self,
        pipeline,
        score: float,
        metric: str,
        config: Dict[str, Any],
        preprocessing_metadata: Optional[Dict[str, Any]] = None,
        path_suffix: str = "best_model"
    ) -> str:
        """
        Save best model with metadata.

        Args:
            pipeline: Trained sklearn pipeline
            score: Metric score achieved
            metric: Metric name
            config: Configuration used
            preprocessing_metadata: Optional preprocessing transformation details
            path_suffix: Suffix for model filename

        Returns:
            Path to saved model file
        """
        model_path = self.registry_dir / f"{path_suffix}.pkl"

        try:
            joblib.dump(pipeline, model_path)
        except Exception as e:
            raise AutoMLError(f"Failed to save model: {e}") from e

        entry = {
            'path': str(model_path),
            'score': float(score),
            'metric': metric,
            'config': config,
            'saved_at': datetime.utcnow().isoformat() + 'Z'
        }

        # Include preprocessing metadata if provided
        if preprocessing_metadata:
            entry['preprocessing'] = preprocessing_metadata

        # Remove old entries with same path to avoid duplicates
        self.metadata['models'] = [m for m in self.metadata['models'] if m['path'] != str(model_path)]
        self.metadata['models'].append(entry)
        self._save_metadata()

        return str(model_path)

    def load_best(self) -> Optional[Any]:
        """
        Load the best model (highest score) from registry.

        Returns:
            Loaded pipeline or None if no models exist
        """
        if not self.metadata['models']:
            return None

        # Find best by score (assumes maximize)
        best_entry = max(self.metadata['models'], key=lambda x: x['score'])
        model_path = Path(best_entry['path'])

        if not model_path.exists():
            print(f"Warning: Model file not found: {model_path}")
            return None

        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_best_info(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata for best model.

        Returns:
            Dictionary with best model info or None
        """
        if not self.metadata['models']:
            return None
        return max(self.metadata['models'], key=lambda x: x['score'])

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return self.metadata['models'].copy()

    def clear(self) -> None:
        """Clear all models (use with caution)"""
        # Delete model files
        for entry in self.metadata['models']:
            try:
                Path(entry['path']).unlink(missing_ok=True)
            except:
                pass
        self.metadata['models'] = []
        self._save_metadata()
