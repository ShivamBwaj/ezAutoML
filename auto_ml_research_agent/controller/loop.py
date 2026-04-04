"""
Controller Loop: Manages iterative improvement with patience-based stopping.
"""
from typing import Optional, Dict, Any


class ControllerLoop:
    """
    Controls the iterative training loop.
    Stops when no improvement for 'patience' iterations or max iterations reached.
    """

    def __init__(self, patience: int = 5):
        """
        Initialize controller.

        Args:
            patience: Number of iterations without improvement before stopping
        """
        self.patience = patience
        self.best_score: Optional[float] = None
        self.no_improvement_count = 0

    def should_continue(self, current_score: float, maximize: bool = True) -> bool:
        """
        Check if iteration should continue.

        Args:
            current_score: Score from current iteration
            maximize: True if higher score is better, False for lower

        Returns:
            True if should continue, False to stop
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        improved = (current_score > self.best_score) if maximize else (current_score < self.best_score)

        if improved:
            self.best_score = current_score
            self.no_improvement_count = 0
            return True
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                return False
            return True

    def reset(self) -> None:
        """Reset controller state (useful for new runs)"""
        self.best_score = None
        self.no_improvement_count = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'patience_remaining': max(0, self.patience - self.no_improvement_count)
        }
