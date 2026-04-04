"""
LLM Analyzer: Uses LLM to analyze experiment history and suggest improvements.
"""
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from auto_ml_research_agent.llm.groq_client import GroqClient
from auto_ml_research_agent.exceptions import LLMError


class AnalysisResponse(BaseModel):
    """
    LLM analysis response.

    Attributes:
        issues: Identified problems (overfitting, poor features, etc.)
        suggestions: Concrete improvements to try
    """
    issues: List[str] = Field(default_factory=list, description="Patterns indicating issues")
    suggestions: List[str] = Field(default_factory=list, description="Specific improvement suggestions")


class LLMAnalyzer:
    """
    Uses LLM to analyze experiment history and suggest next steps.
    """

    def __init__(self, llm_client: GroqClient):
        """
        Initialize analyzer.

        Args:
            llm_client: Configured GroqClient
        """
        self.llm = llm_client

    def analyze(
        self,
        history: List[Dict[str, Any]],
        latest_metrics: Dict[str, Any]
    ) -> AnalysisResponse:
        """
        Analyze recent experiments and provide suggestions.

        Args:
            history: List of experiment tracker entries (recent)
            latest_metrics: Most recent evaluation metrics

        Returns:
            AnalysisResponse with issues and suggestions
        """
        # Prepare concise history summary
        history_summary = self._summarize_history(history[-10:] if len(history) > 10 else history)

        prompt = f"""
Analyze this AutoML experiment history to identify patterns and suggest improvements.

Recent experiments:
{history_summary}

Latest metrics:
{json.dumps(latest_metrics, indent=2)}

Look for:
1. **Overfitting**: Large gap between train/val scores, or CV scores with high variance
2. **Underfitting**: All models have similar low performance
3. **Model suitability**: Some models consistently outperforming others
4. **Convergence**: Scores plateauing over recent iterations
5. **Data issues**: Low baseline scores despite good models

Available model types you can suggest:
- Classical: LogisticRegression, RandomForest, GradientBoosting, ExtraTrees, AdaBoost, DecisionTree
- Linear: Ridge, Lasso, ElasticNet, LinearRegression
- Support Vector: SVC (classification), SVR (regression)
- Neighbors: KNeighbors (KNN)
- Naive Bayes: GaussianNB
- Discriminant: LDA, QDA
- Neural Networks: MLPClassifier, MLPRegressor
- Gradient Boosting (histogram): HistGradientBoostingClassifier, HistGradientBoostingRegressor
- Stochastic: SGDClassifier, SGDRegressor, Perceptron, PassiveAggressive
- Robust: RidgeClassifier, TheilSenRegressor, RANSACRegressor, HuberRegressor
- GLM: PoissonRegressor, GammaRegressor, TweedieRegressor
- External: XGBoost, LightGBM, CatBoost

Suggest 3-5 CONCRETE improvements as simple text strings (not objects). Each suggestion should be one sentence with specific details:
- "Try RandomForest with n_estimators=200"
- "Add polynomial features degree 2 to numeric columns"
- "Use StandardScaler instead of RobustScaler"
- "Increase C parameter to 10.0 for SVC"
- "Switch to XGBoost with learning_rate=0.05"

Be specific and actionable. Include parameter values. Avoid vague suggestions like "try better models".
DO NOT use structured objects - just plain text strings.

Respond with JSON:
{{
    "issues": ["identified issue 1", "issue 2", ...],
    "suggestions": ["suggestion 1", "suggestion 2", ...]
}}
"""

        try:
            response = self.llm.generate_json(prompt, AnalysisResponse)
            return response
        except LLMError as e:
            print(f"Warning: LLM analysis failed: {e}")
            # Return safe default
            return AnalysisResponse(
                issues=["LLM analysis unavailable"],
                suggestions=["Continue with current approach", "Try parameter tuning"]
            )

    def _summarize_history(self, history: List[Dict[str, Any]]) -> str:
        """Create concise text summary of experiment history"""
        if not history:
            return "No previous experiments."

        lines = []
        for exp in history:
            iter_num = exp.get('iteration', '?')
            score = exp.get('score', '?')
            metric = exp.get('metric_name', 'unknown')
            model = exp.get('config', {}).get('model_name', 'unknown')
            params = exp.get('config', {}).get('params', {})
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()]) if params else 'default'
            lines.append(f"Iter {iter_num}: model={model}, params=[{param_str}], {metric}={score:.4f}")

        return '\n'.join(lines)
