"""
LLM Edge Detector: Uses LLM to identify dataset issues and preprocessing suggestions.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from auto_ml_research_agent.llm.groq_client import GroqClient
from auto_ml_research_agent.exceptions import LLMError


class EdgeDetectionResponse(BaseModel):
    """
    LLM response containing issues and suggestions.

    Attributes:
        issues: List of identified problems (e.g., high missing data, leakage)
        suggestions: List of recommended preprocessing improvements
    """
    issues: List[str] = Field(default_factory=list, description="Identified data issues")
    suggestions: List[str] = Field(default_factory=list, description="Preprocessing improvement suggestions")


class LLMEdgeDetector:
    """
    Uses LLM to analyze dataset profile and identify potential issues.
    """

    def __init__(self, llm_client: GroqClient):
        """
        Initialize edge detector.

        Args:
            llm_client: Configured GroqClient
        """
        self.llm = llm_client

    def detect(
        self,
        profiler_stats: Dict[str, Any],
        problem_spec: Dict[str, Any]
    ) -> EdgeDetectionResponse:
        """
        Analyze dataset profile for edge cases and issues.

        Args:
            profiler_stats: Output from DataProfiler.profile()
            problem_spec: ML problem specification

        Returns:
            EdgeDetectionResponse with issues and suggestions
        """
        # Summarize dataset for LLM
        summary = self._summarize_profile(profiler_stats)

        prompt = f"""
Analyze this dataset profile for ML task: {problem_spec.get('task', 'unknown')}
Target column: {problem_spec.get('target_column', 'unknown')}

{summary}

Identify potential issues that could affect model performance:
- High missing data percentages
- Extreme class imbalance (for classification)
- Target leakage (features that shouldn't be available)
- Outliers or data quality problems
- Encoding challenges (high cardinality, rare categories)
- Feature scaling needs
- Skewed distributions

Also suggest preprocessing improvements:
- Specific transformations (log, sqrt, etc.)
- Outlier handling strategies
- Different imputation methods
- Feature engineering ideas
- Handling of categorical variables

Be specific and actionable. Respond with JSON:

{{
    "issues": ["issue1", "issue2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...]
}}

Limit to top 3-5 most important items in each category.
"""

        try:
            response = self.llm.generate_json(prompt, EdgeDetectionResponse)
            return response
        except LLMError as e:
            print(f"Warning: LLM edge detection failed: {e}")
            # Return empty response on failure - don't block pipeline
            return EdgeDetectionResponse(issues=[], suggestions=[])

    def _summarize_profile(self, profiler_stats: Dict[str, Any]) -> str:
        """Create concise summary of profile stats for LLM prompt"""
        lines = []
        lines.append(f"Dataset: {profiler_stats.get('n_rows', 0)} rows, {profiler_stats.get('n_cols', 0)} columns")
        lines.append("\nColumn statistics:")

        for col, stats in profiler_stats.get('columns', {}).items():
            dtype = stats.get('dtype', 'unknown')
            missing = stats.get('missing_pct', 0)
            unique = stats.get('unique_count', 0)
            unique_pct = stats.get('unique_pct', 0)
            samples = stats.get('sample_values', [])[:3]
            sample_str = ', '.join([str(s) for s in samples])

            lines.append(f"- {col}: {dtype}, missing={missing:.1f}%, unique={unique} ({unique_pct:.1f}%), samples=[{sample_str}]")

        return '\n'.join(lines)
