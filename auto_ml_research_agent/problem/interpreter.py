"""
Problem Interpreter: Converts natural language ML problem to structured specification.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from auto_ml_research_agent.llm.groq_client import GroqClient
from auto_ml_research_agent.exceptions import LLMError


class ProblemSpecification(BaseModel):
    """
    Structured ML problem specification.

    Attributes:
        task: "classification" or "regression"
        target_column: Name of the column to predict
        metric: Recommended evaluation metric
            - Classification: "accuracy" or "f1"
            - Regression: "rmse" or "r2"
    """
    task: str = Field(..., description="ML task type: classification or regression")
    target_column: str = Field(..., description="Name of the target column to predict")
    metric: str = Field(..., description="Primary evaluation metric")

    def validate_spec(self) -> None:
        """Validate the specification values"""
        valid_tasks = {"classification", "regression"}
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task '{self.task}'. Must be one of {valid_tasks}")

        if self.task == "classification":
            valid_metrics = {"accuracy", "f1"}
        else:
            valid_metrics = {"rmse", "r2"}
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{self.metric}' for {self.task}. Must be one of {valid_metrics}")


class ProblemInterpreter:
    """Uses LLM to interpret natural language problems into ML specifications"""

    def __init__(self, llm_client: GroqClient):
        """
        Initialize interpreter.

        Args:
            llm_client: Configured GroqClient instance
        """
        self.llm = llm_client

    def interpret(
        self,
        natural_language: str,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> ProblemSpecification:
        """
        Interpret natural language problem description.

        Args:
            natural_language: User's problem description
            dataset_info: Optional dataset context (columns, types, etc.)

        Returns:
            ProblemSpecification with task, target column, and metric

        Raises:
            LLMError: If LLM fails to produce valid response
        """
        dataset_context = ""
        if dataset_info:
            # Extract relevant info for LLM
            columns = dataset_info.get("columns", {})
            col_summary = []
            # Keep prompt compact: send top columns only.
            for col, stats in list(columns.items())[:40]:
                col_summary.append(f"- {col}: {stats.get('dtype', 'unknown')} (missing: {stats.get('missing_pct', 0):.1f}%)")
            dataset_context = f"\nDataset has {len(columns)} columns (showing up to 40):\n" + "\n".join(col_summary)

        prompt = f"""
Given this ML problem description:

"{natural_language}"
{dataset_context}

Determine the following:

1. TASK: Is this a "classification" or "regression" problem?

2. TARGET_COLUMN: Which column should be predicted?
   - If the target column name is mentioned, use that exact name.
   - If the dataset has a common target column (like "target", "label", "price", "class"), suggest that.
   - Otherwise, infer the most reasonable target based on the problem description.

3. METRIC: What metric should be used to evaluate the model?
   - For classification: "accuracy" (balanced classes) or "f1" (imbalanced/precision-recall important)
   - For regression: "rmse" (standard) or "r2" (explained variance)

Consider the dataset context to make informed choices.

Respond with ONLY a JSON object matching this exact schema:
{{
    "task": "classification|regression",
    "target_column": "string",
    "metric": "accuracy|f1|rmse|r2"
}}
"""

        try:
            spec = self.llm.generate_json(prompt, ProblemSpecification)
            spec.validate_spec()
            return spec
        except LLMError as e:
            raise LLMError(f"Failed to interpret problem: {str(e)}") from e
        except Exception as e:
            raise LLMError(f"Unexpected error in problem interpretation: {str(e)}") from e
