"""
Feature selection utilities (LLM-guided + model-driven + hybrid).
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

from auto_ml_research_agent.llm.groq_client import GroqClient


class FeatureSelectionResponse(BaseModel):
    included_features: List[str] = Field(default_factory=list)
    excluded_features: List[str] = Field(default_factory=list)
    reasoning: str = ""


class LLMFeatureSelector:
    """Use LLM to propose a feature subset from profiling metadata."""

    def __init__(self, llm_client: GroqClient):
        self.llm = llm_client

    def select_features(
        self,
        profiler_stats: Dict[str, Any],
        problem_spec: Dict[str, Any],
        candidate_features: List[str],
        correlation_summary: Optional[Dict[str, Any]] = None,
    ) -> FeatureSelectionResponse:
        compact_profile = {
            "n_rows": profiler_stats.get("n_rows"),
            "n_cols": profiler_stats.get("n_cols"),
            "columns": {
                k: {
                    "dtype": v.get("dtype"),
                    "missing_pct": v.get("missing_pct"),
                    "unique_count": v.get("unique_count"),
                    "sample_values": (v.get("sample_values") or [])[:3],
                }
                for k, v in list((profiler_stats.get("columns") or {}).items())[:40]
            },
        }
        compact_corr = {}
        if correlation_summary:
            compact_corr = {
                "highly_correlated_pairs": (correlation_summary.get("highly_correlated_pairs") or [])[:25]
            }

        prompt = f"""
You are selecting predictive features for an AutoML run.
Task context: {problem_spec}
Candidate features: {candidate_features}
Profile summary (compact): {compact_profile}
Correlation summary (compact): {compact_corr}

Return JSON with:
- included_features: list of chosen feature names
- excluded_features: list of dropped feature names
- reasoning: short rationale

Keep 40-80% of input features unless there is clear reason to be stricter.
Only include names from candidate_features.
"""
        response = self.llm.generate_json(prompt, FeatureSelectionResponse)
        allowed = set(candidate_features)
        included = [f for f in response.included_features if f in allowed]
        if not included:
            included = candidate_features
        excluded = [f for f in candidate_features if f not in included]
        return FeatureSelectionResponse(
            included_features=included,
            excluded_features=excluded,
            reasoning=response.reasoning,
        )


class ModelFeatureSelector:
    """Data-driven selector using mutual information."""

    def select_features(
        self, df: pd.DataFrame, target_column: str, task: str, max_features: Optional[int] = None
    ) -> List[str]:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if X_encoded[col].dtype == "object" or str(X_encoded[col].dtype) == "category":
                X_encoded[col] = X_encoded[col].astype(str).factorize()[0]
            elif str(X_encoded[col].dtype) == "bool":
                X_encoded[col] = X_encoded[col].astype(int)
        X_encoded = X_encoded.fillna(0)
        k = max_features or max(1, min(len(X_encoded.columns), int(len(X_encoded.columns) * 0.7)))
        score_func = mutual_info_classif if task == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_encoded, y)
        selected = [c for c, keep in zip(X_encoded.columns, selector.get_support()) if keep]
        return selected or list(X.columns)
