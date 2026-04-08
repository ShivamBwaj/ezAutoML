"""
Data Profiler: Analyzes dataset structure and statistics.
"""
from typing import Dict, List, Any
import pandas as pd


class DataProfiler:
    """Profiles pandas DataFrames to extract column statistics"""

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive profile of a DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with dataset and column-level statistics
        """
        n_rows, n_cols = df.shape

        profile = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": {},
            "correlations": self.compute_correlations(df)
        }

        for col in df.columns:
            series = df[col]
            unique_count = series.nunique()
            missing_count = series.isnull().sum()
            missing_pct = (missing_count / n_rows * 100) if n_rows > 0 else 0.0

            col_stats: Dict[str, Any] = {
                "dtype": str(series.dtype),
                "missing_count": int(missing_count),
                "missing_pct": float(round(missing_pct, 2)),
                "unique_count": int(unique_count),
                "unique_pct": float(round(unique_count / n_rows * 100, 2)) if n_rows > 0 else 0.0
            }

            # Sample non-null values (up to 5)
            non_null = series.dropna()
            if len(non_null) > 0:
                samples = non_null.head(5).tolist()
                # Convert all samples to strings for JSON compatibility
                col_stats["sample_values"] = [str(v) for v in samples]
            else:
                col_stats["sample_values"] = []

            profile["columns"][col] = col_stats

        return profile

    def compute_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute lightweight numeric correlation diagnostics."""
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] < 2:
            return {"correlation_matrix": {}, "highly_correlated_pairs": []}
        corr = numeric_df.corr(method="pearson").fillna(0.0)
        high_pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = float(corr.iloc[i, j])
                if abs(value) >= 0.9:
                    high_pairs.append((cols[i], cols[j], round(value, 4)))
        return {
            "correlation_matrix": corr.round(4).to_dict(),
            "highly_correlated_pairs": high_pairs
        }

    def get_suggested_targets(self, df: pd.DataFrame) -> List[str]:
        """
        Get candidate target column names based on common patterns.

        Args:
            df: Input DataFrame

        Returns:
            List of column names that might be targets
        """
        common_targets = [
            'target', 'label', 'class', 'price', 'value', 'score',
            'outcome', 'result', 'y', 'output', 'response'
        ]
        columns_lower = {col.lower(): col for col in df.columns}
        suggestions = []
        for target in common_targets:
            if target in columns_lower:
                suggestions.append(columns_lower[target])
        return suggestions
