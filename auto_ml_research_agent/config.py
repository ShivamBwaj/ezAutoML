"""
Configuration management for AUTO_ML_RESEARCH_AGENT.
Loads settings from environment variables or .env file.
"""
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config(BaseModel):
    """Main configuration container"""
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field("llama-3.1-8b-instant", description="Groq model to use")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="LLM temperature")
    max_retries: int = Field(3, ge=0, description="Max retries for LLM calls")
    patience: int = Field(50, ge=1, description="Patience for iteration stopping (research: use 50+)")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Holdout validation split size")
    random_state: int = Field(42, description="Random seed for reproducibility")
    cv_threshold: int = Field(500, description="Use CV if n_rows < threshold")
    experiment_db_path: str = Field("experiments.json", description="Path to experiment log")
    model_registry_dir: str = Field("models", description="Directory for saved models")
    download_timeout: int = Field(30, description="Dataset download timeout (seconds)")
    dataset_min_downloads: int = Field(10, description="Minimum HuggingFace downloads for dataset candidates")
    max_dataset_attempts: int = Field(10, description="Maximum number of dataset candidates to try")
    enable_llm_query_expansion: bool = Field(False, description="Use LLM to generate additional search queries")
    max_iterations: int = Field(100, ge=1, description="Maximum total iterations (research: use 100)")
    llm_analysis_interval: int = Field(10, ge=1, description="Call LLM for analysis every N iterations (reduce frequency)")
    enable_kaggle_search: bool = Field(True, description="Enable Kaggle dataset search (unreliable API but can find real datasets)")
    enable_huggingface_search: bool = Field(False, description="Enable HuggingFace dataset search (can return junk)")
    suppress_sklearn_warnings: bool = Field(True, description="Suppress sklearn warnings about class distribution and CV splits")

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be set in environment or .env file. "
                "Get your key from https://console.groq.com/"
            )

        return cls(
            groq_api_key=groq_api_key,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            test_size=float(os.getenv("TEST_SIZE", "0.2")),
            random_state=int(os.getenv("RANDOM_STATE", "42")),
            cv_threshold=int(os.getenv("CV_THRESHOLD", "500")),
            experiment_db_path=os.getenv("EXPERIMENT_DB_PATH", "experiments.json"),
            model_registry_dir=os.getenv("MODEL_REGISTRY_DIR", "models"),
            download_timeout=int(os.getenv("DOWNLOAD_TIMEOUT", "30")),
            dataset_min_downloads=int(os.getenv("DATASET_MIN_DOWNLOADS", "10")),
            max_dataset_attempts=int(os.getenv("MAX_DATASET_ATTEMPTS", "10")),
            enable_llm_query_expansion=os.getenv("ENABLE_LLM_QUERY_EXPANSION", "False").lower() == "true",
            patience=int(os.getenv("PATIENCE", "50")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "100")),
            llm_analysis_interval=int(os.getenv("LLM_ANALYSIS_INTERVAL", "10")),
            enable_kaggle_search=os.getenv("ENABLE_KAGGLE_SEARCH", "True").lower() == "true",
            enable_huggingface_search=os.getenv("ENABLE_HUGGINGFACE_SEARCH", "False").lower() == "true",
            suppress_sklearn_warnings=os.getenv("SUPPRESS_SKLEARN_WARNINGS", "True").lower() == "true",
        )


def load_config() -> Config:
    """Convenience function to load config"""
    return Config.from_env()
