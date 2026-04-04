"""
Custom exceptions for AUTO_ML_RESEARCH_AGENT.
"""


class AutoMLError(Exception):
    """Base exception for AutoML system"""
    pass


class LLMError(AutoMLError):
    """Raised when LLM API call fails or returns invalid data"""
    pass


class DatasetError(AutoMLError):
    """Raised when dataset operations fail"""
    pass


class ConfigurationError(AutoMLError):
    """Raised when configuration is invalid"""
    pass


class ValidationError(AutoMLError):
    """Raised when data validation fails"""
    pass


class DeploymentError(AutoMLError):
    """Raised when deployment operations fail"""
    pass
