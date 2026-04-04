"""
FastAPI Deployment: REST API for model predictions.
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import joblib
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from auto_ml_research_agent.registry.model_registry import ModelRegistry
from auto_ml_research_agent.exceptions import DeploymentError

app = FastAPI(
    title="AutoML Research Agent API",
    description="Deployed model prediction endpoint",
    version="1.0.0"
)

# Global pipeline instance
model_pipeline = None
registry = None


class PredictionRequest(BaseModel):
    """
    Request body for predictions.

    Supports:
    - List of feature dictionaries: [{"feature1": val1, ...}, ...]
    - Single feature dict: {"feature1": val1, ...} (wrapped automatically)
    - List of values: [val1, val2, ...] (requires column order matching training)
    """
    features: Union[List[Dict[str, Any]], Dict[str, Any], List[List[Any]]]

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Ensure features is non-empty"""
        if not v:
            raise ValueError("Features cannot be empty")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                    {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response with predictions"""
    predictions: List[Any]
    model_version: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [0, 1],
                "model_version": "best_model"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None


def get_model():
    """Dependency to get loaded model"""
    global model_pipeline
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_pipeline


@app.on_event("startup")
def startup_event():
    """Load model on startup"""
    global model_pipeline, registry
    try:
        registry = ModelRegistry()
        model_pipeline = registry.load_best()
        if model_pipeline is None:
            print("Warning: No model found in registry. API will accept requests but return 503.")
        else:
            print("Model loaded successfully")
            # Get model info
            info = registry.get_best_info()
            print(f"Model info: {info}")
    except Exception as e:
        print(f"Error loading model on startup: {e}")
        model_pipeline = None


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    info = registry.get_best_info() if registry else None
    return HealthResponse(
        status="ok",
        model_loaded=model_pipeline is not None,
        model_info=info
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make predictions using loaded model.

    Args:
        request: PredictionRequest with features

    Returns:
        Predictions as list
    """
    pipeline = get_model()

    try:
        # Convert request to DataFrame
        features = request.features

        if isinstance(features, dict):
            # Single example as dict -> wrap in list
            X = pd.DataFrame([features])
        elif isinstance(features, list):
            if len(features) == 0:
                raise HTTPException(status_code=400, detail="Empty features list")
            if all(isinstance(item, dict) for item in features):
                # List of dicts
                X = pd.DataFrame(features)
            elif all(isinstance(item, (list, tuple)) for item in features):
                # List of lists (assume uniform length)
                X = pd.DataFrame(features)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mixed feature types. Use list of dicts or list of lists."
                )
        else:
            raise HTTPException(status_code=400, detail="Invalid features format")

        # Make predictions
        predictions = pipeline.predict(X)

        # Convert to list for JSON serialization
        pred_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

        return PredictionResponse(
            predictions=pred_list,
            model_version=Path(registry.list_models()[-1]['path']).stem if registry and registry.list_models() else "unknown"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def root():
    """API root with info"""
    return {
        "message": "AutoML Research Agent API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        },
        "model_loaded": model_pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
