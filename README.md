# ezAutoML

Automated machine learning research agent that uses LLM-guided reasoning to iteratively improve models.

## Features

- **Natural Language Input**: Describe your ML problem in plain English
- **Automatic Dataset Search**: Searches HuggingFace Hub and sklearn built-in datasets
- **Intelligent Preprocessing**: Auto-detects column types and applies appropriate transformations
- **Multi-Variant Training**: Trains 3-5 different models per iteration
- **LLM-Guided Improvement**: Uses LLM analysis to suggest and generate new variants
- **Patience-Based Stopping**: Stops automatically when no improvement for N iterations
- **Model Registry**: Saves best model with full metadata
- **FastAPI Deployment**: Ready-to-use REST API for predictions

## Architecture

```
Natural Language → Problem Interpreter → Dataset Search → Profiling → Preprocessing
     ↓
Pipeline Generation → Training & Evaluation → Experiment Tracking
     ↓
LLM Analysis → Variant Generation → [Iterate] → Registry → API Deployment
```

## Installation

### Prerequisites

- Python 3.10+
- Groq API key (get from https://console.groq.com/)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers** (needed for fallback dataset downloads):
   ```bash
   playwright install chromium
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

## Usage

### Basic Usage

```bash
# With a dataset file
python main.py "classify iris flowers" iris.csv

# Without dataset - auto-searches
python main.py "predict house prices"
python main.py "classify breast cancer"
```

### Example Problems

- `"classify iris flowers"` - automatically uses sklearn iris dataset
- `"predict house prices"` - searches for housing datasets
- `"classify tumor types"` - searches for medical classification datasets
- `"predict diabetes progression"` - searches for regression datasets

### Using the API

After training completes, start the FastAPI server:

```bash
# Option 1: Direct Python
python -m auto_ml_research_agent.deployment.api

# Option 2: Uvicorn
uvicorn auto_ml_research_agent.deployment.api:app --reload --port 8000
```

Then make predictions:

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3}
    ]
  }'
```

Or visit `http://localhost:8000/docs` for interactive Swagger UI.

## How It Works

### 1. Problem Interpretation

The LLM analyzes your natural language description and determines:
- Task type (classification or regression)
- Target column to predict
- Appropriate evaluation metric

### 2. Dataset Acquisition

Searches in priority order:
1. **sklearn built-in** (iris, boston, diabetes, etc.)
2. **HuggingFace Hub** (thousands of datasets)
3. **Direct CSV URLs** (if known)
4. **Browser automation** (Kaggle fallback - requires login)

### 3. Data Profiling

Analyzes:
- Column types and missing values
- Cardinality of categorical features
- Sample values for context

### 4. Preprocessing

Automatically builds sklearn ColumnTransformer:
- **Numeric**: Median imputation + StandardScaler
- **Categorical** (low cardinality): One-hot encoding
- **Categorical** (high cardinality): Frequency encoding

### 5. Initial Variants

Generates 3-5 diverse model configurations:
- Different model types (Logistic, RandomForest, GradientBoosting, SVC, etc.)
- Various parameter settings
- All using the same preprocessor

### 6. Training & Evaluation

- Small datasets (<500 rows): 3-fold cross-validation
- Large datasets: 80-20 holdout split
- Computes relevant metrics (accuracy, F1, RMSE, R²)

### 7. Iterative Improvement

For each iteration:
1. LLM analyzes experiment history
2. Identifies issues (overfitting, underfitting, etc.)
3. Suggests concrete improvements
4. Variant generator creates 3-5 new pipelines
5. Train and evaluate all variants
6. Keep the best

Stops when:
- No improvement for `patience` iterations (default: 5)
- Reached maximum iterations (10)

### 8. Model Registry

Saves best model to `models/best_model.pkl` with metadata in `models/registry.json`.

### 9. API Deployment

FastAPI service provides:
- `GET /health` - health check
- `POST /predict` - make predictions
- `GET /` - API info

## Project Structure

```
auto_ml_research_agent/
├── llm/
│   └── groq_client.py      # Groq API wrapper
├── problem/
│   └── interpreter.py      # Natural language → ML spec
├── dataset/
│   ├── search.py          # Dataset search
│   ├── evaluator.py       # Dataset quality check
│   ├── downloader.py      # Multi-source download
│   └── browser_agent.py   # Playwright fallback
├── data/
│   └── profiler.py        # Dataset profiling
├── preprocessing/
│   ├── rules.py           # Auto ColumnTransformer
│   └── llm_edge.py        # LLM edge detection
├── pipeline/
│   └── generator.py       # Variant generation
├── training/
│   ├── trainer.py         # Training with CV/holdout
│   └── evaluator.py       # Metric extraction
├── experiments/
│   └── tracker.py         # JSON experiment log
├── reasoning/
│   ├── llm_analyzer.py    # LLM analysis
│   └── variant_generator.py  # Suggestion → config
├── controller/
│   └── loop.py            # Iteration control
├── registry/
│   └── model_registry.py  # Model persistence
├── deployment/
│   └── api.py             # FastAPI service
├── main.py                # Orchestration
├── config.py              # Configuration
└── exceptions.py          # Custom exceptions
```

## Configuration

Settings in `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM model |
| `TEMPERATURE` | `0.3` | LLM sampling temperature |
| `MAX_RETRIES` | `3` | LLM API retry attempts |
| `PATIENCE` | `5` | Iterations without improvement before stop |
| `TEST_SIZE` | `0.2` | Holdout validation split |
| `RANDOM_STATE` | `42` | Random seed |
| `CV_THRESHOLD` | `500` | Use CV if n_rows < threshold |
| `EXPERIMENT_DB_PATH` | `experiments.json` | Experiment log path |
| `MODEL_REGISTRY_DIR` | `models` | Model save directory |
| `DOWNLOAD_TIMEOUT` | `30` | Dataset download timeout (seconds) |

## Requirements

See `requirements.txt`. Key dependencies:

- `scikit-learn>=1.3.0` - ML pipelines
- `pandas>=2.0.0` - Data handling
- `fastapi>=0.104.0` - API deployment
- `groq>=0.4.0` - LLM client
- `pydantic>=2.0.0` - Data validation
- `huggingface-hub>=0.17.0` - Dataset search
- `datasets>=2.14.0` - Dataset loading
- `playwright>=1.40.0` - Browser fallback (optional)

## Design Principles

1. **LLM as Advisor**: LLM suggests improvements, metrics decide
2. **Metrics are Truth**: All decisions based on validation scores
3. **Multiple Variants**: 3-5 different configs per iteration
4. **Patience Stopping**: Prevent infinite loops
5. **Structured Output**: All LLM responses validated via Pydantic
6. **Safe Fallbacks**: Browser automation only as last resort
7. **Modular Design**: Each component independently testable

## Error Handling

The system handles:
- LLM API failures (retry with backoff)
- Dataset download failures (try next source)
- Invalid LLM responses (validation + retry)
- Training failures (skip variant, continue)
- Graceful interruption (KeyboardInterrupt saves current best)

## Limitations

- LLM analysis uses context window efficiently but may miss patterns in very long histories
- Dataset search relies on HuggingFace metadata quality
- Preprocessing is automatic but may not handle all edge cases
- No feature engineering beyond basic preprocessing (LLM can suggest, but not auto-apply)
- Browser fallback may fail if sites require login
- XGBoost optional (only if installed separately)

## Future Enhancements

- Feature engineering automation
- Hyperparameter optimization (Bayesian optimization)
- AutoML competitions integration
- Multi-metric optimization
- Distributed training for large datasets
- Cloud storage backends (S3, GCS)
- Web UI for monitoring

## Troubleshooting

### "GROQ_API_KEY must be set"
Create `.env` file with your Groq API key.

### "No datasets found"
Provide a dataset path explicitly: `python main.py "problem" data.csv`

### Browser agent fails
Install Playwright: `playwright install chromium`
Note: Some sites require manual login. Use dataset paths instead.

### Import errors
Make sure you're in the project root directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory issues with large datasets
The system uses holdout validation for datasets >500 rows. For extremely large datasets, consider downsampling first.

## License

MIT

## Author

Shivam
