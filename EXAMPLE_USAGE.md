# Example Usage: AUTO_ML_RESEARCH_AGENT

## Prerequisites

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Get Groq API key**
   - Visit https://console.groq.com/
   - Sign up / log in
   - Copy your API key

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and set:
   GROQ_API_KEY=your_actual_key_here
   ```

---

## Example 1: Classify Iris (with dataset)

```bash
# First, get the iris dataset
python -c "
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('iris.csv', index=False)
print('Created iris.csv')
"

# Run the full pipeline
python -m auto_ml_research_agent.main "classify iris flowers" iris.csv
```

**Expected output** (with GROQ_API_KEY set):
```
============================================================
AUTO_ML_RESEARCH_AGENT
============================================================
[1/9] Initializing components...
✓ Configuration loaded
✓ All components initialized

[2/9] Loading dataset...
✓ Loaded from iris.csv: (150, 5)

[3/9] Interpreting problem...
✓ Problem specification:
    Task: classification
    Target: target
    Metric: accuracy

[4/9] Validating dataset...
✓ Dataset validated (baseline: 0.973)

...

--- Iteration 1 ---
  Generating initial pipeline variants...
  Training 5 variants...
    1. logistic_default: accuracy=0.967
    2. randomforest_default: accuracy=0.973
    ...
  ✓ Best: randomforest_default (score=0.973)
  🎯 New best overall score!

...

============================================================
SAVING BEST MODEL
============================================================
✓ Best model saved to: models/best_model.pkl

============================================================
PIPELINE COMPLETE
============================================================
Best score: 0.973
...
```

---

## Example 2: Auto-Search Dataset (no CSV needed)

```bash
# Let the system find a suitable dataset
python -m auto_ml_research_agent.main "predict Boston housing prices"
```

The system will:
1. Search for "Boston housing" datasets
2. Prefer sklearn built-in (boston dataset)
3. Profile and train
4. Iterate to improve

---

## Example 3: Deploy the API

After training completes:

```bash
# Option 1: Direct python
python -m auto_ml_research_agent.deployment.api

# Option 2: Uvicorn
uvicorn auto_ml_research_agent.deployment.api:app --host 0.0.0.0 --port 8000
```

Test predictions:

```bash
# Health check
curl http://localhost:8000/health

# Classification example for iris
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3}
    ]
  }'

# Expected response
{
  "predictions": [0, 1],
  "model_version": "best_model"
}
```

Interactive docs: http://localhost:8000/docs

---

## Example 4: Run System Tests (no API key needed)

```bash
python test_system.py
```

This validates:
- All module imports
- Dataset loading and profiling
- Preprocessing pipeline generation
- Model training (with iris)

Tests pass without GROQ_API_KEY because they don't require LLM analysis.

---

## Troubleshooting

### "GROQ_API_KEY must be set"
Create `.env` file in project root with your key.

### Import errors
Run from project root directory. Ensure Python can find the package:
```bash
cd /path/to/EZautoML
python -m auto_ml_research_agent.main "problem"
```

### Playwright errors
If browser fallback is needed, install Chromium:
```bash
playwright install chromium
```

### Memory errors with large datasets
System auto-uses CV for datasets <500 rows, holdout for larger. For huge datasets (>100k rows), consider downsampling first.

---

## What's Happening Under the Hood

1. **Problem Interpretation** (LLM)
   - Input: "classify iris flowers"
   - Output: `{"task": "classification", "target_column": "target", "metric": "accuracy"}`

2. **Dataset Validation**
   - Loads iris.csv
   - Quick baseline: LogisticRegression CV → 0.973 accuracy
   - Confirms dataset is suitable

3. **Variants Generation** (Iteration 1)
   - Creates 5 pipelines:
     - Logistic Regression (default)
     - Random Forest (default)
     - Gradient Boosting (default)
     - SVC (default)
     - K-Neighbors (default)

4. **Training**
   - All 5 models trained with CV (n=150 → 3-fold)
   - Best: Random Forest (0.973)

5. **LLM Analysis** (Iteration 2+)
   - If continued, LLM would analyze history and suggest e.g.:
     - "Increase n_estimators for RandomForest"
     - "Try XGBoost"
   - Variant generator creates new configs
   - Process repeats until no improvement for 5 iterations

6. **Registry**
   - Best pipeline saved to `models/best_model.pkl`
   - Metadata in `models/registry.json`

---

## Experiment Log

All runs are saved to `experiments.json`:

```json
[
  {
    "iteration": 1,
    "timestamp": "2026-04-04T18:50:00Z",
    "config": {
      "name": "randomforest_default",
      "model_name": "randomforest",
      "params": {}
    },
    "score": 0.973,
    "metric_name": "accuracy",
    "model_path": "models/iter_1.pkl"
  },
  ...
]
```

---

## Configuration

See `.env` for all tunable parameters:

- `PATIENCE=5` - Stop after 5 iterations without improvement
- `CV_THRESHOLD=500` - Use CV if dataset has <500 rows
- `TEMPERATURE=0.3` - LLM sampling temperature (lower = more deterministic)

---

**Note**: The full LLM-guided iteration loop requires a valid GROQ_API_KEY. Without it, the system will run one iteration and stop (or fail on problem interpretation). Use `test_system.py` to validate components without an API key.
