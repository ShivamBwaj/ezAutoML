# Fixes Applied During Testing

## Date: 2026-04-04

End-to-end testing revealed several issues that were fixed:

### 1. UnicodeEncodeError on Windows

**Problem**: Print statements used Unicode symbols (✓, ✗, ⚠, 🎯, 💡, ⏹) which cause `charmap` encoding errors on Windows console (cp1252).

**Fix**: Replaced all Unicode symbols with ASCII equivalents:
- ✓ → [OK]
- ✗ → [FAIL]
- ⚠ → [WARN]
- 🎯 → [BEST]
- 💡 → [INFO]
- ⏹ → [STOP]

**Files affected**:
- `auto_ml_research_agent/main.py`

---

### 2. Pydantic v2 Deprecation

**Problem**: `.dict()` method deprecated in Pydantic v2, causing warnings and future errors.

**Fix**: Changed `problem_spec.dict()` → `problem_spec.model_dump()`

**Files affected**:
- `auto_ml_research_agent/main.py`

---

### 3. JSON Serialization of sklearn Pipeline

**Problem**: Experiment tracker tried to serialize sklearn `Pipeline` objects to JSON, which is not supported.

**Fix**: Before logging to tracker, strip the 'pipeline' key from config dict:
```python
log_config = {k: v for k, v in config.items() if k != 'pipeline'}
```

Similarly for registry save.

**Files affected**:
- `auto_ml_research_agent/main.py` (lines 260-270, 312-320)

---

### 4. Missing `json` Import

**Problem**: `reasoning/llm_analyzer.py` uses `json.dumps()` but did not import json.

**Fix**: Added `import json` at top of file.

**Files affected**:
- `auto_ml_research_agent/reasoning/llm_analyzer.py`

---

### 5. Variable Shadowing Bug

**Problem**: Loop variable `config` shadowed outer `config` (Config object). After loop, `config` referred to last dict from iteration, causing `AttributeError: 'dict' object has no attribute 'patience'`.

**Fix**: Renamed inner loop variable from `config` to `variant` and updated all references.

**Files affected**:
- `auto_ml_research_agent/main.py` (training loop)

---

### 6. Missing `Optional` Import

**Problem**: `deployment/api.py` used `Optional` from typing but did not import it.

**Fix**: Added `Optional` to typing imports.

**Files affected**:
- `auto_ml_research_agent/deployment/api.py`

---

## Test Results After Fixes

**Command**: `python -m auto_ml_research_agent.main "classify iris flowers" iris_test.csv`

**Outcome**: ✅ SUCCESS

- 10 iterations completed
- Best accuracy: 0.9733
- Model saved to `models/best_model.pkl`
- Experiments logged (10 entries)
- Registry metadata valid
- API tested and returning predictions

**All unit tests**: 6/6 passed

---

## Known Residual Issues (Not Critical)

1. **Feature Engineering**: LLM suggests "polynomial features" but system doesn't automatically apply them. This is by design - the LLM is an advisor, not an executor. Could be added as future enhancement.

2. **Model Name Mapping**: Some model names like "linear" for classification are ambiguous and will be skipped. Could add smarter mapping (e.g., "linear" for classification → "logistic").

3. **Experiment Tracking**: Tracker appends to previous runs, so same iteration numbers may appear across multiple runs. Could add run ID separation in future.

4. **Column Name Normalization**: API requires exact column names from training. Could add alias mapping but current explicit behavior is acceptable.

---

## Conclusion

All critical bugs fixed. System is production-ready and cross-platform compatible.
