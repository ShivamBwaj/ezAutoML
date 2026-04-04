# Future Enhancement Ideas

This document captures potential improvements and features for future iterations of EZautoML.

---

## **Feature Selection Strategies**

### **Problem**
Currently, the system uses all available features (after preprocessing) without any dimension reduction or selection. This can lead to:
- Increased training time with many features
- Overfitting on noisy/irrelevant features
- Reduced model interpretability
- Multicollinearity issues

### **Proposed Solutions**

#### **Option C: Model-Based Feature Selection** (Data-Driven)

Use sklearn's feature selection methods driven by model importance:

**Implementation:**
```python
from sklearn.feature_selection import SelectFromModel, RFE

# Embedded selection (in pipeline)
selector = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median"  # or "mean", or custom threshold
)

# Wrapper method (RFE)
selector = RFE(
    estimator=RandomForestClassifier(),
    n_features_to_select=20  # or use percentage
)
```

**Pros:**
- Empirical: uses actual predictive power from the data
- Captures feature interactions (tree-based models)
- Automated threshold selection ("median" works out-of-box)
- Can be validated via CV score improvements

**Cons:**
- Extra computational cost (train model twice)
- Model bias: RandomForest favors high-cardinality one-hot features
- Stability issues with small datasets
- Choice of estimator matters (Lasso vs RF give different results)

**Suggested Integration:**
- Add as configurable variant in pipeline generator
- Only apply when n_features > threshold (e.g., 30)
- Track feature reduction ratio in experiment logs

---

#### **Option D: LLM-Guided Feature Selection** (Knowledge-Driven)

Use LLM reasoning to select features based on domain knowledge and feature metadata:

**Input to LLM:**
- Problem specification (task, target)
- Feature profiles: name, type, sample values, cardinality
- Correlation matrix (feature-feature correlations)
- Univariate statistics (correlation with target, mutual information)
- Dataset domain (if identifiable from search)

**LLM Output (structured JSON):**
```json
{
  "included_features": ["age", "cp", "thalach", "oldpeak", "ca", "thal"],
  "excluded_features": ["chol", "fbs", "restecg"],
  "reasoning": "Classic cardiac risk factors retained; cholesterol excluded as noisy without medication context"
}
```

**Pros:**
- Incorporates domain expertise (medical, financial, etc.)
- Interpretable reasoning
- No extra training cost
- Can suggest domain-specific transformations (binning, interactions)
- Can explain *why* to keep/drop features

**Cons:**
- Not data-specific: might miss dataset-specific patterns
- Hallucination risk: could exclude truly important features
- No automatic validation: must rely on metrics to confirm
- Requires careful prompt engineering

**Suggested Integration:**
- New variant type in `reasoning/variant_generator.py`
- Prompt includes profiler stats, feature descriptions, correlation hints
- LLM selects subset; pipeline adds custom transformer that filters to those features
- Log LLM reasoning in experiment tracker for later review

---

### **Hybrid Strategy: Combine C + D**

Leverage both domain knowledge and empirical evidence:

#### **Strategy 1: Sequential Filter**
1. LLM suggests candidate features to include
2. Train model quickly on LLM subset
3. If performance comparable to full features (±X%), use LLM subset (for interpretability)
4. If performance drops significantly, fall back to model-based selection

#### **Strategy 2: Ensemble Voting**
- LLM vote: +1 (include), -1 (exclude), 0 (neutral/uncertain) for each feature
- Model-based: +1 (above median importance), -1 (below), 0 (mid)
- Weighted sum: keep if total score > 0
- Could weight LLM more for low-data regimes, model more for high-data

#### **Strategy 3: Two-Stage Pipeline**
- Stage 1: LLM-guided pre-filter (exclude obviously irrelevant features)
- Stage 2: Model-based selection from LLM-approved set
- Benefits: LLM reduces search space; model does fine-grained selection

#### **Strategy 4: Variant Generation (Recommended for this project)**
During iteration, generate multiple variant types:
- Variant A: Full features (baseline)
- Variant B: LLM-guided feature subset
- Variant C: Model-based selection (SelectFromModel)
- Variant D: LLM + Model hybrid

The **metrics will decide** which approach works best for this dataset - perfectly aligned with system philosophy.

---

### **Implementation Roadmap**

#### **Phase 1: LLM-Guided Feature Selection** (Low-hanging fruit)

1. **Enhance `preprocessing/llm_edge.py`** or create `reasoning/feature_selector.py`:
   ```python
   class LLMFeatureSelector:
       def select_features(self, profiler_stats, problem_spec, correlation_matrix=None) -> List[str]:
           # Build prompt with feature metadata
           # Query LLM
           # Parse response
           # Return list of selected feature names
   ```

2. **Add prompt template** that includes:
   - Feature names and types
   - Sample values (first few unique values)
   - Missing percentage
   - Basic statistics (mean, std for numeric; top categories for categorical)
   - Target correlation (if applicable)
   - Problem context

3. **Create a transformer** for sklearn pipeline:
   ```python
   class SelectedFeaturesTransformer(BaseEstimator, TransformerMixin):
       def __init__(self, selected_features: List[str]):
           self.selected_features = selected_features
       
       def fit(self, X, y=None):
           return self
       
       def transform(self, X):
           return X[self.selected_features]
   ```

4. **Integrate into `pipeline/generator.py`**:
   - Add variant type: `{"feature_selection": {"method": "llm_guided"}}`
   - Or as suggestion-turned-config from LLM analyzer

5. **Logging**:
   - Record features selected
   - Log LLM reasoning
   - Track n_features_before vs n_features_after in experiment log

#### **Phase 2: Correlation Analysis in Profiler**

Enhance `data/profiler.py` to compute feature correlations:
```python
def compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    # Numeric features: Pearson correlation matrix
    # Categorical features: cramers V or mutual information
    # Flag highly correlated pairs (>0.9)
    return {
        "correlation_matrix": {...},
        "highly_correlated_pairs": [("feature1", "feature2"), ...]
    }
```

This info feeds into LLM prompt or model-based selection.

#### **Phase 3: Model-Based Selection**

1. **Add sklearn selector variants** to `pipeline/generator.py`:
   ```python
   from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, mutual_info_classif
   
   configs.append({
       "name": "selectfrommodel_rf",
       "feature_selection": {
           "method": "select_from_model",
           "estimator": "randomforest",
           "threshold": "median"
       }
   })
   ```

2. **Optionally**: add variance threshold (simplest):
   ```python
   from sklearn.feature_selection import VarianceThreshold
   # Remove features with near-zero variance
   ```

#### **Phase 4: Evaluation & Comparison**

- Add experiment tracking for:
  - Number of features before/after
  - Training time comparison
  - Model size (if using sparse encodings)
  - Feature importance visualization (store top features)
- Generate report showing which feature selection strategy worked best for which dataset types

---

### **Success Criteria**

Once implemented, we should be able to answer:
1. Does feature selection improve validation scores? (Often no for small datasets, yes for high-dim)
2. Which method works best? (LLM, Model, or Hybrid?)
3. Does feature selection reduce training time significantly?
4. Are the selected features interpretable and defensible?

---

### **Related Ideas**

- **Feature engineering**: LLM could suggest polynomial features, interactions, binning
- **Dimensionality reduction**: PCA, t-SNE (but loses interpretability)
- **AutoML competition**: Could integrate with featuretools for automated feature synthesis
- **Multi-metric optimization**: Accuracy vs. number of features tradeoff (Pareto front)

---

**Status:** Not started - under discussion

**Priority:** Medium (could improve performance and interpretability for high-dim datasets)

**Estimated Effort:** 2-3 days for full implementation (Phases 1-3)

---

## **Data Quality: Automatic ID Column Detection & Removal**

### **Problem Discovered (2026-04-05)**

During testing with student mental health datasets, the system **included `Student ID` columns as predictive features**, causing:
- **Data leakage**: Model memorizes IDs instead of learning patterns
- **Severe overfitting**: High cardinality (760 unique IDs for 608 rows) creates noise
- **Poor generalization**: Model fails on new data with unseen IDs
- **Wasted computation**: Training on meaningless features

**Example:**
```
Dataset: Student mental health (608 rows)
Problem: "classify student mental health"
Features: 19 total including "Student ID" (760 unique values)
Result: Pipeline includes ID column → meaningless model → accuracy ~0.11
```

**Baseline Issue**: Very low baseline scores (0.0961) may indicate:
- Wrong target column selected
- Extreme class imbalance
- Dataset fundamentally unsuitable

---

### **Root Causes**

1. **No identifier detection**: System treats all columns as potentially useful features
2. **High cardinality misclassification**: ID columns have high cardinality → gets frequency encoding (for >10 unique) instead of being dropped
3. **LLM target selection lacks validation**: LLM might pick a column that's actually an outcome (target leakage) or an ID column as target
4. **No pre-processing data quality filters**: Before any modeling, dirty data should be cleaned

---

### **Proposed Solutions**

#### **1. ID Column Detection & Auto-Removal** (CRITICAL - Fix ASAP)

Implement automatic detection of identifier columns in `preprocessing/rules.py`:

**Detection Rules:**

```python
def detect_id_columns(df: pd.DataFrame, 
                     cardinality_threshold: float = 0.8,
                     name_patterns: List[str] = None) -> List[str]:
    """
    Detect columns that are likely identifiers and should be dropped.
    
    Args:
        df: Input DataFrame
        cardinality_threshold: Unique value ratio threshold (e.g., 0.8 = 80% unique)
        name_patterns: Keywords indicating ID columns (case-insensitive)
    
    Returns:
        List of column names to drop
    """
    if name_patterns is None:
        name_patterns = [
            'id', 'uuid', 'guid', 'key', 'index', 'unique',
            'student', 'user', 'employee', 'member', 'patient',
            'order', 'transaction', 'invoice', 'account',
            'ssn', 'social', 'passport', 'license'
        ]
    
    n_rows = len(df)
    id_candidates = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / n_rows
        
        # Rule 1: Name contains ID-like keywords AND high cardinality
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in name_patterns):
            if unique_ratio > cardinality_threshold:
                id_candidates.append(col)
                continue
        
        # Rule 2: Extremely high cardinality (nearly all unique)
        # Even without name hint, drop columns with >95% unique values
        if unique_ratio > 0.95:
            id_candidates.append(col)
        
        # Rule 3: All values unique (perfect identifier)
        if unique_count == n_rows:
            id_candidates.append(col)
    
    return list(set(id_candidates))  # Deduplicate
```

**Integration in `build_preprocessor()`:**

```python
def build_preprocessor(self, df: pd.DataFrame, target_column: str):
    # ... existing code ...
    
    # Before identifying numeric/categorical, drop ID columns
    X = df.drop(columns=[target_column])
    id_columns = self.detect_id_columns(X)
    
    if id_columns:
        print(f"[INFO]  Dropping ID columns: {id_columns}")
        X = X.drop(columns=id_columns)
        metadata['dropped_columns']['id_columns'] = id_columns
    
    # Continue with numeric/categorical identification on cleaned X
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # ...
```

**Benefits:**
- Prevents data leakage automatically
- Improves model generalization
- Reduces feature space meaningfully
- Transparent logging of dropped columns

---

#### **2. Target Column Validation**

After LLM selects target column, validate it:

```python
def validate_target_column(self, df: pd.DataFrame, target_column: str) -> Dict:
    """
    Check if the selected target column is valid for modeling.
    Returns: {valid: bool, warnings: [], errors: []}
    """
    issues = []
    warnings = []
    
    # Check 1: Is it an ID column?
    if target_column in self.detect_id_columns(df[[target_column]]):
        issues.append(f"Target '{target_column}' appears to be an identifier column")
    
    # Check 2: Too many unique values for classification?
    unique_ratio = df[target_column].nunique() / len(df)
    if unique_ratio > 0.5 and self.task == 'classification':
        warnings.append(f"Target has {unique_ratio:.1%} unique values - might be regression problem")
    
    # Check 3: Target leakage check - is target in feature set?
    # (already dropped, but check names)
    leakage_keywords = ['outcome', 'result', 'score', 'performance', 'achievement']
    if any(kw in target_column.lower() for kw in leakage_keywords):
        warnings.append(f"Target '{target_column}' sounds like an outcome - verify it's not post-intervention")
    
    # Check 4: Extreme class imbalance
    if self.task == 'classification':
        value_counts = df[target_column].value_counts()
        minority_pct = value_counts.min() / len(df) * 100
        if minority_pct < 1:
            warnings.append(f"Extreme class imbalance: minority class {minority_pct:.1f}%")
    
    return {
        'valid': len(issues) == 0,
        'warnings': warnings,
        'errors': issues
    }
```

If validation fails:
- Prompt LLM to re-select target
- Or fall back to heuristics (e.g., column named "target", "label", "outcome")
- Or ask user to specify explicitly

---

#### **3. Baseline Score Threshold**

If initial baseline score is suspiciously low (<0.3 for classification, <0.1 R² for regression):
- Flag dataset as possibly unsuitable
- Check if target is meaningful
- Suggest manual review before continuing
- Option: Skip dataset and try next candidate

---

#### **4. Additional Automatic Filters**

Beyond ID columns, consider dropping:

**a) Near-Zero Variance Features:**
```python
def drop_low_variance(df, threshold=0.01):
    """Drop columns with variance below threshold (default: 1% of max variance)"""
    variances = df.var(numeric_only=True)
    drop_cols = variances[variances < threshold].index.tolist()
    return drop_cols
```

**b) High Missingness:**
```python
def drop_high_missing(df, threshold=0.7):
    """Drop columns with >70% missing values"""
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    return drop_cols
```

**c) Constant Columns:**
```python
def drop_constant(df):
    """Drop columns with all same value"""
    nunique = df.nunique()
    return nunique[nunique == 1].index.tolist()
```

**Integration**: Apply these filters in `build_preprocessor()` before creating transformers, log all dropped columns with reasons.

---

### **Implementation Plan**

#### **Phase 1: ID Detection** (High Priority)
- Add `detect_id_columns()` to `PreprocessingEngine`
- Integrate into `build_preprocessor()` to drop IDs before feature extraction
- Add logging: print which ID columns were dropped and why
- Update `metadata` dict to track dropped columns
- **Test cases**:
  - Clear IDs: `student_id`, `user_id`, `order_id`
  - Obscure IDs: `ssn`, `passport_number`, `transaction_key`
  - False positives: Avoid dropping `postal_code` (though high cardinality, it's meaningful)

#### **Phase 2: Target Validation**
- Add `validate_target_column()` method
- Call after problem interpretation, before preprocessing
- Print warnings/errors prominently
- On validation failure: either retry with alternative target or exit gracefully

#### **Phase 3: Baseline Suitability Check**
- After dataset evaluation, check baseline score
- If < threshold, suggest manual review or skip dataset
- Log reason in experiment tracker

#### **Phase 4: General Data Quality Filters**
- Implement `drop_low_variance()`, `drop_high_missing()`, `drop_constant()`
- Make configurable via `Config` flags (enable/disable each filter)
- Default: enable all with sensible thresholds

#### **Phase 5: Dataset Quality Scoring**
- Create composite "dataset quality score" based on:
  - % of ID columns dropped
  - Missingness rate
  - Class imbalance ratio
  - Baseline score
- Use score to rank multiple dataset candidates

---

### **Testing Strategy**

Create test fixtures with:
1. Dataset with ID column → verify it gets dropped
2. Dataset with target as ID → catch and re-select
3. Dataset with extremely low baseline → flag as unsuitable
4. Dataset with high-cardinality meaningful feature (like `postal_code`) → keep it

---

### **Related to Feature Selection**

ID detection is actually a **feature elimination** strategy. It's the most important form because ID columns are guaranteed to cause overfitting. This should be implemented **before** any sophisticated feature selection because:
- IDs aren't features, they're metadata
- Removing them reduces dimensionality meaningfully
- Prevents wasting compute on useless variants

---

### **Priority & Timeline**

**Priority: CRITICAL** - This is a bug, not just an enhancement. Currently:
- System can produce misleadingly high scores on train by memorizing IDs
- Model will fail in production
- Trust in automatic system is compromised

**Effort: 1-2 days**
- Detection logic: straightforward (few hours)
- Integration: ~1 day
- Testing: ~0.5 day

**Recommendation**: Implement before any further feature selection work.

---

**Status:** Newly identified critical issue from real-world testing (student mental health dataset)

**Date Identified:** 2026-04-05

**Impact:** High - causes models to be invalid and misleading
