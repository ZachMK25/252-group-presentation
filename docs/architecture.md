# Code Architecture

Detailed code structure and organization of the cryptocurrency prediction system.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Module Architecture](#module-architecture)
3. [Data Flow](#data-flow)
4. [Module Reference](#module-reference)
5. [Design Patterns](#design-patterns)

---

## Project Structure

```
252-group-presentation/
│
├── src/                          # Modular source code
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration & hyperparameters
│   ├── data_utils.py             # Data loading & preprocessing
│   ├── feature_engineering.py    # Technical indicators
│   ├── model_utils.py            # Models & ensembles
│   ├── evaluation.py             # Metrics & cross-validation
│   └── preprocessing.py          # Feature scaling pipelines
│
├── run_ensemble.py               # Main training script (all symbols)
├── trade_simulator.py            # Trade simulation on BTC
├── visualize_backtest.py         # Generate backtest visualizations
├── load_models_and_simulate.py   # Load saved models for live simulation
├── test_live_simulation.py       # Test script for live simulation
│
├── Preprocessing.ipynb           # Exploratory notebook
├── crypto.csv                    # Input OHLCV data (not in repo)
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
├── DOCUMENTATION.md              # Full technical documentation
├── CLAUDE.md                     # AI assistant guidance
├── .gitignore                    # Git ignore rules
│
├── docs/                         # Documentation directory
│   ├── index.md                  # Documentation index
│   ├── quick-start.md            # Quick start guide
│   ├── implementation.md         # Implementation details
│   ├── algorithms.md             # Algorithm descriptions
│   ├── results.md                # Results and analysis
│   ├── architecture.md           # This file
│   └── ...                       # Other documentation files
│
└── Output Directories (gitignored):
    ├── saved_models/             # Trained models, preprocessors, calibrators
    ├── roi_results/              # Performance metrics per model/symbol
    └── simulation_results/       # Trade logs and backtest results
```

---

## Module Architecture

### Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                      config.py                          │
│  (Constants, hyperparameters - no dependencies)         │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
                    ┌─────┴─────┐
                    │           │
         ┌──────────▼──┐   ┌────▼───────────┐
         │ data_utils  │   │ feature_eng    │
         │             │   │                │
         └──────┬──────┘   └────┬───────────┘
                │               │
                │         ┌─────▼─────────┐
                │         │ preprocessing │
                │         └─────┬─────────┘
                │               │
                └───────┬───────┘
                        │
                   ┌────▼──────────┐
                   │  model_utils  │
                   └────┬──────────┘
                        │
                   ┌────▼──────────┐
                   │  evaluation   │
                   └───────────────┘
```

### Module Responsibility Matrix

| Module | Responsibility | Input | Output | Dependencies |
|--------|---------------|-------|--------|--------------|
| `config.py` | Configuration | None | Constants | None |
| `data_utils.py` | Data loading | CSV file | DataFrame | config |
| `feature_engineering.py` | Feature creation | DataFrame | Features + Labels | config, ta |
| `preprocessing.py` | Feature scaling | Features | Pipeline | data_utils |
| `model_utils.py` | Model definitions | Features, Labels | Trained models | config, sklearn, xgboost |
| `evaluation.py` | Metrics, CV | Predictions, True labels | Metrics | config |

---

## Data Flow

### High-Level Pipeline

```
crypto.csv (Wide Format)
         │
         ▼
┌─────────────────────┐
│  load_crypto_csv()  │ ── data_utils.py
└─────────┬───────────┘
          │
          ▼
MultiIndex DataFrame
(symbol, datetime)
          │
          ▼
┌─────────────────────┐
│ compute_features()  │ ── feature_engineering.py
└─────────┬───────────┘
          │
          ▼
Feature DataFrame (31+ columns)
          │
          ▼
┌─────────────────────┐
│  create_labels()    │ ── feature_engineering.py
└─────────┬───────────┘
          │
          ▼
X (features), y (labels)
          │
          ▼
┌─────────────────────┐
│  Preprocessing      │ ── preprocessing.py
└─────────┬───────────┘
          │
          ▼
Scaled Features
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐  ┌──────────┐
│ Train   │  │  Test    │ ── evaluation.py (Purged K-Fold)
└────┬────┘  └─────┬────┘
     │             │
     ▼             ▼
┌─────────────────────┐
│  Train Models       │ ── model_utils.py
└─────────┬───────────┘
          │
          ▼
Model Predictions
          │
          ▼
┌─────────────────────┐
│  Calibration        │ ── model_utils.py
└─────────┬───────────┘
          │
          ▼
Calibrated Probabilities
          │
          ▼
┌─────────────────────┐
│  Ensembles          │ ── model_utils.py
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Evaluation         │ ── evaluation.py
└─────────┬───────────┘
          │
          ▼
Metrics & ROI Results
```

### Main Script Flow (run_ensemble.py)

```python
# 1. Import modules
from src import data_utils, feature_engineering, model_utils, evaluation, preprocessing

# 2. Load data
df = data_utils.load_crypto_csv('crypto.csv')
# → MultiIndex DataFrame (symbol, datetime)

# 3. Feature engineering (per symbol)
features = feature_engineering.compute_features(df)
labels = feature_engineering.create_labels(features)
# → 31+ features, binary labels

# 4. Split features and labels
X, y = feature_engineering.split_features_labels(features)

# 5. Create preprocessing pipeline
bounded, loggy, rest = feature_engineering.categorize_feature_columns(X.columns)
preprocessor = preprocessing.create_preprocessing_pipeline(bounded, loggy, rest)

# 6. Get base models
models = model_utils.get_base_models()
# → Dict of 6 ML models

# 7. Cross-validation training (4 folds)
cv_predictions = {}
for fold_idx, (train_idx, test_idx) in enumerate(evaluation.purged_kfold_cv(X, y)):
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit preprocessing
    preprocessor.fit(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Train each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict_proba(X_test_scaled)[:, 1]
        cv_predictions[name].append(pred)

# 8. Calibrate probabilities
calibrators = {}
for name in models.keys():
    all_cv_pred = np.concatenate(cv_predictions[name])
    calibrators[name] = model_utils.calibrate_probabilities(y_all, all_cv_pred)

# 9. Create ensembles
ensemble_stacked = model_utils.create_stacked_ensemble(X_meta, y_meta, base_preds)
weights_brier = model_utils.compute_ensemble_weights(brier_scores, 'brier')
weights_logloss = model_utils.compute_ensemble_weights(logloss_scores, 'logloss')

# 10. Evaluate all models
metrics = {}
for name, predictions in all_predictions.items():
    metrics[name] = evaluation.compute_metrics(y_true, predictions)

# 11. Compute ROI per symbol
roi_results = evaluation.compute_roi_per_threshold(y_true, y_pred, returns, thresholds)

# 12. Save models
joblib.dump(models, f'{model_dir}/models.pkl')
joblib.dump(preprocessor, f'{model_dir}/preprocessor.pkl')
joblib.dump(calibrators, f'{model_dir}/calibrators.pkl')
```

---

## Module Reference

### src/config.py

**Purpose:** Centralized configuration and hyperparameters

**Exports:**
```python
# Random seed
RANDOM_SEED = 42

# Data configuration
DATA_FILE = 'crypto.csv'
FIELD_NAMES = {'open', 'high', 'low', 'close', 'volume'}
KEEP_SYMBOLS = []
MAX_ROWS_PER_SYMBOL = None

# Feature engineering parameters
ROC_WINS = [1, 3, 42]
EMA_PAIRS = [(84, 168)]
RSI_WINS = [8, 14, 26]
# ... etc

# Model parameters
USE_GPU_XGB = True
USE_HEAVY_MODELS = False
XGB_N_ESTIMATORS = 300
# ... etc

# Cross-validation
N_FOLDS = 4
EMBARGO_HOURS = 24

# Trade simulation
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
```

**No dependencies**

---

### src/data_utils.py

**Purpose:** Data loading and preprocessing

**Key Functions:**

```python
def load_crypto_csv(csv_path='crypto.csv') -> pd.DataFrame:
    """
    Load and reshape crypto data from wide to long format

    Returns:
        MultiIndex DataFrame (symbol, datetime) with OHLCV columns
    """

def load_single_symbol(csv_path, symbol) -> pd.DataFrame:
    """
    Load data for a single cryptocurrency

    Returns:
        DataFrame indexed by datetime with OHLCV columns
    """

def signed_log1p(df_in) -> pd.DataFrame:
    """
    Apply signed log1p transformation: sign(x) * log(1 + |x|)
    """
```

**Dependencies:** pandas, numpy, config

---

### src/feature_engineering.py

**Purpose:** Technical indicator computation

**Key Functions:**

```python
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 31+ technical indicators

    Input: DataFrame with OHLCV columns
    Output: DataFrame with OHLCV + 31 indicator columns
    """

def create_labels(df: pd.DataFrame, threshold=0.01) -> pd.Series:
    """
    Create binary classification labels

    Returns: Series of 0/1 labels (0=down, 1=up)
    """

def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into features (X) and labels (y)

    Returns: (X, y) where X excludes OHLCV and label
    """

def categorize_feature_columns(cols) -> Tuple[List, List, List]:
    """
    Categorize features for preprocessing

    Returns: (bounded_cols, loggy_cols, the_rest)
    """
```

**Dependencies:** pandas, numpy, ta (technical analysis), config

---

### src/preprocessing.py

**Purpose:** Feature scaling pipeline creation

**Key Functions:**

```python
def create_preprocessing_pipeline(bounded_cols, loggy_cols, the_rest) -> ColumnTransformer:
    """
    Create 3-category preprocessing pipeline

    Returns: sklearn ColumnTransformer
    """
```

**Dependencies:** sklearn, data_utils

---

### src/model_utils.py

**Purpose:** Model definitions and ensemble creation

**Key Functions:**

```python
def get_base_models() -> Dict[str, Model]:
    """
    Initialize all base ML models

    Returns: Dict of {model_name: model_instance}
    """

def calibrate_probabilities(y_true, y_pred) -> IsotonicRegression:
    """
    Calibrate probabilities using isotonic regression

    Returns: Fitted IsotonicRegression calibrator
    """

def create_ensemble_predictions(base_preds, weights=None) -> np.ndarray:
    """
    Create weighted ensemble predictions

    Returns: Array of ensemble predictions
    """

def create_stacked_ensemble(X_train, y_train, base_preds) -> LogisticRegression:
    """
    Train stacked ensemble meta-model

    Returns: Fitted LogisticRegression model
    """

def compute_ensemble_weights(metrics, metric_name) -> Dict[str, float]:
    """
    Compute ensemble weights from metrics

    Returns: Dict of {model_name: weight}
    """
```

**Dependencies:** sklearn, xgboost, config

---

### src/evaluation.py

**Purpose:** Metrics computation and cross-validation

**Key Functions:**

```python
def compute_metrics(y_true, y_pred_proba, y_pred_binary=None) -> Dict:
    """
    Compute all evaluation metrics

    Returns: {'auc': float, 'brier': float, 'logloss': float, 'accuracy': float}
    """

def purged_kfold_cv(X, y, n_splits=4, embargo_hours=24) -> Iterator:
    """
    Purged K-Fold cross-validation generator

    Yields: (train_idx, test_idx) for each fold
    """

def compute_roi_per_threshold(y_true, y_pred, returns, thresholds, transaction_cost=0.001) -> pd.DataFrame:
    """
    Compute ROI across different probability thresholds

    Returns: DataFrame with threshold, gross_pnl, net_pnl, accuracy, num_trades
    """

def compute_sharpe_ratio(returns, risk_free_rate=0.0) -> float:
    """
    Compute annualized Sharpe ratio

    Returns: Sharpe ratio (float)
    """

def compute_max_drawdown(cumulative_returns) -> float:
    """
    Compute maximum drawdown

    Returns: Max drawdown (negative value)
    """

def print_metrics_table(metrics_dict):
    """
    Print formatted comparison table of model metrics
    """
```

**Dependencies:** sklearn.metrics, pandas, numpy, config

---

## Design Patterns

### 1. Module Pattern

**Principle:** Each module has a single, well-defined responsibility

**Benefits:**
- Easy to test in isolation
- Clear separation of concerns
- Reusable across different scripts

**Example:**
```python
# data_utils.py handles ONLY data loading
# feature_engineering.py handles ONLY indicator computation
# model_utils.py handles ONLY model creation
```

### 2. Configuration Centralization

**Principle:** All hyperparameters in one place (config.py)

**Benefits:**
- Easy to modify settings
- Consistent across all scripts
- Version control for experiments

**Example:**
```python
# Instead of hardcoding:
roc_windows = [1, 3, 42]  # BAD

# Import from config:
from src.config import ROC_WINS
roc_windows = ROC_WINS  # GOOD
```

### 3. Pipeline Pattern (sklearn)

**Principle:** Chain transformations in reusable pipelines

**Benefits:**
- Fit once, transform many times
- Prevents data leakage
- Easy to save/load

**Example:**
```python
# Create pipeline
preprocessor = ColumnTransformer([
    ('bounded', 'passthrough', bounded_cols),
    ('loggy', Pipeline([
        ('log', FunctionTransformer(signed_log1p)),
        ('scale', StandardScaler())
    ]), loggy_cols),
    ('standard', StandardScaler(), the_rest)
])

# Fit on train, transform both
preprocessor.fit(X_train)
X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)  # No leakage!
```

### 4. Strategy Pattern (Ensemble Methods)

**Principle:** Encapsulate different algorithms (ensemble strategies)

**Benefits:**
- Easy to add new ensemble methods
- Consistent interface
- Testable

**Example:**
```python
# Different strategies, same interface
ensemble1 = create_stacked_ensemble(X, y, base_preds)
ensemble2 = create_weighted_ensemble(base_preds, weights_brier)
ensemble3 = create_weighted_ensemble(base_preds, weights_logloss)

# All return predictions in the same format
```

### 5. Iterator Pattern (Cross-Validation)

**Principle:** Lazy generation of fold indices

**Benefits:**
- Memory efficient
- Flexible (can break early)
- Clean interface

**Example:**
```python
# Generator yields fold indices one at a time
for train_idx, test_idx in purged_kfold_cv(X, y):
    # Process fold
    ...
    # Can break early if needed
```

---

## Code Quality Practices

### 1. Type Hints (Recommended)

```python
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators"""
    ...

def create_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """Create binary labels"""
    ...
```

### 2. Docstrings

```python
def purged_kfold_cv(X, y, n_splits=4, embargo_hours=24):
    """
    Purged K-Fold cross-validation to avoid look-ahead bias

    Args:
        X: Feature DataFrame with datetime index
        y: Label series
        n_splits: Number of folds (default: 4)
        embargo_hours: Hours to purge between train/test (default: 24)

    Yields:
        train_idx, test_idx: Arrays of row indices for each fold
    """
    ...
```

### 3. Error Handling

```python
def load_crypto_csv(csv_path='crypto.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'{csv_path} not found')

    if 'datetime' not in raw.columns and 'OpenDt' not in raw.columns:
        raise ValueError('CSV must have datetime or OpenDt column')
```

### 4. Constants Over Magic Numbers

```python
# BAD
if rsi < 30:  # What is 30?
    ...

# GOOD
RSI_OVERSOLD_THRESHOLD = 30
if rsi < RSI_OVERSOLD_THRESHOLD:
    ...
```

---

## Testing Strategy (Not Implemented)

**Recommended test structure:**

```
tests/
├── test_data_utils.py        # Test data loading
├── test_feature_eng.py        # Test feature computation
├── test_preprocessing.py      # Test scaling pipelines
├── test_model_utils.py        # Test model creation
├── test_evaluation.py         # Test metrics
└── test_integration.py        # End-to-end tests
```

**Example unit test:**
```python
# tests/test_data_utils.py
import pytest
from src.data_utils import signed_log1p

def test_signed_log1p():
    df = pd.DataFrame({'x': [-10, -1, 0, 1, 10]})
    result = signed_log1p(df)

    assert result['x'].iloc[0] < 0  # Negative stays negative
    assert result['x'].iloc[2] == 0  # Zero stays zero
    assert result['x'].iloc[4] > 0  # Positive stays positive
    assert abs(result['x'].iloc[1]) == abs(result['x'].iloc[3])  # Symmetric
```

---

## Extensibility

### Adding a New Feature

1. **Define parameters in config.py:**
```python
MY_CUSTOM_WINDOW = 30
```

2. **Add computation in feature_engineering.py:**
```python
def compute_features(df):
    # ... existing features ...

    # Add new feature
    feat['my_custom_feature'] = compute_my_indicator(
        df['close'],
        window=MY_CUSTOM_WINDOW
    )

    return feat
```

3. **Categorize for preprocessing:**
```python
def categorize_feature_columns(cols):
    loggy_keywords = [..., 'my_custom']
    # ... rest of function
```

### Adding a New Model

1. **Add to model_utils.py:**
```python
def get_base_models():
    models = {}
    # ... existing models ...

    # Add new model
    models['MyNewModel'] = MyNewModel(
        param1=value1,
        param2=value2
    )

    return models
```

2. **No other changes needed!** The pipeline will automatically:
   - Train the new model
   - Calibrate its predictions
   - Include it in ensembles
   - Evaluate its performance

---

## Performance Optimization Opportunities

### 1. Parallel Symbol Processing

**Current:** Sequential per symbol
```python
for symbol in symbols:
    features[symbol] = compute_features(data[symbol])
```

**Optimized:** Parallel processing
```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(compute_features, symbol_data)
```

**Speedup:** ~4x on 4 cores

### 2. Feature Selection

**Current:** Use all 31 features

**Optimized:** Select top 15 by importance
```python
top_features = feature_importance.nlargest(15).index
X = X[top_features]
```

**Speedup:** ~2x (fewer features to scale/train on)

### 3. Model Caching

**Current:** Retrain every run

**Optimized:** Cache trained models
```python
if os.path.exists('models_cache.pkl'):
    models = joblib.load('models_cache.pkl')
else:
    models = train_models(X, y)
    joblib.dump(models, 'models_cache.pkl')
```

---

## Summary

The codebase follows clean architecture principles:

✅ **Modular design** - Each module has one responsibility
✅ **Centralized config** - All hyperparameters in config.py
✅ **Clear data flow** - Easy to trace from input to output
✅ **Reusable components** - Functions can be imported and used independently
✅ **Extensible** - Easy to add new features, models, ensembles

**Total Source Code:** ~27 KB across 6 modules

**Lines of Code:**
- config.py: ~100 lines
- data_utils.py: ~200 lines
- feature_engineering.py: ~230 lines
- preprocessing.py: ~40 lines
- model_utils.py: ~180 lines
- evaluation.py: ~200 lines
- **Total: ~950 lines** (clean, documented code)

---

**Next:** See [Usage Guide](usage.md) for how to use the system.
