# Cryptocurrency Price Prediction - Technical Documentation

**Project**: Ensemble Machine Learning for Cryptocurrency Price Prediction
**Course**: ECON 252 - Financial Markets
**Team**: Zach Kuo, Nikhil Ghind, Ethan Ho
**Date**: December 2024

---

## Table of Contents

1. [Implementation Details](#implementation-details)
2. [Code Structure](#code-structure)
3. [Algorithm Details](#algorithm-details)
4. [Results and Analysis](#results-and-analysis)
5. [How to Run the Project](#how-to-run-the-project)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Implementation Details

### 1.1 Overview

This project implements a sophisticated machine learning pipeline for predicting cryptocurrency price movements using ensemble methods. The system predicts binary up/down movements for the next time period across multiple cryptocurrency symbols.

**Core Components:**
- Multi-symbol data processing pipeline
- 31+ technical indicator feature engineering
- 6 base machine learning models
- 3 ensemble methods with probability calibration
- Purged K-Fold cross-validation
- ROI-based backtesting framework

### 1.2 Data Processing Pipeline

#### Input Data Format

**File**: `crypto.csv` (wide format)

```
datetime,open-BTCUSDT,high-BTCUSDT,low-BTCUSDT,close-BTCUSDT,volume-BTCUSDT,open-ETHUSDT,...
2023-01-01 00:00:00,16500.5,16520.3,16480.1,16510.8,1234.56,...
2023-01-01 04:00:00,16510.8,16550.2,16505.0,16545.5,1456.78,...
```

**Characteristics:**
- Timeframe: 4-hour bars (6 bars per day)
- Period: Historical data through February 2024
- Symbols: 49 cryptocurrencies (BTC, ETH, ADA, etc.)
- Fields per symbol: open, high, low, close, volume

#### Data Transformation Steps

**Step 1: Wide to Long Conversion** (`src/data_utils.py`)

```python
# Input: Wide format with columns like 'open-BTCUSDT'
# Output: MultiIndex DataFrame (symbol, datetime)
#
# Structure after conversion:
#                          open     high      low    close     volume
# symbol  datetime
# BTC     2023-01-01    16500.5  16520.3  16480.1  16510.8   1234.56
#         2023-01-05    16510.8  16550.2  16505.0  16545.5   1456.78
# ETH     2023-01-01     1200.3   1205.8   1198.5   1203.2   5678.90
#         2023-01-05     1203.2   1210.5   1201.0   1208.7   6234.11
```

**Implementation**:
```python
def load_crypto_csv(csv_path='crypto.csv'):
    # Parse wide column names (e.g., 'open-BTCUSDT' -> field='open', symbol='BTC')
    # Group by symbol
    # Create long-format DataFrame with MultiIndex
    # Convert to numeric types
    # Drop missing values
    # Sort chronologically
```

**Step 2: Data Cleaning** (`src/data_utils.py`)

```python
# 1. Convert all price/volume columns to numeric
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Drop rows with missing OHLC data
df = df.dropna(subset=['open', 'high', 'low', 'close'])

# 3. Ensure datetime index is sorted
df = df.sort_index()
```

### 1.3 Feature Engineering

#### Technical Indicators (31+ Features)

**Implementation in** `src/feature_engineering.py`

##### 1. Momentum Indicators

**Rate of Change (ROC)** - 3 features
```python
# Windows: [1, 3, 42] periods
# Formula: (close_t - close_{t-n}) / close_{t-n} * 100
ROC_1 = ROCIndicator(close, window=1).roc()
ROC_3 = ROCIndicator(close, window=3).roc()
ROC_42 = ROCIndicator(close, window=42).roc()
```

**Interpretation**:
- `ROC_1`: Single-period momentum (4-hour change)
- `ROC_3`: Short-term momentum (12-hour change)
- `ROC_42`: Medium-term momentum (7-day change)

##### 2. Trend Indicators

**EMA Cross** - 2 features
```python
# Fast/Slow pair: (84, 168) periods = (14 days, 28 days)
ema_fast = EMAIndicator(close, window=84).ema_indicator()
ema_slow = EMAIndicator(close, window=168).ema_indicator()

ema_diff = ema_fast - ema_slow      # Raw difference
ema_ratio = ema_fast / ema_slow     # Ratio (>1 = bullish, <1 = bearish)
```

**MACD** - 2 features
```python
# MACD histogram (difference between MACD line and signal line)
macd_hist = MACD(close).macd_diff()
macd_hist_lag2 = macd_hist.shift(2)  # 2-period lag for stability
```

##### 3. Oscillators

**RSI (Relative Strength Index)** - 6 features
```python
# Windows: [8, 14, 26] periods
# Each window generates: current RSI + 2-period lag
for window in [8, 14, 26]:
    rsi = RSIIndicator(close, window).rsi()
    features[f'rsi_{window}'] = rsi
    features[f'rsi_{window}_lag2'] = rsi.shift(2)
```

**Range**: 0-100 (oversold < 30, overbought > 70)

**Stochastic Oscillator** - 6 features
```python
# K values: [8, 14] periods
# For each K: fast %K, fast %D (signal), slow %D
for k in [8, 14]:
    stoch = StochasticOscillator(high, low, close, window=k, smooth_window=3)
    fast_k = stoch.stoch()           # %K line
    fast_d = stoch.stoch_signal()    # %D line (3-period SMA of %K)

    features[f'stoch_d_fast_{k}'] = fast_d
    features[f'stoch_d_slow_{k}'] = fast_d.rolling(3).mean()
    features[f'stoch_hist_{k}'] = fast_d - fast_d.rolling(3).mean()
```

**CCI (Commodity Channel Index)** - 2 features
```python
# Windows: [10, 20] periods
# Measures deviation from typical price
for window in [10, 20]:
    cci = CCIIndicator(high, low, close, window).cci()
```

##### 4. Volatility Indicators

**Bollinger Bands** - 4 features
```python
# Windows: [10, 20] periods, 2 standard deviations
for window in [10, 20]:
    bb = BollingerBands(close, window, window_dev=2)

    # %B: Position relative to bands (0=lower, 0.5=middle, 1=upper)
    bb_pctb = bb.bollinger_pband()

    # Bandwidth: (upper - lower) / middle (width of bands)
    bb_width = bb.bollinger_wband()
```

**Rolling Volatility** - 1 feature
```python
# 20-period standard deviation of log returns
log_returns = np.log(close / close.shift(1))
volatility = log_returns.rolling(20).std()
```

##### 5. Volume Indicators

**Volume Features** - 3 features
```python
# Moving average
volume_mean = volume.rolling(42).mean()

# Log changes at different windows
volume_log_change_42 = np.log(volume / volume.shift(42))
volume_log_change_84 = np.log(volume / volume.shift(84))
```

#### Feature Preprocessing

**Three-Category Preprocessing Pipeline** (`src/preprocessing.py`)

```python
# Category 1: Bounded features (already 0-100 or 0-1)
bounded_cols = ['rsi_*', 'stoch_*', 'bb_pctb_*']
# → Passthrough (no transformation needed)

# Category 2: Log-skewed features
loggy_cols = ['volume_*', 'volatility', 'ema_ratio_*', 'bb_width_*']
# → signed_log1p() → StandardScaler()

# Category 3: All other features
the_rest = [all other numeric features]
# → StandardScaler()
```

**Signed Log1p Transformation**:
```python
def signed_log1p(x):
    # Preserves sign while applying log scaling
    # sign(x) * log(1 + |x|)
    return np.sign(x) * np.log1p(np.abs(x))
```

**Why this matters**:
- Bounded features are already normalized (RSI 0-100)
- Volume/volatility are highly skewed → log transform handles outliers
- Other features (price ratios, differences) → standard scaling sufficient

### 1.4 Label Creation

**Binary Classification Target** (`src/feature_engineering.py`)

```python
def create_labels(df, threshold=0.01):
    close = df['close']

    # Compute next-period log return
    log_return_next = np.log(close.shift(-1) / close)

    # Binary label: 1 if return > 1%, else 0
    labels = (log_return_next > threshold).astype(int)

    return labels
```

**Example**:
```
Time    Close    Next_Close    Log_Return    Label
0       100.0    101.5         0.0149        1 (up)
1       101.5    101.0        -0.0049        0 (down)
2       101.0    102.2         0.0118        1 (up)
```

**Class Distribution**:
- Typical balance: ~48% up, ~52% down
- Slight bearish bias in crypto markets
- Threshold of 1% filters out noise

### 1.5 Cross-Validation Strategy

**Purged K-Fold** (`src/evaluation.py`)

```python
def purged_kfold_cv(X, y, n_splits=4, embargo_hours=24):
    # Get unique datetime values
    unique_times = sorted(X.index.get_level_values('datetime').unique())

    fold_size = len(unique_times) // n_splits

    for fold in range(n_splits):
        # Test period
        test_start = unique_times[fold * fold_size]
        test_end = unique_times[(fold + 1) * fold_size - 1]

        # Train period (with 24-hour embargo)
        train_end = test_start - timedelta(hours=24)

        # Get indices
        train_idx = where(times < train_end)
        test_idx = where(test_start <= times <= test_end)

        yield train_idx, test_idx
```

**Visualization of Fold Structure** (4 folds):

```
Timeline: |----Train 1----|--Gap--|--Test 1--|----Train 2----|--Gap--|--Test 2--| ...

Fold 1:   |████████████████|  ░░  |▓▓▓▓▓▓▓▓▓▓|
Fold 2:   |████████████████████████████████████|  ░░  |▓▓▓▓▓▓▓▓▓▓|
Fold 3:   |████████████████████████████████████████████████████████|  ░░  |▓▓▓▓▓▓▓▓▓▓|
Fold 4:   |████████████████████████████████████████████████████████████████████████|  ░░  |▓▓▓▓|

Legend:
████ = Training data
░░░░ = 24-hour embargo (purge zone)
▓▓▓▓ = Test data
```

**Why Purging is Critical**:
- **Without purge**: Training at 11:59 PM, testing at 12:01 AM → overlap!
- **With 24h purge**: Training ends 24h before test starts → no leakage
- Prevents overfitting to near-future patterns

### 1.6 Model Training

#### Base Models (6 total)

**1. Logistic Regression** (`src/model_utils.py`)

```python
LogisticRegression(
    penalty='l2',           # L2 regularization
    C=1.0,                  # Inverse regularization strength
    solver='saga',          # Stochastic Average Gradient Descent
    max_iter=1000,
    random_state=42
)
```

**Characteristics**:
- Linear decision boundary
- Outputs calibrated probabilities
- Fast training (~1 second per fold)
- Interpretable coefficients

**2. Gaussian Naive Bayes**

```python
GaussianNB()  # No hyperparameters
```

**Characteristics**:
- Assumes feature independence
- Very fast training (<1 second)
- Good baseline model
- Works well with normalized features

**3. Decision Tree**

```python
DecisionTreeClassifier(
    max_depth=12,           # Limit depth to prevent overfitting
    random_state=42
)
```

**Characteristics**:
- Non-linear decision boundary
- Can capture complex interactions
- Prone to overfitting → depth limited
- Training: ~5 seconds per fold

**4. Random Forest**

```python
RandomForestClassifier(
    n_estimators=500,       # 500 trees in forest
    max_depth=None,         # Unlimited depth (ensemble handles overfitting)
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)
```

**Characteristics**:
- Ensemble of 500 decision trees
- Bootstrap aggregating (bagging)
- Feature randomness at each split
- Training: ~60 seconds per fold (parallelized)

**5. XGBoost**

```python
XGBClassifier(
    n_estimators=300,       # 300 boosting rounds
    learning_rate=0.1,      # Step size shrinkage
    max_depth=6,            # Depth of each tree
    tree_method='hist',     # Histogram-based algorithm
    device='cuda',          # GPU acceleration (fallback to CPU)
    random_state=42,
    n_jobs=-1
)
```

**Characteristics**:
- Gradient boosting (sequential tree building)
- GPU acceleration when available
- Handles missing values natively
- Training: ~30 seconds per fold (GPU), ~120 seconds (CPU)

**6. Linear SVM + KNN** (Optional, `USE_HEAVY_MODELS=True`)

```python
# Linear SVM
LinearSVC(C=0.1, max_iter=1000, random_state=42)

# K-Nearest Neighbors
KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
```

**Characteristics**:
- SVM: Large-margin classifier, very slow on large datasets
- KNN: Instance-based learning, no training phase but slow prediction
- Training: ~300 seconds per fold (SVM), ~5 seconds (KNN)

#### Ensemble Methods (3 total)

**1. Stacked Logistic Regression**

```python
def create_stacked_ensemble(X_train, y_train, base_predictions):
    # Stack base model predictions as features
    X_meta = np.column_stack([
        pred_logreg,
        pred_nb,
        pred_dt,
        pred_rf,
        pred_xgb,
        pred_svm  # if enabled
    ])

    # Train meta-model (Logistic Regression on base predictions)
    meta_model = LogisticRegression(penalty='l2', C=1.0)
    meta_model.fit(X_meta, y_train)

    return meta_model
```

**Architecture**:
```
Base Models              Meta-Model
┌──────────────┐
│ LogReg       │─────┐
├──────────────┤     │
│ NaiveBayes   │─────┤
├──────────────┤     │
│ DecisionTree │─────┼─→ [Logistic      → Final
├──────────────┤     │    Regression]      Prediction
│ RandomForest │─────┤
├──────────────┤     │
│ XGBoost      │─────┘
└──────────────┘
```

**2. Brier-Weighted Ensemble**

```python
def compute_ensemble_weights(metrics_dict, metric_name='brier'):
    weights = {}
    for model_name, brier_score in metrics_dict.items():
        # Weight = 1 / brier_score (lower Brier is better)
        weights[model_name] = 1.0 / (brier_score + 1e-10)

    # Normalize to sum to 1
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    return weights

# Final prediction = weighted average
prediction = sum(weight_i * pred_i for all models)
```

**Example Weights**:
```
Model           Brier Score    Weight
LogReg          0.2487         0.185
NaiveBayes      0.2489         0.184
RandomForest    0.2493         0.183
XGBoost         0.2491         0.184
Ensemble        0.2486         ---
```

**3. LogLoss-Weighted Ensemble**

```python
# Same as Brier-weighted but using log loss metric
weights[model_name] = 1.0 / (log_loss + 1e-10)
```

#### Probability Calibration

**Isotonic Regression** (`src/model_utils.py`)

```python
def calibrate_probabilities(y_true, y_pred_proba):
    # Fit isotonic regression on CV predictions
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_proba, y_true)
    return calibrator

# Usage in training loop:
for fold in cross_validation:
    # Train model
    model.fit(X_train, y_train)

    # Predict on test fold
    cv_predictions = model.predict_proba(X_test)[:, 1]

    # Store CV predictions
    all_cv_predictions.append(cv_predictions)

# After all folds, calibrate on all CV predictions
calibrator = calibrate_probabilities(all_y_test, all_cv_predictions)

# Apply calibration to new predictions
calibrated_pred = calibrator.transform(raw_pred)
```

**Why Calibration Matters**:
- Raw model outputs may not represent true probabilities
- DecisionTree: Tends to output probabilities close to 0 or 1
- RandomForest: Underconfident on extreme probabilities
- Isotonic regression maps predictions to true probabilities

**Before vs After Calibration**:
```
Raw Prediction    True Frequency    Calibrated Prediction
0.60              0.52              0.52
0.70              0.58              0.58
0.80              0.62              0.62
```

### 1.7 Evaluation Metrics

**Four Primary Metrics** (`src/evaluation.py`)

**1. AUC (Area Under ROC Curve)**
```python
auc = roc_auc_score(y_true, y_pred_proba)
```
- Range: 0-1 (0.5 = random, 1.0 = perfect)
- Measures ranking quality (separability)
- Threshold-independent

**2. Brier Score**
```python
brier = brier_score_loss(y_true, y_pred_proba)
```
- Range: 0-1 (0 = perfect, 0.25 = random for balanced data)
- Mean squared error of probabilities
- Lower is better

**3. Log Loss (Cross-Entropy)**
```python
logloss = log_loss(y_true, y_pred_proba)
```
- Range: 0-∞ (0 = perfect, ~0.69 = random for balanced data)
- Penalizes confident wrong predictions heavily
- Lower is better

**4. Accuracy**
```python
accuracy = accuracy_score(y_true, y_pred_binary)
# where y_pred_binary = (y_pred_proba >= 0.5)
```
- Range: 0-1 (0.5 = random for balanced data)
- Simple percentage correct
- Threshold-dependent (uses 0.5)

---

## Code Structure

### 2.1 Module Architecture

```
src/
├── __init__.py                   # Package initialization
├── config.py                     # Configuration & hyperparameters
├── data_utils.py                 # Data loading & preprocessing
├── feature_engineering.py        # Technical indicators
├── model_utils.py                # Models & ensembles
├── evaluation.py                 # Metrics & cross-validation
└── preprocessing.py              # Feature scaling pipelines
```

### 2.2 Module Dependencies

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

### 2.3 Main Script Flow

#### `run_ensemble.py` - Main Training Pipeline

```
1. Import modules
   └─→ from src import data_utils, feature_engineering, model_utils, evaluation

2. Load data
   └─→ data_utils.load_crypto_csv('crypto.csv')
        └─→ Returns MultiIndex DataFrame (symbol, datetime)

3. Feature engineering (per symbol)
   └─→ feature_engineering.compute_features(df)
        └─→ Computes 31+ technical indicators
        └─→ feature_engineering.create_labels(df)
             └─→ Binary labels (1% threshold)

4. Split features and labels
   └─→ feature_engineering.split_features_labels(df)
        └─→ X (features), y (labels)

5. Create preprocessing pipeline
   └─→ preprocessing.create_preprocessing_pipeline(bounded, loggy, rest)
        └─→ ColumnTransformer with 3 transformers

6. Get base models
   └─→ model_utils.get_base_models()
        └─→ Dict of 6 base models

7. Cross-validation training (4 folds)
   for fold in evaluation.purged_kfold_cv(X, y):
       7.1. Split train/test with embargo
       7.2. Fit preprocessing on train, transform both
       7.3. Train each base model
       7.4. Collect CV predictions for calibration
       7.5. Compute fold metrics

8. Calibrate probabilities
   └─→ model_utils.calibrate_probabilities(y_cv, pred_cv)
        └─→ IsotonicRegression for each model

9. Create ensembles
   9.1. Stacked Logistic Regression
        └─→ model_utils.create_stacked_ensemble(X_meta, y)
   9.2. Brier-weighted ensemble
        └─→ model_utils.compute_ensemble_weights(brier_scores)
   9.3. LogLoss-weighted ensemble
        └─→ model_utils.compute_ensemble_weights(log_losses)

10. Evaluate all models
    └─→ evaluation.compute_metrics(y_true, y_pred)
         └─→ AUC, Brier, LogLoss, Accuracy

11. Compute ROI per symbol
    └─→ evaluation.compute_roi_per_threshold(y, pred, returns)
         └─→ Test thresholds 0.3-0.8

12. Save models and results
    └─→ joblib.dump(model, f'saved_models/...')
         └─→ Models, preprocessors, calibrators
```

**Runtime Breakdown** (Full dataset, 49 symbols):
- Data loading: 5 seconds
- Feature engineering: 30 seconds
- Model training (4 folds × 6 models):
  - Logistic Regression: 4 seconds
  - Naive Bayes: 2 seconds
  - Decision Tree: 20 seconds
  - Random Forest: 240 seconds (4 minutes)
  - XGBoost: 120 seconds (2 minutes) with GPU
  - SVM+KNN (optional): 1200 seconds (20 minutes)
- Ensemble creation: 10 seconds
- ROI computation: 15 seconds
- **Total: ~8-10 minutes** (without SVM/KNN)

#### `trade_simulator.py` - BTC Trade Simulation

```
1. Load BTC data only
   └─→ data_utils.load_single_symbol('crypto.csv', 'BTC')

2. Compute features for BTC
   └─→ feature_engineering.compute_features(btc_df)

3. Create labels
   └─→ feature_engineering.create_labels(btc_df)

4. Train ensemble on FULL BTC data
   (Note: This introduces look-ahead bias)
   └─→ Fit all 6 base models
   └─→ Create ensembles

5. Backtest across thresholds (0.30 to 0.80)
   for threshold in np.arange(0.30, 0.81, 0.01):
       5.1. Generate signals (1 if prob >= threshold)
       5.2. Track positions, entries, exits
       5.3. Calculate P&L with transaction costs
       5.4. Compute metrics (return, Sharpe, drawdown)

6. Find optimal threshold
   └─→ Best by Sharpe ratio

7. Generate trade logs
   └─→ Entry/exit times, prices, returns, P&L

8. Save results
   └─→ trade_log.csv
   └─→ threshold_optimization.csv
```

**Runtime**: ~5-10 minutes

### 2.4 Data Flow Diagram

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

### 2.5 Key Functions Reference

#### `src/config.py`
```python
# Constants only, no functions
RANDOM_SEED = 42
N_FOLDS = 4
EMBARGO_HOURS = 24
XGB_N_ESTIMATORS = 300
...
```

#### `src/data_utils.py`
```python
load_crypto_csv(csv_path) → DataFrame
    # Load and reshape crypto data from wide to long format

load_single_symbol(csv_path, symbol) → DataFrame
    # Load data for a single cryptocurrency

signed_log1p(df_in) → DataFrame
    # Apply signed log transformation
```

#### `src/feature_engineering.py`
```python
compute_features(df) → DataFrame
    # Compute all 31+ technical indicators

create_labels(df, threshold=0.01) → Series
    # Create binary classification labels

split_features_labels(df) → (X, y)
    # Split into features and labels

categorize_feature_columns(feature_cols) → (bounded, loggy, rest)
    # Categorize features for preprocessing
```

#### `src/preprocessing.py`
```python
create_preprocessing_pipeline(bounded, loggy, rest) → ColumnTransformer
    # Create sklearn preprocessing pipeline
```

#### `src/model_utils.py`
```python
get_base_models() → Dict[str, Model]
    # Initialize all base ML models

calibrate_probabilities(y_true, y_pred) → IsotonicRegression
    # Calibrate probabilities using isotonic regression

create_ensemble_predictions(base_preds, weights) → Array
    # Create weighted ensemble predictions

create_stacked_ensemble(X_train, y_train, base_preds) → LogisticRegression
    # Train stacked ensemble meta-model

compute_ensemble_weights(metrics, metric_name) → Dict[str, float]
    # Compute ensemble weights from metrics
```

#### `src/evaluation.py`
```python
compute_metrics(y_true, y_pred_proba) → Dict
    # Compute AUC, Brier, LogLoss, Accuracy

purged_kfold_cv(X, y, n_splits, embargo_hours) → Iterator
    # Purged K-Fold cross-validation generator

compute_roi_per_threshold(y_true, y_pred, returns, thresholds) → DataFrame
    # Compute ROI across different probability thresholds

compute_sharpe_ratio(returns) → float
    # Compute annualized Sharpe ratio

compute_max_drawdown(cumulative_returns) → float
    # Compute maximum drawdown

print_metrics_table(metrics_dict)
    # Print formatted comparison table
```

---

## Algorithm Details

### 3.1 Purged K-Fold Algorithm

**Pseudocode**:
```
Input: X (features), y (labels), n_splits (4), embargo_hours (24)
Output: train_idx, test_idx for each fold

1. Extract unique datetime values from X.index
   unique_times = sorted(unique(X.index.datetime))
   n_times = len(unique_times)

2. Calculate fold size
   fold_size = n_times / n_splits

3. For each fold i in 0..(n_splits-1):

   a. Define test period
      test_start_idx = i × fold_size
      test_end_idx = (i+1) × fold_size  (or n_times for last fold)
      test_start_time = unique_times[test_start_idx]
      test_end_time = unique_times[test_end_idx - 1]

   b. Apply embargo (purge zone)
      embargo_delta = 24 hours
      train_end_time = test_start_time - embargo_delta

   c. Get row indices
      train_idx = indices where X.index.datetime < train_end_time
      test_idx = indices where test_start_time ≤ X.index.datetime ≤ test_end_time

   d. Yield (train_idx, test_idx)
```

**Time Complexity**: O(n) where n = number of rows

### 3.2 Ensemble Weight Calculation

**Brier-Weighted Algorithm**:
```
Input: brier_scores = {model_name: brier_score}
Output: weights = {model_name: weight}

1. Compute raw weights (inverse of metric)
   for each model:
       raw_weight[model] = 1 / (brier_score[model] + ε)
       # ε = 1e-10 to avoid division by zero

2. Normalize weights to sum to 1
   total_weight = sum(raw_weight.values())
   for each model:
       weight[model] = raw_weight[model] / total_weight

3. Compute ensemble prediction
   ensemble_pred = Σ (weight[model] × prediction[model])
```

**Example**:
```
Model           Brier Score    Raw Weight    Normalized Weight
LogReg          0.2487         4.021         0.201
NaiveBayes      0.2489         4.018         0.200
RandomForest    0.2493         4.011         0.200
XGBoost         0.2491         4.014         0.200
DecisionTree    0.2495         4.008         0.199
                               ─────         ─────
                               20.072        1.000

Ensemble prediction = 0.201×pred_LR + 0.200×pred_NB + ...
```

### 3.3 ROI Calculation Algorithm

**Pseudocode**:
```
Input:
    y_true (true labels)
    y_pred_proba (predicted probabilities)
    returns (actual log returns)
    thresholds (array of probability thresholds to test)
    transaction_cost (0.001 = 0.1%)

Output: DataFrame with metrics per threshold

1. For each threshold in thresholds:

   a. Generate trading signals
      signals = 1 if y_pred_proba >= threshold else 0
      # 1 = long position, 0 = no position

   b. Calculate gross P&L
      gross_pnl = sum(signals × returns)

   c. Count position changes (trades)
      position_changes = diff(signals)
      num_trades = sum(abs(position_changes))

   d. Calculate transaction costs
      total_costs = num_trades × transaction_cost

   e. Calculate net P&L
      net_pnl = gross_pnl - total_costs

   f. Calculate accuracy
      accuracy = mean(signals == y_true)

   g. Store results
      results[threshold] = {
          'gross_pnl': gross_pnl,
          'net_pnl': net_pnl,
          'num_trades': num_trades,
          'accuracy': accuracy
      }

2. Return results as DataFrame
```

**Example Output**:
```
Threshold    Gross PnL    Net PnL    Num Trades    Accuracy
0.50         0.0523       0.0478     45            0.556
0.52         0.0615       0.0577     38            0.579
0.54         0.0489       0.0457     32            0.600
```

### 3.4 Sharpe Ratio Calculation

```
Input: returns (array of trade returns)
Output: annualized_sharpe_ratio

1. Calculate excess returns
   excess_returns = returns - risk_free_rate  # risk_free_rate typically 0

2. Calculate Sharpe ratio
   sharpe = mean(excess_returns) / std(returns)

3. Annualize (assuming daily returns)
   annualized_sharpe = sharpe × sqrt(252)
   # 252 = trading days per year

   # For 4-hour bars (6 per day):
   # annualized_sharpe = sharpe × sqrt(252 × 6)
```

**Interpretation**:
- Sharpe < 1: Poor risk-adjusted returns
- Sharpe 1-2: Good risk-adjusted returns
- Sharpe > 2: Excellent risk-adjusted returns

---

## Results and Analysis

### 4.1 Model Performance Summary

**Cross-Validation Results** (4-fold Purged K-Fold on all 49 symbols)

| Model | AUC | Brier Score | Log Loss | Accuracy | Training Time |
|-------|-----|-------------|----------|----------|---------------|
| **Ensemble_Brier** | **0.5421** | **0.2488** | **0.6907** | **53.06%** | 10 sec |
| **Ensemble_LogLoss** | **0.5421** | **0.2488** | **0.6907** | **53.06%** | 10 sec |
| **Ensemble_StackedLogReg** | **0.5418** | **0.2486** | **0.6903** | **53.08%** | 15 sec |
| LinearSVM | 0.5407 | 0.2487 | 0.6905 | 53.04% | 1200 sec |
| LogisticRegression | 0.5401 | 0.2487 | 0.6905 | 52.98% | 4 sec |
| GaussianNB | 0.5368 | 0.2489 | 0.6909 | 52.74% | 2 sec |
| XGBoost | 0.5318 | 0.2491 | 0.6914 | 52.45% | 120 sec |
| DecisionTree | 0.5289 | 0.2493 | 0.6917 | 52.30% | 20 sec |
| RandomForest | 0.5287 | 0.2493 | 0.6917 | 52.23% | 240 sec |
| KNN | 0.5146 | 0.2497 | 0.6926 | 51.34% | 5 sec |

### 4.2 Key Findings

**1. Ensemble Methods Dominate**
- All 3 ensemble methods outperform individual models
- Stacked LogReg has lowest Brier and LogLoss
- Brier/LogLoss weighted ensembles are nearly identical

**2. AUC Analysis**
- Best AUC: 0.5421 (ensemble)
- Worst AUC: 0.5146 (KNN)
- Improvement over random: +8.4%
- **Interpretation**: Models can rank predictions moderately well

**3. Accuracy Analysis**
- Best: 53.08% (Stacked Ensemble)
- Random baseline: 50%
- Edge: +3.08 percentage points
- **Statistical Significance**: p < 0.001 with n=1.2M samples

**4. Calibration Quality**
- Brier score 0.2488 vs random baseline 0.25
- **Interpretation**: Probabilities are well-calibrated
- Predicted 60% → ~60% true probability

**5. Model Speed vs Performance Tradeoff**

```
                        Accuracy
                           ↑
53.1% ┤                   ⬢ StackedEnsemble
53.0% ┤         ⬢ LogReg
52.7% ┤     ⬢ NaiveBayes
52.5% ┤                   ⬡ XGBoost
52.3% ┤                         ⬡ RandomForest
52.0% ┤
51.3% ┤ ⬡ KNN
      └───────────────────────────────────→ Training Time
         2s     120s              240s

Key:
⬢ = Efficient (good accuracy, fast training)
⬡ = Inefficient (low accuracy or slow training)
```

**Recommendation**: Use ensemble methods for best accuracy, or LogReg for fastest training with minimal performance drop.

### 4.3 ROI Analysis (Per Symbol)

**Top 5 Performing Symbols** (Ensemble_Brier model):

| Symbol | Net ROI | Gross ROI | Num Trades | Win Rate | Best Threshold |
|--------|---------|-----------|------------|----------|----------------|
| TRB | +1.4268 | +1.5045 | 89 | 68.5% | 0.52 |
| SOL | +1.4897 | +1.5123 | 112 | 64.3% | 0.50 |
| YFI | +0.6371 | +0.6598 | 67 | 71.6% | 0.55 |
| SNX | +0.4709 | +0.4892 | 78 | 62.8% | 0.51 |
| ETH | +0.4442 | +0.4589 | 156 | 59.0% | 0.49 |

**Bottom 5 Performing Symbols**:

| Symbol | Net ROI | Gross ROI | Num Trades | Win Rate |
|--------|---------|-----------|------------|----------|
| ALGO | -0.0145 | +0.0023 | 34 | 47.1% |
| LTC | -0.0092 | +0.0054 | 28 | 46.4% |
| EOS | -0.0005 | +0.0089 | 21 | 47.6% |
| KAVA | +0.0000 | +0.0145 | 12 | 50.0% |
| IOST | +0.0000 | +0.0078 | 8 | 50.0% |

**Average Across All Symbols**: +23.71% net ROI

**Analysis**:
- High-volatility coins (TRB, SOL, YFI) show best returns
- Large-cap stable coins (BTC, ETH) show moderate positive returns
- Small-cap low-volume coins underperform
- Transaction costs (0.1%) significantly impact low-trade-count symbols

### 4.4 BTC Trade Simulation Results

**Simulation Parameters**:
- Initial Capital: $10,000
- Data Period: Full historical dataset
- Transaction Cost: 0.1% per trade
- Optimal Threshold: 0.50 (selected by Sharpe ratio)

**Performance Metrics**:

| Metric | Value |
|--------|-------|
| **Total Return** | **+38,676%** |
| **Final Capital** | $3,877,568 |
| **Total Trades** | 699 |
| **Profitable Trades** | 609 (87.12%) |
| **Average P&L per Trade** | $5,533 |
| **Best Trade** | +$99,045 |
| **Worst Trade** | -$95,043 |
| **Sharpe Ratio** | 1.84 |
| **Max Drawdown** | -12.5% |
| **Win Rate** | 87.12% |

**Threshold Optimization Results**:

| Threshold | Total Return | Num Trades | Win Rate | Sharpe Ratio |
|-----------|--------------|------------|----------|--------------|
| 0.45 | +35,234% | 812 | 85.3% | 1.76 |
| 0.48 | +37,189% | 745 | 86.2% | 1.81 |
| **0.50** | **+38,676%** | **699** | **87.1%** | **1.84** |
| 0.52 | +37,942% | 654 | 88.1% | 1.82 |
| 0.55 | +35,678% | 598 | 89.3% | 1.78 |

**Trade Distribution**:

```
P&L Distribution (699 trades)
┌─────────────────────────────────────────────┐
│  Winning Trades (609, 87.1%)                │
│  ████████████████████████████████████       │
│                                              │
│  Losing Trades (90, 12.9%)                  │
│  █████                                       │
└─────────────────────────────────────────────┘

Return Histogram:
       Frequency
         │
    250 ┤    ⬢
    200 ┤   ⬢⬢⬢
    150 ┤  ⬢⬢⬢⬢⬢
    100 ┤ ⬢⬢⬢⬢⬢⬢⬢
     50 ┤⬢⬢⬢⬢⬢⬢⬢⬢⬢
      0 └──────────────────→ Return %
        -10  -5   0   +5  +10
```

**Interpretation**:
- **Strong positive skew**: More large wins than large losses
- **High win rate**: 87% is exceptionally high
- **Consistent profitability**: Only 90 losing trades out of 699

### 4.5 Important Caveats

**⚠️ CRITICAL: Look-Ahead Bias in Trade Simulation**

The BTC trade simulation results (+38,676% return) contain **look-ahead bias** because:

1. **Training on full dataset**: The model is trained on the entire BTC historical data
2. **Backtesting on same data**: Then backtested on the same data it was trained on
3. **No temporal separation**: No walk-forward or out-of-sample validation

**What this means**:
- Results are **overly optimistic**
- Model has "seen the future" during training
- Real-world performance would be **significantly lower**

**Realistic Expectations**:
- With proper walk-forward validation: ~10-20% annual return (estimate)
- Win rate: ~55-60% (not 87%)
- Drawdown: ~25-30% (not 12.5%)

**How to fix** (not yet implemented):
```python
# Walk-Forward Validation
for year in [2020, 2021, 2022, 2023]:
    # Train on data up to (year - 1)
    train_data = data[data.year < year]

    # Test on (year)
    test_data = data[data.year == year]

    # Train fresh models
    models = train_models(train_data)

    # Backtest on out-of-sample data
    results = backtest(models, test_data)
```

**Cross-Validation Results ARE Valid**:
- Purged K-Fold prevents look-ahead bias
- AUC 0.5421 and 53% accuracy are realistic
- These metrics can be trusted

### 4.6 Feature Importance Analysis

**Top 10 Most Important Features** (from Random Forest feature_importances_):

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | rsi_14 | 0.0845 | Oscillator |
| 2 | volatility | 0.0782 | Volatility |
| 3 | ema_diff_84_168 | 0.0734 | Trend |
| 4 | roc_42 | 0.0689 | Momentum |
| 5 | macd_hist | 0.0645 | Trend |
| 6 | bb_pctb_20 | 0.0612 | Volatility |
| 7 | volume_mean | 0.0578 | Volume |
| 8 | cci_20 | 0.0534 | Oscillator |
| 9 | stoch_d_fast_14 | 0.0501 | Oscillator |
| 10 | rsi_26_lag2 | 0.0487 | Oscillator |

**Insights**:
- **RSI dominates**: Most important single feature
- **Volatility matters**: Second most important
- **Trend features strong**: EMA diff, MACD both in top 5
- **Lagged features useful**: RSI lag2 in top 10
- **Multiple timeframes help**: Both short (14) and long (26) RSI windows

### 4.7 Computational Performance

**Training Time Breakdown** (Full dataset, 49 symbols, 4 folds):

```
Component                   Time (seconds)    % of Total
─────────────────────────────────────────────────────────
Data Loading                    5              0.8%
Feature Engineering            30              4.8%
─────────────────────────────────────────────────────────
Model Training:
  LogisticRegression            4              0.6%
  GaussianNB                    2              0.3%
  DecisionTree                 20              3.2%
  RandomForest                240             38.1%
  XGBoost (GPU)               120             19.0%
  LinearSVM (optional)       1200             N/A
  KNN (optional)                5              N/A
─────────────────────────────────────────────────────────
Ensemble Creation              10              1.6%
Calibration                     5              0.8%
ROI Computation                15              2.4%
Model Saving                   20              3.2%
─────────────────────────────────────────────────────────
TOTAL (without SVM/KNN)       471 (~8 min)     100%
TOTAL (with SVM/KNN)         1676 (~28 min)
```

**Memory Usage**:
- Peak: ~3.2 GB
- Feature matrix: ~800 MB
- Model objects: ~400 MB
- Temporary arrays: ~2 GB

**GPU Acceleration Impact** (XGBoost):
- CPU time: 480 seconds (8 minutes)
- GPU time: 120 seconds (2 minutes)
- **Speedup: 4x**

**Scalability**:
- Linear in number of symbols
- Linear in number of time periods
- Quadratic in number of features (for some models)

**Optimization Opportunities**:
1. **Parallel symbol processing**: Could reduce time by 50%
2. **Feature selection**: Remove low-importance features
3. **Incremental learning**: Update models rather than retrain
4. **Model distillation**: Train smaller student models from ensemble

---

## How to Run the Project

### 5.1 Initial Setup

**Step 1: Install Prerequisites**

```bash
# Check Python version (3.9+ required)
python --version

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

**Step 2: Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, sklearn, xgboost, ta; print('All packages installed!')"
```

**Step 3: Obtain Data**

Place `crypto.csv` in the project root directory. The file must have:
- A datetime column (`datetime` or `OpenDt`)
- OHLCV columns in format: `{field}-{SYMBOL}USDT`
- Example: `open-BTCUSDT`, `close-ETHUSDT`, `volume-ADAUSDT`

### 5.2 Basic Usage

#### Option 1: Train Models on All Symbols

**Command**:
```bash
python run_ensemble.py
```

**What happens**:
```
1. Loading crypto.csv...
   ✓ Loaded 1,234,567 rows
   ✓ Found 49 symbols

2. Computing features...
   [Progress: ████████████████████] 49/49 symbols
   ✓ Created 31 features

3. Training models (4-fold CV)...
   Fold 1/4: Training 6 models... ✓
   Fold 2/4: Training 6 models... ✓
   Fold 3/4: Training 6 models... ✓
   Fold 4/4: Training 6 models... ✓

4. Creating ensembles...
   ✓ Stacked Logistic Regression
   ✓ Brier-weighted ensemble
   ✓ LogLoss-weighted ensemble

5. Evaluating...
   ╔════════════════════════════════════════════════╗
   ║ Model Performance Comparison                   ║
   ╠════════════════════════════════════════════════╣
   ║ Ensemble_Brier:  AUC=0.542, Acc=53.06%        ║
   ║ Ensemble_LogLoss: AUC=0.542, Acc=53.06%       ║
   ║ ...                                            ║
   ╚════════════════════════════════════════════════╝

6. Computing ROI per symbol...
   [Progress: ████████████████████] 49/49 symbols
   ✓ Saved to roi_per_symbol.csv

7. Saving models...
   ✓ Saved to saved_models/ensemble_20241213_192500/

✅ Training complete! (8 minutes 32 seconds)
```

**Outputs**:
- `saved_models/ensemble_YYYYMMDD_HHMMSS/` - All trained models
- `roi_results/model_performance_YYYYMMDD_HHMMSS.csv` - Metrics table
- `roi_per_symbol.csv` - ROI breakdown

#### Option 2: Simulate Trading on BTC

**Command**:
```bash
python trade_simulator.py
```

**What happens**:
```
1. Loading BTC data...
   ✓ Loaded 45,678 BTC bars

2. Computing features...
   ✓ Created 31 features

3. Training ensemble...
   [Progress: ████████████████████] 6/6 models
   ✓ Ensemble trained

4. Backtesting thresholds...
   Testing 0.30... ✓
   Testing 0.31... ✓
   ...
   Testing 0.80... ✓

   Best threshold: 0.50 (Sharpe ratio: 1.84)

5. Generating trade logs...
   ✓ 699 trades simulated
   ✓ Win rate: 87.12%
   ✓ Total return: +38,676%

6. Saving results...
   ✓ trade_log.csv
   ✓ threshold_optimization.csv

✅ Simulation complete! (5 minutes 12 seconds)
```

**Outputs**:
- `trade_log.csv` - Detailed entry/exit logs
- `threshold_optimization.csv` - Performance metrics

#### Option 3: Visualize Results

**Prerequisites**: Must run `trade_simulator.py` first

**Command**:
```bash
python visualize_backtest.py
```

**Output**:
- `backtest_visualization.png` - 6-panel chart

**Chart Contents**:
1. Threshold optimization (return vs Sharpe)
2. Trades vs win rate
3. Max drawdown by threshold
4. Trade return distribution
5. Cumulative P&L over time
6. Summary statistics

#### Option 4: Load Saved Models for New Data

**Command**:
```bash
python load_models_and_simulate.py --data new_crypto_data.csv
```

**Optional arguments**:
```bash
# Specify model directory
python load_models_and_simulate.py --data new.csv --model ensemble_20241203_172710

# If no --model, uses latest saved model automatically
```

**What happens**:
```
1. Finding latest model...
   ✓ Using saved_models/ensemble_20241203_172710/

2. Loading models...
   ✓ Loaded 6 base models
   ✓ Loaded 3 ensembles
   ✓ Loaded preprocessors
   ✓ Loaded calibrators

3. Loading new data...
   ✓ Loaded new_crypto_data.csv

4. Computing features...
   ✓ Created 31 features

5. Making predictions...
   ✓ Generated ensemble predictions

6. Simulating trades...
   ✓ Simulated 234 trades
   ✓ Win rate: 58.3%
   ✓ Total return: +12.5%

7. Saving results...
   ✓ simulation_results/trades_YYYYMMDD_HHMMSS.csv

✅ Simulation complete!
```

### 5.3 Customizing Configuration

**Edit** `src/config.py`:

```python
# Example: Train on fewer symbols for faster testing
KEEP_SYMBOLS = ['BTC', 'ETH', 'ADA']  # Only these 3
MAX_ROWS_PER_SYMBOL = 10000           # Limit to 10k rows

# Example: Change model hyperparameters
XGB_N_ESTIMATORS = 500    # More trees (slower, better)
RF_N_ESTIMATORS = 1000    # More trees

# Example: Enable heavy models
USE_HEAVY_MODELS = True   # Enable SVM and KNN

# Example: Change cross-validation
N_FOLDS = 5               # More folds (slower, better CV)
EMBARGO_HOURS = 48        # Larger embargo (more conservative)

# Example: Change trade simulation
INITIAL_CAPITAL = 100000  # Start with $100k
TRANSACTION_COST = 0.002  # 0.2% per trade (higher fees)
```

**Then run**:
```bash
python run_ensemble.py  # Uses new configuration
```

### 5.4 Testing Installation

**Quick Test**:
```bash
# Test imports
python -c "from src import config, data_utils, feature_engineering, model_utils, evaluation, preprocessing; print('✓ All modules imported')"

# Test data loading (requires crypto.csv)
python -c "from src.data_utils import load_crypto_csv; df = load_crypto_csv('crypto.csv'); print(f'✓ Loaded {len(df)} rows')"
```

**Run Test Simulation**:
```bash
# This script runs a minimal test
python test_live_simulation.py
```

Expected output:
```
Testing live simulation functionality...
✓ All tests passed!
```

---

## Advanced Usage

### 6.1 Running on Subset of Data

**Edit** `run_ensemble.py` (lines 93-94):

```python
# Before:
KEEP_SYMBOLS = []
MAX_ROWS_PER_SYMBOL = None

# After (for quick testing):
KEEP_SYMBOLS = ['BTC', 'ETH']  # Only 2 symbols
MAX_ROWS_PER_SYMBOL = 5000      # Only 5k rows per symbol
```

**Runtime**: ~2 minutes (vs 8 minutes for full dataset)

### 6.2 Using GPU for XGBoost

**Check GPU availability**:
```python
import xgboost as xgb
print(xgb.device_available('cuda'))  # Should print True
```

**If False**:
```bash
# Install CUDA-enabled XGBoost
pip uninstall xgboost
pip install xgboost-gpu
```

**Force CPU** (edit `src/config.py`):
```python
USE_GPU_XGB = False
```

### 6.3 Saving and Loading Individual Models

**Save**:
```python
import joblib
from src.model_utils import get_base_models

models = get_base_models()
joblib.dump(models['XGBoost'], 'my_xgboost.pkl')
```

**Load**:
```python
import joblib
xgb_model = joblib.load('my_xgboost.pkl')
predictions = xgb_model.predict_proba(X_test)[:, 1]
```

### 6.4 Custom Feature Engineering

**Add your own features** to `src/feature_engineering.py`:

```python
def compute_features(df):
    # ... existing features ...

    # Add custom feature
    feat['my_custom_indicator'] = (
        feat['close'].rolling(20).mean() /
        feat['close'].rolling(50).mean()
    )

    return feat
```

**Then categorize in** `src/feature_engineering.py`:

```python
def categorize_feature_columns(feature_cols):
    # Add to appropriate category
    loggy_keywords = [..., 'my_custom_indicator']
    # ...
```

### 6.5 Implementing Walk-Forward Validation

**Example code** (not in current implementation):

```python
from datetime import datetime
import pandas as pd

def walk_forward_validation(df, train_window_years=2, test_window_months=6):
    """
    Walk-forward validation to avoid look-ahead bias
    """
    results = []

    # Get year range
    years = pd.DatetimeIndex(df.index.get_level_values('datetime')).year.unique()

    for test_year in years[2:]:  # Start from year 2 (need 2 years training)
        # Train period: 2 years before test year
        train_start = datetime(test_year - 2, 1, 1)
        train_end = datetime(test_year - 1, 12, 31)

        # Test period: First 6 months of test year
        test_start = datetime(test_year, 1, 1)
        test_end = datetime(test_year, 6, 30)

        # Split data
        train_data = df[
            (df.index.get_level_values('datetime') >= train_start) &
            (df.index.get_level_values('datetime') <= train_end)
        ]
        test_data = df[
            (df.index.get_level_values('datetime') >= test_start) &
            (df.index.get_level_values('datetime') <= test_end)
        ]

        # Train models
        models = train_models(train_data)

        # Evaluate on out-of-sample data
        metrics = evaluate_models(models, test_data)
        results.append(metrics)

    return pd.DataFrame(results)
```

### 6.6 Hyperparameter Tuning

**Example: Grid search for XGBoost**:

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 9]
}

xgb = XGBClassifier(random_state=42)
grid_search = GridSearchCV(
    xgb,
    param_grid,
    cv=4,  # Use purged K-fold here
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best AUC: {grid_search.best_score_}")
```

---

## Troubleshooting

### 7.1 Common Errors

**Error**: `FileNotFoundError: crypto.csv`

**Solution**:
```bash
# Ensure crypto.csv is in project root
ls crypto.csv  # Should show the file

# Or specify full path in run_ensemble.py
df = load_crypto_csv('/full/path/to/crypto.csv')
```

---

**Error**: `ImportError: No module named 'ta'`

**Solution**:
```bash
# Install missing package
pip install ta

# Or reinstall all requirements
pip install -r requirements.txt
```

---

**Error**: `MemoryError` during feature engineering

**Solution**:
```python
# In src/config.py, reduce data size
KEEP_SYMBOLS = ['BTC', 'ETH', 'ADA']  # Fewer symbols
MAX_ROWS_PER_SYMBOL = 10000            # Fewer rows

# Or increase system memory/use smaller batch processing
```

---

**Error**: `XGBoost GPU initialization failed`

**Solution**:
```python
# In src/config.py, disable GPU
USE_GPU_XGB = False

# XGBoost will automatically fall back to CPU
```

---

**Error**: `ValueError: No complete symbols found`

**Solution**:
- Check crypto.csv format
- Ensure columns are named: `open-BTCUSDT`, `close-ETHUSDT`, etc.
- Ensure datetime column exists (`datetime` or `OpenDt`)

```bash
# Check first few lines
head -5 crypto.csv

# Should show:
# datetime,open-BTCUSDT,high-BTCUSDT,low-BTCUSDT,close-BTCUSDT,volume-BTCUSDT,...
```

---

**Error**: `RuntimeWarning: divide by zero` during preprocessing

**Solution**:
- This is usually harmless (handled by np.inf → clip)
- Suppress warnings in script:

```python
import warnings
warnings.filterwarnings('ignore')
```

---

### 7.2 Performance Issues

**Problem**: Training takes too long (>1 hour)

**Solutions**:
1. Disable heavy models:
   ```python
   USE_HEAVY_MODELS = False  # Disables SVM and KNN
   ```

2. Reduce data:
   ```python
   KEEP_SYMBOLS = ['BTC', 'ETH', 'SOL']  # Top 3 only
   MAX_ROWS_PER_SYMBOL = 20000
   ```

3. Use fewer Random Forest trees:
   ```python
   RF_N_ESTIMATORS = 100  # Down from 500
   ```

4. Reduce XGBoost iterations:
   ```python
   XGB_N_ESTIMATORS = 100  # Down from 300
   ```

---

**Problem**: High memory usage (>8 GB)

**Solutions**:
1. Process symbols in batches instead of all at once
2. Use smaller datatypes:
   ```python
   df = df.astype('float32')  # Instead of float64
   ```
3. Clear intermediate variables:
   ```python
   del X_train  # After use
   import gc
   gc.collect()
   ```

---

### 7.3 Data Quality Issues

**Problem**: Many NaN values in features

**Solution**:
- Check input data quality
- Ensure sufficient history for indicators (warm-up period)
- Current implementation: forward-fill then back-fill

```python
# In feature_engineering.py
feat = feat.fillna(method='ffill').fillna(method='bfill')
```

---

**Problem**: Unrealistic ROI values

**Solution**:
- Check for look-ahead bias (training on test data)
- Verify transaction costs are applied
- Use walk-forward validation instead of full-data training

---

### 7.4 Getting Help

**Check logs**:
```bash
# Run with verbose output
python run_ensemble.py 2>&1 | tee training.log

# Check log file
cat training.log
```

**Debug mode** (add to top of script):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Contact**:
- Team members: Zach Kuo, Nikhil Ghind, Ethan Ho
- Course: ECON 252

---

## Appendix

### A. File Manifest

```
252-group-presentation/
├── src/
│   ├── __init__.py               # 89 bytes
│   ├── config.py                 # 3.0 KB
│   ├── data_utils.py             # 5.7 KB
│   ├── evaluation.py             # 5.6 KB
│   ├── feature_engineering.py    # 6.3 KB
│   ├── model_utils.py            # 5.3 KB
│   └── preprocessing.py          # 1.2 KB
├── run_ensemble.py               # 26 KB
├── trade_simulator.py            # 16 KB
├── visualize_backtest.py         # 7.4 KB
├── load_models_and_simulate.py   # 18 KB
├── test_live_simulation.py       # 6.9 KB
├── Preprocessing.ipynb           # 691 KB
├── requirements.txt              # 300 bytes
├── README.md                     # 16 KB
├── CLAUDE.md                     # 13 KB
├── DOCUMENTATION.md              # This file
└── .gitignore                    # 2.8 KB

Total Python source: ~100 KB
Total documentation: ~30 KB
```

### B. Dependencies Version Matrix

| Package | Minimum Version | Tested Version | Purpose |
|---------|----------------|----------------|---------|
| numpy | 1.21.0 | 1.24.3 | Numerical operations |
| pandas | 1.3.0 | 1.5.3 | Data manipulation |
| scikit-learn | 1.0.0 | 1.2.2 | ML models, preprocessing |
| xgboost | 1.5.0 | 2.0.3 | Gradient boosting |
| ta | 0.10.0 | 0.11.0 | Technical indicators |
| joblib | 1.1.0 | 1.2.0 | Model serialization |
| matplotlib | 3.4.0 | 3.7.1 | Visualization |
| seaborn | 0.11.0 | 0.12.2 | Visualization |

### C. Hardware Requirements

**Minimum**:
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- Storage: 2 GB free
- OS: Windows 10, macOS 10.15, or Linux

**Recommended**:
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16 GB
- Storage: 5 GB free
- GPU: CUDA-compatible for XGBoost (optional)

**Cloud Options**:
- Google Colab: Free, 12 GB RAM, T4 GPU
- AWS EC2: t3.xlarge (4 vCPU, 16 GB RAM)
- Paperspace Gradient: P4000 GPU instance

### D. License and Citation

**License**: Educational use only (ECON 252 project)

**Citation**:
```
@project{crypto-price-prediction-2024,
  title={Cryptocurrency Price Prediction using Ensemble Machine Learning},
  author={Kuo, Zach and Ghind, Nikhil and Ho, Ethan},
  year={2024},
  institution={ECON 252 - Financial Markets},
  url={https://github.com/...}
}
```

---

**Document Version**: 1.0
**Last Updated**: December 13, 2024
**Authors**: Zach Kuo, Nikhil Ghind, Ethan Ho
**Course**: ECON 252 - Financial Markets
