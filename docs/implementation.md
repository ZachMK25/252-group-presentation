# Implementation Details

Complete technical implementation of the cryptocurrency prediction system.

---

## Table of Contents

1. [Data Processing Pipeline](#data-processing-pipeline)
2. [Feature Engineering](#feature-engineering)
3. [Label Creation](#label-creation)
4. [Preprocessing](#preprocessing)
5. [Model Training](#model-training)
6. [Probability Calibration](#probability-calibration)
7. [Ensemble Methods](#ensemble-methods)

---

## Data Processing Pipeline

### Input Data Format

**File:** `crypto.csv` (wide format)

```csv
datetime,open-BTCUSDT,high-BTCUSDT,low-BTCUSDT,close-BTCUSDT,volume-BTCUSDT,open-ETHUSDT,...
2023-01-01 00:00:00,16500.5,16520.3,16480.1,16510.8,1234.56,1200.3,...
2023-01-01 04:00:00,16510.8,16550.2,16505.0,16545.5,1456.78,1203.2,...
```

**Characteristics:**
- **Timeframe:** 4-hour bars (6 bars per day)
- **Period:** Historical data through February 2024
- **Symbols:** 49 cryptocurrencies (BTC, ETH, ADA, SOL, etc.)
- **Fields per symbol:** open, high, low, close, volume

### Wide to Long Conversion

**Implementation:** `src/data_utils.py::load_crypto_csv()`

**Input:** Wide format
```
datetime         open-BTCUSDT  close-BTCUSDT  open-ETHUSDT  close-ETHUSDT
2023-01-01       16500.5       16510.8        1200.3        1203.2
```

**Output:** Long format with MultiIndex
```
                         open     high      low    close     volume
symbol  datetime
BTC     2023-01-01    16500.5  16520.3  16480.1  16510.8   1234.56
        2023-01-05    16510.8  16550.2  16505.0  16545.5   1456.78
ETH     2023-01-01     1200.3   1205.8   1198.5   1203.2   5678.90
        2023-01-05     1203.2   1210.5   1201.0   1208.7   6234.11
```

**Algorithm:**
```python
def load_crypto_csv(csv_path='crypto.csv'):
    # 1. Parse wide column names (e.g., 'open-BTCUSDT')
    for col in raw.columns:
        if '-' in col:
            field, symbol = parse_column_name(col)
            by_symbol[symbol][field] = raw[col]

    # 2. Create long-format DataFrame
    frames = []
    for symbol, fields in by_symbol.items():
        df_symbol = create_symbol_dataframe(fields)
        df_symbol['symbol'] = symbol
        frames.append(df_symbol)

    # 3. Concatenate and set MultiIndex
    df = pd.concat(frames)
    df = df.set_index(['symbol', 'datetime']).sort_index()

    return df
```

### Data Cleaning

**Steps:**
1. Convert to numeric types
```python
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

2. Drop rows with missing OHLC
```python
df = df.dropna(subset=['open', 'high', 'low', 'close'])
```

3. Sort chronologically
```python
df = df.sort_index()  # Sorts by (symbol, datetime)
```

---

## Feature Engineering

**Implementation:** `src/feature_engineering.py::compute_features()`

### 1. Momentum Indicators

#### Rate of Change (ROC) - 3 features

**Formula:** `ROC = (close_t - close_{t-n}) / close_{t-n} * 100`

**Windows:** [1, 3, 42] periods

```python
from ta.momentum import ROCIndicator

for window in [1, 3, 42]:
    roc = ROCIndicator(close, window=window).roc()
    features[f'roc_{window}'] = roc
```

**Interpretation:**
- `roc_1`: Single-period momentum (4-hour change)
- `roc_3`: Short-term momentum (12-hour change)
- `roc_42`: Medium-term momentum (7-day change)

**Example:**
```
Close: 100 → 102 → 101 → 104
ROC_1: 0% → 2% → -0.98% → 2.97%
ROC_3: 0% → 0% → 1% → 3.92%
```

### 2. Trend Indicators

#### EMA Cross - 2 features

**Parameters:** Fast=84, Slow=168 periods (14 days, 28 days)

```python
from ta.trend import EMAIndicator

ema_fast = EMAIndicator(close, window=84).ema_indicator()
ema_slow = EMAIndicator(close, window=168).ema_indicator()

features['ema_diff_84_168'] = ema_fast - ema_slow
features['ema_ratio_84_168'] = ema_fast / ema_slow
```

**Interpretation:**
- `ema_diff > 0`: Bullish (short-term above long-term)
- `ema_ratio > 1`: Bullish
- `ema_ratio < 1`: Bearish

#### MACD - 2 features

```python
from ta.trend import MACD

macd = MACD(close)
features['macd_hist'] = macd.macd_diff()  # MACD line - Signal line
features['macd_hist_lag2'] = features['macd_hist'].shift(2)
```

**Interpretation:**
- `macd_hist > 0`: Bullish momentum
- `macd_hist < 0`: Bearish momentum
- Lag added for stability

### 3. Oscillators

#### RSI (Relative Strength Index) - 6 features

**Formula:** `RSI = 100 - (100 / (1 + RS))` where `RS = Avg Gain / Avg Loss`

**Windows:** [8, 14, 26] periods

```python
from ta.momentum import RSIIndicator

for window in [8, 14, 26]:
    rsi = RSIIndicator(close, window=window).rsi()
    features[f'rsi_{window}'] = rsi
    features[f'rsi_{window}_lag2'] = rsi.shift(2)
```

**Interpretation:**
- RSI < 30: Oversold (potential buy)
- RSI > 70: Overbought (potential sell)
- RSI = 50: Neutral

**Example:**
```
Price: 100→102→105→103→101→99→98→100
RSI_14: 50→55→62→58→52→45→42→48
```

#### Stochastic Oscillator - 6 features

**Formula:** `%K = (Close - Low_n) / (High_n - Low_n) * 100`

**K values:** [8, 14] periods

```python
from ta.momentum import StochasticOscillator

for k in [8, 14]:
    stoch = StochasticOscillator(high, low, close, window=k, smooth_window=3)

    features[f'stoch_d_fast_{k}'] = stoch.stoch()  # Fast %K
    features[f'stoch_d_slow_{k}'] = stoch.stoch_signal()  # Slow %D
    features[f'stoch_hist_{k}'] = fast_d - slow_d  # Histogram
```

**Range:** 0-100

#### CCI (Commodity Channel Index) - 2 features

**Formula:** `CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)`

**Windows:** [10, 20] periods

```python
from ta.trend import CCIIndicator

for window in [10, 20]:
    cci = CCIIndicator(high, low, close, window=window).cci()
    features[f'cci_{window}'] = cci
```

**Typical range:** -100 to +100 (but unbounded)

### 4. Volatility Indicators

#### Bollinger Bands - 4 features

**Parameters:** Windows=[10, 20], std_dev=2

```python
from ta.volatility import BollingerBands

for window in [10, 20]:
    bb = BollingerBands(close, window=window, window_dev=2)

    # %B: Position within bands (0=lower, 1=upper)
    features[f'bb_pctb_{window}'] = bb.bollinger_pband()

    # Bandwidth: (upper - lower) / middle
    features[f'bb_width_{window}'] = bb.bollinger_wband()
```

**Interpretation:**
- `bb_pctb < 0`: Below lower band (oversold)
- `bb_pctb > 1`: Above upper band (overbought)
- `bb_width` high: High volatility
- `bb_width` low: Low volatility (squeeze)

#### Rolling Volatility - 1 feature

**Formula:** Standard deviation of log returns

```python
log_returns = np.log(close / close.shift(1))
features['volatility'] = log_returns.rolling(20).std()
```

### 5. Volume Indicators - 3 features

```python
# Moving average of volume
features['volume_mean'] = volume.rolling(42).mean()

# Log changes at different windows
features['volume_log_change_42'] = np.log(volume / volume.shift(42))
features['volume_log_change_84'] = np.log(volume / volume.shift(84))
```

**Interpretation:**
- Volume > volume_mean: High activity
- Positive log_change: Increasing volume
- Negative log_change: Decreasing volume

### Feature Summary

**Total Features:** 31+

| Category | Features | Count |
|----------|----------|-------|
| Momentum | ROC | 3 |
| Trend | EMA cross, MACD | 4 |
| Oscillators | RSI, Stochastic, CCI | 14 |
| Volatility | Bollinger Bands, Rolling Vol | 5 |
| Volume | Mean, Log changes | 3 |
| **Total** | | **29** |

Plus OHLCV = 34 total columns before label creation.

---

## Label Creation

**Implementation:** `src/feature_engineering.py::create_labels()`

### Binary Classification Target

**Formula:**
```python
log_return_next = np.log(close.shift(-1) / close)
label = 1 if log_return_next > 0.01 else 0
```

**Threshold:** 1% (0.01)

**Example:**
```
Time    Close    Next_Close    Log_Return    Label    Interpretation
0       100.0    101.5         0.0149        1        Up (>1%)
1       101.5    101.0        -0.0049        0        Down
2       101.0    102.2         0.0118        1        Up (>1%)
3       102.2    101.8        -0.0039        0        Down
4       101.8    103.5         0.0165        1        Up (>1%)
```

**Class Distribution:**
- Typical: ~48% up, ~52% down
- Slight bearish bias in crypto markets
- 1% threshold filters noise

**Why Log Returns?**
- Symmetric: log(100→110) = -log(110→100)
- Additive: log(a→b) + log(b→c) = log(a→c)
- Normally distributed
- Better for statistical modeling

---

## Preprocessing

**Implementation:** `src/preprocessing.py::create_preprocessing_pipeline()`

### Three-Category System

**1. Bounded Features** (already 0-100 or 0-1)
```python
bounded_cols = ['rsi_*', 'stoch_*', 'bb_pctb_*']
# Transformation: Passthrough (no change)
```

**2. Log-Skewed Features**
```python
loggy_cols = ['volume_*', 'volatility', 'ema_ratio_*', 'bb_width_*']
# Transformation: signed_log1p → StandardScaler
```

**3. Standard Features**
```python
the_rest = [all other numeric features]
# Transformation: StandardScaler only
```

### Signed Log1p Transformation

**Formula:** `sign(x) * log(1 + |x|)`

```python
def signed_log1p(df):
    Z = df.copy()
    for col in Z.columns:
        x = Z[col].values
        Z[col] = np.sign(x) * np.log1p(np.abs(x))
    return Z
```

**Why?**
- Preserves sign (positive stays positive)
- Reduces skewness (handles outliers)
- log1p handles x=0 gracefully

**Example:**
```
Original:  [-1000, -100, -10, 0, 10, 100, 1000]
Signed:    [-6.91, -4.62, -2.40, 0, 2.40, 4.62, 6.91]
```

### ColumnTransformer Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ('bounded', 'passthrough', bounded_cols),
    ('loggy', Pipeline([
        ('log', FunctionTransformer(signed_log1p)),
        ('scale', StandardScaler())
    ]), loggy_cols),
    ('standard', StandardScaler(), the_rest)
])

# Fit on training data
preprocessor.fit(X_train)

# Transform both train and test
X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
```

---

## Model Training

**Implementation:** `src/model_utils.py::get_base_models()`

### Base Models (6 total)

#### 1. Logistic Regression

```python
LogisticRegression(
    penalty='l2',           # L2 regularization
    C=1.0,                  # Inverse regularization strength
    solver='saga',          # Stochastic Average Gradient Descent
    max_iter=1000,
    random_state=42
)
```

**Characteristics:**
- Linear decision boundary
- Outputs calibrated probabilities
- Fast: ~1 second per fold
- Interpretable: Can examine coefficients

**Decision Function:** `P(y=1) = σ(β₀ + β₁x₁ + ... + βₙxₙ)`

#### 2. Gaussian Naive Bayes

```python
GaussianNB()
```

**Assumptions:**
- Features are independent (Naive assumption)
- Features follow Gaussian distribution

**Very fast:** <1 second per fold

**Probabilistic:** Uses Bayes' theorem

#### 3. Decision Tree

```python
DecisionTreeClassifier(
    max_depth=12,           # Limit depth
    random_state=42
)
```

**How it works:**
- Binary splits on features
- Greedy algorithm (locally optimal splits)
- Max depth prevents overfitting

**Training:** ~5 seconds per fold

#### 4. Random Forest

```python
RandomForestClassifier(
    n_estimators=500,       # 500 trees
    max_depth=None,         # Unlimited depth
    random_state=42,
    n_jobs=-1               # Parallel processing
)
```

**Ensemble of 500 decision trees:**
- Bootstrap sampling (random rows)
- Feature randomness (random columns)
- Majority vote for prediction
- Resistant to overfitting

**Training:** ~60 seconds per fold (parallelized)

#### 5. XGBoost

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    tree_method='hist',     # Histogram-based algorithm
    device='cuda',          # GPU acceleration
    random_state=42,
    n_jobs=-1
)
```

**Gradient Boosting:**
- Sequential tree building
- Each tree corrects previous errors
- Shrinkage (learning_rate) prevents overfitting

**GPU acceleration:** 4x faster than CPU

**Training:** ~30 seconds (GPU), ~120 seconds (CPU)

#### 6. Linear SVM + KNN (Optional)

```python
# Linear SVM
LinearSVC(C=0.1, max_iter=1000)

# K-Nearest Neighbors
KNeighborsClassifier(n_neighbors=5)
```

**SVM:** Large-margin classifier
**KNN:** Instance-based learning

**Enabled with:** `USE_HEAVY_MODELS = True` in config

**Training:** Very slow (~300 seconds for SVM)

---

## Probability Calibration

**Implementation:** `src/model_utils.py::calibrate_probabilities()`

### Isotonic Regression

**Why calibrate?**
- Raw model outputs may not represent true probabilities
- Decision Tree: Tends to output 0 or 1
- Random Forest: Underconfident on extremes

**Method:** Isotonic Regression

```python
from sklearn.isotonic import IsotonicRegression

def calibrate_probabilities(y_true, y_pred_proba):
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_proba, y_true)
    return calibrator
```

**Training process:**
```
1. Cross-validation (4 folds)
   ├─ Fold 1: Train model, predict on test fold
   ├─ Fold 2: Train model, predict on test fold
   ├─ Fold 3: Train model, predict on test fold
   └─ Fold 4: Train model, predict on test fold

2. Collect all CV predictions

3. Fit isotonic regression on CV predictions
   calibrator.fit(all_cv_predictions, all_y_true)

4. Apply to new predictions
   calibrated_pred = calibrator.transform(raw_pred)
```

**Before vs After:**
```
Raw Prediction    True Frequency    Calibrated
0.60              0.52              0.52
0.70              0.58              0.58
0.80              0.62              0.62
```

---

## Ensemble Methods

**Implementation:** `src/model_utils.py`

### 1. Stacked Logistic Regression

**Architecture:**
```
Base Models              Meta-Model
┌──────────────┐
│ LogReg       │─────┐
├──────────────┤     │
│ NaiveBayes   │─────┤
├──────────────┤     │   ┌─────────────┐
│ DecisionTree │─────┼──→│  Logistic   │──→ Final
├──────────────┤     │   │ Regression  │    Prediction
│ RandomForest │─────┤   └─────────────┘
├──────────────┤     │
│ XGBoost      │─────┘
└──────────────┘
```

**Code:**
```python
def create_stacked_ensemble(X_train, y_train, base_predictions):
    # Stack base predictions as features
    X_meta = np.column_stack(list(base_predictions.values()))

    # Train meta-model
    meta_model = LogisticRegression(penalty='l2', C=1.0)
    meta_model.fit(X_meta, y_train)

    return meta_model
```

**Prediction:**
```python
# Get base predictions
base_preds = []
for model in base_models.values():
    pred = model.predict_proba(X_test)[:, 1]
    base_preds.append(pred)

# Stack and predict
X_meta_test = np.column_stack(base_preds)
final_pred = meta_model.predict_proba(X_meta_test)[:, 1]
```

### 2. Brier-Weighted Ensemble

**Formula:**
```
weight_i = (1 / brier_score_i) / Σ(1 / brier_score_j)
final_pred = Σ(weight_i × pred_i)
```

**Code:**
```python
def compute_ensemble_weights(brier_scores):
    weights = {}
    for model, brier in brier_scores.items():
        weights[model] = 1.0 / (brier + 1e-10)

    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    return weights
```

**Example:**
```
Model           Brier Score    Raw Weight    Normalized Weight
LogReg          0.2487         4.021         0.201
NaiveBayes      0.2489         4.018         0.200
RandomForest    0.2493         4.011         0.200
XGBoost         0.2491         4.014         0.200
DecisionTree    0.2495         4.008         0.199
                               ─────         ─────
                               20.072        1.000
```

### 3. LogLoss-Weighted Ensemble

**Same as Brier-weighted but using log loss:**
```python
weight_i = (1 / log_loss_i) / Σ(1 / log_loss_j)
```

---

## Summary

This implementation includes:

✅ **Data Pipeline:** Wide→Long conversion, cleaning
✅ **Feature Engineering:** 31+ technical indicators
✅ **Preprocessing:** 3-category system with signed log1p
✅ **Models:** 6 base models + 3 ensembles
✅ **Calibration:** Isotonic regression
✅ **Ensemble:** Stacking, Brier-weighted, LogLoss-weighted

**Next:** See [Algorithms](algorithms.md) for detailed algorithms and pseudocode.
