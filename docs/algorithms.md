# Algorithms

Detailed algorithms and pseudocode for the cryptocurrency prediction system.

---

## Table of Contents

1. [Purged K-Fold Cross-Validation](#purged-k-fold-cross-validation)
2. [Ensemble Weight Calculation](#ensemble-weight-calculation)
3. [ROI Calculation](#roi-calculation)
4. [Sharpe Ratio](#sharpe-ratio-calculation)
5. [Max Drawdown](#max-drawdown-calculation)

---

## Purged K-Fold Cross-Validation

**Purpose:** Prevent look-ahead bias in time series by purging data between train/test splits.

**Implementation:** `src/evaluation.py::purged_kfold_cv()`

### Algorithm

```
Input:
    X: Features with datetime index
    y: Labels
    n_splits: Number of folds (default: 4)
    embargo_hours: Hours to purge between train/test (default: 24)

Output:
    Iterator of (train_idx, test_idx) for each fold

Procedure:
1. Extract unique datetime values
   unique_times ← sorted(unique(X.index.datetime))
   n_times ← len(unique_times)

2. Calculate fold size
   fold_size ← n_times ÷ n_splits

3. For each fold i in [0, n_splits-1]:

   a. Define test period
      test_start_idx ← i × fold_size
      test_end_idx ← (i+1) × fold_size  # or n_times for last fold

      test_start_time ← unique_times[test_start_idx]
      test_end_time ← unique_times[test_end_idx - 1]

   b. Apply embargo (purge zone)
      embargo_delta ← TimeDelta(hours=embargo_hours)
      train_end_time ← test_start_time - embargo_delta

   c. Get row indices
      train_idx ← indices where X.index.datetime < train_end_time
      test_idx ← indices where (test_start_time ≤ X.index.datetime ≤ test_end_time)

   d. Validate fold
      if len(train_idx) = 0 or len(test_idx) = 0:
          continue  # Skip invalid fold

   e. Yield fold
      yield (train_idx, test_idx)
```

### Visualization

**Timeline (4 folds):**
```
|----Train 1----|--Gap--|--Test 1--|----Train 2----|--Gap--|--Test 2--| ...

Fold 1:   |████████████████|  ░░  |▓▓▓▓▓▓▓▓▓▓|
Fold 2:   |████████████████████████████████████|  ░░  |▓▓▓▓▓▓▓▓▓▓|
Fold 3:   |████████████████████████████████████████████████████████|  ░░  |▓▓▓▓▓▓▓▓▓▓|
Fold 4:   |████████████████████████████████████████████████████████████████████████|  ░░  |▓▓▓▓|

Legend:
████ = Training data
░░░░ = 24-hour embargo (purge zone)
▓▓▓▓ = Test data
```

### Example

**Input Data:**
- 1000 unique datetimes
- 4 folds
- 24-hour embargo

**Fold 1:**
```
Train: datetime < 2023-06-01 00:00
Gap:   2023-06-01 00:00 to 2023-06-02 00:00  (24 hours purged)
Test:  2023-06-02 00:00 to 2023-09-01 00:00  (250 datetimes)
```

**Fold 2:**
```
Train: datetime < 2023-08-31 00:00
Gap:   2023-08-31 00:00 to 2023-09-01 00:00
Test:  2023-09-01 00:00 to 2023-12-01 00:00
```

### Time Complexity

- **O(n)** where n = number of rows
- One pass through index to extract unique times
- One pass to filter indices per fold

### Why Purging Matters

**Without purge:**
```
Train: [..., 11:00 PM, 11:30 PM]
Test:  [12:00 AM, 12:30 AM, ...]

Problem: Features at 11:30 PM may contain info about 12:00 AM
         (e.g., volume spike starts at 11:50 PM)
```

**With 24h purge:**
```
Train: [..., 11:00 PM (day -1)]
Gap:   [11:00 PM (day -1) to 11:00 PM (day 0)]
Test:  [11:00 PM (day 0), 11:30 PM (day 0), ...]

Result: Clean separation, no information leakage
```

---

## Ensemble Weight Calculation

**Purpose:** Weight models by inverse of error metric (lower error = higher weight).

**Implementation:** `src/model_utils.py::compute_ensemble_weights()`

### Brier-Weighted Algorithm

```
Input:
    brier_scores: Dict[model_name → brier_score]

Output:
    weights: Dict[model_name → normalized_weight]

Procedure:
1. Compute raw weights (inverse of Brier score)
   for each model in brier_scores:
       raw_weight[model] ← 1 / (brier_score[model] + ε)
       # ε = 1e-10 to avoid division by zero

2. Normalize weights to sum to 1
   total_weight ← sum(raw_weight.values())

   for each model in raw_weight:
       weight[model] ← raw_weight[model] / total_weight

3. Return weights
```

### Example Calculation

**Input:**
```
brier_scores = {
    'LogReg': 0.2487,
    'NaiveBayes': 0.2489,
    'RandomForest': 0.2493,
    'XGBoost': 0.2491,
    'DecisionTree': 0.2495
}
```

**Step 1: Raw weights**
```
raw_weight['LogReg']       = 1 / 0.2487 = 4.021
raw_weight['NaiveBayes']   = 1 / 0.2489 = 4.018
raw_weight['RandomForest'] = 1 / 0.2493 = 4.011
raw_weight['XGBoost']      = 1 / 0.2491 = 4.014
raw_weight['DecisionTree'] = 1 / 0.2495 = 4.008

total = 20.072
```

**Step 2: Normalize**
```
weight['LogReg']       = 4.021 / 20.072 = 0.201
weight['NaiveBayes']   = 4.018 / 20.072 = 0.200
weight['RandomForest'] = 4.011 / 20.072 = 0.200
weight['XGBoost']      = 4.014 / 20.072 = 0.200
weight['DecisionTree'] = 4.008 / 20.072 = 0.199

sum = 1.000 ✓
```

### Ensemble Prediction

```
Input:
    base_predictions: Dict[model_name → predictions]
    weights: Dict[model_name → weight]

Output:
    ensemble_prediction: Weighted average

Procedure:
    ensemble_pred ← 0
    for each model in base_predictions:
        ensemble_pred ← ensemble_pred + (weight[model] × base_predictions[model])

    return ensemble_pred
```

**Example:**
```
Base predictions:
    LogReg: [0.60, 0.55, 0.70]
    NaiveBayes: [0.58, 0.52, 0.68]
    RandomForest: [0.62, 0.58, 0.72]

Weights:
    LogReg: 0.34
    NaiveBayes: 0.33
    RandomForest: 0.33

Ensemble:
    Sample 0: 0.34×0.60 + 0.33×0.58 + 0.33×0.62 = 0.600
    Sample 1: 0.34×0.55 + 0.33×0.52 + 0.33×0.58 = 0.550
    Sample 2: 0.34×0.70 + 0.33×0.68 + 0.33×0.72 = 0.700
```

---

## ROI Calculation

**Purpose:** Compute return on investment for different trading strategies.

**Implementation:** `src/evaluation.py::compute_roi_per_threshold()`

### Algorithm

```
Input:
    y_true: True labels (1=up, 0=down)
    y_pred_proba: Predicted probabilities
    returns: Actual log returns
    thresholds: Array of probability thresholds to test
    transaction_cost: Cost per trade (default: 0.001 = 0.1%)

Output:
    results: DataFrame with metrics per threshold

Procedure:
1. Initialize results list
   results ← []

2. For each threshold in thresholds:

   a. Generate trading signals
      signals ← 1 if y_pred_proba ≥ threshold else 0
      # 1 = enter long position, 0 = no position

   b. Calculate gross P&L
      gross_pnl ← sum(signals × returns)

   c. Count position changes (trades)
      position_changes ← diff([0] + signals)
      num_trades ← sum(abs(position_changes))

   d. Calculate transaction costs
      total_costs ← num_trades × transaction_cost

   e. Calculate net P&L
      net_pnl ← gross_pnl - total_costs

   f. Calculate accuracy
      if num_trades > 0:
          accuracy ← mean(signals == y_true)
      else:
          accuracy ← 0

   g. Store results
      results.append({
          'threshold': threshold,
          'gross_pnl': gross_pnl,
          'net_pnl': net_pnl,
          'num_trades': num_trades,
          'accuracy': accuracy
      })

3. Return DataFrame(results)
```

### Example

**Input:**
```
y_pred_proba = [0.45, 0.52, 0.58, 0.48, 0.62]
returns = [0.01, 0.02, -0.01, 0.015, 0.025]
threshold = 0.50
transaction_cost = 0.001
```

**Step-by-step:**

1. **Signals:**
```
Prob:    [0.45, 0.52, 0.58, 0.48, 0.62]
Signals: [0,    1,    1,    0,    1   ]  (1 if prob ≥ 0.50)
```

2. **Gross P&L:**
```
Returns:    [0.01, 0.02, -0.01, 0.015, 0.025]
Signals:    [0,    1,    1,     0,     1    ]
Product:    [0,    0.02, -0.01, 0,     0.025]

gross_pnl = 0 + 0.02 - 0.01 + 0 + 0.025 = 0.035 (3.5%)
```

3. **Position Changes:**
```
Position:  [0, 0, 1, 1, 0, 1]  (prepend 0)
Changes:   [0, 1, 0, -1, 1]
Trades:    |0|+|1|+|0|+|-1|+|1| = 3 trades
```

4. **Transaction Costs:**
```
total_costs = 3 × 0.001 = 0.003 (0.3%)
```

5. **Net P&L:**
```
net_pnl = 0.035 - 0.003 = 0.032 (3.2%)
```

6. **Accuracy:**
```
y_true:   [1, 1, 0, 1, 1]
signals:  [0, 1, 1, 0, 1]
correct:  [0, 1, 0, 0, 1]  (2 out of 5)
accuracy = 2/5 = 0.40 (40%)
```

### Time Complexity

- **O(n × m)** where:
  - n = number of samples
  - m = number of thresholds to test

---

## Sharpe Ratio Calculation

**Purpose:** Measure risk-adjusted returns.

**Formula:** `Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns`

**Implementation:** `src/evaluation.py::compute_sharpe_ratio()`

### Algorithm

```
Input:
    returns: Array of trade returns
    risk_free_rate: Risk-free rate (default: 0)

Output:
    sharpe_ratio: Annualized Sharpe ratio

Procedure:
1. Calculate excess returns
   excess_returns ← returns - risk_free_rate

2. Calculate mean and std
   mean_return ← mean(excess_returns)
   std_return ← std(returns)

3. Handle edge cases
   if len(returns) = 0 or std_return = 0:
       return 0

4. Calculate Sharpe ratio
   sharpe ← mean_return / std_return

5. Annualize (assuming daily returns)
   annualized_sharpe ← sharpe × sqrt(252)
   # 252 = trading days per year

   # For 4-hour bars (6 per day):
   # annualized_sharpe ← sharpe × sqrt(252 × 6)

6. Return annualized_sharpe
```

### Example

**Input:**
```
returns = [0.02, -0.01, 0.03, 0.01, -0.005, 0.025]  # Daily returns
risk_free_rate = 0
```

**Calculation:**

1. **Excess returns:**
```
excess_returns = [0.02, -0.01, 0.03, 0.01, -0.005, 0.025]
```

2. **Mean and std:**
```
mean_return = (0.02 - 0.01 + 0.03 + 0.01 - 0.005 + 0.025) / 6 = 0.0133
std_return = 0.0148
```

3. **Sharpe ratio:**
```
sharpe = 0.0133 / 0.0148 = 0.899
```

4. **Annualize:**
```
annualized_sharpe = 0.899 × sqrt(252) = 0.899 × 15.87 = 14.27
```

### Interpretation

| Sharpe Ratio | Interpretation |
|--------------|----------------|
| < 0 | Losing money |
| 0 - 1 | Poor risk-adjusted returns |
| 1 - 2 | Good risk-adjusted returns |
| 2 - 3 | Very good risk-adjusted returns |
| > 3 | Excellent (possibly too good to be true) |

---

## Max Drawdown Calculation

**Purpose:** Measure worst peak-to-trough decline.

**Formula:** `Max Drawdown = min((Value_t - Peak_t) / Peak_t)`

**Implementation:** `src/evaluation.py::compute_max_drawdown()`

### Algorithm

```
Input:
    cumulative_returns: Array of cumulative returns

Output:
    max_drawdown: Maximum drawdown (negative value)

Procedure:
1. Handle edge case
   if len(cumulative_returns) = 0:
       return 0

2. Calculate running maximum (peak)
   running_max ← zeros(len(cumulative_returns))
   running_max[0] ← cumulative_returns[0]

   for i in 1..len(cumulative_returns):
       running_max[i] ← max(running_max[i-1], cumulative_returns[i])

3. Calculate drawdown at each point
   drawdown ← (cumulative_returns - running_max) / (running_max + ε)
   # ε = 1e-10 to avoid division by zero

4. Return maximum drawdown
   max_drawdown ← min(drawdown)

   return max_drawdown
```

### Example

**Input:**
```
cumulative_returns = [0, 0.05, 0.10, 0.08, 0.06, 0.12, 0.11, 0.15]
```

**Calculation:**

1. **Running maximum:**
```
cum_returns: [0,    0.05, 0.10, 0.08, 0.06, 0.12, 0.11, 0.15]
running_max: [0,    0.05, 0.10, 0.10, 0.10, 0.12, 0.12, 0.15]
```

2. **Drawdown:**
```
Time  Cum_Return  Peak   Drawdown
0     0.00        0.00   0.00
1     0.05        0.05   0.00
2     0.10        0.10   0.00
3     0.08        0.10  -0.20  (down 20% from peak)
4     0.06        0.10  -0.40  (down 40% from peak)
5     0.12        0.12   0.00
6     0.11        0.12  -0.08
7     0.15        0.15   0.00
```

3. **Maximum drawdown:**
```
max_drawdown = min(drawdown) = -0.40 (-40%)
```

### Visualization

```
Cumulative Returns
  │
15%┤                             ●
  │                         ●
12%┤                     ●
  │                 ●   ●
10%┤         ●
  │     ●       ↓
 5%┤ ●           ↓
  │              ↓ Max Drawdown
 0%┼──────────────────────────────→ Time
  │              ↓ = -40%
  │          ●
 ```

---

## Complexity Summary

| Algorithm | Time | Space |
|-----------|------|-------|
| Purged K-Fold | O(n) | O(1) |
| Ensemble Weights | O(k) | O(k) |
| ROI Calculation | O(n × m) | O(m) |
| Sharpe Ratio | O(n) | O(1) |
| Max Drawdown | O(n) | O(n) |

Where:
- n = number of samples
- k = number of models
- m = number of thresholds

---

## References

1. **Purged K-Fold**: Lopez de Prado, M. (2018). Advances in Financial Machine Learning
2. **Sharpe Ratio**: Sharpe, W.F. (1966). Mutual Fund Performance
3. **Isotonic Regression**: Zadrozny & Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates

---

**Next:** See [Results](results.md) for performance analysis using these algorithms.
