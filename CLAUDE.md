# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency price prediction project that uses ensemble machine learning to predict next-bar up/down movements across multiple crypto symbols. The system trains models using 4-fold Purged K-Fold cross-validation to avoid look-ahead bias, evaluates them on multiple metrics (AUC, Brier score, log loss, accuracy), and computes ROI projections. Additionally, it includes a trade simulation program for backtesting strategies on BTC.

## Architecture

### Data Flow
1. **Input**: `crypto.csv` - OHLCV data in wide format (e.g., `open-BTCUSDT`, `close-ETHUSDT`)
2. **Preprocessing**: Reshape from wide to long format per symbol, handle datetime parsing
3. **Feature Engineering**: Compute 31+ technical indicators from OHLCV
4. **Labeling**: Binary classification target (up/down movement next period, threshold 1%)
5. **Model Training**: 4-fold Purged K-Fold with 6+ base models + 3 ensemble methods
6. **Evaluation**: Cross-validation metrics + per-symbol ROI calculation
7. **Trade Simulation** (optional): Backtest ensemble on BTC, optimize probability threshold, generate trade logs

### Key Files
- **`run_ensemble.py`**: Main end-to-end script. Handles data loading, preprocessing, feature engineering, model training, and evaluation. This is the canonical implementation to run.
- **`trade_simulator.py`**: Trade simulation program. Trains ensemble on BTC data, backtests across probability thresholds (0.30-0.80), finds optimal threshold by Sharpe ratio, generates detailed trade logs and performance metrics.
- **`visualize_backtest.py`**: Generates 6-panel visualization of backtest results (threshold optimization, equity curve, drawdown, trade distribution, cumulative PnL, summary stats).
- **`Preprocessing.ipynb`**: Optional exploratory notebook showing data reshaping and feature engineering steps. For reference only.
- **`crypto.csv`**: Raw OHLCV input (wide format, multi-symbol)
- **`roi_per_symbol.csv`**: Output from run_ensemble.py - net PnL for each model-symbol combination
- **`trade_log.csv`**: Output from trade_simulator.py - detailed entry/exit logs (time, price, return, PnL)
- **`threshold_optimization.csv`**: Output from trade_simulator.py - performance metrics across tested thresholds
- **`backtest_visualization.png`**: Output from visualize_backtest.py - 6-panel performance chart

### Data Structure

**Raw data format in crypto.csv**:
- DateTime column: `datetime` or `OpenDt`
- Symbol columns: Formatted as `{FIELD}-{SYMBOL}USDT` (e.g., `open-BTCUSDT`, `close-ETHUSDT`)
- Fields: open, high, low, close, volume

**After preprocessing**:
- MultiIndex DataFrame indexed by (symbol, datetime)
- Columns: open, high, low, close, volume
- Sorted chronologically per symbol

### Feature Engineering Parameters

Located in `run_ensemble.py` lines 89-98 and `trade_simulator.py` lines 41-50. Current parameters target ~4-hourly bars:
- **ROC (Rate of Change)**: Windows [1, 3, 42]
- **EMA Cross**: (84, 168) fast/slow pair → diff and ratio
- **RSI**: Windows [8, 14, 26] with 2-period lag
- **MACD**: Histogram + 2-period lag
- **CCI**: Windows [10, 20]
- **Bollinger Bands**: Windows [10, 20] → %b and bandwidth
- **Stochastic**: K values [8, 14] → fast/slow %D and histogram
- **Volatility**: 20-period rolling std of log returns
- **Volume**: 42-period mean, log changes at [42, 84]

### Preprocessing Pipeline

Three-branch preprocessing (ColumnTransformer in `run_ensemble.py` lines 186-196 and `trade_simulator.py` lines 149-161):
1. **Bounded columns** (bounded_cols): RSI, Stochastic, BB %b → passthrough (already 0-100 or 0-1)
2. **Logarithmic columns** (loggy_cols): Volume, volatility, EMA ratio, BB bandwidth → signed log1p then StandardScaler
3. **Other numeric columns** (the_rest): Everything else → StandardScaler only

This categorization handles skewed distributions without losing information.

### Model Architecture

**Base models** (6 total):
- LogisticRegression (penalty='l2', C=1.0, solver='saga')
- GaussianNB
- DecisionTree (max_depth=12)
- RandomForest (500 estimators, no depth limit)
- XGBoost (300-1000 estimators, 0.1-0.05 learning rate, GPU fallback to CPU)
- LinearSVM + KNN (optional, enabled by `USE_HEAVY_MODELS` flag)

**Ensemble methods** (3 in run_ensemble.py, 2 in trade_simulator.py):
- **Stacked Logistic Regression**: LogisticRegression trained on calibrated base model probabilities
- **Brier-weighted ensemble**: Weighted average using 1/brier_score as weights
- **LogLoss-weighted ensemble**: Weighted average using 1/log_loss as weights

**Probability calibration** (lines 372-379 in run_ensemble.py):
- All base model predictions calibrated via IsotonicRegression on CV fold predictions
- Applied before ensemble computation and ROI evaluation

### Cross-Validation Strategy

**Purged K-Fold** (custom implementation in run_ensemble.py lines 277-298):
- Splits by unique datetime values (not rows)
- 4 folds with 24-hour embargo (purge) between train/test
- Ensures no information leakage from temporal overlap
- Returns indices into the full X, y arrays

**Fold structure**:
- Train: All rows up to (test_start_time - 24h)
- Test: Rows from test_start_time to test_end_time
- Each fold trains all 6+ base models independently
- Collects CV predictions for calibration and ensemble training

### Output Metrics

**Performance table** (lines 398-431 in run_ensemble.py):
- AUC (ROC-AUC score)
- Brier score (MSE of probabilities)
- LogLoss (cross-entropy)
- Accuracy (binary classification at 0.5 threshold)

**ROI calculation** (lines 436-492 in run_ensemble.py):
- Per-model, per-symbol net PnL over all CV folds
- Tests thresholds from 0.3 to 0.8 probability
- Accounts for 0.1% transaction costs on position entry/exit
- Outputs: threshold, gross PnL, net PnL, accuracy, number of trades

### Trade Simulation (trade_simulator.py)

**Workflow**:
1. Load BTC data from crypto.csv
2. Compute 31 technical indicators
3. Train ensemble model on full BTC data (6 base models + average/stacked ensembles)
4. Backtest across probability thresholds 0.30 to 0.80
5. Find optimal threshold using Sharpe ratio
6. Generate trade logs and performance metrics

**Simulation Details**:
- Starting capital: $10,000 (customizable)
- Trading: Long-only, no shorting or pyramiding
- Position management: Buy at probability ≥ threshold, sell at probability < threshold
- Transaction cost: 0.1% per trade (customizable)
- Metrics tracked: Total return, number of trades, win rate, max drawdown, Sharpe ratio

**Output Files**:
- `trade_log.csv`: Entry/exit times, prices, returns, PnL for each trade
- `threshold_optimization.csv`: Performance across all tested thresholds
- `backtest_visualization.png` (optional): 6-panel chart via visualize_backtest.py

## Common Development Tasks

### Running the Full Pipeline
```bash
python run_ensemble.py
```
Runs end-to-end: loads crypto.csv, features, trains all 4 folds, all models, outputs metrics and ROI.

### Running Trade Simulation
```bash
python trade_simulator.py
```
Trains ensemble on BTC, backtests across thresholds, outputs trade logs and performance metrics.

### Visualizing Backtest Results
```bash
python visualize_backtest.py
```
Requires trade_simulator.py output (trade_log.csv, threshold_optimization.csv). Generates backtest_visualization.png.

### Testing with Subset of Data
In `run_ensemble.py` lines 80-85, uncomment and set:
```python
KEEP_SYMBOLS = ['BTC', 'ETH', 'ADA']  # limit to specific coins
MAX_ROWS_PER_SYMBOL = 10000  # limit rows per coin
```

### Adjusting Model Parameters

**Preprocessing thresholds**:
- Line 175 (run_ensemble.py): Adjust column categorization (bounded_cols, loggy_cols, the_rest)

**Feature parameters**:
- Lines 89-98 (run_ensemble.py) or 41-50 (trade_simulator.py): Adjust ROC_WINS, EMA_PAIRS, RSI_WINS, etc.

**Cross-validation**:
- Line 300 (run_ensemble.py): Change n_splits or embargo_hours in PurgedKFold

**Base model hyperparameters**:
- Lines 205-274 (run_ensemble.py) or 107-121 (trade_simulator.py): Modify individual model instantiation

**Trade simulation parameters**:
- Line 176 (trade_simulator.py): Change initial_capital (default 10000)
- Line 223: Change transaction_cost (default 0.001 = 0.1%)
- Line 250: Change threshold range (default 0.30 to 0.80)

**GPU acceleration**:
- Line 201 (run_ensemble.py): Set USE_GPU_XGB = False to force CPU
- Lines 213-253 (run_ensemble.py): Modify XGBoost tree_method or device parameters

### Debugging Data Issues

Common check points:
1. **Data shape after loading** (line 86 in run_ensemble.py): Verify datetime parsing and symbol count
2. **Feature creation** (line 152 in run_ensemble.py): Check NaN counts per feature (indicator warm-up)
3. **Label alignment** (lines 157-168 in run_ensemble.py): Diagnose row mismatches between X and y
4. **Preprocessing output** (line 171 in run_ensemble.py): Confirm no NaN after preprocessing
5. **BTC data availability** (line 205 in trade_simulator.py): Verify BTC in crypto.csv symbols

## Dependencies

**Required packages**:
- numpy, pandas, scikit-learn, xgboost, ta (Technical Analysis)

**Optional**:
- GPU support for XGBoost (CUDA)
- matplotlib, seaborn (for visualization)

**Installation**:
```bash
pip install numpy pandas scikit-learn xgboost ta
pip install matplotlib seaborn  # optional, for visualize_backtest.py
```

## Performance Notes

- **Runtime (run_ensemble.py)**: Full pipeline with all 6+ models × 4 folds can take 30+ minutes on CPU, especially with KNN/SVM
- **Runtime (trade_simulator.py)**: ~5-10 minutes on CPU (trains ensemble once, backtests across thresholds)
- **Memory**: ~2-4 GB during feature engineering and training
- **GPU**: XGBoost attempts GPU (device='cuda' or tree_method='gpu_hist') with graceful CPU fallback
- **Heavy models**: SVM and KNN disabled by default (`USE_HEAVY_MODELS = False`) due to runtime; enable if needed

## Git Workflow

- Current branch: `Ensemble_Branch`
- Main branch: `main`
- Recent commits focus on ensemble model development and ROI metrics

## Important Implementation Details

- **Avoid look-ahead bias** (run_ensemble.py): Purged K-Fold ensures embargo between folds; never use future data for features
- **Trade simulator training bias**: trade_simulator.py trains ensemble on full BTC data before backtesting (not realistic but faster). Use walk-forward approach for production.
- **Calibration order**: Base models calibrated on CV predictions before ensemble (not on ensemble)
- **Return calculation**: Uses log returns (ln(close_t+1 / close_t)) for mathematical correctness
- **Binary classification**: Labels are 0/1 (down/up); multiclass (-1/0/1) commented out in preprocessing
- **NaN handling**: Features forward-filled then back-filled per symbol (lines 151 in run_ensemble.py) to maximize usable rows
- **Threshold optimization**: Selects best threshold by Sharpe ratio (risk-adjusted returns), not max return

## Trade Simulation Interpretation

Example threshold optimization output:
```
Threshold    Return %    Trades    Win Rate    Sharpe
0.50         +5.23%      45        55.6%       0.842
0.52         +6.15%      38        57.9%       0.891  ← Best (Sharpe)
0.54         +4.89%      32        60.0%       0.764
```

**Best threshold 0.52**:
- Enter long when model probability ≥ 52%
- Exit when probability drops below 52%
- Generated 38 trades with 57.9% win rate
- Total $615 profit on $10k (6.15% return)
- Risk-adjusted performance (Sharpe) is best at this level

## Caveats

1. **Training data leakage** (trade_simulator.py): Trains on full data before backtesting. For production, use walk-forward validation.
2. **No slippage**: Assumes perfect execution at OHLCV prices. Real trading has 5-10 bps slippage.
3. **Fixed position sizing**: All-in trades. Real trading should use 2-5% risk per trade.
4. **No shorting**: Only long positions supported.
5. **Historical data only**: crypto.csv ends Feb 2024. To use live data, fetch from exchange API.
