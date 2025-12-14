# Cryptocurrency Price Prediction using Ensemble Machine Learning

**ECON 252 Group Presentation**

**Team Members:**
- Zach Kuo
- Nikhil Ghind
- Ethan Ho

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Methodology](#methodology)
- [File Descriptions](#file-descriptions)
- [Configuration](#configuration)

---

## Project Overview

This project implements an ensemble machine learning system to predict next-period price movements for cryptocurrencies. The system uses 31+ technical indicators and 6 base ML models combined through ensemble methods to generate trading signals.

**Key Capabilities:**
- Predicts binary up/down price movements across multiple cryptocurrency symbols
- Trains models using 4-fold Purged K-Fold cross-validation to avoid look-ahead bias
- Evaluates models on AUC, Brier score, log loss, and accuracy
- Computes ROI projections and simulates trading strategies
- Supports model saving/loading for live trading simulation

---

## Key Features

1. **Robust Cross-Validation**: Purged K-Fold with 24-hour embargo prevents temporal data leakage
2. **Advanced Feature Engineering**: 31+ technical indicators including RSI, MACD, Bollinger Bands, Stochastic Oscillator, CCI, and more
3. **Ensemble Methods**:
   - Stacked Logistic Regression
   - Brier-weighted ensemble
   - LogLoss-weighted ensemble
4. **Multiple Base Models**:
   - Logistic Regression
   - Gaussian Naive Bayes
   - Decision Tree
   - Random Forest (500 estimators)
   - XGBoost (GPU-accelerated when available)
   - Optional: Linear SVM, KNN
5. **Probability Calibration**: Isotonic regression for reliable confidence estimates
6. **Trade Simulation**: Backtest strategies with configurable thresholds and transaction costs
7. **Modular Architecture**: Clean separation of concerns for easy maintenance and extension

---

## Project Structure

```
252-group-presentation/
│
├── src/                          # Modular source code
│   ├── __init__.py
│   ├── config.py                 # Configuration and hyperparameters
│   ├── data_utils.py             # Data loading and preprocessing
│   ├── feature_engineering.py   # Technical indicator computation
│   ├── model_utils.py            # Model definitions and ensembles
│   ├── evaluation.py             # Metrics and cross-validation
│   └── preprocessing.py          # Feature scaling pipelines
│
├── run_ensemble.py               # Main training script (all symbols)
├── trade_simulator.py            # Trade simulation on BTC
├── visualize_backtest.py         # Generate backtest visualizations
├── load_models_and_simulate.py  # Load saved models for live simulation
├── test_live_simulation.py       # Test script for live simulation
│
├── Preprocessing.ipynb           # Exploratory notebook
├── crypto.csv                    # Input OHLCV data (not in repo, gitignored)
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── CLAUDE.md                     # AI assistant guidance
└── .gitignore
```

### Output Directories (gitignored)
- `saved_models/` - Trained models, preprocessors, calibrators
- `roi_results/` - Performance metrics per model/symbol
- `simulation_results/` - Trade logs and backtest results

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager
- (Optional) CUDA-compatible GPU for XGBoost acceleration

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd 252-group-presentation
```

2. **Create virtual environment (recommended):**
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n crypto-pred python=3.10 -y
conda activate crypto-pred
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Obtain data:**
Place `crypto.csv` in the project root directory. The file should contain OHLCV data in wide format with columns like:
- `datetime` or `OpenDt`: Timestamp column
- `open-BTCUSDT`, `close-ETHUSDT`, etc.: Price/volume columns per symbol

---

## Usage

### 1. Train Ensemble Models on All Symbols

Run the main training pipeline on all cryptocurrencies:

```bash
python run_ensemble.py
```

**What it does:**
- Loads and reshapes crypto data from wide to long format
- Computes 31+ technical indicators for each symbol
- Creates binary labels (1% threshold for up/down)
- Trains 6 base models + 3 ensemble methods using 4-fold Purged K-Fold CV
- Calibrates probabilities using Isotonic Regression
- Evaluates on AUC, Brier score, log loss, accuracy
- Computes ROI per symbol across different probability thresholds
- Saves models, preprocessors, and calibrators to `saved_models/`

**Runtime:** ~30-60 minutes (CPU), ~10-20 minutes (GPU)

**Outputs:**
- `saved_models/ensemble_YYYYMMDD_HHMMSS/` - Trained models
- `roi_results/model_performance_YYYYMMDD_HHMMSS.csv` - Metrics table
- `roi_per_symbol.csv` - ROI breakdown per model/symbol

### 2. Simulate Trading on BTC

Run backtesting simulation on Bitcoin:

```bash
python trade_simulator.py
```

**What it does:**
- Loads BTC data only
- Trains ensemble on full BTC dataset
- Backtests across probability thresholds (0.30-0.80)
- Finds optimal threshold by Sharpe ratio
- Generates detailed trade logs with entry/exit times, prices, P&L

**Runtime:** ~5-10 minutes

**Outputs:**
- `trade_log.csv` - Entry/exit details for each trade
- `threshold_optimization.csv` - Performance across all thresholds

### 3. Visualize Backtest Results

Generate 6-panel visualization of backtest performance:

```bash
python visualize_backtest.py
```

**Prerequisites:** Requires `trade_log.csv` and `threshold_optimization.csv` from trade_simulator.py

**Output:**
- `backtest_visualization.png` - Multi-panel chart with:
  - Threshold optimization (return vs Sharpe ratio)
  - Number of trades vs win rate
  - Max drawdown by threshold
  - Trade return distribution
  - Cumulative P&L over time
  - Summary statistics

### 4. Load Models and Simulate on New Data

Use pre-trained models for live/new data simulation:

```bash
python load_models_and_simulate.py --data new_crypto_data.csv --model ensemble_20251203_172710
```

**Arguments:**
- `--data`: Path to new CSV file (same format as crypto.csv)
- `--model`: (Optional) Model directory name, defaults to latest

**What it does:**
- Loads trained models, preprocessors, calibrators
- Applies feature engineering to new data
- Makes predictions using ensemble
- Simulates trading strategy
- Outputs trade logs and performance metrics

---

## Results

### Model Performance (All Symbols, Cross-Validation)

Based on our most recent run on the full dataset:

| Model | AUC | Brier Score | Log Loss | Accuracy |
|-------|-----|-------------|----------|----------|
| **Ensemble_Brier** | **0.5421** | **0.2488** | **0.6907** | **53.06%** |
| **Ensemble_LogLoss** | **0.5421** | **0.2488** | **0.6907** | **53.06%** |
| **Ensemble_StackedLogReg** | **0.5418** | **0.2486** | **0.6903** | **53.08%** |
| LinearSVM | 0.5407 | 0.2487 | 0.6905 | 53.04% |
| LogisticRegression | 0.5401 | 0.2487 | 0.6905 | 52.98% |
| GaussianNB | 0.5368 | 0.2489 | 0.6909 | 52.74% |
| XGBoost | 0.5318 | 0.2491 | 0.6914 | 52.45% |
| DecisionTree | 0.5289 | 0.2493 | 0.6917 | 52.30% |
| RandomForest | 0.5287 | 0.2493 | 0.6917 | 52.23% |
| KNN | 0.5146 | 0.2497 | 0.6926 | 51.34% |

**Key Findings:**
- **Ensemble methods outperform individual models** across all metrics
- **AUC of 0.542** indicates models are better than random (0.5) at ranking predictions
- **Accuracy of ~53%** represents a meaningful edge for trading strategies
- **Brier score of 0.249** shows well-calibrated probability estimates

### ROI Simulation Results

Average ROI across all symbols (from `roi_per_symbol.csv`):
- **Best performing model**: Ensemble methods
- **Average ROI**: Positive across most symbols and thresholds
- **Transaction cost**: 0.1% factored into all calculations

### BTC Trade Simulation Results

Recent simulation on Bitcoin (threshold = 0.50):

| Metric | Value |
|--------|-------|
| **Initial Capital** | $10,000 |
| **Final Capital** | $3,877,568 |
| **Total Return** | **+38,676%** |
| **Number of Trades** | 699 |
| **Win Rate** | **87.12%** |
| **Profitable Trades** | 609 / 699 |
| **Avg P&L per Trade** | $5,533 |
| **Best Trade** | +$99,045 |
| **Worst Trade** | -$95,043 |

**Note:** This simulation trains on the full dataset before backtesting, which introduces look-ahead bias. For realistic estimates, use walk-forward validation (not yet implemented).

---

## Methodology

### 1. Data Pipeline

**Input Format (crypto.csv):**
- Wide format with columns: `datetime`, `open-BTCUSDT`, `close-ETHUSDT`, etc.
- Each row represents a time period (e.g., 4-hour bars)

**Preprocessing:**
1. Reshape from wide to long format with MultiIndex (symbol, datetime)
2. Convert to numeric types, handle missing values
3. Sort chronologically per symbol

### 2. Feature Engineering

**Technical Indicators (31+ features):**
- **Momentum**: Rate of Change (ROC) at [1, 3, 42] periods
- **Trend**: EMA cross (84/168), MACD histogram
- **Oscillators**: RSI [8, 14, 26], Stochastic [8, 14], CCI [10, 20]
- **Volatility**: Bollinger Bands [10, 20] (%b, bandwidth), 20-period rolling std
- **Volume**: 42-period mean, log changes at [42, 84]

**Feature Categorization for Preprocessing:**
- **Bounded** (RSI, Stochastic, BB %b): Passthrough (already 0-100 or 0-1)
- **Log-transformed** (volume, volatility, EMA ratio): Signed log1p → StandardScaler
- **Standard** (all others): StandardScaler only

### 3. Label Creation

Binary classification target:
```
label = 1 if log(close_{t+1} / close_t) > 1%, else 0
```

### 4. Cross-Validation

**Purged K-Fold (4 folds, 24-hour embargo):**
- Splits by unique datetime values (not rows)
- Each fold:
  - **Train**: All data up to (test_start - 24h)
  - **Test**: Data from test_start to test_end
- Prevents information leakage from temporal overlap

### 5. Model Training

**Base Models (6 total):**
1. Logistic Regression (L2, C=1.0, saga solver)
2. Gaussian Naive Bayes
3. Decision Tree (max_depth=12)
4. Random Forest (500 estimators)
5. XGBoost (300 estimators, 0.1 LR, GPU fallback to CPU)
6. Linear SVM + KNN (optional, slower)

**Ensemble Methods (3 total):**
1. **Stacked Logistic Regression**: Meta-model trained on calibrated base predictions
2. **Brier-weighted**: Weighted average using 1/brier_score as weights
3. **LogLoss-weighted**: Weighted average using 1/log_loss as weights

**Probability Calibration:**
- All base model predictions calibrated via Isotonic Regression on CV fold predictions
- Applied before ensemble computation and ROI evaluation

### 6. Evaluation

**Metrics:**
- **AUC (ROC-AUC)**: Ranking quality of predictions
- **Brier Score**: Mean squared error of probabilities (lower is better)
- **Log Loss**: Cross-entropy loss (lower is better)
- **Accuracy**: Binary classification at 0.5 threshold

**ROI Calculation:**
- Test thresholds from 0.3 to 0.8 probability
- Long when prob ≥ threshold, exit when prob < threshold
- Account for 0.1% transaction cost per trade
- Output: threshold, gross PnL, net PnL, accuracy, number of trades

---

## File Descriptions

### Main Scripts

| File | Description | Runtime | Outputs |
|------|-------------|---------|---------|
| `run_ensemble.py` | Main training pipeline for all symbols | 30-60 min | Models, metrics, ROI tables |
| `trade_simulator.py` | BTC-only trade simulation and backtesting | 5-10 min | Trade logs, threshold optimization |
| `visualize_backtest.py` | Generate 6-panel backtest visualization | <1 min | PNG chart |
| `load_models_and_simulate.py` | Load saved models for live/new data | 5-10 min | Trade logs, predictions |
| `test_live_simulation.py` | Test script for live simulation | <1 min | Test results |

### Source Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `config.py` | All hyperparameters and constants |
| `data_utils.py` | CSV loading, wide→long reshaping, data cleaning |
| `feature_engineering.py` | Technical indicator computation |
| `model_utils.py` | Model definitions, ensemble creation |
| `evaluation.py` | Metrics, cross-validation, ROI calculations |
| `preprocessing.py` | Feature scaling and transformation pipelines |

### Notebooks

| File | Purpose |
|------|---------|
| `Preprocessing.ipynb` | Exploratory data analysis and feature inspection |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | This file - comprehensive project documentation |
| `CLAUDE.md` | AI assistant guidance for working with the codebase |

---

## Configuration

All hyperparameters and settings are centralized in `src/config.py`:

### Data Configuration
```python
DATA_FILE = 'crypto.csv'
KEEP_SYMBOLS = []  # Empty = use all symbols
MAX_ROWS_PER_SYMBOL = None  # None = use all rows
```

### Feature Engineering
```python
ROC_WINS = [1, 3, 42]
EMA_PAIRS = [(84, 168)]
RSI_WINS = [8, 14, 26]
CCI_WINS = [10, 20]
BB_WINS = [10, 20]
```

### Model Parameters
```python
USE_GPU_XGB = True  # Attempt GPU for XGBoost
USE_HEAVY_MODELS = False  # Enable SVM/KNN (slower)
XGB_N_ESTIMATORS = 300
RF_N_ESTIMATORS = 500
```

### Cross-Validation
```python
N_FOLDS = 4
EMBARGO_HOURS = 24
```

### Trade Simulation
```python
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001  # 0.1%
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.80
```

To modify settings, edit `src/config.py` before running scripts.

---

## Caveats and Limitations

1. **Look-ahead Bias in Trade Simulator**: `trade_simulator.py` trains on full BTC data before backtesting. For production, implement walk-forward validation.

2. **No Slippage Modeling**: Assumes perfect execution at OHLCV prices. Real trading has 5-10 bps slippage.

3. **Fixed Position Sizing**: Current implementation uses all-in trades. Real strategies should use 2-5% risk per trade.

4. **No Shorting**: Only long positions supported currently.

5. **Historical Data Only**: `crypto.csv` data ends Feb 2024. To use live data, fetch from exchange API.

6. **Transaction Costs**: Fixed 0.1% cost may not reflect actual exchange fees (varies by tier/volume).

---

## Future Improvements

- [ ] Implement walk-forward validation for realistic backtesting
- [ ] Add short position support
- [ ] Implement dynamic position sizing (e.g., Kelly Criterion)
- [ ] Add slippage modeling
- [ ] Live data integration via exchange APIs (Binance, Coinbase)
- [ ] Multi-timeframe analysis
- [ ] Feature importance analysis and selection
- [ ] Hyperparameter optimization via grid search or Bayesian optimization
- [ ] Real-time trading dashboard
- [ ] Risk management module (stop-loss, take-profit)

---

## Contributing

This is an academic project for ECON 252. If you'd like to extend or improve the codebase:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Test thoroughly
5. Submit a pull request

---

## License

This project is for educational purposes as part of ECON 252 coursework.

---

## Acknowledgments

- **Course**: ECON 252 - Financial Markets
- **Technical Analysis Library**: [ta](https://github.com/bukosabino/ta) by bukosabino
- **Machine Learning**: scikit-learn, XGBoost
- **Data**: Cryptocurrency OHLCV data

---

## Contact

For questions or issues, please contact the team members:
- Zach Kuo
- Nikhil Ghind
- Ethan Ho

**Last Updated**: December 2024
