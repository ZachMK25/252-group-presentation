# Quick Start Guide

Get up and running with the cryptocurrency prediction system in 5 minutes.

---

## Prerequisites

- Python 3.9 or higher
- `crypto.csv` data file
- 8 GB RAM minimum

---

## Installation (2 minutes)

```bash
# 1. Navigate to project directory
cd 252-group-presentation

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Basic Usage (3 minutes)

### Option 1: Train Models

```bash
python run_ensemble.py
```

**What it does:**
- Trains 6 ML models + 3 ensembles
- Evaluates on 49 cryptocurrencies
- Outputs performance metrics
- Saves trained models

**Runtime:** 8-10 minutes

**Output:**
```
Model Performance Comparison
════════════════════════════
Ensemble_Brier:  AUC=0.542, Accuracy=53.06%
Ensemble_LogLoss: AUC=0.542, Accuracy=53.06%
...
```

### Option 2: Simulate Trading (Faster)

```bash
python trade_simulator.py
```

**What it does:**
- Trains on BTC only
- Backtests trading strategy
- Generates trade logs

**Runtime:** 5-10 minutes

**Output:**
```
Best threshold: 0.50
Total return: +38,676%
Win rate: 87.12%
Trades: 699
```

### Option 3: Visualize Results

```bash
# First run trade_simulator.py, then:
python visualize_backtest.py
```

**Output:** `backtest_visualization.png` (6-panel chart)

---

## Quick Test (1 minute)

Verify installation:

```bash
# Test imports
python -c "from src import config, data_utils, model_utils; print('✓ Success')"

# Test data loading (requires crypto.csv)
python -c "from src.data_utils import load_crypto_csv; df = load_crypto_csv(); print(f'✓ Loaded {len(df)} rows')"
```

---

## Project Structure

```
252-group-presentation/
├── src/                  # Modular source code
│   ├── config.py         # Hyperparameters
│   ├── data_utils.py     # Data loading
│   ├── feature_engineering.py
│   ├── model_utils.py
│   └── evaluation.py
├── run_ensemble.py       # Main training script
├── trade_simulator.py    # BTC simulation
├── visualize_backtest.py # Visualization
└── crypto.csv            # Your data file (place here)
```

---

## Expected Results

### Model Performance
| Model | AUC | Accuracy |
|-------|-----|----------|
| Ensemble_Brier | 0.5421 | 53.06% |
| Ensemble_LogLoss | 0.5421 | 53.06% |
| LogisticRegression | 0.5401 | 52.98% |

### BTC Simulation
- **Total Return:** +38,676%
- **Win Rate:** 87.12%
- **Trades:** 699
- ⚠️ **Note:** Contains look-ahead bias (see [Results](results.md#caveats))

---

## Next Steps

1. **Understand the results:** Read [Results Documentation](results.md)
2. **Customize settings:** Edit `src/config.py` ([Configuration Guide](configuration.md))
3. **Run on subset:** [Advanced Usage](advanced-usage.md#subset-data)
4. **Issues?** Check [Troubleshooting Guide](troubleshooting.md)

---

## Common Issues

**Error:** `FileNotFoundError: crypto.csv`
- **Fix:** Place `crypto.csv` in project root directory

**Error:** `ModuleNotFoundError: No module named 'ta'`
- **Fix:** Run `pip install -r requirements.txt`

**Memory Error:**
- **Fix:** Reduce data size in `src/config.py`:
  ```python
  KEEP_SYMBOLS = ['BTC', 'ETH', 'ADA']  # Fewer symbols
  MAX_ROWS_PER_SYMBOL = 10000           # Fewer rows
  ```

See full troubleshooting guide: [troubleshooting.md](troubleshooting.md)

---

## Learn More

- [Full Installation Guide](installation.md)
- [Detailed Usage Guide](usage.md)
- [Implementation Details](implementation.md)
- [Architecture Overview](architecture.md)

---

**Estimated time to first results:** 5 minutes setup + 10 minutes training = **15 minutes total**
