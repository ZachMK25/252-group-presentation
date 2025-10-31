# 252 Group Presentation

Members:

- Zach Kuo
- Nikhil Ghind
- Ethan Ho

## Overview
- Predicts next-bar up/down movement across crypto symbols using an ensemble.
- Trains with 4-fold Purged K-Fold cross-validation and reports model metrics.
- Uses technical indicators (ROC, EMA, RSI, CCI, Stochastics, Bollinger Bands, volatility, volume features).

## Files
- `crypto.csv`: Input OHLCV data in wide format (e.g., `open-BTCUSDT`, `close-ETHUSDT`).
- `run_ensemble.py`: End-to-end training script with preprocessing and model training.
- `Preprocessing.ipynb`: Optional exploratory notebook for data prep and feature inspection.

## Requirements
- Python 3.9+
- Packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `ta`

Example setup (Conda):
```
conda create -n xgb-gpu python=3.10 -y
conda activate xgb-gpu
pip install numpy pandas scikit-learn xgboost ta
```

## How to Run
1. Place `crypto.csv` in the project root.
2. Run the ensemble training:
```
python run_ensemble.py
```

Notes:
- The script attempts GPU acceleration for XGBoost if available; otherwise it falls back to CPU `hist`.
- Early stopping parameters are disabled for compatibility with the installed XGBoost.
- Training uses full data and 4 folds; runtime can be long on CPU.

## Outputs
- Console prints a comparison table of base models and ensembles (AUC, Brier, LogLoss, Accuracy).
- `roi_per_symbol.csv` contains ROI data aligned to the original notebook workflow (optional reference).

## Clean-up
- Historical notebooks were removed to keep the repo tidy. Use `run_ensemble.py` for reproducible runs.

