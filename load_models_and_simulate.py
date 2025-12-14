"""
Load trained ensemble models and run live/new data simulation

This script:
1. Loads pre-trained models, preprocessors, and calibrators from saved_models/
2. Applies feature engineering to new crypto data
3. Makes predictions using the ensemble
4. Simulates trading strategy based on ensemble confidence
5. Outputs trade logs and performance metrics

Usage:
    python load_models_and_simulate.py --data new_crypto_data.csv --model ensemble_20251203_172710
    python load_models_and_simulate.py --data new_data.csv  # auto-finds latest model
"""

import pandas as pd
import numpy as np
import joblib
import os
import glob
from datetime import datetime
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering functions
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands

np.random.seed(42)


# Define signed_log1p (needed for deserializing preprocessors)
def signed_log1p(df_in):
    """Apply signed log1p transformation (must match training)"""
    Z = df_in.copy()
    for c in Z.columns:
        x = Z[c].values
        Z[c] = np.sign(x) * np.log1p(np.abs(x))
    return Z


def find_latest_model_dir():
    """Find the latest saved model directory"""
    model_dirs = glob.glob('saved_models/ensemble_*')
    if not model_dirs:
        raise ValueError("No saved models found in saved_models/ directory")
    latest = sorted(model_dirs)[-1]
    print(f"[INFO] Using latest model directory: {latest}")
    return latest


def load_crypto_data(csv_path):
    """Load and reshape crypto data from CSV (same format as training)"""
    print(f"Loading crypto data from {csv_path}...")
    raw = pd.read_csv(csv_path)

    # Detect datetime column
    if 'datetime' in raw.columns:
        dt_col = 'datetime'
    elif 'OpenDt' in raw.columns:
        dt_col = 'OpenDt'
    else:
        raise ValueError('CSV must contain either "datetime" or "OpenDt" column')

    raw[dt_col] = pd.to_datetime(raw[dt_col])

    # Parse wide format to long format
    field_names = {'open', 'high', 'low', 'close', 'volume'}
    by_symbol = {}

    for col in raw.columns:
        if col == dt_col:
            continue
        if '-' in col:
            parts = col.split('-', 1)
            field = parts[0].lower()
            symbol = parts[1]
            if field not in field_names:
                field = parts[1].lower()
                symbol = parts[0]
            if field not in field_names:
                continue
        else:
            continue

        symbol = symbol.replace('USDT', '').replace('_USDT', '')
        by_symbol.setdefault(symbol, {})[field] = raw[col]

    # Build DataFrame
    frames = []
    for sym, fields in by_symbol.items():
        if not field_names.issubset(fields.keys()):
            continue
        df_sym = pd.DataFrame({
            'open': fields['open'],
            'high': fields['high'],
            'low': fields['low'],
            'close': fields['close'],
            'volume': fields['volume'],
            'datetime': raw[dt_col]
        })
        df_sym['symbol'] = sym
        frames.append(df_sym)

    if not frames:
        raise ValueError('No complete symbols found in CSV')

    df = pd.concat(frames, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.set_index(['symbol', 'datetime']).sort_index()

    print(f"Data shape: {df.shape}")
    print(f"Symbols: {sorted(df.index.get_level_values('symbol').unique().tolist())}")

    return df


def compute_features(df):
    """Compute 31 technical indicator features (same as training)"""
    print("Computing features...")

    # Feature parameters (must match run_ensemble.py)
    ROC_WINS = [1, 3, 42]
    EMA_PAIRS = [(84, 168)]
    RSI_WINS = [8, 14, 26]
    CCI_WINS = [10, 20]
    BB_WINS = [10, 20]
    STO_KS = [8, 14]
    VOL_WIN = 20
    VOL_MA_VOL = 42
    VOL_ROC_W = [42, 84]

    fe_list = []

    for sym, g in df.groupby(level='symbol'):
        gi = g.reset_index(level='symbol', drop=True)
        c, h, l, v = gi['close'], gi['high'], gi['low'], gi['volume']

        fe_sym = pd.DataFrame(index=gi.index)

        # ROC
        for n in ROC_WINS:
            fe_sym[f'roc_{n}'] = ROCIndicator(c, n).roc()

        # EMA
        for fast, slow in EMA_PAIRS:
            ema_fast = EMAIndicator(c, fast).ema_indicator()
            ema_slow = EMAIndicator(c, slow).ema_indicator()
            fe_sym[f'ema_diff_{fast}_{slow}'] = ema_fast - ema_slow
            fe_sym[f'ema_ratio_{fast}_{slow}'] = ema_fast / ema_slow

        # RSI
        for n in RSI_WINS:
            rsi = RSIIndicator(c, n).rsi()
            fe_sym[f'rsi_{n}'] = rsi
            fe_sym[f'rsi_{n}_lag2'] = rsi.shift(2)

        # MACD
        macd = MACD(c)
        fe_sym['macd_hist'] = macd.macd_diff()
        fe_sym['macd_hist_lag2'] = fe_sym['macd_hist'].shift(2)

        # CCI
        for n in CCI_WINS:
            fe_sym[f'cci_{n}'] = CCIIndicator(h, l, c, n).cci()

        # Bollinger Bands
        for n in BB_WINS:
            bb = BollingerBands(c, n)
            fe_sym[f'bb_pctb_{n}'] = bb.bollinger_pband()
            fe_sym[f'bb_bw_{n}'] = bb.bollinger_wband()

        # Stochastic
        for K in STO_KS:
            stoch = StochasticOscillator(h, l, c, K, 3)
            fast_k = stoch.stoch()
            fast_d = stoch.stoch_signal()
            slow_d = fast_d.rolling(3, min_periods=3).mean()
            fe_sym[f'stoch_fastk_{K}'] = fast_k
            fe_sym[f'stoch_fastd_{K}'] = fast_d
            fe_sym[f'stoch_slowd_{K}'] = slow_d
            fe_sym[f'stoch_hist_{K}'] = fast_k - slow_d

        # Volatility
        logret = np.log(c / c.shift(1))
        fe_sym[f'volatility_{VOL_WIN}'] = logret.rolling(VOL_WIN, min_periods=VOL_WIN).std()

        # Volume indicators
        fe_sym[f'net_volume_{VOL_MA_VOL}'] = v.rolling(VOL_MA_VOL, min_periods=VOL_MA_VOL).mean()

        def volume_log_change(vs, n, eps=1e-9):
            return np.log1p(vs + eps) - np.log1p(vs.shift(n) + eps)

        for n in VOL_ROC_W:
            fe_sym[f'vol_change_{n}'] = volume_log_change(v, n)

        # Reattach symbol to index
        fe_sym.index = pd.MultiIndex.from_product([[sym], fe_sym.index], names=['symbol', 'datetime'])
        fe_list.append(fe_sym)

    fe = pd.concat(fe_list).sort_index()
    # Fill NaNs from indicator warm-up
    fe = fe.groupby(level='symbol', group_keys=False).apply(lambda t: t.ffill().bfill())

    print(f"Features shape: {fe.shape}, Features: {fe.shape[1]}")

    return fe


def load_models_and_artifacts(model_dir):
    """Load all trained models, preprocessors, and calibrators"""
    print(f"Loading models from {model_dir}...")

    models = {}
    preprocessors = {}
    calibrators = {}

    # Load all base models from all folds
    model_files = glob.glob(f"{model_dir}/*_fold_*.pkl")

    for model_file in model_files:
        basename = os.path.basename(model_file)
        parts = basename.replace('.pkl', '').rsplit('_fold_', 1)
        if len(parts) != 2:
            continue

        model_name, fold_str = parts
        try:
            fold_num = int(fold_str)
        except:
            continue

        if model_name not in models:
            models[model_name] = {}

        models[model_name][fold_num] = joblib.load(model_file)

    # Load preprocessors
    preproc_files = glob.glob(f"{model_dir}/preprocessor_fold_*.pkl")
    for preproc_file in preproc_files:
        basename = os.path.basename(preproc_file)
        fold_num = int(basename.replace('preprocessor_fold_', '').replace('.pkl', ''))
        preprocessors[fold_num] = joblib.load(preproc_file)

    # Load calibrators
    calibrator_files = glob.glob(f"{model_dir}/calibrator_*.pkl")
    for cal_file in calibrator_files:
        basename = os.path.basename(cal_file)
        model_name = basename.replace('calibrator_', '').replace('.pkl', '')
        calibrators[model_name] = joblib.load(cal_file)

    # Load stacker if available
    stacker = None
    stacker_file = f"{model_dir}/stacker_ensemble.pkl"
    if os.path.exists(stacker_file):
        stacker = joblib.load(stacker_file)

    print(f"Loaded {len(models)} model types with {len(preprocessors)} preprocessors and {len(calibrators)} calibrators")
    print(f"Model types: {sorted(models.keys())}")

    return models, preprocessors, calibrators, stacker


def make_predictions_ensemble(X, models, preprocessors, calibrators, stacker):
    """
    Make ensemble predictions on new data

    Strategy: Use fold 1 preprocessor and average predictions from all folds
    """
    fold_num = 1  # Use first fold's preprocessor
    preprocessor = preprocessors[fold_num]

    X_processed = preprocessor.transform(X.copy())

    predictions = {}

    # Get predictions from all folds and average
    for model_name, fold_dict in models.items():
        fold_preds = []

        for fold_id in sorted(fold_dict.keys()):
            model = fold_dict[fold_id]
            try:
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_processed)[:, 1]
                else:
                    preds = model.predict(X_processed)
                fold_preds.append(preds)
            except Exception as e:
                print(f"Warning: Could not get predictions from {model_name} fold {fold_id}: {e}")
                continue

        if fold_preds:
            avg_preds = np.mean(fold_preds, axis=0)

            # Apply calibration
            if model_name in calibrators:
                avg_preds = calibrators[model_name].predict(avg_preds)

            predictions[model_name] = avg_preds

    if not predictions:
        raise ValueError("No predictions generated")

    # Compute ensemble predictions
    ensemble_preds = np.mean([predictions[name] for name in sorted(predictions.keys())], axis=0)

    # Apply stacked ensemble if available
    if stacker:
        try:
            Z = np.column_stack([predictions[name] for name in sorted(predictions.keys())])
            stacked_ensemble = stacker.predict_proba(Z)[:, 1]
            ensemble_preds = 0.5 * ensemble_preds + 0.5 * stacked_ensemble
        except Exception as e:
            print(f"Warning: Could not apply stacker: {e}")

    return ensemble_preds, predictions


def simulate_trading(df, ensemble_preds, threshold=0.50, initial_capital=10000, transaction_cost=0.001):
    """
    Simulate trading strategy based on ensemble predictions

    Args:
        df: DataFrame with OHLCV data
        ensemble_preds: Ensemble probability predictions
        threshold: Probability threshold to enter/exit positions
        initial_capital: Starting capital in USD
        transaction_cost: Cost per transaction as fraction (0.001 = 0.1%)

    Returns:
        trade_log: List of trades with entry/exit details
        metrics: Performance metrics
    """

    df_copy = df.copy()
    df_copy['pred'] = ensemble_preds
    df_copy['signal'] = (df_copy['pred'] >= threshold).astype(int)

    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    capital = initial_capital

    for idx, row in df_copy.iterrows():
        current_signal = row['signal']
        symbol, dt = idx
        price = row['close']

        # Check for position changes
        position_change = int(current_signal) - position

        if position_change == 1:  # Enter long
            entry_price = price
            entry_time = dt
            cost = capital * transaction_cost
            capital -= cost
            position = 1

        elif position_change == -1:  # Exit long
            if entry_time is not None:
                pnl = capital * (price / entry_price - 1)
                cost = capital * transaction_cost
                net_pnl = pnl - cost
                capital += pnl - cost

                trade = {
                    'symbol': symbol,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': dt,
                    'exit_price': price,
                    'return_pct': (price / entry_price - 1) * 100,
                    'pnl': pnl,
                    'net_pnl': net_pnl,
                    'capital': capital
                }
                trades.append(trade)

            position = 0
            entry_time = None

    # Close final position if open
    if position == 1 and len(df_copy) > 0:
        final_price = df_copy.iloc[-1]['close']
        pnl = capital * (final_price / entry_price - 1)
        cost = capital * transaction_cost
        net_pnl = pnl - cost
        capital += pnl - cost

        trade = {
            'symbol': df_copy.index[-1][0],
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': df_copy.index[-1][1],
            'exit_price': final_price,
            'return_pct': (final_price / entry_price - 1) * 100,
            'pnl': pnl,
            'net_pnl': net_pnl,
            'capital': capital
        }
        trades.append(trade)

    # Compute metrics
    if len(trades) > 0:
        trade_df = pd.DataFrame(trades)
        total_return = (capital - initial_capital) / initial_capital * 100
        win_count = len(trade_df[trade_df['net_pnl'] > 0])
        win_rate = win_count / len(trade_df) * 100

        metrics = {
            'threshold': threshold,
            'total_return_pct': total_return,
            'final_capital': capital,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_pnl': trade_df['net_pnl'].mean(),
            'total_pnl': trade_df['net_pnl'].sum()
        }
    else:
        metrics = {
            'threshold': threshold,
            'total_return_pct': 0.0,
            'final_capital': capital,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0,
            'total_pnl': 0.0
        }

    return trades, metrics


def main():
    parser = argparse.ArgumentParser(description='Load trained models and run simulation on new data')
    parser.add_argument('--data', type=str, required=True, help='Path to new crypto data CSV')
    parser.add_argument('--model', type=str, default=None, help='Model directory (auto-finds latest if not provided)')
    parser.add_argument('--threshold', type=float, default=0.50, help='Trading threshold (default 0.50)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital (default 10000)')
    parser.add_argument('--output-dir', type=str, default='simulation_results', help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    df = load_crypto_data(args.data)

    # Compute features
    features = compute_features(df)

    # Select common rows
    common_idx = features.index.intersection(df.index)
    X = features.loc[common_idx]
    df_sim = df.loc[common_idx]

    print(f"Final dataset: {X.shape}")

    # Load models
    model_dir = args.model if args.model else find_latest_model_dir()
    models, preprocessors, calibrators, stacker = load_models_and_artifacts(model_dir)

    # Make predictions
    print("Making ensemble predictions...")
    ensemble_preds, individual_preds = make_predictions_ensemble(X, models, preprocessors, calibrators, stacker)

    # Run simulation
    print(f"Running trading simulation with threshold {args.threshold}...")
    trades, metrics = simulate_trading(df_sim, ensemble_preds, threshold=args.threshold, initial_capital=args.capital)

    # Save results
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_csv = f"{args.output_dir}/trades_{timestamp}.csv"
        trades_df.to_csv(trades_csv, index=False)
        print(f"Trade log saved to: {trades_csv}")

    metrics_csv = f"{args.output_dir}/simulation_metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to: {metrics_csv}")

    # Print summary
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:+.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Total P&L: ${metrics['total_pnl']:+,.2f}")
    print(f"Avg P&L per Trade: ${metrics['avg_trade_pnl']:+,.2f}")
    print("="*60)


if __name__ == '__main__':
    main()
