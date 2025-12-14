"""
Trade Simulation Program
Simulates trading on BTC using the ensemble model from run_ensemble.py
- $10,000 starting capital
- Finds optimal probability threshold for buy/sell signals
- Generates detailed trade logs and performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("TRADE SIMULATION PROGRAM - BTC with Ensemble Model")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/5] Loading and preparing data...")

raw = pd.read_csv('crypto.csv')
if 'datetime' in raw.columns:
    dt_col = 'datetime'
elif 'OpenDt' in raw.columns:
    dt_col = 'OpenDt'
else:
    raise ValueError('crypto.csv must contain either "datetime" or "OpenDt" column')

raw[dt_col] = pd.to_datetime(raw[dt_col])

# Parse wide columns into long format per symbol
field_names = {'open', 'high', 'low', 'close', 'volume'}
by_symbol = {}
for col in raw.columns:
    if col == dt_col:
        continue
    if '-' in col:
        p0, p1 = col.split('-', 1)
        if p0.lower() in field_names:
            field = p0.lower()
            symbol = p1
        elif p1.lower() in field_names:
            field = p1.lower()
            symbol = p0
        else:
            continue
        symbol = symbol.replace('USDT', '').replace('_USDT', '')
    else:
        continue
    by_symbol.setdefault(symbol, {})[field] = raw[col]

# Extract BTC data
if 'BTC' not in by_symbol:
    raise ValueError('BTC not found in crypto.csv. Available symbols: ' +
                     str(list(by_symbol.keys())))

btc_fields = by_symbol['BTC']
btc_df = pd.DataFrame({
    'open': btc_fields['open'],
    'high': btc_fields['high'],
    'low': btc_fields['low'],
    'close': btc_fields['close'],
    'volume': btc_fields['volume'],
    'datetime': raw[dt_col]
})

for col in ['open', 'high', 'low', 'close', 'volume']:
    btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')

btc_df = btc_df.dropna(subset=['open', 'high', 'low', 'close'])
btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
btc_df = btc_df.set_index('datetime').sort_index()

print(f"  BTC data shape: {btc_df.shape}")
print(f"  Date range: {btc_df.index.min()} to {btc_df.index.max()}")

# ============================================================================
# 2. COMPUTE FEATURES
# ============================================================================
print("\n[2/5] Computing technical indicators...")

c, h, l, v = btc_df['close'], btc_df['high'], btc_df['low'], btc_df['volume']

ROC_WINS = [1, 3, 42]
EMA_PAIRS = [(84, 168)]
RSI_WINS = [8, 14, 26]
CCI_WINS = [10, 20]
BB_WINS = [10, 20]
STO_KS = [8, 14]
VOL_WIN = 20
VOL_MA_VOL = 42
VOL_ROC_W = [42, 84]

fe = pd.DataFrame(index=btc_df.index)

for n in ROC_WINS:
    fe[f'roc_{n}'] = ROCIndicator(c, n).roc()
for fast, slow in EMA_PAIRS:
    ema_fast = EMAIndicator(c, fast).ema_indicator()
    ema_slow = EMAIndicator(c, slow).ema_indicator()
    fe[f'ema_diff_{fast}_{slow}'] = ema_fast - ema_slow
    fe[f'ema_ratio_{fast}_{slow}'] = ema_fast / ema_slow
for n in RSI_WINS:
    rsi = RSIIndicator(c, n).rsi()
    fe[f'rsi_{n}'] = rsi
    fe[f'rsi_{n}_lag2'] = rsi.shift(2)

macd = MACD(c)
fe['macd_hist'] = macd.macd_diff()
fe['macd_hist_lag2'] = fe['macd_hist'].shift(2)

for n in CCI_WINS:
    fe[f'cci_{n}'] = CCIIndicator(h, l, c, n).cci()
for n in BB_WINS:
    bb = BollingerBands(c, n)
    fe[f'bb_pctb_{n}'] = bb.bollinger_pband()
    fe[f'bb_bw_{n}'] = bb.bollinger_wband()
for K in STO_KS:
    stoch = StochasticOscillator(h, l, c, K, 3)
    fast_k = stoch.stoch()
    fast_d = stoch.stoch_signal()
    slow_d = fast_d.rolling(3, min_periods=3).mean()
    fe[f'stoch_fastk_{K}'] = fast_k
    fe[f'stoch_fastd_{K}'] = fast_d
    fe[f'stoch_slowd_{K}'] = slow_d
    fe[f'stoch_hist_{K}'] = fast_k - slow_d

logret = np.log(c / c.shift(1))
fe[f'volatility_{VOL_WIN}'] = logret.rolling(VOL_WIN, min_periods=VOL_WIN).std()
fe[f'net_volume_{VOL_MA_VOL}'] = v.rolling(VOL_MA_VOL, min_periods=VOL_MA_VOL).mean()

def volume_log_change(vs: pd.Series, n: int, eps: float = 1e-9) -> pd.Series:
    return np.log1p(vs + eps) - np.log1p(vs.shift(n) + eps)

for n in VOL_ROC_W:
    fe[f'vol_change_{n}'] = volume_log_change(v, n)

# Fill NaNs from indicator warmup
fe = fe.ffill().bfill()
print(f"  Features created: {len(fe.columns)}")

# ============================================================================
# 3. CREATE TARGET LABELS
# ============================================================================
print("\n[3/5] Creating labels...")

rp = np.log(btc_df['close'].shift(-1) / btc_df['close'])
y = (rp > 0).astype(int)

Xy = fe.join(y.rename('label')).dropna()
X = Xy.drop(columns='label')
y = Xy['label'].astype(int)

print(f"  Rows after preprocessing: {len(Xy)}")
print(f"  Up movements: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
print(f"  Down movements: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")

# ============================================================================
# 4. TRAIN ENSEMBLE MODEL
# ============================================================================
print("\n[4/5] Training ensemble model...")

# Preprocessing
num_cols = X.select_dtypes(np.number).columns.tolist()
bounded_cols = [c for c in num_cols if c.startswith(('rsi_', 'stoch_', 'bb_pctb_'))]
loggy_cols = [c for c in num_cols if c.startswith(('net_volume_', 'vol_change_', 'bb_bw_', 'volatility_', 'ema_ratio_'))]
the_rest = sorted(list(set(num_cols) - set(bounded_cols) - set(loggy_cols)))

def signed_log1p(df_in):
    Z = df_in.copy()
    for c in Z.columns:
        x = Z[c].values
        Z[c] = np.sign(x) * np.log1p(np.abs(x))
    return Z

preprocessor = ColumnTransformer(
    transformers=[
        ("bounded_passthrough", "passthrough", bounded_cols),
        ("log_then_standardize", Pipeline([
            ("log", FunctionTransformer(signed_log1p, validate=False)),
            ("standard", StandardScaler())
        ]), loggy_cols),
        ("standardize_rest", StandardScaler(), the_rest)
    ],
    remainder="drop"
)

X_processed = preprocessor.fit_transform(X)

# Train base models
base_models = {
    'LogisticRegression': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='saga', tol=1e-2, random_state=42),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(max_depth=12, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2,
                                           min_samples_leaf=1, n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=1, eval_metric='logloss',
                             tree_method='hist', predictor='auto')
}

print("  Training base models...")
for name, model in base_models.items():
    model.fit(X_processed, y.values)
    print(f"    [OK] {name}")

# Calibrate predictions
print("  Calibrating probabilities...")
calibrator = CalibratedClassifierCV(LogisticRegression(), method='sigmoid', cv=5)
calibrator.fit(X_processed, y.values)

# Generate ensemble predictions (average + stacked)
print("  Creating ensemble...")
cv_preds = {}
for name, model in base_models.items():
    cv_preds[name] = model.predict_proba(X_processed)[:, 1]

ensemble_avg = np.mean([cv_preds[name] for name in base_models.keys()], axis=0)

Z = np.column_stack([cv_preds[name] for name in base_models.keys()])
stacker = LogisticRegression(max_iter=500, solver='lbfgs')
stacker.fit(Z, y.values)
ensemble_stacked = stacker.predict_proba(Z)[:, 1]

print("  Ensemble ready!")

# ============================================================================
# 5. BACKTEST WITH THRESHOLD OPTIMIZATION
# ============================================================================
print("\n[5/5] Backtesting with threshold optimization...")

def backtest_at_threshold(predictions, prices, returns, labels, threshold=0.5, initial_capital=10000,
                         transaction_cost=0.001):
    """
    Simulate trading at a specific threshold

    Returns:
        dict: Backtest metrics
    """
    signals = (predictions >= threshold).astype(int)

    equity = initial_capital
    position = 0  # 0 = flat, 1 = long
    trades = []
    equity_curve = [equity]

    entry_price = None
    entry_time = None

    for i in range(len(signals)):
        current_price = prices[i]
        current_signal = signals[i]
        pnl_change = 0

        # Position change logic
        if current_signal == 1 and position == 0:  # Enter long
            entry_price = current_price
            entry_time = i
            equity *= (1 - transaction_cost)
            position = 1

        elif current_signal == 0 and position == 1:  # Exit long
            if entry_price:
                pnl_change = (current_price - entry_price) / entry_price
                pnl = equity * pnl_change
                equity *= (1 + pnl_change)
                equity *= (1 - transaction_cost)

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_change,
                    'pnl': pnl,
                    'equity_at_exit': equity
                })
            position = 0

        # Apply return if holding
        if position == 1 and i > entry_time:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            equity *= (1 + price_return)

        equity_curve.append(equity)

    # Calculate metrics
    total_return = (equity - initial_capital) / initial_capital
    num_trades = len(trades)

    if num_trades > 0:
        winning_trades = sum(1 for t in trades if t['return'] > 0)
        win_rate = winning_trades / num_trades
        avg_trade_return = np.mean([t['return'] for t in trades])
    else:
        win_rate = 0
        avg_trade_return = 0

    # Drawdown calculation
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Sharpe ratio (annualized, assuming ~365 trading periods)
    if len(equity_curve) > 1:
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365)
    else:
        sharpe = 0

    return {
        'threshold': threshold,
        'final_equity': equity,
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'trades': trades,
        'equity_curve': equity_curve
    }

# Extract prices and calculate returns for backtesting
prices_aligned = btc_df['close'].loc[Xy.index].values
returns_aligned = rp.loc[Xy.index].values

# Test multiple thresholds
print("  Testing thresholds 0.30 to 0.80...")
results = []

for threshold in np.arange(0.30, 0.81, 0.02):
    result = backtest_at_threshold(
        ensemble_avg,
        prices_aligned,
        returns_aligned,
        y.values,
        threshold=threshold,
        initial_capital=10000
    )
    results.append(result)

# Find best threshold by Sharpe ratio
best_result = max(results, key=lambda x: x['sharpe_ratio'])

print("\n" + "=" * 80)
print("BACKTEST RESULTS - ENSEMBLE AVERAGE")
print("=" * 80)

print(f"\nBest Threshold (Sharpe): {best_result['threshold']:.2f}")
print(f"  Final Equity: ${best_result['final_equity']:,.2f}")
print(f"  Total Return: {best_result['total_return']*100:+.2f}%")
print(f"  Number of Trades: {best_result['num_trades']}")
print(f"  Win Rate: {best_result['win_rate']*100:.1f}%")
print(f"  Avg Trade Return: {best_result['avg_trade_return']*100:+.2f}%")
print(f"  Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
print(f"  Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")

# Show threshold comparison
print("\n" + "-" * 80)
print("Threshold Comparison:")
print("-" * 80)
print(f"{'Threshold':<12} {'Return %':<12} {'Trades':<10} {'Win Rate':<12} {'Sharpe':<10}")
print("-" * 80)
for result in results[::2]:  # Show every other threshold for readability
    print(f"{result['threshold']:<12.2f} {result['total_return']*100:>10.2f}% {result['num_trades']:>8} "
          f"{result['win_rate']*100:>10.1f}% {result['sharpe_ratio']:>9.3f}")

# ============================================================================
# 6. DETAILED TRADE LOG
# ============================================================================
print("\n" + "=" * 80)
print("TRADE LOG (Best Threshold)")
print("=" * 80)

if best_result['trades']:
    print(f"\nTotal Trades: {len(best_result['trades'])}\n")

    trade_df = pd.DataFrame(best_result['trades'])

    # Map indices back to datetimes
    index_to_datetime = {i: dt for i, dt in enumerate(Xy.index)}
    trade_df['entry_time'] = trade_df['entry_time'].map(index_to_datetime)
    trade_df['exit_time'] = trade_df['exit_time'].map(index_to_datetime)

    # Display first 20 trades
    print(f"{'Entry Time':<20} {'Exit Time':<20} {'Entry Price':<12} {'Exit Price':<12} "
          f"{'Return %':<10} {'PnL $':<12}")
    print("-" * 100)

    for _, trade in trade_df.head(20).iterrows():
        print(f"{str(trade['entry_time']):<20} {str(trade['exit_time']):<20} "
              f"${trade['entry_price']:>10,.2f} ${trade['exit_price']:>10,.2f} "
              f"{trade['return']*100:>8.2f}% ${trade['pnl']:>10,.2f}")

    if len(trade_df) > 20:
        print(f"\n... and {len(trade_df) - 20} more trades")

    # Save trade log
    trade_df.to_csv('trade_log.csv', index=False)
    print(f"\n[SAVED] Full trade log saved to: trade_log.csv")
else:
    print("\nNo trades generated at best threshold!")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
summary_df = pd.DataFrame(results)
summary_df.to_csv('threshold_optimization.csv', index=False)
print(f"[SAVED] Threshold optimization results saved to: threshold_optimization.csv")

print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)
print(f"\nInitial Capital: $10,000.00")
print(f"Final Equity: ${best_result['final_equity']:,.2f}")
print(f"Total Return: {best_result['total_return']*100:+.2f}%")
print(f"Optimal Threshold: {best_result['threshold']:.2f}")
print("\n")
