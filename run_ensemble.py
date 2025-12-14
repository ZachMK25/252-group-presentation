import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands
from sklearn.isotonic import IsotonicRegression
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# CREATE DIRECTORIES FOR SAVING MODELS AND RESULTS
# ============================================================================
os.makedirs('saved_models', exist_ok=True)
os.makedirs('roi_results', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"saved_models/ensemble_{timestamp}"
os.makedirs(model_dir, exist_ok=True)
print(f"\n[INFO] Model directory: {model_dir}\n")

print("Loading crypto.csv and reshaping to long format...")
raw = pd.read_csv('crypto.csv')
# Detect datetime-like column
if 'datetime' in raw.columns:
    dt_col = 'datetime'
elif 'OpenDt' in raw.columns:
    dt_col = 'OpenDt'
else:
    raise ValueError('crypto.csv must contain either "datetime" or "OpenDt" column')
raw[dt_col] = pd.to_datetime(raw[dt_col])

# Parse wide columns into long format per symbol
field_names = {'open','high','low','close','volume'}
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

frames = []
for sym, fields in by_symbol.items():
    if not field_names.issubset(fields.keys()):
        continue
    g = pd.DataFrame({
        'open':   fields['open'],
        'high':   fields['high'],
        'low':    fields['low'],
        'close':  fields['close'],
        'volume': fields['volume'],
        'datetime': raw[dt_col]
    })
    g['symbol'] = sym
    frames.append(g)
if not frames:
    raise ValueError('No complete symbols found in crypto.csv (need open/high/low/close/volume).')

df = pd.concat(frames, ignore_index=True)
df['datetime'] = pd.to_datetime(df['datetime'])
# Ensure numeric types for price/volume columns
for col in ['open','high','low','close','volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['open','high','low','close']).set_index(['symbol','datetime']).sort_index()
# Use a moderate set of symbols and rows to keep SVM/KNN runtime reasonable
KEEP_SYMBOLS = []
MAX_ROWS_PER_SYMBOL = None  # use all available rows per symbol
if KEEP_SYMBOLS:
    df = df[df.index.get_level_values('symbol').isin(KEEP_SYMBOLS)]
if MAX_ROWS_PER_SYMBOL:
    df = df.groupby(level='symbol', group_keys=False).tail(MAX_ROWS_PER_SYMBOL)
print(f"Data shape: {df.shape}")
print(f"Symbols: {df.index.get_level_values('symbol').unique().tolist()}")

print("Computing features per symbol...")
ROC_WINS   = [1, 3, 42]
EMA_PAIRS  = [(84, 168)]
RSI_WINS   = [8, 14, 26]
CCI_WINS   = [10, 20]
BB_WINS    = [10, 20]
STO_KS     = [8, 14]
VOL_WIN    = 20
VOL_MA_VOL = 42
VOL_ROC_W  = [42, 84]

fe_list = []
for idx, (sym, g) in enumerate(df.groupby(level='symbol')):
    gi = g.reset_index(level='symbol', drop=True)
    c, h, l, v = gi['close'], gi['high'], gi['low'], gi['volume']
    if idx == 0:
        print(f"Debug [{sym}] close dtype={c.dtype}, head={c.head(3).tolist()}")
        print(f"Debug [{sym}] close describe: min={c.min()}, max={c.max()}, nunique={c.nunique()}, na={c.isna().sum()}")
    # Build features on datetime-only index, then attach symbol back
    fe_sym = pd.DataFrame(index=gi.index)
    for n in ROC_WINS:
        fe_sym[f'roc_{n}'] = ROCIndicator(c, n).roc()
    for fast, slow in EMA_PAIRS:
        ema_fast = EMAIndicator(c, fast).ema_indicator()
        ema_slow = EMAIndicator(c, slow).ema_indicator()
        fe_sym[f'ema_diff_{fast}_{slow}'] = ema_fast - ema_slow
        fe_sym[f'ema_ratio_{fast}_{slow}'] = ema_fast / ema_slow
    for n in RSI_WINS:
        rsi = RSIIndicator(c, n).rsi()
        fe_sym[f'rsi_{n}'] = rsi
        fe_sym[f'rsi_{n}_lag2'] = rsi.shift(2)
    macd = MACD(c)
    fe_sym['macd_hist'] = macd.macd_diff()
    fe_sym['macd_hist_lag2'] = fe_sym['macd_hist'].shift(2)
    for n in CCI_WINS:
        fe_sym[f'cci_{n}'] = CCIIndicator(h, l, c, n).cci()
    for n in BB_WINS:
        bb = BollingerBands(c, n)
        fe_sym[f'bb_pctb_{n}'] = bb.bollinger_pband()
        fe_sym[f'bb_bw_{n}'] = bb.bollinger_wband()
    for K in STO_KS:
        stoch = StochasticOscillator(h, l, c, K, 3)
        fast_k = stoch.stoch()
        fast_d = stoch.stoch_signal()
        slow_d = fast_d.rolling(3, min_periods=3).mean()
        fe_sym[f'stoch_fastk_{K}'] = fast_k
        fe_sym[f'stoch_fastd_{K}'] = fast_d
        fe_sym[f'stoch_slowd_{K}'] = slow_d
        fe_sym[f'stoch_hist_{K}']  = fast_k - slow_d
    logret = np.log(c / c.shift(1))
    fe_sym[f'volatility_{VOL_WIN}'] = logret.rolling(VOL_WIN, min_periods=VOL_WIN).std()
    fe_sym[f'net_volume_{VOL_MA_VOL}'] = v.rolling(VOL_MA_VOL, min_periods=VOL_MA_VOL).mean()
    def volume_log_change(vs: pd.Series, n: int, eps: float = 1e-9) -> pd.Series:
        return np.log1p(vs + eps) - np.log1p(vs.shift(n) + eps)
    for n in VOL_ROC_W:
        fe_sym[f'vol_change_{n}'] = volume_log_change(v, n)
    # Attach back the MultiIndex with symbol
    fe_sym.index = pd.MultiIndex.from_product([[sym], fe_sym.index], names=['symbol','datetime'])
    fe_list.append(fe_sym)

fe = pd.concat(fe_list).sort_index()
# Fill indicator warm-up NaNs within each symbol to maximize usable rows
fe = fe.groupby(level='symbol', group_keys=False).apply(lambda t: t.ffill().bfill())
print(f"Features created: {len(fe.columns)}")

print("Labeling up/down movements...")
rp = df['close'].groupby(level='symbol').transform(lambda s: np.log(s.shift(-1) / s))
y = (rp > 0).astype(int)
print(f"Features rows: {len(fe)}, Labels rows (raw): {len(y)}, Labels non-NaN: {y.notna().sum()}")
common_idx = fe.index.intersection(y.dropna().index)
print(f"Common index rows: {len(common_idx)}")
Xy = fe.join(y.rename('label')).dropna()
print(f"Rows after join+dropna: {len(Xy)}")
if len(Xy) == 0:
    # Diagnose per-column NaNs count
    joined = fe.join(y.rename('label'))
    na_counts = joined.isna().sum().sort_values(ascending=False)
    print("Top NaN counts per column:")
    print(na_counts.head(10))
    raise SystemExit("No rows remained after join; printed diagnostics above.")
X = Xy.drop(columns='label')
y = Xy['label'].astype(int)
print(f"Final dataset shape: {X.shape}")

print("Setting up preprocessing...")
num_cols = X.select_dtypes(np.number).columns.tolist()
bounded_cols   = [c for c in num_cols if c.startswith(('rsi_', 'stoch_', 'bb_pctb_'))]
loggy_cols     = [c for c in num_cols if c.startswith(('net_volume_', 'vol_change_', 'bb_bw_', 'volatility_', 'ema_ratio_'))]
the_rest       = sorted(list(set(num_cols) - set(bounded_cols) - set(loggy_cols)))

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

print("Defining base models...")
USE_HEAVY_MODELS = True  # enable heavy models per user request
HEAVY_SAMPLE_SIZE = 30000  # subsample heavy model training per fold to keep runtime feasible
USE_GPU_XGB = True  # attempt GPU; will gracefully fall back
xgb_version = getattr(xgb, '__version__', 'unknown')
print(f"XGBoost version detected: {xgb_version}")

base_models = {
    'LogisticRegression': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='saga', tol=1e-2, random_state=42),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(max_depth=12, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, random_state=42)
}

# Add XGBoost with GPU if available (v2+: device='cuda'), else try v1 GPU ('gpu_hist'), else CPU
def make_xgb_gpu_v2():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda'
    )

def make_xgb_gpu_v1():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor'
    )

def make_xgb_cpu():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss',
        tree_method='hist',
        predictor='auto'
    )

if USE_GPU_XGB:
    try:
        base_models['XGBoost'] = make_xgb_gpu_v2()
        print("XGBoost configured for GPU via device='cuda' (v2 style)")
    except Exception as e_v2:
        print(f"XGBoost v2 GPU config failed ({e_v2}); trying v1 'gpu_hist'.")
        try:
            base_models['XGBoost'] = make_xgb_gpu_v1()
            print("XGBoost configured for GPU via tree_method='gpu_hist' (v1 style)")
        except Exception as e_v1:
            print(f"XGBoost GPU not available ({e_v1}); using CPU 'hist'.")
            base_models['XGBoost'] = make_xgb_cpu()
else:
    base_models['XGBoost'] = make_xgb_cpu()

if USE_HEAVY_MODELS:
    base_models.update({
        'LinearSVM': CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=42), method='sigmoid', cv=3),
        'KNN': KNeighborsClassifier(n_neighbors=25, weights='distance', n_jobs=-1)
    })

print("Implementing Purged K-Fold...")
class PurgedKFold:
    def __init__(self, n_splits=5, embargo_hours=24):
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours
    def split(self, X, y=None, groups=None):
        dt_index = X.index.get_level_values('datetime')
        unique_times = sorted(dt_index.unique())
        n_times = len(unique_times)
        test_size = max(1, n_times // self.n_splits)
        embargo_periods = pd.Timedelta(hours=self.embargo_hours)
        for i in range(self.n_splits):
            test_start_idx = i * test_size
            test_end_idx = min((i + 1) * test_size, n_times)
            test_start_time = unique_times[test_start_idx]
            test_end_time = unique_times[test_end_idx - 1]
            train_end_time = test_start_time - embargo_periods
            test_mask = (dt_index >= test_start_time) & (dt_index <= test_end_time)
            train_mask = dt_index <= train_end_time
            train_indices = X.index[train_mask]
            test_indices = X.index[test_mask]
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield (X.index.get_indexer(train_indices), X.index.get_indexer(test_indices))

pkf = PurgedKFold(n_splits=4, embargo_hours=24)

print("Training models and collecting CV predictions...")
cv_predictions = {name: [] for name in base_models.keys()}
cv_true_labels = []
cv_returns = []
cv_symbols = []

# Storage for trained models per fold
trained_models_per_fold = {}
trained_preprocessors = {}

fold_num = 0
for train_idx, val_idx in pkf.split(X, y):
    fold_num += 1
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Save preprocessor for this fold
    trained_preprocessors[fold_num] = joblib.dump(preprocessor, f"{model_dir}/preprocessor_fold_{fold_num}.pkl", compress=3)

    cv_true_labels.extend(y_val.values)
    # collect returns aligned to validation indices
    returns_aligned = rp.loc[Xy.index]
    cv_returns.extend(returns_aligned.iloc[val_idx].values)
    cv_symbols.extend(Xy.index.get_level_values('symbol').values[val_idx])

    fold_models = {}
    for name, model in base_models.items():
        train_X = X_train_processed
        train_y = y_train.values
        train_X_use = train_X
        train_y_use = train_y
        try:
            if name == 'XGBoost':
                # stronger config using validation fold (no early stopping for compatibility)
                pos = float(train_y_use.sum())
                neg = float(len(train_y_use) - pos)
                pos_weight = (neg / pos) if pos > 0 else 1.0
                model.set_params(n_estimators=1000, learning_rate=0.05, n_jobs=-1, scale_pos_weight=pos_weight)
                model.fit(
                    train_X_use, train_y_use,
                    eval_set=[(X_val_processed, y_val.values)],
                    verbose=False
                )
            else:
                model.fit(train_X_use, train_y_use)
        except Exception as e:
            if name == 'XGBoost':
                print(f"XGBoost GPU fit failed ({e}); retrying on CPU 'hist' (no early stopping).")
                model = XGBClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss',
                    tree_method='hist',
                    predictor='auto',
                    scale_pos_weight=pos_weight
                )
                base_models[name] = model
                model.fit(
                    train_X_use, train_y_use,
                    eval_set=[(X_val_processed, y_val.values)],
                    verbose=False
                )
            else:
                raise

        # Save trained model
        model_path = f"{model_dir}/{name}_fold_{fold_num}.pkl"
        joblib.dump(model, model_path, compress=3)
        fold_models[name] = model_path

        y_pred_proba = model.predict_proba(X_val_processed)[:, 1]
        cv_predictions[name].extend(y_pred_proba)

    trained_models_per_fold[fold_num] = fold_models
    print(f"Fold {fold_num} done: Train={len(train_idx)}, Val={len(val_idx)}")

cv_predictions = {name: np.array(preds) for name, preds in cv_predictions.items()}
cv_true_labels = np.array(cv_true_labels)
cv_returns = np.array(cv_returns)
cv_symbols = np.array(cv_symbols)

print("\nCalibrating probabilities (Isotonic)...")
calibrated_predictions = {}
calibrators = {}
for name, preds in cv_predictions.items():
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(preds, cv_true_labels)
    calibrated_predictions[name] = calibrator.predict(preds)
    calibrators[name] = calibrator
    # Save calibrator
    calibrator_path = f"{model_dir}/calibrator_{name}.pkl"
    joblib.dump(calibrator, calibrator_path, compress=3)

print("Computing ensemble weights (Brier and LogLoss)...")
def calc_weights(calibrated_predictions, cv_true_labels, method='brier'):
    scores = {}
    for name, preds in calibrated_predictions.items():
        if method == 'brier':
            score = 1 / (brier_score_loss(cv_true_labels, preds) + 1e-8)
        else:
            preds_clipped = np.clip(preds, 1e-8, 1 - 1e-8)
            score = 1 / (log_loss(cv_true_labels, preds_clipped) + 1e-8)
        scores[name] = score
    total = sum(scores.values())
    weights = {name: sc / total for name, sc in scores.items()}
    return weights, scores

brier_weights, _ = calc_weights(calibrated_predictions, cv_true_labels, method='brier')
logloss_weights, _ = calc_weights(calibrated_predictions, cv_true_labels, method='log_loss')

print("Evaluating base and ensemble performance...")
def evaluate(preds, labels):
    auc = roc_auc_score(labels, preds)
    brier = brier_score_loss(labels, preds)
    ll = log_loss(labels, np.clip(preds, 1e-8, 1 - 1e-8))
    acc = accuracy_score(labels, (preds >= 0.5).astype(int))
    return auc, brier, ll, acc

rows = []
for name in base_models.keys():
    auc, brier, ll, acc = evaluate(calibrated_predictions[name], cv_true_labels)
    rows.append({'model': name, 'auc': auc, 'brier': brier, 'logloss': ll, 'accuracy': acc})

# Compute ensemble probabilities
ensemble_brier = sum(brier_weights[name] * calibrated_predictions[name] for name in base_models.keys())
ensemble_logloss = sum(logloss_weights[name] * calibrated_predictions[name] for name in base_models.keys())

# Stacked logistic regression over calibrated base-model predictions
Z = np.column_stack([calibrated_predictions[name] for name in base_models.keys()])
stacker = LogisticRegression(max_iter=500, solver='lbfgs')
stacker.fit(Z, cv_true_labels)
stacked_preds = stacker.predict_proba(Z)[:, 1]
auc, brier, ll, acc = evaluate(stacked_preds, cv_true_labels)
rows.append({'model': 'Ensemble_StackedLogReg', 'auc': auc, 'brier': brier, 'logloss': ll, 'accuracy': acc})

# Save stacker
stacker_path = f"{model_dir}/stacker_ensemble.pkl"
joblib.dump(stacker, stacker_path, compress=3)

# Evaluate ensembles and include in results
auc, brier, ll, acc = evaluate(ensemble_brier, cv_true_labels)
rows.append({'model': 'Ensemble_Brier', 'auc': auc, 'brier': brier, 'logloss': ll, 'accuracy': acc})
auc, brier, ll, acc = evaluate(ensemble_logloss, cv_true_labels)
rows.append({'model': 'Ensemble_LogLoss', 'auc': auc, 'brier': brier, 'logloss': ll, 'accuracy': acc})

results_df = pd.DataFrame(rows).sort_values('auc', ascending=False)
print("\nModel Performance Comparison:")
print(results_df.to_string(index=False, float_format='%.4f'))

# Save model performance results
results_df.to_csv(f'roi_results/model_performance_{timestamp}.csv', index=False)
print(f"✓ Model performance saved to: roi_results/model_performance_{timestamp}.csv")

# ============================================================================
# ENHANCED ROI CALCULATION SECTION
# ============================================================================
print("\nEnhanced ROI Calculation:")

def compute_roi_detailed(probs, returns, labels, transaction_cost=0.001, thresholds=None):
    """
    Compute ROI with detailed metrics for all thresholds

    Returns:
        dict: Best result
        list: All results for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.8, 0.01)

    all_results = []
    best = {'threshold': None, 'net_pnl': -np.inf, 'gross_pnl': 0.0, 'accuracy': 0.0, 'n_trades': 0}

    for t in thresholds:
        signals = (probs >= t).astype(int)
        position_returns = signals * returns
        # transaction costs on position changes
        position_changes = np.diff(np.concatenate([[0], signals]))
        transaction_costs = np.abs(position_changes) * transaction_cost
        gross_pnl = float(np.sum(position_returns))
        net_pnl = gross_pnl - float(np.sum(transaction_costs))
        acc = accuracy_score(labels, signals)
        n_trades = int(np.sum(np.abs(position_changes)))

        result = {
            'threshold': float(t),
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'transaction_costs': float(np.sum(transaction_costs)),
            'accuracy': float(acc),
            'n_trades': n_trades
        }
        all_results.append(result)

        if net_pnl > best['net_pnl']:
            best = result.copy()

    return best, all_results

# Compute ROI for all ensemble methods
print("\nOptimizing thresholds for ensemble methods...")
best_brier, roi_brier_all = compute_roi_detailed(ensemble_brier, cv_returns, cv_true_labels)
best_logloss, roi_logloss_all = compute_roi_detailed(ensemble_logloss, cv_returns, cv_true_labels)
best_stacked, roi_stacked_all = compute_roi_detailed(stacked_preds, cv_returns, cv_true_labels)

# Also compute for base models
base_roi_results = {}
for name in base_models.keys():
    best, all_results = compute_roi_detailed(calibrated_predictions[name], cv_returns, cv_true_labels)
    base_roi_results[name] = {'best': best, 'all_thresholds': all_results}

print(f"  Brier Ensemble: threshold={best_brier['threshold']:.3f}, net_pnl={best_brier['net_pnl']:+.6f}, accuracy={best_brier['accuracy']:.3f}, trades={best_brier['n_trades']}")
print(f"  LogLoss Ensemble: threshold={best_logloss['threshold']:.3f}, net_pnl={best_logloss['net_pnl']:+.6f}, accuracy={best_logloss['accuracy']:.3f}, trades={best_logloss['n_trades']}")
print(f"  Stacked Ensemble: threshold={best_stacked['threshold']:.3f}, net_pnl={best_stacked['net_pnl']:+.6f}, accuracy={best_stacked['accuracy']:.3f}, trades={best_stacked['n_trades']}")

# Save detailed ROI results for all thresholds
roi_brier_df = pd.DataFrame(roi_brier_all)
roi_logloss_df = pd.DataFrame(roi_logloss_all)
roi_stacked_df = pd.DataFrame(roi_stacked_all)

roi_brier_df.to_csv(f'roi_results/roi_brier_thresholds_{timestamp}.csv', index=False)
roi_logloss_df.to_csv(f'roi_results/roi_logloss_thresholds_{timestamp}.csv', index=False)
roi_stacked_df.to_csv(f'roi_results/roi_stacked_thresholds_{timestamp}.csv', index=False)

# ROI per symbol matrix (net_pnl)
unique_symbols = sorted(np.unique(cv_symbols))
row_order = list(base_models.keys()) + ['Ensemble_StackedLogReg', 'Ensemble_Brier', 'Ensemble_LogLoss']
roi_table = {}
for name in row_order:
    if name in base_models:
        preds = calibrated_predictions[name]
    elif name == 'Ensemble_Brier':
        preds = ensemble_brier
    elif name == 'Ensemble_LogLoss':
        preds = ensemble_logloss
    else:  # Ensemble_StackedLogReg
        preds = stacked_preds

    vals = []
    for sym in unique_symbols:
        mask = (cv_symbols == sym)
        if not np.any(mask):
            vals.append(np.nan)
        else:
            best, _ = compute_roi_detailed(preds[mask], cv_returns[mask], cv_true_labels[mask])
            vals.append(float(best['net_pnl']))
    roi_table[name] = vals

roi_df = pd.DataFrame(roi_table, index=unique_symbols).T
# Average net_pnl across all coins per model
roi_df['AVG'] = roi_df.mean(axis=1, skipna=True)
print("\nROI per Symbol (net_pnl) with AVG:")
print(roi_df.to_string(float_format='%.6f'))

# Save ROI per symbol
roi_df.to_csv(f'roi_results/roi_per_symbol_{timestamp}.csv')

# ============================================================================
# SAVE SUMMARY AND METADATA
# ============================================================================
summary_data = {
    'timestamp': timestamp,
    'model_directory': model_dir,
    'data_shape': str(X.shape),
    'n_features': len(fe.columns),
    'n_symbols': len(unique_symbols),
    'symbols': ','.join(unique_symbols),
    'cv_folds': 4,
    'embargo_hours': 24,
    'total_models_trained': len(base_models) * 4 + len(calibrators) + 1,  # base*folds + calibrators + stacker
    'best_brier_threshold': best_brier['threshold'],
    'best_brier_net_pnl': best_brier['net_pnl'],
    'best_logloss_threshold': best_logloss['threshold'],
    'best_logloss_net_pnl': best_logloss['net_pnl'],
    'best_stacked_threshold': best_stacked['threshold'],
    'best_stacked_net_pnl': best_stacked['net_pnl'],
}

summary_df = pd.DataFrame([summary_data])
summary_df.to_csv(f'roi_results/summary_{timestamp}.csv', index=False)

print("\n" + "=" * 80)
print("TRAINING AND EVALUATION COMPLETE")
print("=" * 80)
print(f"\n✓ Models saved to: {model_dir}/")
print(f"✓ ROI results saved to: roi_results/")
print(f"\nGenerated Files:")
print(f"  - saved_models/{os.path.basename(model_dir)}/ (trained models)")
print(f"  - roi_results/model_performance_{timestamp}.csv")
print(f"  - roi_results/roi_brier_thresholds_{timestamp}.csv")
print(f"  - roi_results/roi_logloss_thresholds_{timestamp}.csv")
print(f"  - roi_results/roi_stacked_thresholds_{timestamp}.csv")
print(f"  - roi_results/roi_per_symbol_{timestamp}.csv")
print(f"  - roi_results/summary_{timestamp}.csv")
print("\n")
