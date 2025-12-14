"""
Feature engineering module
Computes technical indicators from OHLCV data
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import EMAIndicator, MACD, CCIIndicator
from ta.volatility import BollingerBands

from src.config import (
    ROC_WINS, EMA_PAIRS, RSI_WINS, CCI_WINS, BB_WINS,
    STOCH_K_VALS, VOL_WIN, VOL_MEAN_WIN, VOL_LOG_WINS,
    LABEL_THRESHOLD
)


def compute_features(df):
    """
    Compute all technical indicator features from OHLCV data

    Args:
        df: DataFrame indexed by (symbol, datetime) or datetime with OHLCV columns

    Returns:
        DataFrame with original OHLCV + all computed features
    """
    print("Computing technical indicators...")

    feat = df.copy()
    close = feat['close']
    high = feat['high']
    low = feat['low']
    volume = feat['volume']

    # ========================================================================
    # Rate of Change (ROC)
    # ========================================================================
    for w in ROC_WINS:
        roc = ROCIndicator(close=close, window=w)
        feat[f'roc_{w}'] = roc.roc()

    # ========================================================================
    # EMA Cross (difference and ratio)
    # ========================================================================
    for fast, slow in EMA_PAIRS:
        ema_fast = EMAIndicator(close=close, window=fast).ema_indicator()
        ema_slow = EMAIndicator(close=close, window=slow).ema_indicator()
        feat[f'ema_diff_{fast}_{slow}'] = ema_fast - ema_slow
        feat[f'ema_ratio_{fast}_{slow}'] = ema_fast / ema_slow

    # ========================================================================
    # RSI with lag
    # ========================================================================
    for w in RSI_WINS:
        rsi = RSIIndicator(close=close, window=w).rsi()
        feat[f'rsi_{w}'] = rsi
        feat[f'rsi_{w}_lag2'] = rsi.shift(2)

    # ========================================================================
    # MACD histogram with lag
    # ========================================================================
    macd_obj = MACD(close=close)
    feat['macd_hist'] = macd_obj.macd_diff()
    feat['macd_hist_lag2'] = feat['macd_hist'].shift(2)

    # ========================================================================
    # CCI (Commodity Channel Index)
    # ========================================================================
    for w in CCI_WINS:
        cci = CCIIndicator(high=high, low=low, close=close, window=w)
        feat[f'cci_{w}'] = cci.cci()

    # ========================================================================
    # Bollinger Bands (%b and bandwidth)
    # ========================================================================
    for w in BB_WINS:
        bb = BollingerBands(close=close, window=w, window_dev=2)
        feat[f'bb_pctb_{w}'] = bb.bollinger_pband()
        feat[f'bb_width_{w}'] = bb.bollinger_wband()

    # ========================================================================
    # Stochastic Oscillator (fast/slow %D and histogram)
    # ========================================================================
    for k in STOCH_K_VALS:
        stoch = StochasticOscillator(high=high, low=low, close=close, window=k, smooth_window=3)
        feat[f'stoch_d_fast_{k}'] = stoch.stoch()
        feat[f'stoch_d_slow_{k}'] = stoch.stoch_signal()
        feat[f'stoch_hist_{k}'] = feat[f'stoch_d_fast_{k}'] - feat[f'stoch_d_slow_{k}']

    # ========================================================================
    # Volatility (rolling std of log returns)
    # ========================================================================
    log_ret = np.log(close / close.shift(1))
    feat['volatility'] = log_ret.rolling(VOL_WIN).std()

    # ========================================================================
    # Volume features
    # ========================================================================
    feat['volume_mean'] = volume.rolling(VOL_MEAN_WIN).mean()
    for w in VOL_LOG_WINS:
        feat[f'volume_log_change_{w}'] = np.log(volume / volume.shift(w))

    # Handle NaN values from indicator warm-up
    feat = feat.fillna(method='ffill').fillna(method='bfill')

    print(f"  Features computed: {feat.shape[1]} total columns")

    return feat


def create_labels(df, threshold=LABEL_THRESHOLD):
    """
    Create binary classification labels based on next-period price change

    Args:
        df: DataFrame with 'close' column
        threshold: Minimum return % to classify as 'up' (default 1%)

    Returns:
        Series of binary labels (0=down, 1=up)
    """
    close = df['close']
    log_ret_next = np.log(close.shift(-1) / close)

    # Binary: 1 if next return > threshold, else 0
    labels = (log_ret_next > threshold).astype(int)

    return labels


def split_features_labels(df):
    """
    Split DataFrame into features (X) and labels (y)
    Drops OHLCV columns and the label column from features

    Args:
        df: DataFrame with features and 'label' column

    Returns:
        X (features), y (labels)
    """
    # Drop price/volume columns and label
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['label'] if 'label' in df.columns else None

    return X, y


def categorize_feature_columns(feature_cols):
    """
    Categorize feature columns for preprocessing

    Returns:
        bounded_cols: Features already bounded (0-100 or 0-1)
        loggy_cols: Features needing log transformation
        the_rest: Features needing only standard scaling
    """
    bounded_keywords = ['rsi', 'stoch', 'bb_pctb']
    loggy_keywords = ['volume', 'volatility', 'ema_ratio', 'bb_width']

    bounded_cols = [c for c in feature_cols if any(k in c for k in bounded_keywords)]
    loggy_cols = [c for c in feature_cols if any(k in c for k in loggy_keywords)]
    the_rest = [c for c in feature_cols if c not in bounded_cols and c not in loggy_cols]

    return bounded_cols, loggy_cols, the_rest
