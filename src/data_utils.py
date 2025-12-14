"""
Data loading and preprocessing utilities
Handles CSV loading, reshaping from wide to long format, and data cleaning
"""

import pandas as pd
import numpy as np
from src.config import FIELD_NAMES, KEEP_SYMBOLS, MAX_ROWS_PER_SYMBOL


def load_crypto_csv(csv_path='crypto.csv'):
    """
    Load cryptocurrency data from CSV file

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with MultiIndex (symbol, datetime) and OHLCV columns
    """
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

    # Parse wide format columns into symbol-specific dictionaries
    by_symbol = {}
    for col in raw.columns:
        if col == dt_col:
            continue
        if '-' in col:
            p0, p1 = col.split('-', 1)
            # Determine field and symbol
            if p0.lower() in FIELD_NAMES:
                field = p0.lower()
                symbol = p1
            elif p1.lower() in FIELD_NAMES:
                field = p1.lower()
                symbol = p0
            else:
                continue
            # Clean symbol name
            symbol = symbol.replace('USDT', '').replace('_USDT', '')
        else:
            continue

        by_symbol.setdefault(symbol, {})[field] = raw[col]

    # Build long-format DataFrame
    frames = []
    for sym, fields in by_symbol.items():
        if not FIELD_NAMES.issubset(fields.keys()):
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
        raise ValueError(f'No complete symbols found in {csv_path} (need open/high/low/close/volume).')

    df = pd.concat(frames, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing OHLC data
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Set MultiIndex and sort
    df = df.set_index(['symbol', 'datetime']).sort_index()

    # Apply optional filtering
    if KEEP_SYMBOLS:
        df = df[df.index.get_level_values('symbol').isin(KEEP_SYMBOLS)]

    if MAX_ROWS_PER_SYMBOL:
        df = df.groupby(level='symbol', group_keys=False).tail(MAX_ROWS_PER_SYMBOL)

    print(f"  Data shape: {df.shape}")
    print(f"  Symbols: {df.index.get_level_values('symbol').unique().tolist()}")

    return df


def load_single_symbol(csv_path='crypto.csv', symbol='BTC'):
    """
    Load data for a single cryptocurrency symbol

    Args:
        csv_path: Path to the CSV file
        symbol: Symbol to extract (e.g., 'BTC', 'ETH')

    Returns:
        DataFrame indexed by datetime with OHLCV columns
    """
    print(f"Loading {symbol} data from {csv_path}...")
    raw = pd.read_csv(csv_path)

    # Detect datetime column
    if 'datetime' in raw.columns:
        dt_col = 'datetime'
    elif 'OpenDt' in raw.columns:
        dt_col = 'OpenDt'
    else:
        raise ValueError('CSV must contain either "datetime" or "OpenDt" column')

    raw[dt_col] = pd.to_datetime(raw[dt_col])

    # Parse columns for the requested symbol
    by_symbol = {}
    for col in raw.columns:
        if col == dt_col:
            continue
        if '-' in col:
            p0, p1 = col.split('-', 1)
            if p0.lower() in FIELD_NAMES:
                field = p0.lower()
                sym = p1
            elif p1.lower() in FIELD_NAMES:
                field = p1.lower()
                sym = p0
            else:
                continue
            sym = sym.replace('USDT', '').replace('_USDT', '')
        else:
            continue

        by_symbol.setdefault(sym, {})[field] = raw[col]

    # Check if symbol exists
    if symbol not in by_symbol:
        available = list(by_symbol.keys())
        raise ValueError(f'{symbol} not found in {csv_path}. Available: {available}')

    # Build DataFrame for the symbol
    fields = by_symbol[symbol]
    df = pd.DataFrame({
        'open': fields['open'],
        'high': fields['high'],
        'low': fields['low'],
        'close': fields['close'],
        'volume': fields['volume'],
        'datetime': raw[dt_col]
    })

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()

    print(f"  {symbol} data shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def signed_log1p(df_in):
    """
    Apply signed log1p transformation for skewed features
    Preserves sign while applying log scaling: sign(x) * log(1 + |x|)

    Args:
        df_in: Input DataFrame

    Returns:
        Transformed DataFrame
    """
    Z = df_in.copy()
    for c in Z.columns:
        x = Z[c].values
        Z[c] = np.sign(x) * np.log1p(np.abs(x))
    return Z
