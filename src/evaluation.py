"""
Evaluation metrics and cross-validation utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

from src.config import N_FOLDS, EMBARGO_HOURS


def compute_metrics(y_true, y_pred_proba, y_pred_binary=None):
    """
    Compute all evaluation metrics

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred_binary: Predicted binary labels (optional, will use 0.5 threshold if None)

    Returns:
        Dictionary of metrics
    """
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'brier': brier_score_loss(y_true, y_pred_proba),
        'logloss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred_binary)
    }

    return metrics


def purged_kfold_cv(X, y, n_splits=N_FOLDS, embargo_hours=EMBARGO_HOURS):
    """
    Purged K-Fold cross-validation to avoid look-ahead bias

    Splits data by unique datetime values with embargo period between train/test

    Args:
        X: Feature DataFrame with datetime index
        y: Label series with datetime index
        n_splits: Number of folds
        embargo_hours: Hours to purge between train/test splits

    Yields:
        train_idx, test_idx: Arrays of row indices for each fold
    """
    # Get unique datetime values
    if isinstance(X.index, pd.MultiIndex):
        unique_times = sorted(X.index.get_level_values('datetime').unique())
    else:
        unique_times = sorted(X.index.unique())

    n_times = len(unique_times)
    fold_size = n_times // n_splits

    for i in range(n_splits):
        # Define test period
        test_start_idx = i * fold_size
        test_end_idx = (i + 1) * fold_size if i < n_splits - 1 else n_times

        test_start_time = unique_times[test_start_idx]
        test_end_time = unique_times[test_end_idx - 1]

        # Apply embargo: train only up to (test_start - embargo)
        embargo_delta = pd.Timedelta(hours=embargo_hours)
        train_end_time = test_start_time - embargo_delta

        # Get row indices
        if isinstance(X.index, pd.MultiIndex):
            times = X.index.get_level_values('datetime')
        else:
            times = X.index

        train_idx = np.where(times < train_end_time)[0]
        test_idx = np.where((times >= test_start_time) & (times <= test_end_time))[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx


def compute_roi_per_threshold(y_true, y_pred_proba, returns, thresholds=None,
                                transaction_cost=0.001):
    """
    Compute ROI for different probability thresholds

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        returns: Actual returns for each sample
        thresholds: List of thresholds to test (default: 0.3 to 0.8)
        transaction_cost: Cost per trade (default 0.1%)

    Returns:
        DataFrame with threshold, gross PnL, net PnL, accuracy, num_trades
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.81, 0.01)

    results = []

    for thresh in thresholds:
        # Signal: 1 if prob >= threshold
        signals = (y_pred_proba >= thresh).astype(int)

        # Calculate PnL
        gross_pnl = (signals * returns).sum()

        # Count trades (position changes)
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        num_trades = position_changes.sum()

        # Net PnL after transaction costs
        net_pnl = gross_pnl - (num_trades * transaction_cost)

        # Accuracy
        acc = accuracy_score(y_true, signals) if num_trades > 0 else 0

        results.append({
            'threshold': thresh,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'accuracy': acc,
            'num_trades': int(num_trades)
        })

    return pd.DataFrame(results)


def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Compute Sharpe ratio from returns

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default 0)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    sharpe = excess_returns.mean() / returns.std()

    # Annualize (assuming daily returns)
    sharpe_annual = sharpe * np.sqrt(252)

    return sharpe_annual


def compute_max_drawdown(cumulative_returns):
    """
    Compute maximum drawdown from cumulative returns

    Args:
        cumulative_returns: Array of cumulative returns

    Returns:
        Maximum drawdown (negative value)
    """
    if len(cumulative_returns) == 0:
        return 0.0

    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-10)

    return drawdown.min()


def print_metrics_table(metrics_dict):
    """
    Print formatted table of model metrics

    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
    """
    df = pd.DataFrame(metrics_dict).T
    df = df[['auc', 'brier', 'logloss', 'accuracy']]  # Reorder columns

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
