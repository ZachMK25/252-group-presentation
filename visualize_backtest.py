"""
Visualization Script for Trade Simulation
Generates equity curve and performance charts from backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Note: seaborn not installed, using matplotlib defaults")

# Check if trade_log.csv and threshold_optimization.csv exist
try:
    trade_log = pd.read_csv('trade_log.csv')
    threshold_opt = pd.read_csv('threshold_optimization.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run trade_simulator.py first to generate results.")
    exit(1)

print("Generating visualizations...")

fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: Threshold Optimization - Return vs Sharpe Ratio
# ============================================================================
ax1 = plt.subplot(3, 2, 1)
ax1_twin = ax1.twinx()

line1 = ax1.plot(threshold_opt['threshold'], threshold_opt['total_return'] * 100,
                  'b-o', linewidth=2, markersize=6, label='Total Return %')
line2 = ax1_twin.plot(threshold_opt['threshold'], threshold_opt['sharpe_ratio'],
                       'r-s', linewidth=2, markersize=6, label='Sharpe Ratio')

ax1.set_xlabel('Probability Threshold', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total Return (%)', color='b', fontsize=11, fontweight='bold')
ax1_twin.set_ylabel('Sharpe Ratio', color='r', fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1.set_title('Threshold Optimization: Return vs Sharpe Ratio', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=9)

# ============================================================================
# Plot 2: Number of Trades vs Win Rate
# ============================================================================
ax2 = plt.subplot(3, 2, 2)
ax2_twin = ax2.twinx()

line1 = ax2.bar(threshold_opt['threshold'], threshold_opt['num_trades'],
                 alpha=0.7, color='steelblue', label='Number of Trades')
ax2_twin.plot(threshold_opt['threshold'], threshold_opt['win_rate'] * 100,
              'g-o', linewidth=2, markersize=6, label='Win Rate %')

ax2.set_xlabel('Probability Threshold', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Trades', color='steelblue', fontsize=11, fontweight='bold')
ax2_twin.set_ylabel('Win Rate (%)', color='g', fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax2_twin.tick_params(axis='y', labelcolor='g')
ax2.set_title('Threshold Optimization: Trades and Win Rate', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 3: Max Drawdown by Threshold
# ============================================================================
ax3 = plt.subplot(3, 2, 3)
colors = ['red' if dd < -0.10 else 'orange' if dd < -0.05 else 'green'
          for dd in threshold_opt['max_drawdown']]
ax3.bar(threshold_opt['threshold'], threshold_opt['max_drawdown'] * 100,
        color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_xlabel('Probability Threshold', fontsize=11, fontweight='bold')
ax3.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
ax3.set_title('Max Drawdown by Threshold', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: Trade Return Distribution
# ============================================================================
ax4 = plt.subplot(3, 2, 4)
trade_returns = trade_log['return'].values * 100
winning = trade_returns[trade_returns > 0]
losing = trade_returns[trade_returns <= 0]

ax4.hist(winning, bins=20, alpha=0.7, label=f'Wins (n={len(winning)})', color='green')
ax4.hist(losing, bins=20, alpha=0.7, label=f'Losses (n={len(losing)})', color='red')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax4.set_xlabel('Trade Return (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 5: Cumulative PnL Over Time
# ============================================================================
ax5 = plt.subplot(3, 2, 5)
trade_log_copy = trade_log.copy()
trade_log_copy['exit_time'] = pd.to_datetime(trade_log_copy['exit_time'])
trade_log_copy = trade_log_copy.sort_values('exit_time')
trade_log_copy['cumulative_pnl'] = trade_log_copy['pnl'].cumsum()

ax5.plot(trade_log_copy['exit_time'], trade_log_copy['cumulative_pnl'],
         linewidth=2, color='darkblue', marker='o', markersize=4)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax5.fill_between(trade_log_copy['exit_time'], trade_log_copy['cumulative_pnl'], 0,
                  where=(trade_log_copy['cumulative_pnl'] >= 0), alpha=0.3, color='green', label='Profit')
ax5.fill_between(trade_log_copy['exit_time'], trade_log_copy['cumulative_pnl'], 0,
                  where=(trade_log_copy['cumulative_pnl'] < 0), alpha=0.3, color='red', label='Loss')
ax5.set_xlabel('Exit Date', fontsize=11, fontweight='bold')
ax5.set_ylabel('Cumulative PnL ($)', fontsize=11, fontweight='bold')
ax5.set_title('Cumulative PnL Over Time', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)
ax5.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# ============================================================================
# Plot 6: Statistics Summary
# ============================================================================
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

best_idx = threshold_opt['sharpe_ratio'].idxmax()
best_result = threshold_opt.iloc[best_idx]

stats_text = f"""
BEST STRATEGY SUMMARY

Optimal Threshold: {best_result['threshold']:.2f}
Total Return: {best_result['total_return']*100:+.2f}%
Number of Trades: {int(best_result['num_trades'])}
Win Rate: {best_result['win_rate']*100:.1f}%
Avg Trade Return: {best_result['avg_trade_return']*100:+.2f}%
Max Drawdown: {best_result['max_drawdown']*100:.2f}%
Sharpe Ratio: {best_result['sharpe_ratio']:.3f}

TRADE STATISTICS

Total Winning Trades: {len(winning)}
Total Losing Trades: {len(losing)}
Best Trade: +{trade_returns.max():.2f}%
Worst Trade: {trade_returns.min():.2f}%
Avg Winning Trade: +{winning.mean():.2f}%
Avg Losing Trade: {losing.mean():.2f}%
Profit Factor: {abs(winning.sum() / losing.sum()):.2f}x
"""

ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('backtest_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved to: backtest_visualization.png")

plt.show()
