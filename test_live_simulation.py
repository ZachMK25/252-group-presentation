#!/usr/bin/env python3
"""
Quick test script for load_models_and_simulate.py

This demonstrates how to use the live simulation script with various thresholds
and analyzes the results to find optimal parameters.
"""

import subprocess
import pandas as pd
import glob
import os
from pathlib import Path


def run_simulation(data_file, threshold, output_dir, model_dir=None):
    """Run a single simulation"""
    cmd = [
        'python',
        'load_models_and_simulate.py',
        '--data', data_file,
        '--threshold', str(threshold),
        '--output-dir', output_dir
    ]

    if model_dir:
        cmd.extend(['--model', model_dir])

    print(f"\n{'='*60}")
    print(f"Running simulation with threshold: {threshold}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def test_single_threshold():
    """Test with a single threshold"""
    print("\n" + "="*80)
    print("TEST 1: Single Threshold Simulation")
    print("="*80)

    success = run_simulation(
        data_file='crypto.csv',
        threshold=0.50,
        output_dir='test_results/single'
    )

    if success:
        print("\n✓ Test 1 PASSED")
        metrics_files = glob.glob('test_results/single/simulation_metrics_*.csv')
        if metrics_files:
            metrics = pd.read_csv(metrics_files[0])
            print(f"\nResults:")
            print(f"  Total Return: {metrics['total_return_pct'].values[0]:+.2f}%")
            print(f"  Number of Trades: {metrics['num_trades'].values[0]:.0f}")
            print(f"  Win Rate: {metrics['win_rate'].values[0]:.1f}%")
    else:
        print("\n✗ Test 1 FAILED")

    return success


def test_threshold_optimization():
    """Test multiple thresholds and find optimal"""
    print("\n" + "="*80)
    print("TEST 2: Threshold Optimization (0.45 to 0.65)")
    print("="*80)

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]
    output_dir = 'test_results/optimization'
    os.makedirs(output_dir, exist_ok=True)

    for threshold in thresholds:
        success = run_simulation(
            data_file='crypto.csv',
            threshold=threshold,
            output_dir=output_dir
        )

        if not success:
            print(f"\n✗ Test 2 FAILED at threshold {threshold}")
            return False

    print("\n" + "="*80)
    print("Optimization Complete - Analyzing Results")
    print("="*80)

    # Load all results
    metrics_files = glob.glob(f'{output_dir}/simulation_metrics_*.csv')
    all_metrics = pd.concat([pd.read_csv(f) for f in metrics_files], ignore_index=True)
    all_metrics = all_metrics.sort_values('threshold')

    print("\nThreshold Comparison:")
    print("-" * 80)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

    display_cols = ['threshold', 'total_return_pct', 'num_trades', 'win_rate', 'avg_trade_pnl']
    print(all_metrics[display_cols].to_string(index=False))

    # Find best by return
    best_return_idx = all_metrics['total_return_pct'].idxmax()
    best_return = all_metrics.loc[best_return_idx]

    # Find best by win rate
    best_winrate_idx = all_metrics['win_rate'].idxmax()
    best_winrate = all_metrics.loc[best_winrate_idx]

    print("\n" + "-"*80)
    print(f"\nBest by Total Return:")
    print(f"  Threshold: {best_return['threshold']:.2f}")
    print(f"  Return: {best_return['total_return_pct']:+.2f}%")
    print(f"  Win Rate: {best_return['win_rate']:.1f}%")
    print(f"  Trades: {best_return['num_trades']:.0f}")

    print(f"\nBest by Win Rate:")
    print(f"  Threshold: {best_winrate['threshold']:.2f}")
    print(f"  Return: {best_winrate['total_return_pct']:+.2f}%")
    print(f"  Win Rate: {best_winrate['win_rate']:.1f}%")
    print(f"  Trades: {best_winrate['num_trades']:.0f}")

    print("\n✓ Test 2 PASSED")
    return True


def test_custom_parameters():
    """Test with custom capital and model"""
    print("\n" + "="*80)
    print("TEST 3: Custom Parameters ($50k Capital)")
    print("="*80)

    cmd = [
        'python',
        'load_models_and_simulate.py',
        '--data', 'crypto.csv',
        '--threshold', '0.52',
        '--capital', '50000',
        '--output-dir', 'test_results/custom'
    ]

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✓ Test 3 PASSED")
        metrics_files = glob.glob('test_results/custom/simulation_metrics_*.csv')
        if metrics_files:
            metrics = pd.read_csv(metrics_files[0])
            print(f"\nResults with $50,000 capital:")
            print(f"  Final Capital: ${metrics['final_capital'].values[0]:,.2f}")
            print(f"  Total Return: {metrics['total_return_pct'].values[0]:+.2f}%")
            print(f"  Total P&L: ${metrics['total_pnl'].values[0]:+,.2f}")
    else:
        print("\n✗ Test 3 FAILED")

    return result.returncode == 0


def display_sample_output():
    """Show sample trade log"""
    print("\n" + "="*80)
    print("Sample Trade Log")
    print("="*80)

    trade_files = glob.glob('test_results/single/trades_*.csv')
    if trade_files:
        trades = pd.read_csv(trade_files[0]).head(5)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.2f}'.format)

        print("\nFirst 5 trades:")
        print(trades[['symbol', 'entry_time', 'exit_time', 'return_pct', 'net_pnl']].to_string(index=False))


def main():
    print("\n" + "="*80)
    print("LIVE CRYPTO SIMULATION - TEST SUITE")
    print("="*80)

    # Create test results directory
    os.makedirs('test_results', exist_ok=True)

    tests_passed = 0
    tests_total = 3

    # Test 1: Single threshold
    if test_single_threshold():
        tests_passed += 1

    # Test 2: Threshold optimization
    if test_threshold_optimization():
        tests_passed += 1

    # Test 3: Custom parameters
    if test_custom_parameters():
        tests_passed += 1

    # Show sample output
    display_sample_output()

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests Passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("\n✓ ALL TESTS PASSED")
        print("\nResults saved to test_results/ directory")
        print("\nNext steps:")
        print("1. Analyze results in test_results/optimization/")
        print("2. Review trade logs in test_results/*/trades_*.csv")
        print("3. For live data, use: python load_models_and_simulate.py --data YOUR_DATA.csv")
    else:
        print(f"\n✗ {tests_total - tests_passed} TEST(S) FAILED")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
