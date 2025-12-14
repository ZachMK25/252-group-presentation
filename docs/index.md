# Documentation Index

**Cryptocurrency Price Prediction - Technical Documentation**

**Project**: Ensemble Machine Learning for Cryptocurrency Price Prediction
**Course**: ECON 252 - Financial Markets
**Team**: Zach Kuo, Nikhil Ghind, Ethan Ho
**Date**: December 2024

---

## 📚 Documentation Structure

### Getting Started
- [**Quick Start Guide**](quick-start.md) - Get up and running in 5 minutes
- [**Installation**](installation.md) - Detailed setup instructions
- [**Usage Guide**](usage.md) - How to run all scripts

### Technical Documentation
- [**Implementation Details**](implementation.md) - Data pipeline, feature engineering, model training
- [**Architecture**](architecture.md) - Code structure and module organization
- [**Algorithms**](algorithms.md) - Detailed algorithm explanations and pseudocode

### Results and Analysis
- [**Results**](results.md) - Performance metrics, ROI analysis, findings
- [**Feature Importance**](feature-importance.md) - Most important technical indicators

### Reference
- [**API Reference**](api-reference.md) - Function and module documentation
- [**Configuration**](configuration.md) - Hyperparameter guide
- [**Troubleshooting**](troubleshooting.md) - Common issues and solutions

### Advanced Topics
- [**Advanced Usage**](advanced-usage.md) - Custom features, hyperparameter tuning
- [**Best Practices**](best-practices.md) - Recommendations and tips

---

## 🎯 Documentation by Role

### For Users (Want to Run the System)
1. [Quick Start Guide](quick-start.md)
2. [Installation](installation.md)
3. [Usage Guide](usage.md)
4. [Troubleshooting](troubleshooting.md)

### For Developers (Want to Understand/Modify Code)
1. [Architecture](architecture.md)
2. [Implementation Details](implementation.md)
3. [API Reference](api-reference.md)
4. [Advanced Usage](advanced-usage.md)

### For Researchers (Want to Understand Methodology)
1. [Algorithms](algorithms.md)
2. [Implementation Details](implementation.md)
3. [Results](results.md)
4. [Feature Importance](feature-importance.md)

---

## 📊 Quick Links

- [Main README](../README.md)
- [Full Technical Documentation](../DOCUMENTATION.md)
- [Source Code](../src/)
- [Requirements](../requirements.txt)

---

## 🔍 Search by Topic

| Topic | Document |
|-------|----------|
| Installation | [installation.md](installation.md) |
| Running scripts | [usage.md](usage.md) |
| Data pipeline | [implementation.md#data-pipeline](implementation.md) |
| Feature engineering | [implementation.md#feature-engineering](implementation.md) |
| Model training | [implementation.md#model-training](implementation.md) |
| Cross-validation | [algorithms.md#purged-k-fold](algorithms.md) |
| Ensemble methods | [algorithms.md#ensemble-methods](algorithms.md) |
| Performance metrics | [results.md#model-performance](results.md) |
| ROI calculation | [algorithms.md#roi-calculation](algorithms.md) |
| Code structure | [architecture.md](architecture.md) |
| Function reference | [api-reference.md](api-reference.md) |
| Hyperparameters | [configuration.md](configuration.md) |
| GPU setup | [installation.md#gpu-setup](installation.md) |
| Common errors | [troubleshooting.md](troubleshooting.md) |
| Custom features | [advanced-usage.md#custom-features](advanced-usage.md) |

---

## 📈 Key Metrics Summary

| Metric | Value |
|--------|-------|
| **Best Model** | Ensemble_Brier |
| **AUC** | 0.5421 |
| **Accuracy** | 53.06% |
| **Brier Score** | 0.2488 |
| **Symbols Traded** | 49 cryptocurrencies |
| **Features** | 31+ technical indicators |
| **Models** | 6 base + 3 ensembles |
| **Training Time** | ~8-10 minutes |

---

## 🚀 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train models on all symbols
python run_ensemble.py

# Simulate BTC trading
python trade_simulator.py

# Visualize results
python visualize_backtest.py

# Load saved models
python load_models_and_simulate.py --data new_data.csv
```

---

**Last Updated**: December 2024
**Version**: 1.0
