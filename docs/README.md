# Documentation

Comprehensive documentation for the Cryptocurrency Price Prediction system.

---

## 📚 Documentation Files

### Getting Started
| File | Description | Lines |
|------|-------------|-------|
| [**index.md**](index.md) | Documentation index and navigation | 129 |
| [**quick-start.md**](quick-start.md) | Get started in 5 minutes | 179 |

### Technical Documentation
| File | Description | Lines |
|------|-------------|-------|
| [**implementation.md**](implementation.md) | Complete implementation details | 677 |
| [**architecture.md**](architecture.md) | Code structure and organization | 809 |
| [**algorithms.md**](algorithms.md) | Detailed algorithms and pseudocode | 560 |

### Results and Analysis
| File | Description | Lines |
|------|-------------|-------|
| [**results.md**](results.md) | Performance metrics and ROI analysis | 498 |

**Total:** ~2,850 lines of documentation

---

## 🎯 Quick Navigation

### I want to...

**...run the project quickly**
→ Start with [quick-start.md](quick-start.md)

**...understand how it works**
→ Read [implementation.md](implementation.md) → [algorithms.md](algorithms.md)

**...see the results**
→ Check [results.md](results.md)

**...modify the code**
→ Study [architecture.md](architecture.md)

**...find a specific function**
→ Use [index.md](index.md) search by topic

---

## 📖 Reading Order

### For Users
1. [Quick Start](quick-start.md) - Get it running
2. [Results](results.md) - See what it can do
3. [Index](index.md) - Find specific topics

### For Developers
1. [Architecture](architecture.md) - Understand code structure
2. [Implementation](implementation.md) - Learn technical details
3. [Algorithms](algorithms.md) - Deep dive into math
4. [Index](index.md) - Function reference

### For Researchers
1. [Algorithms](algorithms.md) - Mathematical foundation
2. [Implementation](implementation.md) - Technical methodology
3. [Results](results.md) - Performance analysis
4. [Index](index.md) - Citations and references

---

## 📊 What's Covered

### Implementation Details
- ✅ Data processing pipeline (wide → long format)
- ✅ 31+ technical indicators (formulas included)
- ✅ Feature preprocessing (3-category system)
- ✅ 6 base models + 3 ensemble methods
- ✅ Probability calibration
- ✅ Purged K-Fold cross-validation

### Algorithms
- ✅ Purged K-Fold (pseudocode + visualization)
- ✅ Ensemble weight calculation
- ✅ ROI calculation (step-by-step examples)
- ✅ Sharpe ratio computation
- ✅ Maximum drawdown algorithm

### Results
- ✅ Model performance comparison (10 models)
- ✅ ROI analysis across 49 symbols
- ✅ BTC trade simulation results
- ✅ Feature importance rankings
- ✅ Computational performance breakdown
- ✅ Key findings and insights
- ⚠️ Caveats and limitations

### Code Structure
- ✅ Project directory tree
- ✅ Module dependency graph
- ✅ Data flow diagrams
- ✅ Function reference for all modules
- ✅ Design patterns used
- ✅ Extensibility guide

---

## 🔍 Search by Topic

### Data & Features
- **Data loading:** [implementation.md#data-processing](implementation.md#data-processing-pipeline)
- **Feature engineering:** [implementation.md#feature-engineering](implementation.md#feature-engineering)
- **RSI:** [implementation.md#rsi](implementation.md#3-oscillators)
- **MACD:** [implementation.md#macd](implementation.md#2-trend-indicators)
- **Bollinger Bands:** [implementation.md#bollinger-bands](implementation.md#4-volatility-indicators)

### Models & Training
- **Model definitions:** [architecture.md#model-utils](architecture.md#srcmodel_utilspy)
- **Base models:** [implementation.md#base-models](implementation.md#base-models-6-total)
- **Ensemble methods:** [implementation.md#ensemble-methods](implementation.md#ensemble-methods)
- **Calibration:** [implementation.md#probability-calibration](implementation.md#probability-calibration)
- **Cross-validation:** [algorithms.md#purged-k-fold](algorithms.md#purged-k-fold-cross-validation)

### Results & Analysis
- **Model performance:** [results.md#model-performance](results.md#model-performance-summary)
- **ROI by symbol:** [results.md#roi-analysis](results.md#roi-analysis-by-symbol)
- **BTC simulation:** [results.md#btc-simulation](results.md#btc-trade-simulation)
- **Feature importance:** [results.md#feature-importance](results.md#feature-importance)
- **Caveats:** [results.md#caveats](results.md#caveats-and-limitations)

### Code Structure
- **Module architecture:** [architecture.md#module-architecture](architecture.md#module-architecture)
- **Data flow:** [architecture.md#data-flow](architecture.md#data-flow)
- **Function reference:** [architecture.md#module-reference](architecture.md#module-reference)
- **Design patterns:** [architecture.md#design-patterns](architecture.md#design-patterns)

---

## 📈 Key Metrics (Quick Reference)

### Model Performance
| Metric | Best Model | Value |
|--------|-----------|-------|
| **AUC** | Ensemble_Brier | 0.5421 |
| **Accuracy** | Ensemble_StackedLogReg | 53.08% |
| **Brier Score** | Ensemble_StackedLogReg | 0.2486 |
| **Training Time** | GaussianNB | 2 sec |

### ROI Performance
| Symbol | Net ROI | Win Rate | Trades |
|--------|---------|----------|--------|
| **SOL** | +148.97% | 64.3% | 112 |
| **TRB** | +142.68% | 68.5% | 89 |
| **YFI** | +63.71% | 71.6% | 67 |
| **BTC** | +49.09% | 59.0% | 156 |

### Feature Importance
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | rsi_14 | 8.45% |
| 2 | volatility | 7.82% |
| 3 | ema_diff_84_168 | 7.34% |
| 4 | roc_42 | 6.89% |
| 5 | macd_hist | 6.45% |

---

## 🚀 Quick Commands

```bash
# Install
pip install -r requirements.txt

# Train models
python run_ensemble.py

# Simulate BTC trading
python trade_simulator.py

# Visualize results
python visualize_backtest.py

# Load saved models
python load_models_and_simulate.py --data new_data.csv
```

---

## 📝 Document Conventions

### Code Blocks
```python
# Python code examples with syntax highlighting
def example_function():
    return "Hello, World!"
```

### Formulas
Mathematical formulas in plain text:
```
Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev
ROI = (Final Value - Initial Value) / Initial Value
```

### Diagrams
ASCII art for visualizations:
```
Data Pipeline:
Input → Processing → Features → Model → Predictions → Results
```

### Tables
Formatted tables for data:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |

---

## 🔄 Documentation Versions

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial comprehensive documentation |

---

## 📞 Support

- **Issues:** Check [troubleshooting.md](troubleshooting.md) (when created)
- **Questions:** Refer to [index.md](index.md) search
- **Contact:** Team members (Zach Kuo, Nikhil Ghind, Ethan Ho)

---

## 📚 Related Files

- [Main README](../README.md) - Project overview
- [Full Documentation](../DOCUMENTATION.md) - Single-file version
- [Source Code](../src/) - Modular implementation
- [Requirements](../requirements.txt) - Dependencies

---

**Last Updated:** December 2024
**Total Documentation:** ~5,000+ lines (including DOCUMENTATION.md)
