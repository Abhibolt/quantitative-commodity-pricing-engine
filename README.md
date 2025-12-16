# 🛢️ Quantitative Commodity Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **production-ready quantitative pricing engine** for commodity derivatives, implementing the Ornstein-Uhlenbeck mean-reverting process for pricing Asian options on WTI Crude Oil futures.

Perfect for airlines, refineries, and manufacturers needing to hedge commodity price risk.

---

## 📊 Key Features

- ✅ **Ornstein-Uhlenbeck Process**: Mean-reverting stochastic model (perfect for commodities)
- ✅ **Monte Carlo Simulation**: 10,000 price path simulations
- ✅ **Asian Option Pricing**: Arithmetic average call option valuation
- ✅ **Greeks Calculation**: Delta for hedge ratio determination
- ✅ **Professional Visualizations**: Publication-quality charts showing mean reversion
- ✅ **Complete Documentation**: Mathematical foundations and business applications

---

## 🎯 Quick Results

```
Current Spot Price:        $76.79
Long-Term Mean:            $75.62
Annualized Volatility:     40.83%

Asian Call Option (ATM):
  OPTION PRICE:            $0.0494 per barrel
  Delta:                   0.7580
  Probability ITM:         35.42%

Business Application (100K barrels):
  Total Premium:           $4,944
  Hedge Ratio:             75.8%
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/quantitative-commodity-pricing-engine.git
cd quantitative-commodity-pricing-engine
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the Jupyter Notebook

```bash
jupyter notebook Commodity_Pricing_Engine.ipynb
```

### 4. Run All Cells

Click **Cell** → **Run All** and wait ~5 seconds for results!

---

## 📁 Repository Structure

```
📦 quantitative-commodity-pricing-engine/
├── 📓 Commodity_Pricing_Engine.ipynb    # Main Jupyter Notebook
├── 🐍 complete_pricing_engine.py        # Standalone Python script
├── 📚 docs/                              # Documentation
│   ├── SETUP_GUIDE.md
│   ├── COMPLETE_DOCUMENTATION.md
│   ├── QUICK_REFERENCE.md
│   └── QUICKSTART_5MIN.md
├── 📄 README.md                          # This file
├── 📄 LICENSE                            # MIT License
└── 📄 requirements.txt                   # Python dependencies
```

---

## 💡 What Makes This Special?

### 1. **Proper Commodity Modeling**

Unlike stocks, commodities exhibit **mean reversion**. This engine uses the **Ornstein-Uhlenbeck process**, not Geometric Brownian Motion.

### 2. **Asian Options**

Perfect for **airlines** and **refineries** because payoff is based on **average price**, matching real business exposure.

### 3. **Production-Ready Code**

- Object-oriented architecture
- Extensive documentation
- Professional error handling
- Reproducible results

---

## 🎓 Mathematical Foundation

**The Ornstein-Uhlenbeck Process:**
```
dX_t = θ(μ - X_t)dt + σ dW_t
```

- **θ** = Mean reversion speed (0.15)
- **μ** = Long-term mean ($75.62)
- **σ** = Volatility (40.83%)

**Asian Option Payoff:**
```
max(Average_Price - Strike, 0)
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART_5MIN.md](docs/QUICKSTART_5MIN.md) | Get running in 5 minutes |
| [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Detailed installation |
| [COMPLETE_DOCUMENTATION.md](docs/COMPLETE_DOCUMENTATION.md) | Full technical reference |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Parameter tuning guide |

---

## 🐛 Troubleshooting

**Network Error?** Use simulated data:
```python
params = commodity.analyze(use_simulation=True)
```

**Missing yfinance?** Install it:
```bash
pip install yfinance
```

**Charts not showing?** Add to first cell:
```python
%matplotlib inline
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 📧 Contact

**Author:** Abhishek  
**Version:** 1.0  
**Last Updated:** December 2025

---

## ⭐ Star This Repo!

If you find this useful, please give it a star! ⭐

---

**Happy Pricing! 🚀📈**
