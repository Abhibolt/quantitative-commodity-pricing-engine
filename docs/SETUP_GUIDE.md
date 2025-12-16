# 🚀 SETUP GUIDE: Running the Commodity Pricing Engine in Jupyter Notebook

## Complete Installation & Setup Instructions

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installing Required Packages](#installing-required-packages)
3. [Setting Up Jupyter Notebook](#setting-up-jupyter-notebook)
4. [Loading the Code](#loading-the-code)
5. [Running the Analysis](#running-the-analysis)
6. [Troubleshooting](#troubleshooting)

---

## 1️⃣ Prerequisites

### What You Need:
- ✅ **Anaconda Distribution** installed (includes Jupyter Notebook, pandas, numpy, matplotlib)
- ✅ **Internet connection** (for downloading data, or use simulated mode)
- ✅ **10-15 minutes** to complete setup

### Check Your Anaconda Installation:
1. Open **Anaconda Navigator** (should be in your Applications/Programs)
2. If it opens successfully, you're good to go!
3. If not installed, download from: https://www.anaconda.com/download

---

## 2️⃣ Installing Required Packages

Anaconda comes with most packages pre-installed, but we need to add **yfinance** for downloading commodity data.

### Method 1: Using Anaconda Prompt (Recommended)

**Windows:**
1. Click Start menu
2. Type "Anaconda Prompt"
3. Click "Anaconda Prompt (Anaconda3)"

**Mac:**
1. Open Terminal
2. It should automatically have Anaconda loaded

**In the Anaconda Prompt/Terminal:**
```bash
pip install yfinance
```

Wait for it to finish (should take 10-20 seconds), then you should see:
```
Successfully installed yfinance-X.X.XX
```

### Method 2: Using Jupyter Notebook (Alternative)

If you prefer to install from within Jupyter:
1. Open Jupyter Notebook (instructions below)
2. Create a new cell
3. Type: `!pip install yfinance`
4. Press **Shift + Enter** to run
5. Wait for installation to complete

---

## 3️⃣ Setting Up Jupyter Notebook

### Step 1: Launch Jupyter Notebook

**Option A: Via Anaconda Navigator (Easiest)**
1. Open **Anaconda Navigator**
2. Find the **Jupyter Notebook** tile
3. Click **Launch**
4. Your browser will open automatically

**Option B: Via Command Line**
1. Open Anaconda Prompt (Windows) or Terminal (Mac)
2. Type: `jupyter notebook`
3. Press Enter
4. Your browser will open automatically

### Step 2: Navigate to Your Working Folder
1. In the Jupyter browser window, navigate to where you want to save your project
   - Example: Documents → Projects → CommodityPricing
2. If you need to create a new folder:
   - Click **New** (top right) → **Folder**
   - Check the new folder and click **Rename** to name it

### Step 3: Upload the Notebook
1. Download the `Commodity_Pricing_Engine.ipynb` file
2. In Jupyter, click **Upload** button
3. Select the downloaded .ipynb file
4. Click **Upload** to confirm

### Step 4: Open and Run
1. Click on the uploaded notebook
2. Click **Cell** → **Run All**
3. Wait ~10 seconds for results!

---

## 4️⃣ Loading the Code

### Option 1: Use the Provided Notebook (Easiest)

Simply upload the `Commodity_Pricing_Engine.ipynb` file and run it!

### Option 2: Manual Cell-by-Cell Setup

If you want to build it yourself, create these 6 cells:

#### **CELL 1: Imports and Setup**

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100

print("✅ All libraries imported successfully!")
```

---

## 5️⃣ Running the Analysis

### Complete Workflow:

1. Open the notebook
2. Click **Cell** → **Run All**
3. Wait ~5-10 seconds
4. See results!

### What You'll See:

**Console Output:**
```
📊 MODULE 1: PARAMETER ESTIMATION RESULTS
  Current Price (S₀):         $76.79
  Mean Reversion Level (μ):   $75.62
  Annualized Volatility (σ):  40.83%

💰 MODULE 3: ASIAN CALL OPTION PRICING
  OPTION PRICE:  $0.0494 per barrel
  Delta (Δ):     0.7580
```

**Charts:**
- Price paths showing mean reversion
- Option payoff distribution

---

## 6️⃣ Troubleshooting

### Problem 1: "ModuleNotFoundError: No module named 'yfinance'"

**Solution:**
```python
!pip install yfinance
```
Then restart kernel and run again.

---

### Problem 2: "No data downloaded" or Network Error

**Solution:** Use simulated data

In Cell 6, change:
```python
params = commodity.analyze(use_simulation=False)
```
to:
```python
params = commodity.analyze(use_simulation=True)
```

---

### Problem 3: Charts Not Displaying

**Solution:** Add to Cell 1:
```python
%matplotlib inline
```

---

### Problem 4: Code is Slow

**Solution:** Reduce simulations

In Cell 6, change:
```python
n_simulations = 10000
```
to:
```python
n_simulations = 5000
```

---

## ✅ Success Checklist

After running, you should have:

- [ ] All cells executed without errors
- [ ] Option price displayed (~$0.04-0.05)
- [ ] Delta calculated (~0.75-0.80)
- [ ] Two charts visible
- [ ] No error messages

---

## 🎯 Next Steps

1. **Experiment with parameters**
2. **Try other commodities** (NG=F, GC=F, SI=F)
3. **Calculate more Greeks**
4. **Upload to GitHub**

---

## 💬 Need Help?

Common fixes:
1. Restart kernel (Kernel → Restart & Clear Output)
2. Run cells in order (1→2→3→4→5→6)
3. Use simulated data if network issues
4. Check all parameters are positive

---

**Happy pricing! 🚀📊**

---

**Version:** 1.0  
**Last Updated:** December 2025
