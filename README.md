# How Far Can You Get With Autocorrelation Alone?
## Single-Ticker Momentum Detection via Machine Learning


> **Research Question:** Using only each stock’s own daily OHLCV data (no cross‑sectional, fundamental, or other data), how far can I go relying purely on autocorrelation and single‑series structure?

### Key Results
- **+7% classification accuracy lift** over baseline (40.1% PR-AUC vs 33.3% random)
- **22% reduction in maximum drawdown** compared to buy-and-hold
- **0.4 Sharpe ratio** with a simple probability-weighted equities trading strat 

### Technical Highlights
- **Volatility-scaled ternary classification** replacing traditional regression approaches
- **PCA-based feature engineering** on rolling RSI and price slope bases
- **Time-decay sample weighting** to mitigate training window sensitivity
- **Bayesian hyperparameter optimization** via Hyperopt with carefully bounded search spaces
- **Comprehensive backtesting framework** with transaction costs and realistic constraints
---

## Overview

This project tackles a deliberately constrained problem: can we extract predictive signals from a single stock's price and volume history alone? 

The main implementation is in [`Autocorr-alpha-pipeline.ipynb`](Autocorr-alpha-pipeline.ipynb), with my first-pass Random Forest approach preserved in [`RFC-autocorrelation.ipynb`](RFC-autocorrelation.ipynb) for comparison.

### 1. Target Engineering: Volatility-Adjusted Ternary Classification

Instead of predicting raw returns, I engineered a **market-regime-aware target**:

Let:
- $\sigma_t$: 1‑step EWM volatility at time $t$
- $h$: forecast horizon (e.g., 20 trading days)
- $k$: sensitivity (default $0.3$, optionally tuned)
- $r_{t \to t+h}$: forward return over horizon $h$

Define the volatility threshold:

$$
\tau_t = k \cdot \sigma_t \cdot \sqrt{h}
$$

and the ternary target:

$$
y_t =
\begin{cases}
+1 & \text{if } r_{t \to t+h} > +\tau_t \\
-1 & \text{if } r_{t \to t+h} < -\tau_t \\
0  & \text{otherwise.}
\end{cases}
$$

This makes “significant moves” relative to prevailing volatility, not in fixed points/percent.

### 2. Feature Engineering: Autocorrelation-Focused Transformations

Developed high-alpha features emphasizing single-series structure:

#### Core Features
- **Volatility metrics**: EWM volatility, Garman-Klass estimator
- **Price positioning**: Donchian channel position
- **Volume dynamics**: Volume/volatility ratio
- **Trend indicators**: Distance from 100-day SMA, MACD histogram

#### Advanced Composites 
- **PCA on RSI spectrum**: Regressed principal components of a collection of RSI(n) values against future returns 
- **PCA on price slopes**: Multi-scale momentum via PCA on slopes of various lookbacks 
- **Torque indicator**: Bounce off of support level based on RSI-price-slope interactions
- **Hurst × RSI interaction**: Combining fractal dimension with mean reversion
- **OBV slope**: Directional volume accumulation via linear regression
- **SMA ratio**: Fast/slow moving average convergence

### 3. Model Architecture & Evolution: From Random Forest to XGBoost

#### Initial Approach (RFC)
- Random Forest with recursive feature elimination
- Randomized search over hyperparameters
- **Result:** Conservative predictions, ~0 Sharpe

#### Final Approach (XGBoost)

- **Bayesian optimization** via Hyperopt over refined search space
- **Time-decay sample weighting** using exponential combined with class-based weighting
- **Early stopping** separately callibrated for train and refit

### 4. Robust Backtesting Framework
- **Probability-weighted position sizing**: Position size ∝ |P(up) - P(down)|
- **Dual-threshold entry/exit logic**: Enter at 15% confidence, exit below 5%
- **Transaction costs**: 5 bps per trade
- **Proper train/validation/test splits**: No lookahead bias
---

## Main Difficulties & Subsequent Innovations

### Pathological Hyperparameter Search Space 
**Problem:** With too large of a search space the hyperparameter optimization arrived in 'dead-zones' where the model performance was seriously degraded (collapse outside of training data, trivial models, etc.).

**Solution:** This was fixed by constructing a carefully bounded search space to ensure meaningful models:
```python
space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),  # Wide enough for convergence
    'max_depth': hp.quniform('max_depth', 2, 5, 1),           # Shallow trees for stability
    'min_child_weight': hp.uniform('min_child_weight', 1, 15), # Regularization
    'subsample': hp.uniform('subsample', 0.4, 0.9),           # Stochasticity
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),                        # Minimum loss reduction
    'reg_alpha': hp.loguniform('reg_alpha', -7, 1),           # L1 regularization
    'reg_lambda': hp.loguniform('reg_lambda', -3, 2)          # L2 regularization
}
```

### Extreme Sensitivity to Training Window
**Problem:** Model performance varied chaotically with training window length (e.g., excellent at 40 months, poor at 45).

**Solution:** Time-decay weighting prioritizes recent observations while maintaining sufficient history:
```python
days_from_end = (train_dates[-1] - train_dates).days
time_weights = np.exp(-days_from_end / 250)  # 250 trading days decay
sample_weights = balanced_class_weights * time_weights
```

### Overfitting to Training Data
**Problem:** Overfitting.

**Solution:** Allow for many trees during hyperparamater optimization to find the best configuration, and then severely restrict `early_stopping_rounds` during final refit.

---

## Results (universe of 66 stocks)

| Metric | XGBoost Model |
|---|---:|
| Ternary accuracy | **+7% lift** over naive|
| Sharpe | **0.4**|
| Max drawdown | reduced by **22%** compared to buy-and-hold|

---
