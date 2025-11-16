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

```python
τ_t = k · σ_t · √h  # Adaptive threshold based on current volatility

y_t = { +1 if r_{t→t+h} > +τ_t    # Significant up move
        -1 if r_{t→t+h} < -τ_t    # Significant down move
         0 otherwise               # No significant momentum
```

The threshold scales with prevailing volatility, making "significant moves" relative to market conditions rather than fixed percentages.

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
**Problem:** Model 






## Approach

![pipeline](https://github.com/S-D-Willis/autocorr-alpha/blob/5c073da1d6217457f8fbc9421b756a63cb095cbb/pipeline.png)

1. **Data**: Fetch daily OHLCV for a single ticker (Yahoo Finance via `yfinance`).  
2. **Features**: Autocorr‑oriented and single‑series transforms derived from OHLCV (momentum windows, slope/volatility state, rolling residual structure, volume/OBV dynamics, etc.).  
3. **First iteration**: Random Forest with a “feature factory” and RFE; randomized search on hyperparams.  
4. **Model shift**: Move to **XGBoost** to handle nonlinear interactions, regularize better, and generate more trade-friendly predicted probabilities
5. **Tuning**: Bayesian optimization via **Hyperopt** with a bounded, practical search space and careful early stopping.  
6. **Stability**: **Time‑decay sample weighting** to reduce chaotic sensitivity to the exact train window length.  
7. **Trading test**: Probability‑weighted position sizing + evidence gating; compare against buy‑and‑hold.  

---

## Results (universe of 66 stocks; daily data)

| Metric | XGBoost Model |
|---|---:|
| Ternary accuracy | **+7% lift** over naive|
| Sharpe | **0.4**|
| Max drawdown | **-22%** compared to buy-and-hold|

**Interpretation:** The simple, single‑ticker strategy does not beat buy‑and‑hold Sharpe in this setup, which is nearly 0.2 points higher, but it does improve drawdown meaningfully while achieving a consistent accuracy lift under the volatility‑scaled target. Given the tight “autocorrelation‑only” constraint, the pipeline is a useful sandbox to demonstrate modeling rigor, diagnostics, and engineering choices likely to transfer to more realistic multi‑signal settings.

---

## Details of Progress

Traditional time‑series tools (ARMA/GARCH) were not particularly useful for prediction; however, examining AR residuals alongside rolling trade volume revealed connections that informed later feature engineering. Regularized regression using typical trading indicators also underperformed, so I changed approaches.

I reframed the problem as **classification** with a volatility‑adjusted ternary target.

### Target definition

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

### Iteration

As a first pass, I built a feature factory and used **recursive feature elimination** to select an optimal subset per stock, training a **random forest classifier** with randomized search. See the RFC pipeline and results **[`here`](RFC-autocorrelation.ipynb)**. In short, the model erred on the side of caution, often predicting “no momentum”; however, when it did call momentum, direction accuracy was strong. A simple trading layer with probability‑weighted sizing and entry gating achieved ~0 Sharpe; naive variants did worse.

To improve, I generated a **smaller set of higher‑alpha features** emphasizing multi‑scale momentum, persistence, and regime‑aware volatility, switched the model to **XGBoost**, and upgraded hyperparameter optimization to **Hyperopt**. Early experiments sometimes produced zero or very few trees (no predictive power) due to an overly expansive search space; constraining bounds for key parameters, such as the learning‑rate (`eta`), enabled non‑trivial ensembles, after which I controlled capacity with **early stopping** and a **bounded search region**.

A major issue was **sensitivity to the exact training span** (e.g., great at 40 months, poor at 45, good again at 50). I mitigated this by adding **time‑decay sample weights**, prioritizing recent observations.

### Outcome

Across 66 stocks, the final pipeline delivered a **+7% accuracy lift**, **Sharpe ~0.4** on the simple strategy, and **22% lower max drawdown** than buy-and-hold.

This constrained single‑ticker approach demonstrates robust ML methodology applicable to broader multi‑asset strategies.

---
