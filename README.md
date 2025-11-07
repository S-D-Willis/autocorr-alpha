# How Far Can You Get With Autocorrelation Alone?

> **Research Question:** Using only each stock’s own daily OHLCV data (no cross‑sectional or fundamentals), how far can I go relying purely on autocorrelation and single‑series structure?

---

## Why this is interesting

- **Severe constraint:** Single‑ticker, daily data only. No market/breadth/factor inputs.
- **Autocorr focus:** Leverages persistence/mean‑reversion structure present in one price/volume series.
- **Full pipeline:** Data → features → target → model → backtest → diagnostics, all in Python.

---

## Approach

1. **Data**: Fetch daily OHLCV for a single ticker (Yahoo Finance via `yfinance`).  
2. **Features**: Autocorr‑oriented and single‑series transforms derived from OHLCV (momentum windows, slope/volatility state, rolling residual structure, volume/OBV dynamics, etc.).  
3. **Baseline**: Random Forest with a “feature factory” and RFE; randomized search on hyperparams.  
4. **Model shift**: Move to **XGBoost** to handle nonlinear interactions, regularize better, and generate more trade-friendly predicted probabilities
5. **Tuning**: Bayesian optimization via **Hyperopt** with a bounded, practical search space and careful early stopping.  
6. **Stability**: **Time‑decay sample weighting** to reduce chaotic sensitivity to the exact train window length.  
7. **Trading test**: Probability‑weighted position sizing + evidence gating; compare against buy‑and‑hold.  

---

## Results (universe of 66 stocks; daily data)

| Metric | This pipeline |
|---|---:|---:|
| Ternary accuracy (vs. naive baseline) | **+7% lift** |
| Sharpe (simple strategy) | **0.4** |
| Buy‑and‑hold Sharpe (context) | **0.5** |
| Max drawdown | **22% reduction** vs baseline strategy |

**Interpretation:** The simple, single‑ticker strategy does not beat buy‑and‑hold Sharpe in this setup, but it does improve drawdown meaningfully while achieving a consistent accuracy lift under the volatility‑scaled target. Given the tight “autocorrelation‑only” constraint, the pipeline is a useful sandbox to demonstrate modeling rigor, diagnostics, and engineering choices likely to transfer to more realistic multi‑signal settings.

---

## Details of Progress

Traditional time‑series tools (ARMA/GARCH) were not particularly useful for prediction; however, examining AR residuals alongside rolling trade volume revealed connections that informed later feature engineering. Regularized regression using typical trading indicators also underperformed, so I changed approaches.

I reframed the problem as **classification** with a volatility‑adjusted ternary target.

### Target definition

Let:

- \( \sigma_t \): 1‑step EWM volatility at time \( t \)  
- \( h \): forecast horizon (e.g., 20 trading days)  
- \( k \): sensitivity (default \(0.3\), optionally tuned)  
- \( r_{t \to t+h} \): forward return over horizon \(h\)  

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

### Baseline → iteration

As a baseline, I built a feature factory and used **recursive feature elimination** to select an optimal subset per stock, training a **random forest classifier** with randomized search. See the baseline pipeline and results in **[`ADD`](ADD)**. In short, the model erred on the side of caution, often predicting “no momentum”; however, when it did call momentum, direction accuracy was strong. A simple trading layer with probability‑weighted sizing and entry gating achieved ~0 Sharpe; naive variants did worse.

To improve, I generated a **smaller set of higher‑alpha features** emphasizing multi‑scale momentum, persistence, and regime‑aware volatility, switched the model to **XGBoost**, and upgraded hyperparameter optimization to **Hyperopt**. Early experiments sometimes produced zero or very few trees (no predictive power) due to an overly restrictive search space; widening the learning‑rate (`eta`) bounds enabled non‑trivial ensembles, after which I controlled capacity with **early stopping** and a **bounded search region**.

A major issue was **sensitivity to the exact training span** (e.g., great at 40 months, poor at 45, good again at 50). I mitigated this by adding **time‑decay sample weights**, prioritizing recent observations.

**Outcome.** Across 66 stocks, the final pipeline delivered a **+7% accuracy lift**, **Sharpe ~0.40** on the simple strategy, and **22% lower max drawdown** than the baseline strategy (still below buy‑and‑hold Sharpe ~0.50).

This constrained single‑ticker approach demonstrates robust ML methodology applicable to broader multi‑asset strategies.

---
