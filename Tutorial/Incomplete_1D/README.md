# 🐍 PG-DPO Portfolio Optimization — Experiment Scripts

Welcome! 🎉  
This repository is a **tutorial-style** collection of Python scripts that implement and test **Pontryagin-Guided Direct Policy Optimization (PG-DPO)** and its variants — step-by-step, from the most basic to more advanced forms.  
It’s designed not just for research, but also as a **hands-on learning resource** 📚 so you can follow along, modify, and experiment.

---

## 🧪 Test Environment

All experiments here are run on the **Kim and Omberg (1996)** continuous-time portfolio model 🏦.  
This tutorial uses the simplest non-trivial setting — **one risky asset 💹** and **one exogenous state variable 📈** (the risk premium / Sharpe ratio \(X_t\), coded as `Y` in the scripts) — which makes it easier to understand while still capturing intertemporal hedging.

✨ **Exogenous state dynamics** (OU process for risk premium):  
`dX_t = -lambda_X * (X_t - X_bar) * dt + sigma_X * dZ_t^X`

💰 **Wealth dynamics**:  
`dW_t = r * W_t * dt + y_t * ( (mu_t - r) * dt + sigma_t * dZ_t )`

🎯 **Evaluation metrics**:  
- 📏 RMSE between learned policy and the **closed-form Kim–Omberg optimal policy**  
- 💡 (Optional) Expected utility difference from the analytical optimum  
  *(the default runner reports RMSE only; EU difference can be added if needed).*

---

## 📂 File Descriptions

### `pgdpo_base_single.py` — **Baseline PG-DPO**
🚀 The starting point.  
Trains a Stage-1 PG-DPO policy **without** variance reduction or other enhancements. Reports:
- Stage-1 RMSE vs closed-form policy

**Snippet (minimal):**
```python
def simulate(policy_module, B, train=True, W0=None, Y0=None, Tval=None, rng=None, seed=None):
    logW = W.clamp(min=lb_W).log()
    for _ in range(m):
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)
        risk_prem = sigma * (alpha * Y)
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2
        zW, zY = correlated_normals(W.shape[0], rho, gen=gen)
        dBW, dBY = math.sqrt(dt) * zW, math.sqrt(dt) * zY
        logW = logW + (driftW - 0.5*varW)*dt + (pi_t * sigma)*dBW
        Y    = Y + kappaY*(thetaY - Y)*dt + sigmaY*dBY
        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt
    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0)<1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U
```

---

### `pgdpo_run_single.py` — **Antithetic + Richardson**
🔄 This replaces the old `pgdpo_antithetic_single.py`.  
It applies **antithetic sampling** (variance reduction) and **Richardson extrapolation** (bias reduction) in one consistent runtime.  

**Snippet (minimal):**
```python
def simulate_run(policy, B, W0=None, Y0=None, Tval=None, seed_local=None):
    # coarse (m steps)
    ZWc, ZYc = _draw_correlated_normals(B, m, make_generator(seed_local))
    WTc_p = _forward_path(policy, W0, Tval, Y0, +ZWc, +ZYc, m)
    WTc_m = _forward_path(policy, W0, Tval, Y0, -ZWc, -ZYc, m)
    Uc = 0.5 * (_crra_utility(WTc_p, gamma) + _crra_utility(WTc_m, gamma))

    # fine (2m steps)
    ZWf, ZYf = _draw_correlated_normals(B, 2*m, make_generator((seed_local or 0)+8191))
    WTf_p = _forward_path(policy, W0, Tval, Y0, +ZWf, +ZYf, 2*m)
    WTf_m = _forward_path(policy, W0, Tval, Y0, -ZWf, -ZYf, 2*m)
    Uf = 0.5 * (_crra_utility(WTf_p, gamma) + _crra_utility(WTf_m, gamma))

    # Richardson extrapolation
    return 2.0*Uf - Uc
```

**Explanation:**  
- Antithetic: pairs noise ± for variance reduction.  
- Richardson: combines coarse/fine paths as \( 2U_f - U_c \) for bias reduction.  
- Output: extrapolated, antithetic utility per path.

---

### `pgdpo_residual_single.py` — **Residual PG-DPO**
🛠 Builds a residual network on top of a **myopic baseline policy**.  
The residual learns only **hedging demand corrections**, using the RUN simulator.

---

### `pgdpo_cv_single.py` — **Residual + Control Variate**
🎚 Adds a **control variate (CV)** based on **myopic utility**, simulated under identical CRNs.  
This further reduces variance and improves robustness.

---

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Simple Euler)**
🎯 Applies Pontryagin-based projection with costates estimated using the **simple Euler simulator**.  
Provides a straightforward baseline for comparing projection effects versus the variance-reduced RUN version.

---

## 📊 Stage-by-Stage RMSE Results

| Variant                                      | Stage 1 RMSE | Stage 2 RMSE |
|----------------------------------------------|--------------|--------------|
| Baseline PG-DPO                              | 0.134747     | –            |
| P-PGDPO (Simple Euler)                       | 0.134747     | 0.001600     |
| P-PGDPO (Antithetic+Richardson)              | 0.081309     | 0.001094     |
| Residual PG-DPO (Antithetic+Richardson)      | 0.027062     | 0.000712     |
| Residual + Control Variate (Antithetic+Richardson) | 0.004917     | 0.000668     |

---

### 📝 Interpretation
- 📉 Stage-1 RMSE drops steadily: Baseline → Residual → Residual+CV.  
- 🏆 P-PGDPO projection consistently yields RMSE < 0.002.  
- 💡 Antithetic+Richardson improves both baseline and residual training.  

---

💡 **Note:** While this repo is research-grade, it’s deliberately structured as a **learning-friendly tutorial** — you can run each script independently, compare results, and see exactly how each enhancement changes the outcome.
