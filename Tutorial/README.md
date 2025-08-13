# 🐍 PG-DPO Portfolio Optimization — Experiment Scripts

Welcome! 🎉  
This repository is a **tutorial-style** collection of Python scripts that implement and test **Pontryagin-Guided Direct Policy Optimization (PG-DPO)** and its variants — step-by-step, from the most basic to more advanced forms.  
It’s designed not just for research, but also as a **hands-on learning resource** 📚 so you can follow along, modify, and experiment.

---

## 🧪 Test Environment

All experiments are run on the **Kim and Omberg (1996)** continuous-time portfolio model 🏦. For tutorial clarity we use the simplest non-trivial setting — **one risky asset** and **one exogenous state variable** (the risk premium / Sharpe ratio X_t). This keeps the model easy to follow while still capturing intertemporal hedging.

**Risky asset and state dynamics.**  
Let the risky asset price S_t follow:  
    dS_t / S_t = mu_t * dt + sigma_t * dZ_t  
The **risk premium** is:  
    X_t = (mu_t - r) / sigma_t  
Assume X_t follows an Ornstein–Uhlenbeck (OU) process:  
    dX_t = -lambda_X * (X_t - X_bar) * dt + sigma_X * dZ_t^X  
with correlation:  
    E[dZ_t * dZ_t^X] = rho_mX * dt  
(See KO eqs. (1)–(4), pp. 143–145.)

It is convenient to define the **normalized return**:  
    dR_t = (dS_t / S_t) - r * dt = X_t * dt + dZ_t  
So all opportunity-set information is summarized by the single state X_t, together with constants (r, sigma_t).

**Wealth dynamics.**  
With monetary position y_t in the risky asset (so the risk-free holding is W_t - y_t):  
    dW_t = r * W_t * dt + y_t * ( (mu_t - r) * dt + sigma_t * dZ_t )  
Equivalently:  
    dW_t = r * W_t * dt + theta_t * dR_t, where theta_t = y_t / sigma_t  
(KO eqs. (6) and (8), pp. 144–145.)

**Preferences.**  
Terminal-wealth HARA utility; in our code we use CRRA (power) utility, a special case. (KO §1 and Fig. 1, pp. 145–147.)

**Closed-form optimal policy.**  
Kim–Omberg show the optimal investment is **linear in the state X_t**. In normalized units theta_t = y_t / sigma_t:  
    theta*(W, X, T) = T_J(W, T) * [ X + rho_mX * sigma_X * ( C(T) * X + B(T) ) ]  
Here T_J(W, T) = -J_W / J_WW is absolute risk tolerance, and B(T), C(T) solve a Riccati ODE system (KO eqs. (16)–(20), p. 147; Appendix, pp. 158–160).

Decomposition:  
- **Myopic term**: T_J * X → collapses to the Merton rule when hedging terms vanish.  
- **Intertemporal-hedging term**: T_J * rho_mX * sigma_X * ( C(T) * X + B(T) ) → adjusts position based on predictability in X_t.

**CRRA specialization (what we use).**  
For U(W) = W^(1-gamma) / (1-gamma), T_J(W, T) = W / gamma. Then:  
    y*(W, X, T) = (W / gamma) * [ (mu_t - r) / sigma_t^2 + (rho_mX * sigma_X / sigma_t) * ( C(T) * X + B(T) ) ]  
When rho_mX = 0 or sigma_X = 0 or gamma = 1 (log utility), the hedging term drops out and the policy reduces to the myopic rule (KO discussion after eqs. (19)–(23), pp. 148–149).

**Why this is a great tutorial bed.**  
The model has one risky asset and one state X_t, so you can clearly see how each enhancement (projection, antithetic, residual, CV, Richardson) changes Stage-1 and Stage-2 results, while benchmarking against a closed-form solution (B(T), C(T)).

---

## 📂 File Descriptions

### `pgdpo_base_single.py` — **Baseline PG-DPO**
🚀 The starting point.  
Trains a Stage-1 PG-DPO policy **without** variance reduction or other enhancements. Reports:
- Stage-1 RMSE vs closed-form policy
- Expected utility difference vs closed-form

---

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Direct Costates)**
🎯 Takes the baseline policy and applies **PMP-based projection** using costates from BPTT.  
Shows how projection alone boosts accuracy.

---

### `pgdpo_antithetic_single.py` — **Antithetic PG-DPO**
🔄 Adds **antithetic sampling** to cut Monte Carlo variance in training and costate estimation.  
Improves both RMSE and utility.

---

### `pgdpo_residual_single.py` — **Residual PG-DPO**
🛠 Builds a residual network on top of the closed-form baseline, learning only **hedging demand corrections**.  
Training becomes more stable and RMSE drops dramatically.

---

### `pgdpo_cv_single.py` — **Residual + Control Variate**
🎚 Adds a **control variate** based on closed-form utility to residual PG-DPO.  
Further reduces variance and improves robustness under noisy conditions.

---

### `pgdpo_richardson_single.py` — **Residual + CV + Richardson Extrapolation**
⏩ Adds **Richardson extrapolation** to the CV setup to reduce timestep bias in utility estimates.  
⚠️ In low-dimensional Kim–Omberg, effect is minimal (and can be slightly negative) due to already tiny bias, but **in high dimensions it can shine** ✨.

---

## 📊 Stage-by-Stage RMSE Results

| Variant                                   | Stage 1 RMSE | Stage 2 RMSE |
|-------------------------------------------|--------------|--------------|
| Baseline PG-DPO                           | 0.233300     | –            |
| Projected PG-DPO (P-PGDPO)                 | 0.233300     | 0.005522     |
| P-PGDPO + Antithetic                       | 0.164274     | 0.004254     |
| P-PGDPO + Residual                         | 0.005663     | 0.003651     |
| P-PGDPO + Residual + Control Variate (CV)  | 0.036626     | 0.003179     |
| P-PGDPO + Residual + CV + Richardson       | 0.037177     | 0.003411     |

---

### 📝 Interpretation
- 📉 **Stage-1 RMSE** drops sharply from BASE → Residual.  
- 🏆 **P-PGDPO projection** consistently gives RMSE < 0.006 across all variants.  
- 💡 Antithetic and Residual improve Stage-1 accuracy well beyond BASE.  
- ⏳ CV and Richardson are more valuable in **high-dimensional problems**.

---

💡 **Note:** While this repo is research-grade, it’s deliberately structured as a **learning-friendly tutorial** — you can run each script independently, compare results, and see exactly how each enhancement changes the outcome.

