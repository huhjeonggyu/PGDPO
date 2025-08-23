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

### `pgdpo_run_single.py` — **ALWAYS RUN: Antithetic + Richardson**
🔄 This replaces the old `pgdpo_antithetic_single.py`. The **RUN** simulator combines **true antithetic pairing** with **Richardson extrapolation** to reduce Monte Carlo bias and variance in a single, consistent runtime.  
It returns the **Richardson-extrapolated, antithetic utility**
\(
U_{\mathrm{run}} \;=\; 2\,U_{\mathrm{fine}} \;-\; U_{\mathrm{coarse}}
\)
where each of \(U_{\mathrm{fine}}, U_{\mathrm{coarse}}\) is itself an **antithetic** average.

**Key idea:** simulate the same initial states with a coarse grid \(m\) and a fine grid \(2m\), using correlated Brownian increments (and their sign-flipped antithetic pairs), then combine them as \(2U_f - U_c\).

**Core snippet (minimal):**
```python
def simulate_run(policy, B=None, *, W0=None, Y0=None, Tval=None, rng=None, seed_local=None):
    # domain-time sampling if (W0,Y0,Tval) not provided
    if W0 is None or Y0 is None or Tval is None:
        assert B is not None
        W0, Y0, TmT, _ = sample_initial_states(B, rng=rng)
    else:
        TmT = Tval if Tval.ndim == 2 else Tval.unsqueeze(1)
        B = W0.size(0)

    # --- coarse path (m steps), with true antithetic pairing ---
    ZWc, ZYc = _draw_correlated_normals(B, m, rng)
    WTc_p = _forward_path(policy, W0, TmT, Y0, +ZWc, +ZYc, m)
    WTc_m = _forward_path(policy, W0, TmT, Y0, -ZWc, -ZYc, m)
    Uc = 0.5 * (_crra_utility(WTc_p, gamma) + _crra_utility(WTc_m, gamma))

    # --- fine path (2m steps), decorrelated from coarse but antithetic-paired ---
    rng_f = make_generator((seed_local or 0) + 8191) if seed_local is not None else None
    ZWf, ZYf = _draw_correlated_normals(B, 2*m, rng_f)
    WTf_p = _forward_path(policy, W0, TmT, Y0, +ZWf, +ZYf, 2*m)
    WTf_m = _forward_path(policy, W0, TmT, Y0, -ZWf, -ZYf, 2*m)
    Uf = 0.5 * (_crra_utility(WTf_p, gamma) + _crra_utility(WTf_m, gamma))

    # --- Richardson-extrapolated antithetic utility ---
    return 2.0*Uf - Uc
```

In addition, the RUN module provides **vectorized, memory-safe costate estimation** for PMP projection:
```python
def estimate_costates_run(policy, T0, W0, Y0, *, repeats=REPEATS, sub_batch=SUBBATCH, seed_eval=None):
    # replicate each eval point (W0,Y0,T0) r_chunk times, run simulate_run once per chunk,
    # average per-point, then take gradients wrt W0 (for J_W) and second-derivatives (J_WW, J_WY).
    # policy params are temporarily frozen for speed and memory.
    ...
```
Combined with the PMP projector, this yields the P‑PGDPO teacher:
```python
def ppgdpo_pi_run(policy, T0, W0, Y0, *, repeats=REPEATS, sub_batch=SUBBATCH, seed_eval=None):
    J_W, J_WW, J_WY = estimate_costates_run(...)
    return project_pmp(J_W, J_WW, J_WY, W0, Y0)
```

---

### `pgdpo_residual_single.py` — **Residual PG-DPO**
🛠 Builds a residual network on top of a **myopic baseline policy**.  
The residual learns only **hedging demand corrections**.

**Snippet (minimal):**
```python
def myopic_from_sharpe(X_t, sigma, gamma):
    return (1.0/gamma) * (X_t / (sigma + 1e-12))

class MyopicPolicy(nn.Module):
    def forward(self, state):
        W, X_t = state[:, [0]], state[:, [1]]
        pi = myopic_from_sharpe(X_t, sigma, gamma)
        return pi

class ResidualPolicy(nn.Module):
    def __init__(self, base_policy):
        super().__init__()
        self.base = base_policy
        self.res  = nn.Sequential(
            nn.Linear(2,128), nn.LeakyReLU(),
            nn.Linear(128,128), nn.LeakyReLU(),
            nn.Linear(128,1)
        )
    def forward(self, state):
        pi_base = self.base(state).detach()
        pi_res  = self.res(state)
        return pi_base + pi_res
```

---

### `pgdpo_cv_single.py` — **Residual + Control Variate**
🎚 Adds a **control variate (CV)** based on **myopic utility**, simulated under identical CRNs.  
This further reduces variance and improves robustness.

**Snippet (minimal):**
```python
def cv_adjust(u_pol, u_my, W0):
    u_pol_c = u_pol - u_pol.mean()
    u_my_c  = u_my - u_my.mean()
    beta = (u_pol_c*u_my_c).mean() / (u_my_c.square().mean() + 1e-8)
    return (u_pol - beta*u_my), beta

U_pol = simulate(policy, batch_size, train=True)
with torch.no_grad():
    U_my = simulate(myopic_policy, batch_size, train=False)
U_adj, beta = cv_adjust(U_pol, U_my, W0)
loss = -U_adj.mean()
```

---

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Direct Costates)**
🎯 Takes the baseline policy and applies **PMP-based projection** using costates from BPTT.  
Projection alone already gives a sharp RMSE reduction. The RUN path above provides a *variance-reduced* teacher; this direct version uses the base simulator.

**Snippet (minimal):**
```python
def estimate_costates(policy_net, T0, W0, Y0, repeats, sub_batch):
    # average chunk utilities per state, backprop to get J_W, then J_WW & J_WY
    ...
def project_pmp(J_W, J_WW, J_WY, W, Y):
    # π* = -1/(W J_WW) [ J_W ((μ-r)/σ^2) + J_WY (ρ σ_Y)/σ ]
    ...
```

---

## 📊 Stage-by-Stage RMSE Results

| Variant                                   | Stage 1 RMSE | Stage 2 RMSE |
|-------------------------------------------|--------------|--------------|
| Baseline PG-DPO                           | 0.233300     | –            |
| Projected PG-DPO (P-PGDPO)                | 0.233300     | 0.005522     |
| P-PGDPO + Antithetic                      | 0.164274     | 0.004254     |
| P-PGDPO + Residual                        | 0.005663     | 0.003651     |
| P-PGDPO + Residual + Control Variate (CV) | 0.036626     | 0.003179     |

---

### 📝 Interpretation
- 📉 **Stage-1 RMSE** drops sharply from BASE → Residual.  
- 🏆 **P-PGDPO projection** consistently gives RMSE < 0.006 across all variants.  
- 💡 Antithetic and Residual improve Stage-1 accuracy well beyond BASE.  
- ⏳ CV is more valuable in **high-dimensional problems**.

---

💡 **Note:** While this repo is research-grade, it’s deliberately structured as a **learning-friendly tutorial** — you can run each script independently, compare results, and see exactly how each enhancement changes the outcome.
