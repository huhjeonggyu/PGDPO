# 🐍 PG-DPO Portfolio Optimization — Experiment Scripts

Welcome! 🎉  
This repository is a **tutorial-style** collection of Python scripts that implement and test **Pontryagin-Guided Direct Policy Optimization (PG-DPO)** and its variants — step-by-step, from the most basic to more advanced forms.  
It’s designed not just for research, but also as a **hands-on learning resource** 📚 so you can follow along, modify, and experiment.

---

## 🧪 Test Environment

All experiments here are run on the **Kim and Omberg (1996)** continuous-time portfolio model 🏦.  
This tutorial uses the simplest non-trivial setting — **one risky asset 💹** and **one exogenous state variable 📈** (the risk premium / Sharpe ratio X_t) — which makes it easier to understand while still capturing intertemporal hedging.

✨ **Exogenous state dynamics** (risk premium OU process):  
`dX_t = -lambda_X * (X_t - X_bar) * dt + sigma_X * dZ_t^X`

💰 **Wealth dynamics**:  
`dW_t = r * W_t * dt + y_t * ( (mu_t - r) * dt + sigma_t * dZ_t )`

🎯 **Evaluation metrics**:  
- 📏 RMSE between learned policy and the **closed-form Kim–Omberg optimal policy**  
- 💡 Expected utility difference from the analytical optimum

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

**Snippet (minimal):**
```python
from torch.distributions import Normal

def simulate_with_sign(net_pi, T, W, dt, noise_sign=+1.0):
    logW = W.log(); sampler = Normal(0.0, 1.0)
    for k in range(m):
        t = k * dt
        state_t = torch.cat([logW.exp(), T - t], dim=1)
        pi_t = net_pi(state_t)
        mu_p = r + pi_t * (mu - r)
        sig2 = (pi_t * sigma)**2
        dZ = sampler.sample((len(W), 1)).to(dev) * noise_sign
        logW = logW + (mu_p - 0.5*sig2)*dt + torch.sqrt(sig2)*dZ*dt.sqrt()
        logW = logW.exp().clamp(min=lb_w).log()
    return logW.exp()

# paired evaluation (same CRNs, opposite signs)
W_T_pos = simulate_with_sign(net_pi, T, W, dt, +1.0)
W_T_neg = simulate_with_sign(net_pi, T, W, dt, -1.0)
U = 0.5 * (crra(W_T_pos) + crra(W_T_neg))  # lower-variance utility
```

- **Key:** Use the **same CRNs** with opposite signs and average. Apply to both Stage‑1 and costate BPTT.

---

### `pgdpo_residual_single.py` — **Residual PG-DPO**
🛠 Builds a residual network on top of the closed-form baseline, learning only **hedging demand corrections**.  
Training becomes more stable and RMSE drops dramatically.

**Snippet (minimal):**
```python
class ResidualPolicy(nn.Module):
    def __init__(self, base_policy):
        super().__init__()
        self.base = base_policy                  # closed-form or teacher
        self.res  = nn.Sequential(nn.Linear(2,128), nn.LeakyReLU(),
                                  nn.Linear(128,128), nn.LeakyReLU(),
                                  nn.Linear(128,1))
    def forward(self, state):                    # state = [W, X_t or τ]
        return self.base(state).detach() + self.res(state)  # π = π_base + δπ_θ
```
- **Tip:** Detach the base (teacher/closed‑form) so gradients update **only** the residual.

---

### `pgdpo_cv_single.py` — **Residual + Control Variate**
🎚 Adds a **control variate** based on closed-form utility to residual PG-DPO.  
Further reduces variance and improves robustness under noisy conditions.

**Snippet (minimal):**
```python
@torch.no_grad()
def u_cf_same_crn(T, W, dt):
    return simulate_and_utility(closed_form_policy, T, W, dt)  # SAME CRNs

def cv_adjust(u_pol, u_cf):
    u_pol_c = u_pol - u_pol.mean()
    u_cf_c  = u_cf  - u_cf.mean()
    beta = (u_pol_c*u_cf_c).mean() / (u_cf_c.square().mean() + 1e-8)
    return (u_pol - beta*u_cf), beta

u_pol = simulate_and_utility(net_pi, T, W, dt)   # requires grad
u_cf  = u_cf_same_crn(T, W, dt)                  # no grad
u_adj, beta = cv_adjust(u_pol, u_cf)             # lower-variance target
loss = -u_adj.mean()
```
- **Key:** Use **identical CRNs** for policy and closed-form utility to maximize correlation.

---

### `pgdpo_richardson_single.py` — **Residual + CV + Richardson Extrapolation**
⏩ Adds **Richardson extrapolation** to the CV setup to reduce timestep bias in utility estimates.  
⚠️ In low-dimensional Kim–Omberg, effect is minimal (and can be slightly negative) due to already tiny bias, but **in high dimensions it can shine** ✨.

**Snippet (minimal):**
```python
from torch.distributions import Normal

def util_with_steps(net_pi, T, W, dt, refine=1):
    sub_dt = dt / refine
    logW = W.log(); sampler = Normal(0.0, 1.0)
    for k in range(int(m*refine)):
        t = k * sub_dt
        state_t = torch.cat([logW.exp(), T - t], dim=1)
        pi_t = net_pi(state_t)
        mu_p = r + pi_t * (mu - r)
        sig2 = (pi_t * sigma)**2
        dZ = sampler.sample((len(W), 1)).to(dev)
        logW = logW + (mu_p - 0.5*sig2)*sub_dt + torch.sqrt(sig2)*dZ*sub_dt.sqrt()
        logW = logW.exp().clamp(min=lb_w).log()
    return crra(logW.exp())

U_coarse = util_with_steps(net_pi, T, W, dt, refine=1)
U_fine   = util_with_steps(net_pi, T, W, dt, refine=2)
U_star   = 2.0*U_fine - U_coarse                  # cancel O(dt) bias
loss = -U_star.mean()
```
- **Note:** Combine with antithetic/CV when time‑step bias is non‑negligible.

---

## 📊 Stage-by-Stage RMSE Results

| Variant                                   | Stage 1 RMSE | Stage 2 RMSE |
|-------------------------------------------|--------------|--------------|
| Baseline PG-DPO                           | 0.233300     | –            |
| Projected PG-DPO (P-PGDPO)                | 0.233300     | 0.005522     |
| P-PGDPO + Antithetic                      | 0.164274     | 0.004254     |
| P-PGDPO + Residual                        | 0.005663     | 0.003651     |
| P-PGDPO + Residual + Control Variate (CV) | 0.036626     | 0.003179     |
| P-PGDPO + Residual + CV + Richardson      | 0.037177     | 0.003411     |

---

### 📝 Interpretation
- 📉 **Stage-1 RMSE** drops sharply from BASE → Residual.  
- 🏆 **P-PGDPO projection** consistently gives RMSE < 0.006 across all variants.  
- 💡 Antithetic and Residual improve Stage-1 accuracy well beyond BASE.  
- ⏳ CV and Richardson are more valuable in **high-dimensional problems**.

---

💡 **Note:** While this repo is research-grade, it’s deliberately structured as a **learning-friendly tutorial** — you can run each script independently, compare results, and see exactly how each enhancement changes the outcome.
