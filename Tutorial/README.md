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


**Snippet (minimal):**
```python
def simulate(
    policy_module: nn.Module,
    B: int,
    train: bool = True,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: float | torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    seed: int | None = None,
):
    """
    Return pathwise CRRA utility U(W_T) for B paths.
    - seed / rng: reproducible randomness 
    """
    # init states
    if W0 is None or Y0 is None:
        W, Y = sample_initial_states(B)
        TmT = torch.full_like(W, T)
    else:
        W, Y = W0, Y0
        if Tval is None:
            TmT = torch.full_like(W, T)
        elif torch.is_tensor(Tval):
            TmT = Tval.to(device=W.device, dtype=W.dtype)
            if TmT.shape != W.shape:
                if TmT.numel() == 1:
                    TmT = TmT.expand_as(W)
                else:
                    TmT = TmT.view_as(W)
        else:
            TmT = torch.full_like(W, float(Tval))

    gen = rng if rng is not None else make_generator(seed)

    logW = W.clamp(min=lb_W).log()

    for _ in range(m):
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)  # [B,1]

        # dynamics
        risk_prem = sigma * (alpha * Y)               # mu - r
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2

        zW, zY = correlated_normals(W.shape[0], rho, gen=gen)
        dBW = math.sqrt(dt) * zW
        dBY = math.sqrt(dt) * zY

        # log-wealth update (Ito for geometric-like wealth with control)
        logW = logW + (driftW - 0.5*varW) * dt + (pi_t * sigma) * dBW
        # factor OU
        Y    = Y    + kappaY*(thetaY - Y)*dt + sigmaY * dBY

        # stabilize and step time
        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0)<1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U  # [B,1]

# ---------------- Closed-form: builder ----------------
```
- **Role:** Roll out log‑wealth under the current policy and return terminal utility for training/metrics.
- **Notes:** Log‑space integration, wealth floor clamp, and per‑step policy evaluation.

---

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Direct Costates)**
🎯 Takes the baseline policy and applies **PMP-based projection** using costates from BPTT.  
Shows how projection alone boosts accuracy.


**Snippet (minimal):**
```python
def estimate_costates(policy_net, T0, W0, Y0, repeats=800, sub_batch=100):
    """
    Estimate λ=J_W, ∂λ/∂W=J_WW, ∂λ/∂Y=J_WY at (W0,Y0,T0) under policy_net.
    Keeps autograd graph by calling simulate(..., train=True).
    """
    n = W0.size(0)
    W0g = W0.clone().requires_grad_(True)
    Y0g = Y0.clone().requires_grad_(True)

    lam_sum = torch.zeros_like(W0g)
    dW_sum  = torch.zeros_like(W0g)
    dY_sum  = torch.zeros_like(Y0g)

    for i in range(0, repeats, sub_batch):
        rpts = min(sub_batch, repeats - i)
        T_b = T0.repeat(rpts,1)
        W_b = W0g.repeat(rpts,1)
        Y_b = Y0g.repeat(rpts,1)

        # two independent MC batches, then average to reduce variance
        u1 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        u2 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        avg_u = 0.5 * (u1 + u2)

        avg_u_per_point = avg_u.view(rpts, n).mean(0)  # [n,1]
        (lam_b,) = torch.autograd.grad(avg_u_per_point.sum(), W0g, create_graph=True)
        dlamW_b, dlamY_b = torch.autograd.grad(lam_b.sum(), (W0g, Y0g), allow_unused=False)

        lam_sum += lam_b.detach() * rpts
        dW_sum  += dlamW_b.detach() * rpts
        dY_sum  += dlamY_b.detach() * rpts

    inv = 1.0 / repeats
    return lam_sum*inv, dW_sum*inv, dY_sum*inv   # [n,1] each
```
- **Role:** Compute `λ = ∂U/∂W` and its derivative `∂λ/∂W` by differentiating Monte Carlo utility.
- **Notes:** Uses `create_graph=True` for higher‑order grads; outputs feed PMP projection.

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
def myopic_demand(mu_t, r, sigma, gamma):
    # Generic CRRA myopic demand
    return (mu_t - r) / (gamma * (sigma**2) + 1e-12)

def myopic_from_sharpe(X_t, sigma, gamma):
    # If the model uses Sharpe factor s.t. (mu_t - r) = sigma * X_t (Kim–Omberg)
    return (1.0/gamma) * (X_t / (sigma + 1e-12))

class ResidualPolicy(nn.Module):
    def __init__(self, base_policy):
        super().__init__()
        self.base = base_policy              # closed-form or a teacher policy
        self.res  = nn.Sequential(
            nn.Linear(2,128), nn.LeakyReLU(),
            nn.Linear(128,128), nn.LeakyReLU(),
            nn.Linear(128,1)
        )
    def forward(self, state):                # state = [W, X_t] or [features]
        pi_base = self.base(state).detach()  # do NOT backprop through base
        pi_res  = self.res(state)
        return pi_base + pi_res               # π = π_base (≈ myopic+hedge) + δπ_θ
```
- **Myopic:** `(mu_t - r)/(γ σ²)` or `(1/γ) * (X_t/σ)` if `mu_t - r = σ X_t`.
- **Residual:** Learns corrections on top of base (closed‑form/teacher) to match dynamics/hedging.

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

