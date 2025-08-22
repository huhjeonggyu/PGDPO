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

### `pgdpo_with_ppgdpo_single.py` — **P-PGDPO (Direct Costates)**
🎯 Takes the baseline policy and applies **PMP-based projection** using costates from BPTT.  
Projection alone already gives a sharp RMSE reduction.

**Snippet (minimal):**
```python
def estimate_costates(policy_net, T0, W0, Y0, repeats=800, sub_batch=100):
    W0g = W0.clone().requires_grad_(True)
    Y0g = Y0.clone().requires_grad_(True)
    lam_sum = torch.zeros_like(W0g)
    dW_sum  = torch.zeros_like(W0g)
    dY_sum  = torch.zeros_like(Y0g)
    for i in range(0, repeats, sub_batch):
        rpts = min(sub_batch, repeats - i)
        T_b, W_b, Y_b = T0.repeat(rpts,1), W0g.repeat(rpts,1), Y0g.repeat(rpts,1)
        u1 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        u2 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        avg_u = 0.5 * (u1 + u2)
        avg_u_per_point = avg_u.view(rpts, n).mean(0)
        (lam_b,) = torch.autograd.grad(avg_u_per_point.sum(), W0g, create_graph=True)
        dlamW_b, dlamY_b = torch.autograd.grad(lam_b.sum(), (W0g, Y0g))
        lam_sum += lam_b.detach() * rpts
        dW_sum  += dlamW_b.detach() * rpts
        dY_sum  += dlamY_b.detach() * rpts
    inv = 1.0 / repeats
    return lam_sum*inv, dW_sum*inv, dY_sum*inv
```

---

### `pgdpo_antithetic_single.py` — **Antithetic PG-DPO**
🔄 Adds **antithetic sampling** to cut Monte Carlo variance in both training and costate estimation.  
Same random numbers with opposite signs are paired and averaged.

**Snippet (minimal):**
```python
def simulate_with_sign(net_pi, T, W, dt, noise_sign=+1.0):
    logW = W.log()
    for k in range(m):
        pi_t = net_pi(torch.cat([logW.exp(), T - k*dt], dim=1))
        mu_p = r + pi_t * (mu - r)
        sig2 = (pi_t * sigma)**2
        dZ = torch.randn(len(W), 1, device=W.device) * noise_sign
        logW = logW + (mu_p - 0.5*sig2)*dt + torch.sqrt(sig2)*dZ*dt.sqrt()
        logW = logW.exp().clamp(min=lb_W).log()
    return logW.exp()

W_T_pos = simulate_with_sign(net_pi, T, W, dt, +1.0)
W_T_neg = simulate_with_sign(net_pi, T, W, dt, -1.0)
U = 0.5 * (crra(W_T_pos) + crra(W_T_neg))
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
