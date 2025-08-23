# 🎓 Educational Version — PG-DPO (Basic Merton Model)

We also provide a simplified educational version of PG-DPO **`pgdpo_basic.py`**.  
This version is **single-asset**, based on the Merton model with **constant coefficients**, and has been simplified in various ways to better suit educational purposes — for example:

- Removes exogenous state variables, control variates, and residual learning  
- Focuses only on the core idea of PG-DPO and the comparison between the Stage-1 direct policy and the P-PGDPO projected policy

The goal is to make it easier to understand the core mechanics without the complexity of the full framework.

---

## 🔧 Core Functions — Minimal PG-DPO ➜ P-PGDPO

### 🎲 `generate_uniform_domain`

```python
def generate_uniform_domain(n, T_max, W_min, W_max, m, dev, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    T  = T_max * torch.rand([n, 1], device=dev)
    dt = T / m                       # per-path constant step size
    W  = W_min + (W_max - W_min) * torch.rand([n, 1], device=dev)
    return T, W, dt
```

**Purpose:** Sampling initial states for training and evaluation.  

**How it works:**
1. Sample maturity `T ~ Uniform(0, T_max)`  
2. Set `dt = T/m` so that each path maintains a constant step size  
3. Sample initial wealth `W` uniformly from `[W_min, W_max]` for stability of utility scaling  

**Note:** `seed` ensures reproducibility; keep the domain compact to avoid extreme values of `W`.

---

### 🌀 `sim`

```python
def sim(net_pi, T, W, dt, train=True):
    batch_size = len(W)
    logW = W.log()
    sampler = Normal(0.0, 1.0)
    for k_step in range(m):
        t = k_step * dt
        state_t = torch.cat([logW.exp(), T - t], dim=1)
        pi_t = net_pi(state_t) if train else net_pi(state_t).detach()
        mu_p  = r + pi_t * (mu - r)
        var_p = (pi_t * sigma) ** 2
        sigma_p = torch.sqrt(var_p)
        dZ = sampler.sample(sample_shape=(batch_size, 1)).to(dev)
        logW = logW + (mu_p - 0.5 * var_p) * dt + sigma_p * dZ * dt.sqrt()
        logW = logW.exp().clamp(min=lb_w).log()  # stabilize: enforce wealth floor
    W_final = logW.exp()
    U_theta = (W_final ** (1.0 - gamma)) / (1.0 - gamma)  # CRRA utility
    return U_theta
```

**Purpose:** Monte Carlo rollout of the log-wealth SDE under policy `π_t`.  

**Implementation:**  

`d \log W_t = [ r + π_t (μ − r) − 0.5 (π_t σ)^2 ] dt + (π_t σ) dB_t`  

**Stability:** Operates in log space and clamps `W` at each step to stay above a wealth floor.  

**Output (Utility):** Terminal CRRA utility `U(W_T)`, which provides the training signal.  

---

### 📐 `estimate_costates`

```python
def estimate_costates(net_pi, T0, W0, dt0, repeats, sub_batch_size):
    W0_grad = W0.detach().clone().requires_grad_(True)
    lamb_accum   = torch.zeros_like(W0_grad)  # λ = ∂U/∂W
    dx_lamb_accum = torch.zeros_like(W0_grad) # ∂λ/∂W
    total_repeats_done = 0
    for i in range(0, repeats, sub_batch_size):
        current_repeats = min(sub_batch_size, repeats - i)
        T_b  = T0.repeat(current_repeats, 1)
        W_b  = W0_grad.repeat(current_repeats, 1)
        dt_b = dt0.repeat(current_repeats, 1)
        U = sim(net_pi, T_b, W_b, dt_b, train=False)
        U_mean_per_point = U.view(current_repeats, W0.shape[0]).mean(dim=0)
        lamb_batch,    = torch.autograd.grad(U_mean_per_point.sum(), W0_grad, create_graph=True, retain_graph=True)
        dx_lamb_batch, = torch.autograd.grad(lamb_batch.sum(), W0_grad)
        lamb_accum    += lamb_batch.detach()    * current_repeats
        dx_lamb_accum += dx_lamb_batch.detach() * current_repeats
        total_repeats_done += current_repeats
    inv_N = 1.0 / total_repeats_done
    return (lamb_accum * inv_N, dx_lamb_accum * inv_N)
```

**Purpose:** Estimate costates via BPTT: `λ = ∂U/∂W` and its derivative at sampled states.  

**Trick:** Tile `(T0, W0, dt0)` `repeats` times, average utilities across rollouts, then compute first/second derivatives.  

**First gradient:** `torch.autograd.grad(U_mean_per_point.sum(), W0_grad, create_graph=True, ...)` → `λ = ∂U/∂W`.  
**Second gradient:** `torch.autograd.grad(lamb_batch.sum(), W0_grad)` → `∂λ/∂W`.  

**Use case:** These costates are later fed into the Pontryagin projection.  

**Caution:** If the second derivative `∂²U/∂W² ≈ 0`, projection can diverge. In practice, add small epsilon-guards or winsorization.

---

### 🎯 `get_optimal_pi`

```python
def get_optimal_pi(W, lam, dlam_dx, mu, r, sigma, device):
    W_t      = torch.as_tensor(W,        dtype=torch.float32, device=device)
    lam_t    = torch.as_tensor(lam,      dtype=torch.float32, device=device)
    dlamdx_t = torch.as_tensor(dlam_dx,  dtype=torch.float32, device=device)
    scalar_coeff   = -lam_t / (W_t * dlamdx_t + 1e-8)   # ≈ 1/γ in the Merton model
    myopic_demand  = (mu - r) / (sigma ** 2)
    return scalar_coeff * myopic_demand
```

**Purpose:** Project costates into the Pontryagin-optimal control.  

**Formula (Single-asset Merton):**  

`π_PMP = [ -λ / ( W * (∂λ/∂W) ) ] * (μ − r) / σ²`  

The bracketed term converges to `1/γ`.  

**Safety:** Division by zero avoided with `1e-8`; mild output clipping is also common in practice.

---

## 📊 Example Results (Educational Version)

We ran the simplified **Merton single-asset model** (`pgdpo_basic.py`).  
Here, the **direct neural policy** gradually learns towards the optimal strategy, while the **PG-DPO projection** (Pontryagin-guided) is *exact* from the start.

```
--- Merton Model (Single Asset) ---
mu=0.0800, sigma=0.2000, pi*=0.7500
-----------------------------------
Starting training loop...
[Iter     0] Loss=1.615353 AvgU=-1.615353 RMSE_direct=0.667845 RMSE_pgdpo=0.000000
[Iter    50] Loss=1.567789 AvgU=-1.553477 RMSE_direct=0.605164 RMSE_pgdpo=0.000000
[Iter   100] Loss=1.533107 AvgU=-1.557038 RMSE_direct=0.544532 RMSE_pgdpo=0.000000
[Iter   150] Loss=1.530957 AvgU=-1.563560 RMSE_direct=0.486697 RMSE_pgdpo=0.000000
[Iter   200] Loss=1.543046 AvgU=-1.551114 RMSE_direct=0.430354 RMSE_pgdpo=0.000000
[Iter   250] Loss=1.528895 AvgU=-1.549450 RMSE_direct=0.377386 RMSE_pgdpo=0.000000
[Iter   300] Loss=1.600487 AvgU=-1.555689 RMSE_direct=0.329234 RMSE_pgdpo=0.000000
[Iter   350] Loss=1.541713 AvgU=-1.539673 RMSE_direct=0.285151 RMSE_pgdpo=0.000000
[Iter   400] Loss=1.486467 AvgU=-1.559407 RMSE_direct=0.248315 RMSE_pgdpo=0.000000
[Iter   450] Loss=1.562397 AvgU=-1.541803 RMSE_direct=0.217388 RMSE_pgdpo=0.000000
[Iter   500] Loss=1.585522 AvgU=-1.548761 RMSE_direct=0.194905 RMSE_pgdpo=0.000000
Training completed.
```

**Takeaway:**  
- The **direct policy** improves slowly, but always lags behind the analytic optimum.  
- The **PG-DPO projection** matches the closed-form Merton solution **perfectly from the very beginning**.  

This highlights the power of **Pontryagin-guided learning**: it enforces the correct structure of the optimal control, even when the neural network has not yet converged.

---
