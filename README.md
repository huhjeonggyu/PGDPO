# PG-DPO: Pontryagin-Guided Direct Policy Optimization

This repository is the **official implementation** of the following paper:

[**Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization for Continuous-Time Multi-Asset Portfolio**](https://arxiv.org/abs/2504.11116) <!-- arXiv link 고정 -->

📄 Detailed code-level instructions are provided in the associated PDF guide:  
<https://drive.google.com/file/d/1b-HoFFeEu1tmXH0LWuaD-wszEic0GKfu/view?usp=sharing>

---

## Citation

```bibtex
@article{huh2025breaking,
  title  = {Breaking the Dimensional Barrier: A Pontryagin-Guided Direct Policy Optimization for Continuous-Time Multi-Asset Portfolio},
  author = {Huh, Jeonggyu and Jeon, Jaegi and Koo, Hyeng Keun},
  journal = {arXiv preprint arXiv:2504.11116},
  year   = {2025}
}
```

---

## 🎓 Educational Version (Colab-ready)

We also provide a simplified educational version of PG-DPO that can be run directly in Google Colab.  
This version is **single-asset**, based on the Merton model with **constant coefficients**, and has been simplified in various ways to better suit educational purposes — for example:

- Removes exogenous state variables, control variates, and residual learning  
- Focuses only on the core idea of PG-DPO and the comparison between the Stage-1 direct policy and the P-PGDPO projected policy

The goal is to make it easier to understand the core mechanics without the complexity of the full framework.

📎 **Colab-ready educational version**: [Google Drive Link](https://drive.google.com/file/d/1JfheqSXIIq2pZY8nLnbgYx_9E-xVb9re/view?usp=sharing)

### 🔧 Core Functions — Minimal PG-DPO ➜ P-PGDPO

#### 🎲 `generate_uniform_domain`

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

#### 🌀 `sim`

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

`d log W_t = [ r + π_t (μ − r) − 0.5 (π_t σ)^2 ] dt + (π_t σ) dB_t`  

**Stability:** Operates in log space and clamps `W` at each step to stay above a wealth floor.  

**Output:** Terminal CRRA utility `U(W_T)`, which provides the training signal.  

**Tip:** For variance/bias reduction, antithetic noise pairs and Richardson extrapolation can be added.

---

#### 📐 `estimate_costates`

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
        dx_lamb_batch, = torch.autograd.grad(lamb_batch.sum(),        W0_grad)
        lamb_accum    += lamb_batch.detach()    * current_repeats
        dx_lamb_accum += dx_lamb_batch.detach() * current_repeats
        total_repeats_done += current_repeats
    inv_N = 1.0 / total_repeats_done
    return (lamb_accum * inv_N, dx_lamb_accum * inv_N)
```

**Purpose:** Estimate costates via BPTT: `λ = ∂U/∂W` and its derivative at sampled states.  

**Trick:** Tile `(T0, W0, dt0)` `repeats` times, average utilities across rollouts, then compute first/second derivatives.  

**Use case:** These costates are later fed into the Pontryagin projection.  

**Caution:** If the second derivative `∂²U/∂W² ≈ 0`, projection can diverge. In practice, add small epsilon-guards or winsorization.

---

#### 🎯 `get_optimal_pi`

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

<details>
<summary>📜 Show Full PG-DPO Educational Version Code</summary>

```python
import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions.normal import Normal

# ====================== 1. 기본 설정 ======================
dev = "cuda" if torch.cuda.is_available() else "cpu"

# 금융 파라미터
r = 0.02
gamma = 2.0
mu = 0.08
sigma = 0.2

# 시뮬레이션 파라미터
T_max = 1.0
m = 20
W_min = 0.1
W_max = 2.0
lb_w = 0.001

# 학습 파라미터
n = 2048
total_epoch = 500
learning_rate = 1e-5
seed = 1

# 평가 파라미터
n_eval = 5000
repeats = 2000
sub_batch_size = 500


# 분석적 머튼 최적 정책
pi_optimal_merton = (1.0 / gamma) * (1 / (sigma**2)) * (mu - r)
print(f"--- Merton Model (Single Asset) ---")
print(f"mu={mu:.4f}, sigma={sigma:.4f}, pi*={pi_optimal_merton:.4f}")
print("-----------------------------------")

# ====================== 2. 모델 정의 ======================
class TradeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1a = nn.Linear(2, 128)
        self.linear2a = nn.Linear(128, 128)
        self.linear3a = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, state):
        x = self.activation(self.linear1a(state))
        x = self.activation(self.linear2a(x))
        return self.linear3a(x)

# ====================== 3. 헬퍼 함수 ======================
def generate_uniform_domain(n, T_max, W_min, W_max, m, dev, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    T = T_max * torch.rand([n, 1], device=dev)
    dt = T / m
    W = W_min + (W_max - W_min) * torch.rand([n, 1], device=dev)
    return T, W, dt

def sim(net_pi, T, W, dt, train=True):
    batch_size = len(W)
    logW = W.log()
    sampler = Normal(0.0, 1.0)
    for k_step in range(m):
        t = k_step * dt
        state_t = torch.cat([logW.exp(), T - t], dim=1)
        pi_t = net_pi(state_t) if train else net_pi(state_t).detach()
        mu_p = r + pi_t * (mu - r)
        var_p = (pi_t * sigma) ** 2
        sigma_p = torch.sqrt(var_p)
        dZ = sampler.sample(sample_shape=(batch_size, 1)).to(dev)  # 안티테틱 제거
        logW = logW + (mu_p - 0.5 * var_p) * dt + sigma_p * dZ * dt.sqrt()
        logW = logW.exp().clamp(min=lb_w).log()
    W_final = logW.exp()
    U_theta = (W_final**(1.0 - gamma)) / (1.0 - gamma)
    return U_theta

def estimate_costates(net_pi, T0, W0, dt0, repeats, sub_batch_size):
    W0_grad = W0.detach().clone().requires_grad_(True)
    lamb_accum = torch.zeros_like(W0_grad)
    dx_lamb_accum = torch.zeros_like(W0_grad)
    total_repeats_done = 0
    for i in range(0, repeats, sub_batch_size):
        current_repeats = min(sub_batch_size, repeats - i)
        T_b = T0.repeat(current_repeats, 1)
        W_b = W0_grad.repeat(current_repeats, 1)
        dt_b = dt0.repeat(current_repeats, 1)
        U = sim(net_pi, T_b, W_b, dt_b, train=False)
        U_mean_per_point = U.view(current_repeats, W0.shape[0]).mean(dim=0)
        lamb_batch, = torch.autograd.grad(U_mean_per_point.sum(), W0_grad, create_graph=True, retain_graph=True)
        dx_lamb_batch, = torch.autograd.grad(lamb_batch.sum(), W0_grad)
        lamb_accum += lamb_batch.detach() * current_repeats
        dx_lamb_accum += dx_lamb_batch.detach() * current_repeats
        total_repeats_done += current_repeats
    inv_N = 1.0 / total_repeats_done
    return (lamb_accum * inv_N, dx_lamb_accum * inv_N)

def get_optimal_pi(W, lam, dlam_dx, mu, r, sigma, device):
    W_t = torch.as_tensor(W, dtype=torch.float32, device=device)
    lam_t = torch.as_tensor(lam, dtype=torch.float32, device=device)
    dlamdx_t = torch.as_tensor(dlam_dx, dtype=torch.float32, device=device)
    scalar_coeff = -lam_t / (W_t * dlamdx_t + 1e-8)
    myopic_demand = (1/(sigma**2)) * (mu - r)
    return scalar_coeff * myopic_demand

# ====================== 4. 학습 루프 ======================
torch.manual_seed(seed); np.random.seed(seed)
net_pi = TradeNet().to(dev)
opt_pi = torch.optim.Adam(net_pi.parameters(), lr=learning_rate)
T_eval, W_eval, dt_eval = generate_uniform_domain(n_eval, T_max, W_min, W_max, m, dev, seed=123)
utility_history = []

print("Starting training loop...")
for i in range(total_epoch + 1):
    net_pi.train()
    opt_pi.zero_grad()
    T0, W0, dt0 = generate_uniform_domain(n, T_max, W_min, W_max, m, dev, i)
    U_theta_sim = sim(net_pi, T0, W0, dt0, train=True)
    if torch.isfinite(U_theta_sim.mean()):
        utility_history.append(U_theta_sim.mean().item())
    loss = -U_theta_sim.mean()
    if torch.isfinite(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_pi.parameters(), 1.0)
        opt_pi.step()

    # P-PGDPO 및 RMSE 비교
    if i % 50 == 0:
        net_pi.eval()
        lamb_hat, dlam_dx_hat = estimate_costates(net_pi, T_eval, W_eval, dt_eval, repeats, sub_batch_size)
        pi_pgdpo = get_optimal_pi(W_eval, lamb_hat, dlam_dx_hat, mu, r, sigma, dev)
        # Direct policy eval
        pi_net_eval = []
        with torch.no_grad():
            for j in range(0, n_eval, sub_batch_size):
                state_batch = torch.cat([W_eval[j:j+sub_batch_size], T_eval[j:j+sub_batch_size]], dim=1)
                pi_net_eval.append(net_pi(state_batch))
        pi_net_eval = torch.cat(pi_net_eval, dim=0)

        rmse_direct = torch.sqrt(((pi_net_eval - pi_optimal_merton)**2).mean()).item()
        rmse_pgdpo = torch.sqrt(((pi_pgdpo - pi_optimal_merton)**2).mean()).item()
        avg_util = np.mean(utility_history[-50:]) if utility_history else float('nan')
        print(f"[Iter {i:5d}] Loss={loss.item():.6f} AvgU={avg_util:.6f} RMSE_direct={rmse_direct:.6f} RMSE_pgdpo={rmse_pgdpo:.6f}")

print("Training completed.")
