# pgdpo_base_single.py
# BASE version + Closed-form benchmark
# - Single asset, single factor
# - Euler–Maruyama simulator
# - Stage-1 direct policy optimization ONLY (no P-PGDPO)
# - Adds closed-form (Kim–Omberg-style for CRRA + OU factor) policy pi_cf(t,Y)
# - Compares learned policy vs closed-form: RMSE and expected utility (common random numbers)
# Requirements: torch, numpy
#
# Usage:
#   python pgdpo_base_single.py

import math, numpy as np, torch
import torch.nn as nn
import torch.optim as optim

# --------------------------- Config ---------------------------
seed = 7
torch.manual_seed(seed); np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Market & utility
r = 0.02                 # risk-free rate
gamma = 5.0              # CRRA risk aversion
sigma = 0.2              # asset volatility
kappaY = 1.2             # OU mean-reversion speed
thetaY = 0.0             # OU long-run mean
sigmaY = 0.3             # factor volatility
rho = 0.3                # Corr(dB_W, dB_Y)
alpha = 0.8              # risk-premium loading: mu - r = sigma * (alpha * Y)

# Simulation
T = 1.5
m = 60
dt = T / m
batch_size = 1024
W0_range = (0.8, 1.2)
Y0_range = (-1.0, 1.0)
lb_W = 1e-3              # log-wealth floor

# Training
epochs = 300
lr = 1e-3
pi_cap = 2.0             # clamp policy within [-pi_cap, pi_cap]

# Evaluation
N_eval_states = 200
N_eval_paths  = 5000     # MC paths for EU comparison (common random numbers)

# --------------------------- Policy ---------------------------
class DirectPolicy(nn.Module):
    """Direct policy: (W, TmT, Y) -> pi. (No residual/myopic split in BASE version.)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # small init to avoid explosive early control
        for mmod in self.net.modules():
            if isinstance(mmod, nn.Linear):
                nn.init.xavier_uniform_(mmod.weight, gain=0.8)
                nn.init.zeros_(mmod.bias)

    def forward(self, W, TmT, Y):
        x = torch.cat([W, TmT, Y], dim=1)
        pi = self.net(x)
        return torch.clamp(torch.tanh(pi), -pi_cap, pi_cap)

# ----------------------- Sampling utils -----------------------
def sample_initial_states(B):
    W0 = torch.empty(B, 1, device=device).uniform_(*W0_range)
    Y0 = torch.empty(B, 1, device=device).uniform_(*Y0_range)
    return W0, Y0

def correlated_normals(B, rho):
    z1 = torch.randn(B, 1, device=device)
    z2 = torch.randn(B, 1, device=device)
    zW = z1
    zY = rho * z1 + math.sqrt(max(1.0 - rho*rho, 0.0)) * z2
    return zW, zY

def generate_noise(B):
    """Pre-generate (m, B, 2) standard normals with desired correlation for common-RN eval."""
    dBW = torch.empty(m, B, 1, device=device)
    dBY = torch.empty(m, B, 1, device=device)
    for t in range(m):
        zW, zY = correlated_normals(B, rho)
        dBW[t] = math.sqrt(dt) * zW
        dBY[t] = math.sqrt(dt) * zY
    return dBW, dBY

# ----------------------- Simulator (Euler) ---------------------
def simulate(policy_module: nn.Module, B: int, train=True, W0=None, Y0=None, Tval=None):
    """Return pathwise CRRA utility U(W_T) for B paths."""
    if W0 is None or Y0 is None:
        W, Y = sample_initial_states(B)
        TmT = torch.full_like(W, T)
    else:
        W, Y = W0, Y0
        TmT = torch.full_like(W, T if Tval is None else Tval)

    logW = W.clamp(min=lb_W).log()

    for _ in range(m):
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)  # [B,1]

        # dynamics
        risk_prem = sigma * (alpha * Y)               # mu - r
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2

        zW, zY = correlated_normals(W.size(0), rho)
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

@torch.no_grad()
def simulate_with_noise(policy_module: nn.Module, W0, Y0, dBW, dBY):
    """Simulate with pre-generated (m,B,1) noise arrays for common-RN comparisons."""
    B = W0.shape[0]
    TmT = torch.full_like(W0, T)
    logW = W0.clamp(min=lb_W).log()
    Y = Y0.clone()

    for t in range(m):
        pi_t = policy_module(logW.exp(), TmT, Y)
        risk_prem = sigma * (alpha * Y)
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2

        logW = logW + (driftW - 0.5*varW) * dt + (pi_t * sigma) * dBW[t]
        Y    = Y    + kappaY*(thetaY - Y)*dt + sigmaY * dBY[t]

        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0)<1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U

# ---------------- Closed-form B(τ), C(τ) via Riccati (RK4) ----------------
# From CRRA + OU factor ansatz V = x^{1-γ}/(1-γ) * exp(A(τ) + B(τ) y + 0.5 C(τ) y^2)
# the ODEs in time-to-go τ with C(0)=B(0)=0 are:
#   C' = 2κ C - σ_Y^2 C^2 - ((1-γ)/γ) (α + σ_Y ρ C)^2
#   B' = κ B - κ θ C - σ_Y^2 B C - ((1-γ)/γ) B (σ_Y ρ) (α + σ_Y ρ C)
# and the optimal policy is
#   π_cf(τ,y) = (1/γ)(1/σ^2) [ σ α y + σ σ_Y ρ (B(τ) + C(τ) y) ]
#             = (1/γ)[ (α/σ) y + (ρ σ_Y / σ) (B(τ) + C(τ) y) ].

def _C_rhs(C):
    return 2.0*kappaY*C - (sigmaY**2)*(C**2) - ((1.0 - gamma)/gamma) * (alpha + sigmaY*rho*C)**2

def _B_rhs(B, C):
    return kappaY*B - kappaY*thetaY*C - (sigmaY**2)*B*C - ((1.0 - gamma)/gamma) * B * (sigmaY*rho) * (alpha + sigmaY*rho*C)

def _rk4_step(B, C, h):
    k1C = _C_rhs(C)
    k1B = _B_rhs(B, C)

    k2C = _C_rhs(C + 0.5*h*k1C)
    k2B = _B_rhs(B + 0.5*h*k1B, C + 0.5*h*k1C)

    k3C = _C_rhs(C + 0.5*h*k2C)
    k3B = _B_rhs(B + 0.5*h*k2B, C + 0.5*h*k2C)

    k4C = _C_rhs(C + h*k3C)
    k4B = _B_rhs(B + h*k3B, C + h*k3C)

    C_next = C + (h/6.0)*(k1C + 2*k2C + 2*k3C + k4C)
    B_next = B + (h/6.0)*(k1B + 2*k2B + 2*k3B + k4B)
    return B_next, C_next

def precompute_BC(n_grid=1024):
    """Precompute B(τ), C(τ) on [0,T] by RK4 with C(0)=B(0)=0."""
    taus = np.linspace(0.0, T, n_grid+1, dtype=np.float64)
    Btab = np.zeros_like(taus)
    Ctab = np.zeros_like(taus)
    B = 0.0; C = 0.0
    for i in range(n_grid):
        h = taus[i+1] - taus[i]
        B, C = _rk4_step(B, C, h)
        Btab[i+1] = B; Ctab[i+1] = C
    # torch buffers for fast GPU usage
    return torch.tensor(taus, device=device, dtype=torch.float64), \
           torch.tensor(Btab, device=device, dtype=torch.float64), \
           torch.tensor(Ctab, device=device, dtype=torch.float64)

@torch.no_grad()
def interp_BC(tau, taus, Btab, Ctab):
    """Linear interpolation of B(τ), C(τ) for given tau tensor (float32/64)."""
    tau = tau.to(dtype=torch.float64)
    # clamp tau into [0,T]
    tau = torch.clamp(tau, min=0.0, max=float(T))
    # find fractional index
    s = (tau / float(T)) * (taus.shape[0]-1)
    idx0 = torch.floor(s).long().clamp(max=taus.shape[0]-2)
    frac = (s - idx0.to(s.dtype)).unsqueeze(-1)  # [B,1]
    B0 = Btab[idx0]; B1 = Btab[idx0+1]
    C0 = Ctab[idx0]; C1 = Ctab[idx0+1]
    B = (1.0-frac)*B0 + frac*B1
    C = (1.0-frac)*C0 + frac*C1
    return B.to(dtype=torch.float32), C.to(dtype=torch.float32)

class ClosedFormPolicy(nn.Module):
    """Closed-form optimal policy π*(τ,Y) using precomputed B(τ), C(τ)."""
    def __init__(self, taus, Btab, Ctab):
        super().__init__()
        # register as buffers so they move with .to(device)
        self.register_buffer('taus', taus)
        self.register_buffer('Btab', Btab)
        self.register_buffer('Ctab', Ctab)

    def forward(self, W, TmT, Y):
        B, C = interp_BC(TmT, self.taus, self.Btab, self.Ctab)  # each [B,1]
        # π* = (1/γ)[ (α/σ) Y + (ρ σ_Y / σ) (B + C Y) ]
        pi = (1.0/gamma) * ( (alpha/sigma)*Y + (rho*sigmaY/sigma)*(B + C*Y) )
        return torch.clamp(pi, -pi_cap, pi_cap)

# ----------------------------- Train ---------------------------
def train_stage1_base():
    policy = DirectPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        loss = -simulate(policy, batch_size, train=True).mean()  # maximize E[U]
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            with torch.no_grad():
                U_est = simulate(policy, batch_size, train=False).mean().item()
            print(f"[{ep:04d}] loss={loss.item():.6f}  E[U]_policy={U_est:.6f}")
    return policy

# ------------------------- Comparisons -------------------------
@torch.no_grad()
def compare_policy_functions(trained_policy: nn.Module, cf_policy: nn.Module):
    W = torch.empty(N_eval_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(N_eval_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)  # compare at start-of-horizon

    pi_learn = trained_policy(W, TmT, Y)
    pi_cf    = cf_policy(W, TmT, Y)
    rmse = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()

    print(f"[Policy RMSE] ||π_learn - π_closed-form||_RMSE over {N_eval_states} states: {rmse:.6f}")
    # few samples
    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_learn={pi_learn[i].item():.4f}, π_cf={pi_cf[i].item():.4f})")

@torch.no_grad()
def compare_expected_utility(trained_policy: nn.Module, cf_policy: nn.Module):
    # common random numbers
    W0, Y0 = sample_initial_states(N_eval_paths)
    dBW, dBY = generate_noise(N_eval_paths)

    U_learn = simulate_with_noise(trained_policy, W0, Y0, dBW, dBY)
    U_cf    = simulate_with_noise(cf_policy,     W0, Y0, dBW, dBY)

    EU_learn = U_learn.mean().item()
    EU_cf    = U_cf.mean().item()
    std_learn = U_learn.std(unbiased=True).item()
    std_cf    = U_cf.std(unbiased=True).item()

    print(f"[EU]  E[U]_learn = {EU_learn:.6f}  (std≈{std_learn:.6f})")
    print(f"      E[U]_closed-form = {EU_cf:.6f}  (std≈{std_cf:.6f})")
    print(f"      Δ (learn - closed-form) = {EU_learn - EU_cf:.6f}")

# ------------------------------ Run ----------------------------
def main():
    # Precompute closed-form B(τ), C(τ) and build policy
    taus, Btab, Ctab = precompute_BC(n_grid=2048)
    cf_policy = ClosedFormPolicy(taus.to(device), Btab.to(device), Ctab.to(device)).to(device)

    # Train Stage-1 policy
    policy = train_stage1_base()
    with torch.no_grad():
        U_final = simulate(policy, batch_size, train=False).mean().item()
    print(f"[After Train] E[U] (BASE policy): {U_final:.6f}")

    # 1) Policy-function RMSE @ τ=T
    compare_policy_functions(policy, cf_policy)

    # 2) Expected-utility comparison under common random numbers
    compare_expected_utility(policy, cf_policy)

if __name__ == "__main__":
    main()
