# pgdpo_richardson_single.py
# Version 4 (updated):
# - Single asset, single factor
# - Euler–Maruyama with Richardson extrapolation (dt vs dt/2 two-step)
# - Stage-1 direct policy optimization
# - Stage-2 P-PGDPO projection (costates via BPTT)
# - Adds closed-form (Kim–Omberg-style) policy via Riccati ODE (RK4)
# - Comparisons:
#     (i) Policy-function RMSEs: Stage-1 vs CF, P-PGDPO vs CF, Stage-1 vs P-PGDPO
#     (ii) Expected utility under common random numbers (with Richardson simulator): Stage-1 vs CF
#         [P-PGDPO EU optional & heavy]
# - No antithetic sampling, no control variates, no residual learning split
# Requirements: torch, numpy
#
# Usage:
#   python pgdpo_richardson_single.py

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

# P-PGDPO costate estimation
eval_repeats = 256       # MC repeats per evaluation state
sub_repeat = 128         # chunk size to limit memory

# Evaluation
N_eval_states = 200
N_eval_paths  = 5000     # MC paths for EU comparison (common random numbers)
EVAL_PPGDPO_EU = False   # WARNING: very heavy if True (costates each step)

# --------------------------- Policy ---------------------------
class DirectPolicy(nn.Module):
    """Direct policy: (W, TmT, Y) -> pi. (No residual/myopic split.)"""
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

def correlated_2d_normals(B, rho):
    # Cholesky for [[1, rho], [rho, 1]]
    a = math.sqrt(max(1.0 - rho*rho, 0.0))
    z1 = torch.randn(B, 1, device=device)
    z2 = torch.randn(B, 1, device=device)
    zW = z1
    zY = rho * z1 + a * z2
    return zW, zY

def generate_noise_pairs(B):
    """
    Pre-generate (m,B,1) standard normals for Richardson:
      z_base_W, z_base_Y, z_noise_W, z_noise_Y
    Coarse increment uses z_base; fine half-steps use (z_base ± z_noise)/sqrt(2).
    """
    zW_base = torch.empty(m, B, 1, device=device)
    zY_base = torch.empty(m, B, 1, device=device)
    zW_noise = torch.empty(m, B, 1, device=device)
    zY_noise = torch.empty(m, B, 1, device=device)
    for t in range(m):
        zbw, zby = correlated_2d_normals(B, rho)
        znw, zny = correlated_2d_normals(B, rho)
        zW_base[t], zY_base[t] = zbw, zby
        zW_noise[t], zY_noise[t] = znw, zny
    return zW_base, zY_base, zW_noise, zY_noise

# ----------------------- Simulator (Richardson) ----------------
def simulate_richardson(policy_module: nn.Module, B: int, train=True, W0=None, Y0=None, Tval=None):
    """Return pathwise CRRA utility U(W_T) for B paths using Richardson extrapolation."""
    if W0 is None or Y0 is None:
        W, Y = sample_initial_states(B)
        TmT = torch.full_like(W, T)
    else:
        W, Y = W0, Y0
        TmT = torch.full_like(W, T if Tval is None else Tval)

    logW = W.clamp(min=lb_W).log()

    invsqrt2 = 1.0 / math.sqrt(2.0)
    half = 0.5 * dt

    for _ in range(m):
        # policy at t
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)

        risk_prem_t = sigma * (alpha * Y)
        driftW_t = r + pi_t * risk_prem_t
        varW_t   = (pi_t * sigma)**2

        # noises: base and extra to form two half steps
        zW_base, zY_base   = correlated_2d_normals(W.size(0), rho)
        zW_noise, zY_noise = correlated_2d_normals(W.size(0), rho)

        # coarse increment
        dBW_coarse = math.sqrt(dt) * zW_base
        dBY_coarse = math.sqrt(dt) * zY_base

        # fine increments (two halves, preserving covariance)
        zW_half1 = (zW_base + zW_noise) * invsqrt2
        zY_half1 = (zY_base + zY_noise) * invsqrt2
        zW_half2 = (zW_base - zW_noise) * invsqrt2
        zY_half2 = (zY_base - zY_noise) * invsqrt2

        dBW_half1 = math.sqrt(half) * zW_half1
        dBY_half1 = math.sqrt(half) * zY_half1
        dBW_half2 = math.sqrt(half) * zW_half2
        dBY_half2 = math.sqrt(half) * zY_half2

        # coarse step: t -> t+dt
        logW_coarse = logW + (driftW_t - 0.5*varW_t) * dt + (pi_t * sigma) * dBW_coarse
        Y_coarse    = Y    + kappaY*(thetaY - Y) * dt   + sigmaY * dBY_coarse

        # fine step 1: t -> t+dt/2
        logW_half = logW + (driftW_t - 0.5*varW_t) * half + (pi_t * sigma) * dBW_half1
        Y_half    = Y    + kappaY*(thetaY - Y) * half    + sigmaY * dBY_half1

        # policy at mid-step
        with torch.set_grad_enabled(train):
            pi_half = policy_module(logW_half.exp(), TmT - half, Y_half)

        risk_prem_half = sigma * (alpha * Y_half)
        driftW_half = r + pi_half * risk_prem_half
        varW_half   = (pi_half * sigma)**2

        # fine step 2: t+dt/2 -> t+dt
        logW_fine = logW_half + (driftW_half - 0.5*varW_half) * half + (pi_half * sigma) * dBW_half2
        Y_fine    = Y_half    + kappaY*(thetaY - Y_half) * half      + sigmaY * dBY_half2

        # Richardson extrapolation
        logW = 2.0 * logW_fine - logW_coarse
        Y    = 2.0 * Y_fine    - Y_coarse

        # stabilize and step time
        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0)<1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U  # [B,1]

@torch.no_grad()
def simulate_richardson_with_noise(policy_module: nn.Module, W0, Y0, zW_base, zY_base, zW_noise, zY_noise):
    """
    Richardson simulation with pre-generated base/noise normals:
      coarse: dB = sqrt(dt) * z_base
      fine halves: dB_half{1,2} = sqrt(dt/2) * (z_base ± z_noise)/sqrt(2)
    """
    B = W0.shape[0]
    TmT = torch.full_like(W0, T)
    logW = W0.clamp(min=lb_W).log()
    Y = Y0.clone()

    invsqrt2 = 1.0 / math.sqrt(2.0)
    half = 0.5 * dt

    for t in range(m):
        pi_t = policy_module(logW.exp(), TmT, Y)

        risk_prem_t = sigma * (alpha * Y)
        driftW_t = r + pi_t * risk_prem_t
        varW_t   = (pi_t * sigma)**2

        dBW_coarse = math.sqrt(dt) * zW_base[t]
        dBY_coarse = math.sqrt(dt) * zY_base[t]

        zW_half1 = (zW_base[t] + zW_noise[t]) * invsqrt2
        zY_half1 = (zY_base[t] + zY_noise[t]) * invsqrt2
        zW_half2 = (zW_base[t] - zW_noise[t]) * invsqrt2
        zY_half2 = (zY_base[t] - zY_noise[t]) * invsqrt2

        dBW_half1 = math.sqrt(half) * zW_half1
        dBY_half1 = math.sqrt(half) * zY_half1
        dBW_half2 = math.sqrt(half) * zW_half2
        dBY_half2 = math.sqrt(half) * zY_half2

        # coarse
        logW_coarse = logW + (driftW_t - 0.5*varW_t) * dt + (pi_t * sigma) * dBW_coarse
        Y_coarse    = Y    + kappaY*(thetaY - Y) * dt   + sigmaY * dBY_coarse

        # fine half 1
        logW_half = logW + (driftW_t - 0.5*varW_t) * half + (pi_t * sigma) * dBW_half1
        Y_half    = Y    + kappaY*(thetaY - Y) * half    + sigmaY * dBY_half1

        # mid policy
        pi_half = policy_module(logW_half.exp(), TmT - half, Y_half)

        risk_prem_half = sigma * (alpha * Y_half)
        driftW_half = r + pi_half * risk_prem_half
        varW_half   = (pi_half * sigma)**2

        # fine half 2
        logW_fine = logW_half + (driftW_half - 0.5*varW_half) * half + (pi_half * sigma) * dBW_half2
        Y_fine    = Y_half    + kappaY*(thetaY - Y_half) * half      + sigmaY * dBY_half2

        # extrapolation
        logW = 2.0 * logW_fine - logW_coarse
        Y    = 2.0 * Y_fine    - Y_coarse

        # stabilize and time step
        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0)<1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U

# -------------------- Stage 1: Train policy -------------------
def train_stage1():
    policy = DirectPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        loss = -simulate_richardson(policy, batch_size, train=True).mean()  # maximize E[U]
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            with torch.no_grad():
                U_est = simulate_richardson(policy, batch_size, train=False).mean().item()
            print(f"[{ep:04d}] loss={loss.item():.6f}  E[U]_policy={U_est:.6f}")
    return policy

# ------------- Stage 2: costates via BPTT (Richardson) --------
def estimate_costates(policy_for_sim: nn.Module, W_pts: torch.Tensor, Y_pts: torch.Tensor,
                      repeats=eval_repeats, sub_batch=sub_repeat):
    """
    Returns lambda_hat=dU/dW0, dlam_dW=d^2U/dW0^2, dlam_dY=d^2U/(dW0 dY0) for each eval state.
    Uses the Richardson simulator for variance and bias reduction.
    """
    n = W_pts.size(0)
    lam_sum = torch.zeros_like(W_pts)
    dW_sum  = torch.zeros_like(W_pts)
    dY_sum  = torch.zeros_like(Y_pts)

    done = 0
    while done < repeats:
        rpts = min(sub_batch, repeats - done)
        # Tile states
        W0 = W_pts.detach().repeat(rpts, 1).clone().to(device).requires_grad_(True)
        Y0 = Y_pts.detach().repeat(rpts, 1).clone().to(device).requires_grad_(True)

        U = simulate_richardson(policy_for_sim, W0.size(0), train=True, W0=W0, Y0=Y0)
        U_avg = U.view(rpts, n, 1).mean(dim=0)  # [n,1]

        # First-order costate
        lam, = torch.autograd.grad(U_avg.sum(), W0, create_graph=True, retain_graph=True)
        lam = lam.view(rpts, n, 1).mean(dim=0)  # [n,1]

        # Second-order
        dlamW, dlamY = torch.autograd.grad(lam.sum(), (W0, Y0), retain_graph=False)
        dlamW = dlamW.view(rpts, n, 1).mean(dim=0)
        dlamY = dlamY.view(rpts, n, 1).mean(dim=0)

        lam_sum += lam.detach()
        dW_sum  += dlamW.detach()
        dY_sum  += dlamY.detach()
        done += rpts

    inv = 1.0 / repeats
    return lam_sum*inv, dW_sum*inv, dY_sum*inv  # each [n,1]

# -------- Stage 2: Projected policy (single-asset PMP) --------
def project_pmp(lambda_hat, dlamW_hat, dlamY_hat, W, Y):
    """
    Single-asset PMP projection:
    pi = - (lambda_hat / (W * dlamW_hat)) * (mu - r)/sigma^2
         - (1.0 / (W * dlamW_hat)) * ( (sigma * rho * sigmaY) * dlamY_hat ) / sigma^2
    """
    mu_minus_r = sigma * (alpha * Y)                    # [n,1]
    coeff = -1.0 / (W * dlamW_hat + 1e-8)              # stabilize denom
    myo   = coeff * (lambda_hat * (mu_minus_r / (sigma**2)))
    hedge = coeff * ((sigma * rho * sigmaY) * dlamY_hat / (sigma**2))
    pi = myo + hedge
    return torch.clamp(pi, -pi_cap, pi_cap)

class PPGDPOPolicy(nn.Module):
    """On-the-fly projection using costates estimated from the trained Stage-1 policy."""
    def __init__(self, trained_policy: nn.Module, repeats=eval_repeats, sub_batch=sub_repeat):
        super().__init__()
        self.trained = trained_policy
        self.repeats = repeats
        self.sub_batch = sub_batch

    def forward(self, W, TmT, Y):
        lam, dlamW, dlamY = estimate_costates(self.trained, W.detach(), Y.detach(),
                                              repeats=self.repeats, sub_batch=self.sub_batch)
        return project_pmp(lam, dlamW, dlamY, W.detach(), Y.detach())

# ---------------- Closed-form B(τ), C(τ) via Riccati (RK4) ----------------
# V = x^{1-γ}/(1-γ) * exp(A(τ) + B(τ) y + 0.5 C(τ) y^2)
# ODEs (τ=time-to-go) with C(0)=B(0)=0:
#   C' = 2κ C - σ_Y^2 C^2 - ((1-γ)/γ) (α + σ_Y ρ C)^2
#   B' = κ B - κ θ C - σ_Y^2 B C - ((1-γ)/γ) B (σ_Y ρ) (α + σ_Y ρ C)
# Optimal policy:
#   π_cf(τ,y) = (1/γ)[ (α/σ) y + (ρ σ_Y / σ) (B(τ) + C(τ) y) ]

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
    return torch.tensor(taus, device=device, dtype=torch.float64), \
           torch.tensor(Btab, device=device, dtype=torch.float64), \
           torch.tensor(Ctab, device=device, dtype=torch.float64)

@torch.no_grad()
def interp_BC(tau, taus, Btab, Ctab):
    """Linear interpolation of B(τ), C(τ) for given tau tensor."""
    tau = tau.to(dtype=torch.float64)
    tau = torch.clamp(tau, min=0.0, max=float(T))
    s = (tau / float(T)) * (taus.shape[0]-1)
    idx0 = torch.floor(s).long().clamp(max=taus.shape[0]-2)
    frac = (s - idx0.to(s.dtype)).unsqueeze(-1)
    B0 = Btab[idx0]; B1 = Btab[idx0+1]
    C0 = Ctab[idx0]; C1 = Ctab[idx0+1]
    B = (1.0-frac)*B0 + frac*B1
    C = (1.0-frac)*C0 + frac*C1
    return B.to(dtype=torch.float32), C.to(dtype=torch.float32)

class ClosedFormPolicy(nn.Module):
    """Closed-form optimal policy π*(τ,Y) using precomputed B(τ), C(τ)."""
    def __init__(self, taus, Btab, Ctab):
        super().__init__()
        self.register_buffer('taus', taus)
        self.register_buffer('Btab', Btab)
        self.register_buffer('Ctab', Ctab)

    def forward(self, W, TmT, Y):
        B, C = interp_BC(TmT, self.taus, self.Btab, self.Ctab)  # [B,1]
        pi = (1.0/gamma) * ( (alpha/sigma)*Y + (rho*sigmaY/sigma)*(B + C*Y) )
        return torch.clamp(pi, -pi_cap, pi_cap)

# ------------------------- Comparisons -------------------------
@torch.no_grad()
def compare_policy_functions(stage1_policy: nn.Module, proj_policy: nn.Module, cf_policy: nn.Module):
    W = torch.empty(N_eval_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(N_eval_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)

    pi_s1 = stage1_policy(W, TmT, Y)
    pi_pp = proj_policy(W, TmT, Y)
    pi_cf = cf_policy(W, TmT, Y)

    rmse_s1_cf = torch.sqrt(((pi_s1 - pi_cf)**2).mean()).item()
    rmse_pp_cf = torch.sqrt(((pi_pp - pi_cf)**2).mean()).item()
    rmse_s1_pp = torch.sqrt(((pi_s1 - pi_pp)**2).mean()).item()

    print(f"[Policy RMSE] Stage-1 vs CF:      {rmse_s1_cf:.6f}")
    print(f"[Policy RMSE] P-PGDPO vs CF:      {rmse_pp_cf:.6f}")
    print(f"[Policy RMSE] Stage-1 vs P-PGDPO: {rmse_s1_pp:.6f}")

    # few samples
    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_s1={pi_s1[i].item():.4f}, π_pp={pi_pp[i].item():.4f}, π_cf={pi_cf[i].item():.4f})")

@torch.no_grad()
def compare_expected_utility(stage1_policy: nn.Module, cf_policy: nn.Module, proj_policy: nn.Module=None):
    # common random numbers compatible with Richardson
    zW_base, zY_base, zW_noise, zY_noise = generate_noise_pairs(N_eval_paths)
    W0, Y0 = sample_initial_states(N_eval_paths)

    U_s1 = simulate_richardson_with_noise(stage1_policy, W0, Y0, zW_base, zY_base, zW_noise, zY_noise)
    U_cf = simulate_richardson_with_noise(cf_policy,     W0, Y0, zW_base, zY_base, zW_noise, zY_noise)

    EU_s1 = U_s1.mean().item()
    EU_cf = U_cf.mean().item()
    print(f"[EU]  E[U]_Stage-1 (Richardson) = {EU_s1:.6f}")
    print(f"      E[U]_Closed-form (Richardson) = {EU_cf:.6f}")
    print(f"      Δ (Stage-1 - Closed-form) = {EU_s1 - EU_cf:.6f}")

    if EVAL_PPGDPO_EU and proj_policy is not None:
        # WARNING: extremely heavy — costates each step
        U_pp = simulate_richardson_with_noise(proj_policy, W0, Y0, zW_base, zY_base, zW_noise, zY_noise)
        EU_pp = U_pp.mean().item()
        print(f"[EU]  E[U]_P-PGDPO (Richardson) = {EU_pp:.6f}")
        print(f"      Δ (P-PGDPO - Closed-form) = {EU_pp - EU_cf:.6f}")

# ------------------------------ Run ---------------------------
def main():
    # Closed-form policy precompute
    taus, Btab, Ctab = precompute_BC(n_grid=2048)
    cf_policy = ClosedFormPolicy(taus.to(device), Btab.to(device), Ctab.to(device)).to(device)

    # Stage 1: train direct policy with Richardson simulator
    stage1 = train_stage1()
    with torch.no_grad():
        U_s1 = simulate_richardson(stage1, batch_size, train=False).mean().item()
    print(f"[After Train] E[U] Stage-1 policy (Richardson): {U_s1:.6f}")

    # Stage 2: projector using Richardson costates
    projector = PPGDPOPolicy(stage1, repeats=eval_repeats, sub_batch=sub_repeat).to(device)

    # 1) Policy-function RMSE comparisons
    compare_policy_functions(stage1, projector, cf_policy)

    # 2) Expected-utility comparisons (common random numbers compatible with Richardson)
    compare_expected_utility(stage1, cf_policy, projector if EVAL_PPGDPO_EU else None)

if __name__ == "__main__":
    main()
