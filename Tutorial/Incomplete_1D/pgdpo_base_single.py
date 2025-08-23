# pgdpo_base_single.py
import math, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from closed_form_ref import precompute_BC, ClosedFormPolicy

# --------------------------- Config ---------------------------
seed = 7
torch.manual_seed(seed); np.random.seed(seed)
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# Market & utility
r = 0.02
gamma = 5.0
sigma = 0.3
kappaY = 1.2
thetaY = 0.2
sigmaY = 0.3
rho = 0.3
alpha = 1.0  # mu - r = sigma * (alpha * Y)

# Simulation (default steps; per-path dt will be tau/m)
T = 1.5
m = 20

# Ranges
W0_range = (0.1, 3.0)
Y0_range = (-1.0, 1.0)

# Caps & floors
pi_cap = 1.5
lb_W   = 1e-5

# Training hyperparams
epochs     = 100
batch_size = 1024
lr         = 3e-4

# Eval
N_eval_states = 3000
CRN_SEED_EU   = 12345

# Closed-form mode (both scripts share)
CF_MODE = "X"  # KO X-mode ODE then map to Y

# --------------------------- Policy ---------------------------
class DirectPolicy(nn.Module):
    """Direct policy: (W, TmT, Y) -> pi, inputs are [B,1] each."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        for mmod in self.net.modules():
            if isinstance(mmod, nn.Linear):
                nn.init.xavier_uniform_(mmod.weight, gain=0.8)
                nn.init.zeros_(mmod.bias)

    def forward(self, W, TmT, Y):
        x = torch.cat([W, TmT, Y], dim=1)  # [B,3]
        pi = self.net(x)
        return torch.clamp(torch.tanh(pi), -pi_cap, pi_cap)

# ----------------------- RNG helper -----------------------
def make_generator(seed_val: int | None):
    """Create a torch.Generator on the current device with a given seed."""
    if seed_val is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed_val))
    return g

# ----------------------- Samplers --------------------------
def sample_TmT(B: int, *, rng: torch.Generator | None = None):
    # Uniform(0,T)
    return torch.rand(B, 1, device=device, generator=rng) * T

def sample_initial_states(B: int, *, rng: torch.Generator | None = None):
    W0_lo, W0_hi = W0_range
    Y0_lo, Y0_hi = Y0_range

    W0 = torch.rand((B, 1), device=device, generator=rng) * (W0_hi - W0_lo) + W0_lo
    Y0 = torch.rand((B, 1), device=device, generator=rng) * (Y0_hi - Y0_lo) + Y0_lo

    TmT0 = sample_TmT(B, rng=rng)          # [B,1]
    dt_vec = TmT0 / float(m)               # [B,1], per-path Δt

    return W0, Y0, TmT0, dt_vec

def correlated_normals(B, rho, gen: torch.Generator | None = None):
    """Return correlated N(0,1) draws using optional generator."""
    z1 = torch.randn(B, 1, device=device, generator=gen)
    z2 = torch.randn(B, 1, device=device, generator=gen)
    zW = z1
    zY = rho * z1 + math.sqrt(max(1.0 - rho*rho, 0.0)) * z2
    return zW, zY

# ----------------------- Simulator (Euler) ---------------------
def simulate(
    policy_module: nn.Module,
    B: int,
    train: bool = True,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: float | torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    seed: int | None = None,
    per_path_dt: torch.Tensor | None = None,
):
    """
    Domain-time sampling (random tau per path), Euler–Maruyama in log-space.
    Returns utility [B,1].
    """
    if W0 is None or Y0 is None or Tval is None:
        W, Y, TmT, dt_vec = sample_initial_states(B, rng=rng)
    else:
        # shape harmonization
        W = W0 if W0.ndim == 2 else W0.unsqueeze(1)
        Y = Y0 if Y0.ndim == 2 else Y0.unsqueeze(1)
        # --- Updated scalar-handling for Tval ---
        if isinstance(Tval, torch.Tensor):
            TmT = Tval if Tval.ndim == 2 else Tval.unsqueeze(1)
        elif Tval is not None:
            TmT = torch.full((B,1), float(Tval), device=device)
        else:
            TmT = sample_TmT(B, rng=rng)

        dt_vec = TmT / float(m) if per_path_dt is None else per_path_dt

    # antithetic off here (base simulate) – antithetic/Richardson are used in (run/richardson) modules
    gen_local = rng if rng is not None else (make_generator(seed) if seed is not None else None)

    # draw all normals upfront (vectorized)
    ZW = torch.randn(B, m, device=device, generator=gen_local)
    ZY = rho*ZW + math.sqrt(max(1.0 - rho*rho, 0.0)) * torch.randn(B, m, device=device, generator=gen_local)

    logW = W.clamp_min(lb_W).log()   # [B,1]
    Ycur = Y.clone()                 # [B,1]
    dt   = dt_vec                    # [B,1]

    for k in range(m):
        pi = policy_module(logW.exp(), TmT - k*dt, Ycur)  # [B,1]
        dBW = ZW[:, k:k+1] * dt.sqrt()                    # [B,1]
        dBY = ZY[:, k:k+1] * dt.sqrt()

        drift = r + pi * sigma * alpha * Ycur
        vol   = pi * sigma
        logW  = logW + (drift - 0.5 * vol*vol) * dt + vol * dBW
        Ycur  = Ycur + kappaY * (thetaY - Ycur) * dt + sigmaY * dBY

    WT = logW.exp().clamp_min(lb_W)
    # CRRA utility
    if abs(gamma - 1.0) < 1e-8:
        U = torch.log(WT.clamp(min=1e-12))
    else:
        U = (WT.clamp(min=1e-12).pow(1.0 - gamma) - 1.0) / (1.0 - gamma)
    return U  # [B,1]

# ------------------------- Closed-form builder ------------------
def build_closed_form_policy(mode: str = CF_MODE):
    taus, Btab, Ctab = precompute_BC(
        T=T, kappaY=kappaY, thetaY=thetaY, sigmaY=sigmaY,
        gamma=gamma, rho=rho, alpha=alpha,
        n_grid=2048, mode=mode, device=device
    )
    cf_policy = ClosedFormPolicy(
        taus, Btab, Ctab,
        sigma=sigma, sigmaY=sigmaY, alpha=alpha, rho=rho, gamma=gamma, T=T,
        pi_cap=pi_cap
    ).to(device)
    return cf_policy, (taus, Btab, Ctab)

# ----------------------------- Train (log only loss) ------------
def train_stage1_base(epochs_override=None, lr_override=None, seed_train: int | None = None):
    _epochs = epochs if epochs_override is None else int(epochs_override)
    _lr = lr if lr_override is None else float(lr_override)

    policy = DirectPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=_lr)

    gen = make_generator(seed_train)

    for ep in range(1, _epochs+1):
        opt.zero_grad()
        # DEFAULT: domain-time sampling (random tau per path)
        loss = -simulate(policy, batch_size, train=True, rng=gen).mean()  # maximize E[U]
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            # 평균효용만 출력
            print(f"[{ep:04d}] loss={loss.item():.6f}")
    return policy

# ------------------------- RMSE comparison ----------------------
@torch.no_grad()
def compare_policy_functions(trained_policy: nn.Module, cf_policy: nn.Module, seed_eval: int | None = None):
    gen = make_generator(seed_eval)
    W, Y, TmT, _dt = sample_initial_states(N_eval_states, rng=gen)
    pi_learn = trained_policy(W, TmT, Y)
    pi_cf    = cf_policy(W, TmT, Y)
    rmse = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
    print(f"[Policy RMSE] ||π_learn - π_closed-form||_RMSE over {N_eval_states} states: {rmse:.6f}")
    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_learn={pi_learn[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})")

# ------------------------------ Runner -------------------------
def run_common(train_fn, rmse_fn, *, seed_train=seed, rmse_kwargs=None):
    """
    Shared entry point:
      - Builds closed-form policy
      - Trains via train_fn(seed_train=...)
      - Evaluates via rmse_fn(policy, cf_policy, **rmse_kwargs)
    """
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)
    policy = train_fn(seed_train=seed_train)
    rmse_kwargs = {} if rmse_kwargs is None else dict(rmse_kwargs)
    rmse_fn(policy, cf_policy, **rmse_kwargs)

# ------------------------------ Run ----------------------------
def main():
    # 공용 러너 재활용: BASE 정책 학습 + RMSE 비교
    run_common(
        train_fn=train_stage1_base,
        rmse_fn=compare_policy_functions,
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU},
    )

if __name__ == "__main__":
    main()

__all__ = [
    # constants
    "device","r","gamma","sigma","kappaY","thetaY","sigmaY","rho","alpha","T","m",
    "W0_range","Y0_range","lb_W","pi_cap","CF_MODE","N_eval_states","CRN_SEED_EU","epochs",
    "batch_size","lr","seed",
    # classes
    "DirectPolicy",
    # rng helper
    "make_generator",
    # utils/sim
    "sample_TmT","sample_initial_states","correlated_normals","simulate",
    # closed-form builder
    "build_closed_form_policy",
    # training
    "train_stage1_base",
    # evaluation
    "compare_policy_functions",
    # runner
    "run_common",
]