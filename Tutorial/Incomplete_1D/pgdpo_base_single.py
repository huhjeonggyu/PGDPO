import math, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from closed_form_ref import precompute_BC, ClosedFormPolicy

# --------------------------- Config ---------------------------
seed = 7
torch.manual_seed(seed); np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
T = 10.0
m = 50
batch_size = 1024
W0_range = (0.1, 2.0)
Y0_range = (-0.5, 1.5)
lb_W = 1e-3

# Training
epochs = 100
lr = 1e-3
pi_cap = 2.0

# Evaluation
N_eval_states = 1000
CRN_SEED_EU = 24680

# Closed-form mode (both scripts share)
CF_MODE = "X"  # KO X-mode ODE then map to Y

# --------------------------- Policy ---------------------------
class DirectPolicy(nn.Module):
    """Direct policy: (W, TmT, Y) -> pi."""
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
        x = torch.cat([W, TmT, Y], dim=1)
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

# ----------------------- Sampling utils -----------------------
T0_range = (0.0, T)  # tau in [0, T]

def sample_TmT(B, rng: torch.Generator | None = None):
    # tau ~ U[0, T]
    return torch.rand((B, 1), device=device, generator=rng) * (T0_range[1] - T0_range[0]) + T0_range[0]

def sample_initial_states(B, *, rng: torch.Generator | None = None):
    """
    Return (W0, Y0, TmT0, dt_vec) with per-path tau sampling.
    W0 ~ U[W0_range], Y0 ~ U[Y0_range], tau ~ U[0, T], dt_i = tau/m
    """
    W0_lo, W0_hi = W0_range
    Y0_lo, Y0_hi = Y0_range

    # torch.rand(..., generator=rng) 는 generator를 지원합니다.
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
    m_steps: int = m,
):
    """
    Return pathwise CRRA utility U(W_T) for B paths.
    By DEFAULT this simulator uses domain-time sampling:
      - If W0/Y0/Tval are not provided, it samples (W0, Y0, TmT0) and sets dt_i = TmT0/m.
      - If Tval is provided, per-path dt defaults to (Tval/m).
    You can override the step size via per_path_dt=[B,1].
    - seed / rng: reproducible randomness 
    """
    gen = rng if rng is not None else make_generator(seed)

    # init states
    if W0 is None or Y0 is None:
        W, Y, TmT, dt_vec = sample_initial_states(B, rng=gen)
    else:
        W, Y = W0, Y0
        if Tval is None:
            TmT = sample_TmT(B, rng=gen)
            dt_vec = TmT / float(m_steps)
        else:
            if torch.is_tensor(Tval):
                TmT = Tval.to(device=W.device, dtype=W.dtype)
                if TmT.shape != W.shape:
                    if TmT.numel() == 1:
                        TmT = TmT.expand_as(W)
                    else:
                        TmT = TmT.view_as(W)
            else:
                TmT = torch.full_like(W, float(Tval))
            dt_vec = TmT / float(m_steps)

    if per_path_dt is not None:
        dt_vec = per_path_dt.to(device=W.device, dtype=W.dtype)

    logW = W.clamp(min=lb_W).log()

    for _ in range(int(m_steps)):
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)  # [B,1]

        # dynamics
        risk_prem = sigma * (alpha * Y)               # mu - r
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2

        zW, zY = correlated_normals(W.shape[0], rho, gen=gen)
        dBW = dt_vec.sqrt() * zW
        dBY = dt_vec.sqrt() * zY

        # log-wealth update (Ito for geometric-like wealth with control)
        logW = logW + (driftW - 0.5*varW) * dt_vec + (pi_t * sigma) * dBW
        # factor OU
        Y    = Y    + kappaY*(thetaY - Y)*dt_vec + sigmaY * dBY

        # stabilize and step time
        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt_vec

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0)<1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U  # [B,1]

# ---------------- Closed-form: builder ----------------
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

# ----------------------------- Train (no E[U] logs) ------------
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
            # 평균효용 출력 제거 요청 반영: loss만 출력
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

# ------------------------- Common runner ------------------------
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

# -------- Public API for import --------
__all__ = [
    # config
    "device","r","gamma","sigma","kappaY","thetaY","sigmaY","rho","alpha",
    "T","m","batch_size","W0_range","Y0_range","lb_W","epochs","lr","pi_cap","CF_MODE",
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
