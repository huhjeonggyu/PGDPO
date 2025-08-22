import math
import torch
import torch.nn as nn

# ===== (1) BASE: configs, CF builder, samplers, runner =====
from pgdpo_base_single import (
    # device & env params
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, CRN_SEED_EU, epochs,
    # rng + utils (domain-time sampling)
    make_generator, sample_initial_states,
    # closed-form builder
    build_closed_form_policy,
    # shared runner + base seed
    run_common, seed,
)

# ===== (2) P-PGDPO core: projection formula + eval hyperparams =====
from pgdpo_with_ppgdpo_single import (
    project_pmp, REPEATS, SUBBATCH
)

# ========================= Local correlated normals =========================
def _correlated_normals(B: int, rho: float, gen: torch.Generator | None):
    """Return (zW, zY) ~ N(0, Σ), Σ=[[1,rho],[rho,1]] using optional generator."""
    z1 = torch.randn(B, 1, device=device, generator=gen)
    z2 = torch.randn(B, 1, device=device, generator=gen)
    zW = z1
    zY = rho * z1 + math.sqrt(max(1.0 - rho*rho, 0.0)) * z2
    return zW, zY

# ========================= Domain-time simulator (TRUE antithetic) =========================
def _simulate_core_antithetic(
    policy_module: nn.Module,
    B: int,
    *,
    train: bool,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: torch.Tensor | None = None,   # [B,1] tau if provided
    seed_local: int | None = None,
    noise_sign: float = 1.0,            # +1 or -1 (true antithetic)
    m_steps: int = m,
):
    """
    Domain-time simulator (per-path tau, dt_i=tau/m) with TRUE antithetic flips.
    - If W0/Y0/Tval are None: sample (W0,Y0,tau) via sample_initial_states(...)
    - Else: use provided W0/Y0 and Tval, and set dt_vec = Tval/m_steps
    """
    gen = make_generator(seed_local)

    # init states (domain-time sampling)
    if W0 is None or Y0 is None or Tval is None:
        W, Y, TmT, dt_vec = sample_initial_states(B, rng=gen)  # [B,1] each (TmT=tau, dt_vec=tau/m)
    else:
        W, Y = W0, Y0
        TmT = Tval
        dt_vec = TmT / float(m_steps)

    logW = W.clamp(min=lb_W).log()

    for _ in range(int(m_steps)):
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)  # [B,1]

        # dynamics
        risk_prem = sigma * (alpha * Y)                # mu - r
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2

        zW, zY = _correlated_normals(W.size(0), rho, gen)
        dBW = noise_sign * (dt_vec.sqrt() * zW)
        dBY = noise_sign * (dt_vec.sqrt() * zY)

        # log-wealth update (Ito for geometric-like wealth with control)
        logW = logW + (driftW - 0.5*varW) * dt_vec + (pi_t * sigma) * dBW
        # factor OU
        Y    = Y    + kappaY*(thetaY - Y)*dt_vec + sigmaY * dBY

        # stabilize and step time
        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt_vec

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0) < 1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U  # [B,1]

def simulate_antithetic(
    policy_module: nn.Module, B: int, *,
    train: bool, W0=None, Y0=None, Tval=None, seed_local: int | None = None
):
    """
    True antithetic wrapper (domain-time). Same local seed, ±noise_sign, then average.
    """
    U_pos = _simulate_core_antithetic(policy_module, B, train=train, W0=W0, Y0=Y0, Tval=Tval,
                                      seed_local=seed_local, noise_sign=+1.0)
    U_neg = _simulate_core_antithetic(policy_module, B, train=train, W0=W0, Y0=Y0, Tval=Tval,
                                      seed_local=seed_local, noise_sign=-1.0)
    return 0.5 * (U_pos + U_neg)

# ========================= Antithetic costates (BPTT) =========================
def estimate_costates_antithetic(
    policy_net: nn.Module,
    T0: torch.Tensor, W0: torch.Tensor, Y0: torch.Tensor,
    *, repeats: int, sub_batch: int,
    seed_eval: int | None = None,   # 평가/CRN 시드에서 파생
):
    """
    Returns (λ=J_W, J_WW, J_WY) at (W0,Y0,T0) using TRUE antithetic:
    SAME seed with noise_sign=±1.0, then average. Independent pairs via seed++.
    Uses simulate_antithetic for clarity and correctness.
    """
    n = W0.size(0)
    W0g = W0.clone().requires_grad_(True)
    Y0g = Y0.clone().requires_grad_(True)

    lam_sum = torch.zeros_like(W0g)
    dW_sum  = torch.zeros_like(W0g)
    dY_sum  = torch.zeros_like(Y0g)

    # freeze params
    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    try:
        seed_ctr = int(seed_eval) if (seed_eval is not None) else None
        done = 0
        while done < repeats:
            rpts = min(sub_batch, repeats - done)
            T_b = T0.repeat(rpts, 1)
            W_b = W0g.repeat(rpts, 1)
            Y_b = Y0g.repeat(rpts, 1)

            U = simulate_antithetic(
                policy_net, n * rpts, train=True,
                W0=W_b, Y0=Y_b, Tval=T_b, seed_local=seed_ctr
            )
            seed_ctr = None if seed_ctr is None else seed_ctr + 1

            U_avg = U.view(rpts, n, 1).mean(dim=0)  # [n,1]

            (lam_b,) = torch.autograd.grad(U_avg.sum(), W0g, create_graph=True, retain_graph=True)
            lam_b = lam_b.view(n, 1)

            dlamW_b, dlamY_b = torch.autograd.grad(lam_b.sum(), (W0g, Y0g), retain_graph=False)

            lam_sum += lam_b.detach()
            dW_sum  += dlamW_b.detach()
            dY_sum  += dlamY_b.detach()
            done += rpts
    finally:
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)

    inv = 1.0 / repeats
    return lam_sum * inv, dW_sum * inv, dY_sum * inv

def ppgdpo_pi_antithetic(
    policy_s1: nn.Module,
    W: torch.Tensor, TmT: torch.Tensor, Y: torch.Tensor,
    *, repeats: int, sub_batch: int,
    seed_eval: int | None = None,
):
    """
    Antithetic costates + PMP projection -> π_pp(W, τ, Y). Always antithetic.
    """
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates_antithetic(
            policy_s1, TmT, W, Y,
            repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval
        )
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()

# ========================= Stage-1 training (always antithetic) =========================
def train_stage1_antithetic(epochs: int = epochs, lr: float = 1e-3, seed_train: int | None = seed):
    """
    Antithetic Stage-1 training using domain-time simulate_antithetic(...).
    Uses independent antithetic pairs per epoch by incrementing the local seed.
    """
    from pgdpo_base_single import DirectPolicy  # same net as base
    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        pair_seed = None if seed_train is None else int(seed_train) + ep  # vary across epochs
        U = simulate_antithetic(policy, batch_size, train=True, seed_local=pair_seed)
        loss = -U.mean()  # maximize E[U]
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")

    return policy

# ========================= Comparisons =========================
@torch.no_grad()
def print_policy_rmse_and_samples_antithetic(pol_s1: nn.Module, pol_cf: nn.Module,
                                             *, seed_eval: int | None = CRN_SEED_EU,
                                             repeats: int = REPEATS, sub_batch: int = SUBBATCH):
    # 도메인-시간 샘플링으로 평가 상태 통일 (CRN_SEED_EU 사용)
    gen = make_generator(seed_eval)
    W, Y, TmT, _dt = sample_initial_states(N_eval_states, rng=gen)

    pi_learn = pol_s1(W, TmT, Y)
    pi_cf    = pol_cf(W, TmT, Y)

    # Function-form projector (teacher)
    pi_pp = ppgdpo_pi_antithetic(pol_s1, W, TmT, Y,
                                 repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval)

    rmse_learn = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
    rmse_pp    = torch.sqrt(((pi_pp    - pi_cf)**2).mean()).item()
    rmse_l_pp  = torch.sqrt(((pi_learn - pi_pp)**2).mean()).item()

    print(f"[Policy RMSE] Stage-1 vs CF:      {rmse_learn:.6f}")
    print(f"[Policy RMSE] P-PGDPO(anti) vs CF:{rmse_pp:.6f}")
    print(f"[Policy RMSE] Stage-1 vs P-PGDPO: {rmse_l_pp:.6f}")

    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_s1={pi_learn[i,0].item():.4f}, π_pp(anti)={pi_pp[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})")

# ========================= Run (reuse common runner) =========================
def main():
    # 공용 러너 재활용: (항상 안티테틱) Stage-1 학습 + RMSE 비교
    run_common(
        train_fn=lambda seed_train=seed: train_stage1_antithetic(seed_train=seed_train),
        rmse_fn=print_policy_rmse_and_samples_antithetic,
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU, "repeats": REPEATS, "sub_batch": SUBBATCH},
    )

if __name__ == "__main__":
    main()

__all__ = [
    # simulate (with domain-time antithetic hook)
    "simulate_antithetic",
    # costates + function-form projector
    "estimate_costates_antithetic", "ppgdpo_pi_antithetic",
    # evaluation helpers
    "print_policy_rmse_and_samples_antithetic",
    # training
    "train_stage1_antithetic",
]