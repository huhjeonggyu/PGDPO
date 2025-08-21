import math
import torch
import torch.nn as nn

# ===== Import from (1) BASE: configs, simple training, CF builder, and state samplers =====
from pgdpo_base_single import (
    # device & env params
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, dt, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, N_eval_paths, CRN_SEED_EU, epochs,
    # utils
    sample_initial_states,
    # training (non-antithetic baseline)
    train_stage1_base,  # 남겨두되 사용은 안 함
    # closed-form builder
    build_closed_form_policy,
)

# ===== Import from (2) P-PGDPO core: projection formula only =====
from pgdpo_with_ppgdpo_single import project_pmp

# --------------------------- Hyperparams ---------------------------
SEED_TRAIN_BASE  = 12345    # training base seed; epoch offset으로 페어 변경
SEED_COSTATE_BASE = 98765   # costate 추정용 base seed
EVAL_REPEATS      = 256     # MC repeats per evaluation state
SUB_REPEAT        = 128     # chunk size to limit memory

# ========================= Local RNG & correlated draws =========================
def _make_generator(seed: int | None):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def _correlated_normals(B: int, rho: float, gen: torch.Generator | None):
    """Return (zW, zY) ~ N(0, Σ), Σ=[[1,rho],[rho,1]] using optional generator."""
    z1 = torch.randn(B, 1, device=device, generator=gen)
    z2 = torch.randn(B, 1, device=device, generator=gen)
    zW = z1
    zY = rho * z1 + math.sqrt(max(1.0 - rho*rho, 0.0)) * z2
    return zW, zY

# ========================= simulate (supports antithetic via noise_sign) =========================
def simulate(
    policy_module: nn.Module,
    B: int,
    train: bool = True,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: float | torch.Tensor | None = None,
    seed: int | None = None,
    noise_sign: float = 1.0,
):
    """
    Return pathwise CRRA utility U(W_T) for B paths.

    Features:
      - 'seed' for reproducibility (CRN across policies)
      - 'noise_sign' in {+1.0, -1.0} to form true antithetic pairs
        (flip BOTH zW and zY so the corr structure is preserved)
    """
    if W0 is None or Y0 is None:
        W, Y = sample_initial_states(B)
        TmT = torch.full_like(W, T)
    else:
        W, Y = W0, Y0
        if Tval is None:
            TmT = torch.full_like(W, T)
        elif torch.is_tensor(Tval):
            TmT = Tval.clone()
            if TmT.shape != W.shape:
                if TmT.numel() == 1:
                    TmT = TmT.expand_as(W)
                else:
                    TmT = TmT.view_as(W)
        else:
            TmT = torch.full_like(W, float(Tval))

    gen = _make_generator(seed)
    logW = W.clamp(min=lb_W).log()

    for _ in range(m):
        with torch.set_grad_enabled(train):
            pi_t = policy_module(logW.exp(), TmT, Y)  # [B,1]

        # dynamics
        risk_prem = sigma * (alpha * Y)    # μ - r = σ α Y
        driftW = r + pi_t * risk_prem
        varW   = (pi_t * sigma)**2

        zW, zY = _correlated_normals(W.size(0), rho, gen)
        dBW = noise_sign * (math.sqrt(dt) * zW)
        dBY = noise_sign * (math.sqrt(dt) * zY)

        logW = logW + (driftW - 0.5*varW) * dt + (pi_t * sigma) * dBW
        Y    = Y    + kappaY*(thetaY - Y)*dt + sigmaY * dBY

        logW = logW.exp().clamp(min=lb_W).log()
        TmT  = TmT - dt

    W_T = logW.exp()
    U = W_T.log() if abs(gamma-1.0) < 1e-8 else (W_T.pow(1.0-gamma))/(1.0-gamma)
    return U

# Convenience: exact antithetic average with same seed
def simulate_pair(
    policy_module: nn.Module, B: int, train=True, W0=None, Y0=None, Tval=None, seed: int | None = None
):
    U_pos = simulate(policy_module, B, train=train, W0=W0, Y0=Y0, Tval=Tval, seed=seed, noise_sign=+1.0)
    U_neg = simulate(policy_module, B, train=train, W0=W0, Y0=Y0, Tval=Tval, seed=seed, noise_sign=-1.0)
    return 0.5 * (U_pos + U_neg)

# ========================= Costates (always antithetic averaged) =========================
def estimate_costates(
    policy_net, T0, W0, Y0,
    repeats=EVAL_REPEATS, sub_batch=SUB_REPEAT,
    seed_base: int | None = SEED_COSTATE_BASE
):
    """
    Returns (λ=J_W, J_WW, J_WY) at (W0,Y0,T0), using TRUE antithetic averaging:
    SAME seed with noise_sign=±1.0, then average. Independent pairs via seed++.
    """
    n = W0.size(0)
    W0g = W0.clone().requires_grad_(True)
    Y0g = Y0.clone().requires_grad_(True)

    lam_sum = torch.zeros_like(W0g)
    dW_sum  = torch.zeros_like(W0g)
    dY_sum  = torch.zeros_like(Y0g)

    # turn OFF parameter grads (policy fixed for costates)
    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params: 
        p.requires_grad_(False)

    try:
        seed_ctr = int(seed_base) if (seed_base is not None) else None
        done = 0
        while done < repeats:
            rpts = min(sub_batch, repeats - done)
            T_b = T0.repeat(rpts, 1)
            W_b = W0g.repeat(rpts, 1)
            Y_b = Y0g.repeat(rpts, 1)

            if seed_ctr is not None:
                U_pos = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b,
                                 seed=seed_ctr, noise_sign=+1.0)
                U_neg = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b,
                                 seed=seed_ctr, noise_sign=-1.0)
                U = 0.5 * (U_pos + U_neg)
                seed_ctr += 1
            else:
                U1 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
                U2 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
                U = 0.5 * (U1 + U2)

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
    return lam_sum*inv, dW_sum*inv, dY_sum*inv

def ppgdpo_pi(
    policy_s1: nn.Module,
    W: torch.Tensor, TmT: torch.Tensor, Y: torch.Tensor,
    repeats: int = EVAL_REPEATS, sub_batch: int = SUB_REPEAT,
    seed_base: int | None = SEED_COSTATE_BASE,
):
    """
    Costates (antithetic averaged) + PMP projection -> π_pp(W, τ, Y)
    """
    # no_grad 문맥에서도 안전하게 state-grad만 켜기
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates(
            policy_s1, TmT, W, Y,
            repeats=repeats, sub_batch=sub_batch,
            seed_base=seed_base
        )
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()

# ========================= Stage-1 training (always antithetic) =========================
def train_stage1(epochs=300, lr=1e-3, seed_base: int = SEED_TRAIN_BASE):
    """
    Stage-1 training using exact antithetic pairs per epoch (seed increments).
    """
    from pgdpo_base_single import DirectPolicy  # same net as base
    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        pair_seed = None if seed_base is None else int(seed_base) + ep  # vary across epochs
        U_pos = simulate(policy, batch_size, train=True, seed=pair_seed, noise_sign=+1.0)
        U_neg = simulate(policy, batch_size, train=True, seed=pair_seed, noise_sign=-1.0)
        loss = -0.5 * (U_pos.mean() + U_neg.mean())  # maximize E[U]
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            with torch.no_grad():
                U_est = simulate_pair(policy, batch_size, train=False, seed=pair_seed).mean().item()
            print(f"[{ep:04d}] loss={loss.item():.6f}  E[U]_policy={U_est:.6f}")

    return policy

# ========================= Comparisons =========================
@torch.no_grad()
def print_policy_rmse_and_samples(pol_s1: nn.Module, pol_cf: nn.Module):
    W = torch.empty(N_eval_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(N_eval_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)

    pi_learn = pol_s1(W, TmT, Y)
    pi_cf    = pol_cf(W, TmT, Y)

    # Function-form projector (teacher)
    pi_pp = ppgdpo_pi(pol_s1, W, TmT, Y)

    rmse_learn = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
    rmse_pp    = torch.sqrt(((pi_pp    - pi_cf)**2).mean()).item()
    rmse_l_pp  = torch.sqrt(((pi_learn - pi_pp)**2).mean()).item()

    print(f"[Policy RMSE] Stage-1 vs CF:      {rmse_learn:.6f}")
    print(f"[Policy RMSE] P-PGDPO vs CF:      {rmse_pp:.6f}")
    print(f"[Policy RMSE] Stage-1 vs P-PGDPO: {rmse_l_pp:.6f}")

    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_s1={pi_learn[i,0].item():.4f}, π_pp={pi_pp[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})")

@torch.no_grad()
def compare_expected_utility(pol_s1: nn.Module, pol_cf: nn.Module, eval_seed: int = CRN_SEED_EU):
    # CRN via shared seed and identical initial states; always antithetic average
    W0, Y0 = sample_initial_states(N_eval_paths)
    U_s1 = simulate_pair(pol_s1, N_eval_paths, train=False, W0=W0, Y0=Y0, seed=eval_seed)
    U_cf = simulate_pair(pol_cf,   N_eval_paths, train=False, W0=W0, Y0=Y0, seed=eval_seed)

    EU_s1 = U_s1.mean().item()
    EU_cf = U_cf.mean().item()
    print(f"[EU]  E[U]_Stage-1 = {EU_s1:.6f}")
    print(f"      E[U]_Closed-form = {EU_cf:.6f}")
    print(f"      Δ (Stage-1 - Closed-form) = {EU_s1 - EU_cf:.6f}")

# ========================= Run =========================
def main():
    # 1) Closed-form from BASE
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)

    # 2) Stage-1 training (always antithetic)
    stage1 = train_stage1(epochs=epochs, lr=1e-3, seed_base=SEED_TRAIN_BASE)
    with torch.no_grad():
        U_s1 = simulate_pair(stage1, batch_size, train=False, seed=SEED_TRAIN_BASE+999).mean().item()
    print(f"[After Train] E[U] Stage-1 policy (antithetic eval): {U_s1:.6f}")

    # 3) Policy RMSEs (function-form projector)
    print_policy_rmse_and_samples(stage1, cf_policy)

    # 4) EU comparisons (CRNs, antithetic)
    compare_expected_utility(stage1, cf_policy)

if __name__ == "__main__":
    main()

__all__ = [
    # simulate (with antithetic hook support)
    "simulate", "simulate_pair",
    # costates + function-form projector
    "estimate_costates", "ppgdpo_pi",
    # evaluation helpers
    "print_policy_rmse_and_samples", "compare_expected_utility",
    # training
    "train_stage1",
]
