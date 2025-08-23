import math
import torch
import torch.nn as nn
import torch.optim as optim

from pgdpo_base_single import (
    # config & device
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, CRN_SEED_EU,
    # rng & env/sim
    make_generator, sample_initial_states, simulate,
    # stage-1 & closed-form
    train_stage1_base, build_closed_form_policy,
    # common runner
    run_common, seed,
)

# ------------------------------------------------------------------
# P-PGDPO evaluation hyperparameters (caller may override if imported)
REPEATS  = 2560
SUBBATCH = 2560 // 16

# ------------------------------------------------------------------

# ------------------ Costate estimation (direct) -------------------
def estimate_costates(policy_net: nn.Module,
                      T0: torch.Tensor, W0: torch.Tensor, Y0: torch.Tensor,
                      repeats: int, sub_batch: int):
    """
    Estimate λ=J_W, ∂λ/∂W=J_WW, ∂λ/∂Y=J_WY at (W0,Y0,T0) under policy_net.
    Memory-safe chunking:
      - per-chunk: replicate (W0,Y0,T0) r_chunk times -> simulate once ->
                   average over repeats per state -> take grads -> detach & accumulate
    Shapes:
      W0, Y0, T0 : [n, 1]  (n evaluation states)
    Returns:
      (λ_bar, J_WW_bar, J_WY_bar) with shape [n, 1]
    """
    assert W0.ndim == 2 and Y0.ndim == 2
    n = W0.size(0)

    # Make leafs
    W0g = W0.detach().clone().requires_grad_(True)
    Y0g = Y0.detach().clone().requires_grad_(True)
    T0g = T0.detach().clone()  # time is not a grad target

    # Accumulators (graph-free)
    lam_sum = torch.zeros_like(W0g)  # dJ/dW0
    dW_sum  = torch.zeros_like(W0g)  # d²J/dW0²
    dY_sum  = torch.zeros_like(Y0g)  # d²J/(dY0 dW0)

    # --- Freeze policy params to shrink graph and save memory ---
    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    try:
        done = 0
        while done < repeats:
            rpts = min(sub_batch, repeats - done)

            # replicate batch (rpts blocks): shapes [n*rpts, 1]
            W_b = W0g.repeat(rpts, 1)
            Y_b = Y0g.repeat(rpts, 1)
            T_b = T0g.repeat(rpts, 1)

            # Single simulate per chunk
            U = simulate(policy_net, n * rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)   # [n*rpts,1]

            # average per point (across rpts dim)
            U_bar = U.reshape(rpts, n, -1).mean(dim=0)    # [n,1]
            J     = U_bar.mean()                          # scalar

            # (1) dJ/dW0 : create_graph=True (we need second derivatives)
            lam_b = torch.autograd.grad(
                J, W0g,
                retain_graph=True,
                create_graph=True
            )[0]                           # [n,1]

            # (2) d²J/dW0², d²J/(dY0 dW0) : one autograd call (graph freed here)
            dW_b, dY_b = torch.autograd.grad(
                lam_b.sum(), (W0g, Y0g),
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
            if dW_b is None: dW_b = torch.zeros_like(W0g)
            if dY_b is None: dY_b = torch.zeros_like(Y0g)

            # (3) detach & accumulate (weight by rpts)
            lam_sum += lam_b.detach() * rpts
            dW_sum  += dW_b.detach()  * rpts
            dY_sum  += dY_b.detach()  * rpts

            done += rpts
    finally:
        # restore param requires_grad flags
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)

    inv = 1.0 / float(repeats)
    return lam_sum * inv, dW_sum * inv, dY_sum * inv   # [n,1] each

def project_pmp(lambda_hat: torch.Tensor, dlamW_hat: torch.Tensor, dlamY_hat: torch.Tensor,
                W: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    π = -1/(W J_WW) [ J_W * ((μ-r)/σ^2) + J_WY * (ρ σ_Y)/σ ].
    (μ - r) = σ α Y  => ((μ - r)/σ^2) = (α/σ) Y
    """
    mu_minus_r = sigma * (alpha * Y)          # [n,1]
    denom = W * dlamW_hat                     # W * J_WW
    eps = 1e-6
    sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    denom_safe = torch.where(denom.abs() < eps, eps*sign, denom)
    coeff = -1.0 / denom_safe

    myo   = (lambda_hat * mu_minus_r) / (sigma**2)   # J_W * ((μ-r)/σ^2)
    hedge = (dlamY_hat * (rho * sigmaY)) / sigma     # J_WY * (ρ σ_Y)/σ
    pi = coeff * (myo + hedge)
    return torch.clamp(pi, -pi_cap, pi_cap)

def ppgdpo_pi_direct(policy_s1: nn.Module,
                     W: torch.Tensor, TmT: torch.Tensor, Y: torch.Tensor,
                     repeats: int, sub_batch: int) -> torch.Tensor:
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates(policy_s1, TmT, W, Y, repeats, sub_batch)
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()

# ------------------------- Helpers -------------------------
def _divisors_desc(n: int):
    ds = []
    for d in range(1, n + 1):
        if n % d == 0:
            ds.append(d)
    ds.sort(reverse=True)
    return ds

# ------------------------- Comparisons -------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_direct(pol_s1: nn.Module, pol_cf: nn.Module,
                                         *, repeats: int, sub_batch: int,
                                         seed_eval: int | None = None,
                                         tile: int | None = None) -> None:
    # 도메인-시간 샘플링과 동일하게 평가 상태 샘플
    gen = make_generator(seed_eval)
    W, Y, TmT, _dt = sample_initial_states(N_eval_states, rng=gen)

    # Stage-1 vs CF
    pi_learn = pol_s1(W, TmT, Y)
    pi_cf    = pol_cf(W, TmT, Y)
    rmse_learn = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
    print(f"[Policy RMSE] ||π_learn - π_closed-form||_RMSE over {N_eval_states} states: {rmse_learn:.6f}")

    # Adaptive tiling for direct P-PGDPO teacher
    divisors = _divisors_desc(N_eval_states)
    if tile is None:
        start_idx = 0  # start at full batch
    else:
        start_idx = 0
        for i, d in enumerate(divisors):
            if d <= int(tile):
                start_idx = i
                break

    exc_msg_printed = False
    for idx in range(start_idx, len(divisors)):
        cur_tile = divisors[idx]
        try:
            pi_pp_dir = torch.empty_like(pi_learn)
            B = W.size(0)
            for s in range(0, B, cur_tile):
                e = min(B, s + cur_tile)
                # enable_grad slice compute
                with torch.enable_grad():
                    pi_pp_dir[s:e] = ppgdpo_pi_direct(pol_s1, W[s:e], TmT[s:e], Y[s:e], repeats, sub_batch)
            rmse_ppdir = torch.sqrt(((pi_pp_dir - pi_cf)**2).mean()).item()
            print(f"[Policy RMSE-PP(direct)] ||π_pp(direct) - π_closed-form||_RMSE over {N_eval_states} states: {rmse_ppdir:.6f} (tile={cur_tile})")

            # 3-state samples
            idxs = [0, N_eval_states//2, N_eval_states-1]
            for i in idxs:
                s = f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
                s += f" -> (π_learn={pi_learn[i,0].item():.4f}, π_pp(dir)={pi_pp_dir[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})"
                print(s)
            break
        except RuntimeError as e:
            emsg = str(e).lower()
            if "out of memory" in emsg or ("cuda" in emsg and "memory" in emsg):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_idx = idx + 1
                if next_idx < len(divisors):
                    next_tile = divisors[next_idx]
                    print(f"[Eval] OOM; reducing tile -> {next_tile}")
                    continue
                else:
                    print("[Eval] OOM; could not reduce tile further (tile=1).")
                    raise
            else:
                raise

# ------------------------------ Run ----------------------------
def main():
    # 공용 러너 재활용: BASE 학습 + Direct P-PGDPO RMSE 비교 
    run_common(
        train_fn=train_stage1_base,
        rmse_fn=print_policy_rmse_and_samples_direct,
        seed_train=seed,
        rmse_kwargs={"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "REPEATS",
    "SUBBATCH",
    "project_pmp",
    "ppgdpo_pi_direct",
    "print_policy_rmse_and_samples_direct",
]