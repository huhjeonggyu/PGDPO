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
SUBBATCH = 2560 // 64  # for 16GB GPU RAM, can be increased if memory allows
# ------------------------------------------------------------------

# ------------------ Costate estimation (direct) -------------------
def estimate_costates(policy_net: nn.Module,
                      T0: torch.Tensor, W0: torch.Tensor, Y0: torch.Tensor,
                      repeats: int, sub_batch: int):
    """
    Estimate λ=J_W, ∂λ/∂W=J_WW, ∂λ/∂Y=J_WY at (W0,Y0,T0) under policy_net.
    Keeps autograd graph by calling simulate(..., train=True).
    """
    n = W0.size(0)
    W0g = W0.clone().requires_grad_(True)
    Y0g = Y0.clone().requires_grad_(True)

    lam_sum = torch.zeros_like(W0g)
    dW_sum  = torch.zeros_like(W0g)
    dY_sum  = torch.zeros_like(Y0g)

    done = 0
    while done < repeats:
        rpts = min(sub_batch, repeats - done)

        T_b = T0.repeat(rpts, 1)
        W_b = W0g.repeat(rpts, 1)
        Y_b = Y0g.repeat(rpts, 1)

        # two independent MC batches, then average (variance reduction)
        u1 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        u2 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        avg_u = 0.5 * (u1 + u2)

        avg_u_per_point = avg_u.view(rpts, n).mean(0)  # [n,1]
        (lam_b,) = torch.autograd.grad(avg_u_per_point.sum(), W0g, create_graph=True)
        dlamW_b, dlamY_b = torch.autograd.grad(lam_b.sum(), (W0g, Y0g), allow_unused=False)

        lam_sum += lam_b.detach() * rpts
        dW_sum  += dlamW_b.detach() * rpts
        dY_sum  += dlamY_b.detach() * rpts
        done += rpts

    inv = 1.0 / repeats
    return lam_sum*inv, dW_sum*inv, dY_sum*inv   # [n,1] each

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

# ------------------------- Comparisons -------------------------
@torch.no_grad()
def print_policy_rmse_and_samples_direct(pol_s1: nn.Module, pol_cf: nn.Module,
                                         *, repeats: int, sub_batch: int,
                                         seed_eval: int | None = None) -> None:
    # 도메인-시간 샘플링과 동일하게 평가 상태 샘플
    gen = make_generator(seed_eval)
    W, Y, TmT, _dt = sample_initial_states(N_eval_states, rng=gen)

    pi_learn = pol_s1(W, TmT, Y)
    pi_cf    = pol_cf(W, TmT, Y)

    # Direct P-PGDPO (teacher)
    pi_pp_dir = ppgdpo_pi_direct(pol_s1, W, TmT, Y, repeats, sub_batch)

    rmse_learn = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
    rmse_ppdir = torch.sqrt(((pi_pp_dir - pi_cf)**2).mean()).item()
    print(f"[Policy RMSE] ||π_learn - π_closed-form||_RMSE over {N_eval_states} states: {rmse_learn:.6f}")
    print(f"[Policy RMSE-PP(direct)] ||π_pp(direct) - π_closed-form||_RMSE over {N_eval_states} states: {rmse_ppdir:.6f}")

    # 3-state samples
    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        s = f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
        s += f" -> (π_learn={pi_learn[i,0].item():.4f}, π_pp(dir)={pi_pp_dir[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})"
        print(s)

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
