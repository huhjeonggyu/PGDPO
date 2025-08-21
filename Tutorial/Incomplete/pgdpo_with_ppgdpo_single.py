import math
import torch
import torch.nn as nn
import torch.optim as optim

from pgdpo_base_single import (
    # config & device
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, dt, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, N_eval_paths, CRN_SEED_EU,
    # env/sim
    sample_initial_states, correlated_normals, simulate,
    # stage-1 & closed-form
    train_stage1_base, build_closed_form_policy
)

# ------------------ Costate estimation (direct) -------------------
def estimate_costates(policy_net, T0, W0, Y0, repeats=800, sub_batch=100):
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

    for i in range(0, repeats, sub_batch):
        rpts = min(sub_batch, repeats - i)
        T_b = T0.repeat(rpts,1)
        W_b = W0g.repeat(rpts,1)
        Y_b = Y0g.repeat(rpts,1)

        # two independent MC batches, then average to reduce variance
        u1 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        u2 = simulate(policy_net, n*rpts, train=True, W0=W_b, Y0=Y_b, Tval=T_b)
        avg_u = 0.5 * (u1 + u2)

        avg_u_per_point = avg_u.view(rpts, n).mean(0)  # [n,1]
        (lam_b,) = torch.autograd.grad(avg_u_per_point.sum(), W0g, create_graph=True)
        dlamW_b, dlamY_b = torch.autograd.grad(lam_b.sum(), (W0g, Y0g), allow_unused=False)

        lam_sum += lam_b.detach() * rpts
        dW_sum  += dlamW_b.detach() * rpts
        dY_sum  += dlamY_b.detach() * rpts

    inv = 1.0 / repeats
    return lam_sum*inv, dW_sum*inv, dY_sum*inv   # [n,1] each

def project_pmp(lambda_hat, dlamW_hat, dlamY_hat, W, Y):
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

def ppgdpo_pi_direct(policy_s1, W, TmT, Y, repeats=4000, sub_batch=200):
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates(policy_s1, TmT, W, Y, repeats=repeats, sub_batch=sub_batch)
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()

# ------------------------- Comparisons -------------------------
def print_policy_rmse_and_samples_direct(pol_s1, pol_cf):
    W = torch.empty(N_eval_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(N_eval_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)

    with torch.no_grad():
        pi_learn = pol_s1(W, TmT, Y)
        pi_cf    = pol_cf(W, TmT, Y)

    # Direct P-PGDPO (teacher)
    pi_pp_dir = ppgdpo_pi_direct(pol_s1, W, TmT, Y)  # detached inside

    with torch.no_grad():
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

@torch.no_grad()
def print_eu_comparisons(pol_s1, pol_cf, crn_seed: int = CRN_SEED_EU):
    # Common initial states + same seed (CRN)
    W0, Y0 = sample_initial_states(N_eval_paths)
    U_learn = simulate(pol_s1, N_eval_paths, train=False, W0=W0, Y0=Y0, seed=crn_seed)
    U_cf    = simulate(pol_cf,     N_eval_paths, train=False, W0=W0, Y0=Y0, seed=crn_seed)

    EU_learn, std_learn = U_learn.mean().item(), U_learn.std(unbiased=True).item()
    EU_cf,    std_cf    = U_cf.mean().item(),    U_cf.std(unbiased=True).item()

    print(f"[EU]  E[U]_learn = {EU_learn:.6f}  (std≈{std_learn:.6f})")
    print(f"      E[U]_closed-form = {EU_cf:.6f}  (std≈{std_cf:.6f})")
    print(f"      Δ (learn - closed-form) = {EU_learn - EU_cf:.6f}")

# ------------------------------ Run ----------------------------
def main():
    # Closed-form from BASE (same mode as BASE; recommended CF_MODE="X")
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)

    # Stage 1: reuse BASE training
    policy_s1 = train_stage1_base()

    with torch.no_grad():
        U_final = simulate(policy_s1, batch_size, train=False).mean().item()
    print(f"[After Train] E[U] (BASE policy): {U_final:.6f}")

    # Direct P-PGDPO RMSE check
    print_policy_rmse_and_samples_direct(policy_s1, cf_policy)

    # EU comparisons
    print_eu_comparisons(policy_s1, cf_policy, crn_seed=CRN_SEED_EU)

if __name__ == "__main__":
    main()

__all__ = [
    "project_pmp",
    "ppgdpo_pi_direct",
    "print_policy_rmse_and_samples_direct",
    "print_eu_comparisons",
]
