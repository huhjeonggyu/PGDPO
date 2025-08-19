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

# --------------------------- Stage-2 hyperparams ---------------------------
REPEATS_DIRECT   = 4000
SUBBATCH_DIRECT  = 200
N_DISTILL        = 4000
DISTILL_BATCH    = 256
DISTILL_EPOCHS   = 10
DISTILL_LR       = 1e-3
REPEATS_TARGET   = 2000
SUBBATCH_TARGET  = 200

# Evaluation (same scale as BASE)
N_eval_states = 200
N_eval_paths  = 5000
CRN_SEED_EU   = 12345   # common-random-numbers seed for EU comparisons

# --------------------------- Stage-2 policies ---------------------------
class DistillPolicy(nn.Module):
    """Distilled P-PGDPO policy π_phi(W, τ, Y) to mimic direct P-PGDPO."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.9)
                nn.init.zeros_(m.bias)
    def forward(self, W, TmT, Y):
        x = torch.cat([W, TmT, Y], dim=1)
        pi = self.net(x)
        return torch.clamp(torch.tanh(pi), -pi_cap, pi_cap)

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

def ppgdpo_pi_direct(policy_s1, W, TmT, Y, repeats=REPEATS_DIRECT, sub_batch=SUBBATCH_DIRECT):
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates(policy_s1, TmT, W, Y, repeats=repeats, sub_batch=sub_batch)
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()

# ------------------ Distillation -------------------
def train_ppgdpo_distilled(policy_s1):
    """Train a small policy net to mimic direct π_pp(W,τ,Y)."""
    net = DistillPolicy().to(device)
    opt = optim.Adam(net.parameters(), lr=DISTILL_LR)

    # State-time cover for teacher querying
    W_all = torch.empty(N_DISTILL, 1, device=device).uniform_(*W0_range)
    Y_all = torch.empty_like(W_all).uniform_(*Y0_range)
    T_all = torch.empty_like(W_all).uniform_(0.0, T)

    for ep in range(1, DISTILL_EPOCHS+1):
        idx = torch.randperm(N_DISTILL, device=device)
        running = 0.0
        seen = 0
        for k in range(0, N_DISTILL, DISTILL_BATCH):
            sel = idx[k:k+DISTILL_BATCH]
            Wb, Yb, Tb = W_all[sel], Y_all[sel], T_all[sel]

            # Teacher: direct P-PGDPO (stochastic target; use decent repeats)
            pi_target = ppgdpo_pi_direct(
                policy_s1, Wb, Tb, Yb,
                repeats=REPEATS_TARGET, sub_batch=SUBBATCH_TARGET
            )

            opt.zero_grad()
            pi_hat = net(Wb, Tb, Yb)
            loss = nn.functional.mse_loss(pi_hat, pi_target)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            running += loss.item() * Wb.size(0)
            seen += Wb.size(0)

        print(f"[Distill {ep:02d}/{DISTILL_EPOCHS}] MSE={running/seen:.6f}")

    return net

# ------------------------- Comparisons -------------------------
def print_policy_rmse_and_samples_direct_and_distill(pol_s1, pol_cf, pol_pp_distill=None):
    W = torch.empty(N_eval_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(N_eval_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)

    with torch.no_grad():
        pi_learn = pol_s1(W, TmT, Y)
        pi_cf    = pol_cf(W, TmT, Y)

    # Direct P-PGDPO (teacher)
    pi_pp_dir = ppgdpo_pi_direct(pol_s1, W, TmT, Y)  # detached inside

    # Optional: distilled PP policy
    pi_pp_dis = None
    if pol_pp_distill is not None:
        with torch.no_grad():
            pi_pp_dis = pol_pp_distill(W, TmT, Y)

    with torch.no_grad():
        rmse_learn = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
        rmse_ppdir = torch.sqrt(((pi_pp_dir - pi_cf)**2).mean()).item()
        print(f"[Policy RMSE] ||π_learn - π_closed-form||_RMSE over {N_eval_states} states: {rmse_learn:.6f}")
        print(f"[Policy RMSE-PP(direct)] ||π_pp(direct) - π_closed-form||_RMSE over {N_eval_states} states: {rmse_ppdir:.6f}")
        if pi_pp_dis is not None:
            rmse_ppdis = torch.sqrt(((pi_pp_dis - pi_cf)**2).mean()).item()
            print(f"[Policy RMSE-PP(distill)] ||π_pp(distill) - π_closed-form||_RMSE over {N_eval_states} states: {rmse_ppdis:.6f}")

        # 3-state samples
        idxs = [0, N_eval_states//2, N_eval_states-1]
        for i in idxs:
            s = f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
            s += f" -> (π_learn={pi_learn[i,0].item():.4f}, π_pp(dir)={pi_pp_dir[i,0].item():.4f}"
            if pi_pp_dis is not None:
                s += f", π_pp(dist)={pi_pp_dis[i,0].item():.4f}"
            s += f", π_cf={pi_cf[i,0].item():.4f})"
            print(s)

@torch.no_grad()
def print_eu_comparisons(pol_s1, pol_pp_distill, pol_cf, crn_seed: int = CRN_SEED_EU):
    # Common initial states + same seed (CRN)
    W0, Y0 = sample_initial_states(N_eval_paths)
    U_learn = simulate(pol_s1, N_eval_paths, train=False, W0=W0, Y0=Y0, seed=crn_seed)
    U_pp    = simulate(pol_pp_distill, N_eval_paths, train=False, W0=W0, Y0=Y0, seed=crn_seed)
    U_cf    = simulate(pol_cf,     N_eval_paths, train=False, W0=W0, Y0=Y0, seed=crn_seed)

    EU_learn, std_learn = U_learn.mean().item(), U_learn.std(unbiased=True).item()
    EU_pp,    std_pp    = U_pp.mean().item(),    U_pp.std(unbiased=True).item()
    EU_cf,    std_cf    = U_cf.mean().item(),    U_cf.std(unbiased=True).item()

    print(f"[EU]  E[U]_learn = {EU_learn:.6f}  (std≈{std_learn:.6f})")
    print(f"      E[U]_closed-form = {EU_cf:.6f}  (std≈{std_cf:.6f})")
    print(f"      Δ (learn - closed-form) = {EU_learn - EU_cf:.6f}")
    print(f"[EU-PP(distill)]  E[U]_ppgdpo = {EU_pp:.6f}  (std≈{std_pp:.6f})")
    print(f"                 E[U]_closed-form = {EU_cf:.6f}  (std≈{std_cf:.6f})")
    print(f"                 Δ (ppgdpo - closed-form) = {EU_pp - EU_cf:.6f}")

# ------------------------------ Run ----------------------------
def main():
    # Closed-form from BASE (same mode as BASE; recommended CF_MODE="X")
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)

    # Stage 1: reuse BASE training
    policy_s1 = train_stage1_base()

    with torch.no_grad():
        U_final = simulate(policy_s1, batch_size, train=False).mean().item()
    print(f"[After Train] E[U] (BASE policy): {U_final:.6f}")

    # Direct P-PGDPO RMSE check; then Distilled PP training
    print_policy_rmse_and_samples_direct_and_distill(policy_s1, cf_policy, pol_pp_distill=None)
    pp_distill = train_ppgdpo_distilled(policy_s1)

    # RMSE again incl. distilled policy + samples
    print_policy_rmse_and_samples_direct_and_distill(policy_s1, cf_policy, pol_pp_distill=pp_distill)

    # EU comparisons with distilled policy (CRN via seed)
    print_eu_comparisons(policy_s1, pp_distill, cf_policy, crn_seed=CRN_SEED_EU)

if __name__ == "__main__":
    main()

__all__ = [
    "project_pmp",
    "ppgdpo_pi_direct",
    "train_ppgdpo_distilled",
    "print_policy_rmse_and_samples_direct_and_distill",
    "print_eu_comparisons",
]
