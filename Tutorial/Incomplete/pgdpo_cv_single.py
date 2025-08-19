import torch
import torch.nn as nn
import torch.optim as optim

# ===== (1) BASE: configs, CF builder, sampler =====
from pgdpo_base_single import (
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, dt, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, N_eval_paths, CRN_SEED_EU, epochs,
    sample_initial_states, build_closed_form_policy,
)

# ===== (3) Antithetic layer: simulate & costates =====
from pgdpo_antithetic_single import (
    simulate, simulate_antithetic, estimate_costates_antithetic,
)

# ===== (4) Residual layer: policy + teacher + print helpers =====
from pgdpo_residual_single import (
    ResidualPolicy,                      # 그대로 재사용
    ppgdpo_pi_direct_antithetic,         # teacher (PMP via antithetic costates)
    compare_policy_functions as _compare_pf_v4,  # 출력 포맷 그대로 쓰기
    compare_expected_utility as _compare_eu_v4,  # 출력 포맷 그대로 쓰기
)

# --------------------------- Hyperparams & toggles ---------------------------
# Residual stage-1 training (with CV)
RESIDUAL_CAP          = 1.0
EPOCHS_RESIDUAL       = epochs
LR_RESIDUAL           = 1e-3
USE_ANTITHETIC_TRAIN  = True
SEED_TRAIN_BASE       = 13579   # epoch마다 +ep로 변주

# Control Variate (closed-form utility) settings
USE_CV                = True
CV_BETA_CLIP          = 5.0     # |beta| clip to avoid outliers
CV_EPS                = 1e-8     # denom stabilization

# Evaluation (CRN seed for EU print)
CRN_SEED_EU           = 202_409_01

# --------------------------- CV helpers ---------------------------
@torch.no_grad()
def _compute_cv_beta(u_pol_detached: torch.Tensor, u_cf: torch.Tensor, eps: float = CV_EPS, clip: float = CV_BETA_CLIP):
    """
    Per-batch OLS beta = Cov(U_pol, U_cf) / Var(U_cf)
    - u_pol_detached: U(policy) but **detached** to stop grad through beta
    - u_cf: U(closed-form), no grad path
    """
    z_cf = u_cf - u_cf.mean()
    var_cf = (z_cf * z_cf).mean() + eps
    cov_pc = ((u_pol_detached - u_pol_detached.mean()) * z_cf).mean()
    beta = cov_pc / var_cf
    if clip is not None:
        beta = torch.clamp(beta, -abs(clip), abs(clip))
    return beta

# --------------------------- Training (Residual + CV) ---------------------------
def train_residual_stage1_cv(cf_policy: nn.Module,
                             epochs=EPOCHS_RESIDUAL, lr=LR_RESIDUAL,
                             use_antithetic=USE_ANTITHETIC_TRAIN, seed_base=SEED_TRAIN_BASE,
                             use_cv: bool = USE_CV):
    """
    Residual training with antithetic pairs and CF-based control variate.
    - Loss uses: L = - mean( U_pol - beta * (U_cf - mean(U_cf)) )
      where beta is per-batch OLS, computed with stop-grad (detached).
    - NOTE: Pathwise gradient of CV term is zero; CV는 분산/로그 안정화 목적.
    """
    pol = ResidualPolicy(cf_policy, residual_cap=RESIDUAL_CAP).to(device)
    opt = optim.Adam(pol.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        pair_seed = None if seed_base is None else int(seed_base) + ep

        # ----- antithetic utility for policy -----
        U_pos = simulate(pol, batch_size, train=True, seed=pair_seed, noise_sign=+1.0)
        U_neg = simulate(pol, batch_size, train=True, seed=pair_seed, noise_sign=-1.0)
        U_pol = 0.5 * (U_pos + U_neg)

        if use_cv:
            # closed-form utility with same CRNs (no grad)
            with torch.no_grad():
                Ucf_pos = simulate(cf_policy, batch_size, train=False, seed=pair_seed, noise_sign=+1.0)
                Ucf_neg = simulate(cf_policy, batch_size, train=False, seed=pair_seed, noise_sign=-1.0)
                U_cf = 0.5 * (Ucf_pos + Ucf_neg)

            beta = _compute_cv_beta(U_pol.detach(), U_cf)
            loss = - (U_pol - beta * (U_cf - U_cf.mean())).mean()
        else:
            loss = - U_pol.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            with torch.no_grad():
                U_est = simulate_antithetic(pol, batch_size, train=False, seed=pair_seed).mean().item()
            # 출력 포맷은 3/4와 동일하게 유지
            print(f"[{ep:04d}] loss={loss.item():.6f}  E[U]_policy={U_est:.6f}")

    return pol

# --------------------------- Wrappers to reuse v4 prints ---------------------------
@torch.no_grad()
def compare_policy_functions(stage1_policy: nn.Module, cf_policy: nn.Module):
    # v4의 포맷을 그대로 재사용
    _compare_pf_v4(stage1_policy, cf_policy)

@torch.no_grad()
def compare_expected_utility(stage1_policy: nn.Module, cf_policy: nn.Module):
    # v4의 포맷을 그대로 재사용 (CRN+antithetic)
    _compare_eu_v4(stage1_policy, cf_policy)

# --------------------------- Run ---------------------------
def main():
    # 1) Closed-form base
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)

    # 2) Stage-1 residual PG-DPO with Control-Variate
    stage1 = train_residual_stage1_cv(cf_policy,
                                      epochs=EPOCHS_RESIDUAL, lr=LR_RESIDUAL,
                                      use_antithetic=USE_ANTITHETIC_TRAIN, seed_base=SEED_TRAIN_BASE,
                                      use_cv=USE_CV)
    with torch.no_grad():
        U_s1 = simulate_antithetic(stage1, batch_size, train=False, seed=SEED_TRAIN_BASE+777).mean().item()
    print(f"[After Train] E[U] Stage-1 policy (antithetic eval): {U_s1:.6f}")

    # 3) Policy RMSEs (Stage-1 / P-PGDPO / CF) 
    compare_policy_functions(stage1, cf_policy)

    # 4) EU comparisons (CRN + antithetic) 
    compare_expected_utility(stage1, cf_policy)

if __name__ == "__main__":
    main()

__all__ = [
    "train_residual_stage1_cv",
    "compare_policy_functions",
    "compare_expected_utility",
]
