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

# ===== (2) P-PGDPO core: projection formula =====
from pgdpo_with_ppgdpo_single import project_pmp

# ===== (3) Antithetic layer (default): simulate & costates =====
from pgdpo_antithetic_single import (
    simulate, simulate_pair, estimate_costates,
)

# --------------------------- Hyperparams ---------------------------
# Residual stage-1 training
RESIDUAL_CAP     = 1.0
EPOCHS_RESIDUAL  = epochs
LR_RESIDUAL      = 1e-3
SEED_TRAIN_BASE  = 13579   # epoch마다 +ep로 변주

# Teacher (direct P-PGDPO via costates)
EVAL_REPEATS     = 256
SUB_REPEAT       = 128
SEED_COSTATE_BASE = 97531

# Evaluation
N_eval_states    = 200
N_eval_paths     = 5000
CRN_SEED_EU      = 202_409_01

# ========================= Residual policy =========================
class ResidualPolicy(nn.Module):
    """
    πθ(W,τ,Y) = π_cf(τ,Y) + δπθ(W,τ,Y),
    |δπθ| ≤ RESIDUAL_CAP, 이후 전체 π를 [-pi_cap, pi_cap]으로 clamp.
    """
    def __init__(self, cf_policy: nn.Module, residual_cap: float = RESIDUAL_CAP):
        super().__init__()
        self.cf = cf_policy  # 비학습(교사) 모듈
        self.cap = float(residual_cap)
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        # δπ를 0 근처에서 시작하도록 작은 초기화
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.4)
                nn.init.zeros_(m.bias)

    def forward(self, W, TmT, Y):
        base = self.cf(W, TmT, Y)
        x = torch.cat([W, TmT, Y], dim=1)
        delta = torch.tanh(self.net(x)) * self.cap
        pi = base + delta
        return torch.clamp(pi, -pi_cap, pi_cap)

# ========================= Stage-1: residual PG-DPO =========================
def train_residual_stage1(cf_policy: nn.Module,
                          epochs=EPOCHS_RESIDUAL, lr=LR_RESIDUAL,
                          seed_base=SEED_TRAIN_BASE):
    """
    Stage-1 training with antithetic sampling as default.
    """
    pol = ResidualPolicy(cf_policy, residual_cap=RESIDUAL_CAP).to(device)
    opt = optim.Adam(pol.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        pair_seed = None if seed_base is None else int(seed_base) + ep
        U_pos = simulate(pol, batch_size, train=True, seed=pair_seed, noise_sign=+1.0)
        U_neg = simulate(pol, batch_size, train=True, seed=pair_seed, noise_sign=-1.0)
        loss = -0.5 * (U_pos.mean() + U_neg.mean())

        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            with torch.no_grad():
                U_est = simulate_pair(pol, batch_size, train=False, seed=pair_seed).mean().item()
            print(f"[{ep:04d}] loss={loss.item():.6f}  E[U]_policy={U_est:.6f}")

    return pol

# ========================= Teacher (direct PP via costates) =========================
def ppgdpo_pi_direct(policy_for_sim: nn.Module,
                     W: torch.Tensor, TmT: torch.Tensor, Y: torch.Tensor,
                     repeats=EVAL_REPEATS, sub_batch=SUB_REPEAT,
                     seed_base=SEED_COSTATE_BASE):
    """
    Functional teacher: π_pp(W,τ,Y) via PMP using antithetic costates (default).
    """
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates(
            policy_for_sim, TmT, W, Y,
            repeats=repeats, sub_batch=sub_batch,
            seed_base=seed_base
        )
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()

# ========================= Comparisons =========================
@torch.no_grad()
def compare_policy_functions(stage1_policy: nn.Module, cf_policy: nn.Module):
    W = torch.empty(N_eval_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(N_eval_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)

    pi_s1 = stage1_policy(W, TmT, Y)
    pi_pp = ppgdpo_pi_direct(stage1_policy, W, TmT, Y)
    pi_cf = cf_policy(W, TmT, Y)

    rmse_s1_cf = torch.sqrt(((pi_s1 - pi_cf)**2).mean()).item()
    rmse_pp_cf = torch.sqrt(((pi_pp - pi_cf)**2).mean()).item()
    rmse_s1_pp = torch.sqrt(((pi_s1 - pi_pp)**2).mean()).item()

    print(f"[Policy RMSE] Stage-1 vs CF:      {rmse_s1_cf:.6f}")
    print(f"[Policy RMSE] P-PGDPO vs CF:      {rmse_pp_cf:.6f}")
    print(f"[Policy RMSE] Stage-1 vs P-PGDPO: {rmse_s1_pp:.6f}")

    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_s1={pi_s1[i,0].item():.4f}, π_pp={pi_pp[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})")

@torch.no_grad()
def compare_expected_utility(stage1_policy: nn.Module, cf_policy: nn.Module):
    # 공통 난수(CRN) + antithetic 평균
    W0, Y0 = sample_initial_states(N_eval_paths)
    U_s1 = simulate_pair(stage1_policy, N_eval_paths, train=False, W0=W0, Y0=Y0, seed=CRN_SEED_EU)
    U_cf = simulate_pair(cf_policy,   N_eval_paths, train=False, W0=W0, Y0=Y0, seed=CRN_SEED_EU)

    EU_s1 = U_s1.mean().item()
    EU_cf = U_cf.mean().item()
    print(f"[EU]  E[U]_Stage-1 = {EU_s1:.6f}")
    print(f"      E[U]_Closed-form = {EU_cf:.6f}")
    print(f"      Δ (Stage-1 - Closed-form) = {EU_s1 - EU_cf:.6f}")

# ========================= Run =========================
def main():
    # 1) Closed-form base
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)

    # 2) Stage-1 residual PG-DPO
    stage1 = train_residual_stage1(cf_policy,
                                   epochs=EPOCHS_RESIDUAL, lr=LR_RESIDUAL,
                                   seed_base=SEED_TRAIN_BASE)
    with torch.no_grad():
        U_s1 = simulate_pair(stage1, batch_size, train=False, seed=SEED_TRAIN_BASE+777).mean().item()
    print(f"[After Train] E[U] Stage-1 policy: {U_s1:.6f}")

    # 3) Policy RMSEs
    compare_policy_functions(stage1, cf_policy)

    # 4) EU comparisons
    compare_expected_utility(stage1, cf_policy)

if __name__ == "__main__":
    main()

__all__ = [
    "ResidualPolicy",
    "train_residual_stage1",
    "ppgdpo_pi_direct",
    "compare_policy_functions",
    "compare_expected_utility",
]
