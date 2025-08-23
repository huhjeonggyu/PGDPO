# pgdpo_residual_single.py  (의존성: (1) base, (2) run, (3) with_ppgdpo_single)
import torch
import torch.nn as nn
import torch.optim as optim

# ===== (1) BASE: configs, runner =====
from pgdpo_base_single import (
    # device & env params
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, CRN_SEED_EU, epochs,
    # shared runner + base seed
    run_common, seed,
)

# ===== (2) RUN runtime: simulator & (reused) comparison =====
# 항상 안티테틱+리처드슨(2*Uf-Uc) 시뮬레이션 및 평가 함수
from pgdpo_run_single import (
    simulate_run,                                  # domain-time + true antithetic + Richardson
    print_policy_rmse_and_samples_run,  # 재사용
)

# ===== (3) Direct P-PGDPO eval hyperparams (for repeats/subbatch defaults) =====
from pgdpo_with_ppgdpo_single import REPEATS, SUBBATCH

# --------------------------- Hyperparams (Residual) ---------------------------
RESIDUAL_CAP       = 1.0
EPOCHS_RESIDUAL    = epochs
LR_RESIDUAL        = 1e-3

# ========================= Myopic base policy =========================
class MyopicPolicy(nn.Module):
    """
    π_myopic(W, τ, Y) = (1/γ) * ((μ - r)/σ^2) = (1/γ) * (α/σ) * Y
    (hedging term 없음; τ 무관)
    입력은 [B,1]씩 (W, TmT, Y)
    """
    def __init__(self):
        super().__init__()
        self.coeff = (alpha / sigma) / gamma
    def forward(self, W, TmT, Y):
        pi = self.coeff * Y
        return torch.clamp(pi, -pi_cap, pi_cap)

# ========================= Residual policy (base-agnostic) =========================
class ResidualPolicy(nn.Module):
    """
    πθ(W,τ,Y) = π_base(W,τ,Y) + δπθ(W,τ,Y),  |δπθ| ≤ RESIDUAL_CAP
    """
    def __init__(self, base_policy: nn.Module, residual_cap: float = RESIDUAL_CAP):
        super().__init__()
        self.base = base_policy
        self.cap  = float(residual_cap)
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.4)
                nn.init.zeros_(m.bias)

    def forward(self, W, TmT, Y):
        base = self.base(W, TmT, Y)
        x = torch.cat([W, TmT, Y], dim=1)
        delta = torch.tanh(self.net(x)) * self.cap
        return torch.clamp(base + delta, -pi_cap, pi_cap)

# ========================= Stage-1: residual PG-DPO (ALWAYS RUN) =========================
def train_residual_stage1(
    *,
    epochs: int = EPOCHS_RESIDUAL,
    lr: float = LR_RESIDUAL,
    seed_train: int | None = seed
):
    """
    Residual training assuming myopic base is known.
    - Simulator: ALWAYS RUN (TRUE antithetic + Richardson), domain-time per-path τ with Δt=τ/m
    - Logs: loss only
    """
    base_policy = MyopicPolicy().to(device)
    pol = ResidualPolicy(base_policy, residual_cap=RESIDUAL_CAP).to(device)
    opt = optim.Adam(pol.parameters(), lr=lr)

    # ensure training mode (good practice even without dropout/bn)
    pol.train()

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        # 에폭마다 로컬 시드 변주 (TRUE antithetic은 같은 시드의 ±잡음쌍; Richardson은 coarse/fine)
        pair_seed = None if seed_train is None else int(seed_train) + ep

        # 정책 유틸리티 (RUN 평균) — grad 필요
        U_pol = simulate_run(pol, batch_size, W0=None, Y0=None, Tval=None, seed_local=pair_seed)
        loss = - U_pol.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")

    # set eval mode before returning (safety for any eval-time differences)
    pol.eval()
    return pol

# ========================= Run (reuse common runner) =========================
def main():
    # 공용 러너 재활용:
    #  - 학습: residual (myopic base) + simulate_run (항상 안티테틱+리처드슨)
    #  - 평가: run 모듈의 compare 함수를 재사용 (CRN_SEED_EU, REPEATS/SUBBATCH 적용)
    run_common(
        train_fn=lambda seed_train=seed: train_residual_stage1(seed_train=seed_train),
        rmse_fn=print_policy_rmse_and_samples_run,
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU, "repeats": REPEATS, "sub_batch": SUBBATCH},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "MyopicPolicy",
    "ResidualPolicy",
    "train_residual_stage1",
]