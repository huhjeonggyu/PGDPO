# pgdpo_cv_single.py  (5번: Residual + CV, ALWAYS RUN runtime)
import torch
import torch.nn as nn
import torch.optim as optim

# ===== (1) BASE: configs, samplers, runner =====
from pgdpo_base_single import (
    device, gamma, alpha, sigma,
    batch_size, lb_W, pi_cap,
    epochs, seed, CRN_SEED_EU,
    make_generator, sample_initial_states,
    run_common,
)

# ===== (2) P-PGDPO: eval hyperparams만 재사용 =====
from pgdpo_with_ppgdpo_single import (
    REPEATS, SUBBATCH
)

# ===== (3) RUN 런타임 (항상 TRUE antithetic + Richardson) & 비교 함수 재사용 =====
from pgdpo_run_single import (
    simulate_run, print_policy_rmse_and_samples_run,
)

# ===== (4) 4번의 마이오픽/레지듀얼 정책 + SUBBATCH 재사용 =====
from pgdpo_residual_single import (
    MyopicPolicy, ResidualPolicy
)

# --------------------------- CV 하이퍼파라미터 ---------------------------
USE_CV       = True        # CV 사용/미사용 스위치
CV_BETA_CLIP = 5.0         # 배치 OLS 계수 클리핑
CV_EMA_DECAY = 0.1         # EMA 업데이트 계수 (0<δ≤1)
CV_EPS       = 1e-8        # 수치 안정화

# --------------------------- CV: 배치 OLS 계수만 (no-grad) ---------------------------
@torch.no_grad()
def _cv_coeff_only(u_pol: torch.Tensor, u_my: torch.Tensor, W0: torch.Tensor):
    """
    배치 OLS 계수 c_hat만 추정 (여기서만 detach, no-grad).
    c_hat = Cov(U_pol_norm, U_my_centered_norm) / Var(U_my_centered_norm)
    """
    scale       = W0.pow(1.0 - gamma)                    # [B,1]
    u_pol_n_det = u_pol.detach() / (scale + CV_EPS)      # detach: 계수 추정에만 사용
    u_my_cn     = (u_my - u_my.mean()) / (scale + CV_EPS)

    var_z  = (u_my_cn * u_my_cn).mean() + CV_EPS
    cov_pz = ((u_pol_n_det - u_pol_n_det.mean()) * u_my_cn).mean()
    c_hat  = cov_pz / var_z
    if CV_BETA_CLIP is not None:
        c_hat = torch.clamp(c_hat, -abs(CV_BETA_CLIP), abs(CV_BETA_CLIP))
    return c_hat

# --------------------------- 학습 (Residual + CV, ALWAYS RUN) ---------------------------
def train_residual_stage1_cv(
    *,
    epochs: int = epochs,
    lr: float = 1e-3,
    residual_cap: float = 1.0,
    seed_train: int | None = seed,
    use_cv: bool = USE_CV,
) -> nn.Module:
    """
    Residual 학습: 마이오픽 베이스를 고정, 잔차만 학습.
    - 도메인-시간 샘플링(경로별 τ, Δt=τ/m), ALWAYS RUN(안티테틱+리처드슨)
    - CV: 마이오픽 유틸리티 기반, W0^(1-gamma) 정규화 + 배치 OLS + EMA
    - 코스테이트에는 CV 사용하지 않음
    - 로그: loss만 출력
    """
    base_policy = MyopicPolicy().to(device)
    pol = ResidualPolicy(base_policy, residual_cap=residual_cap).to(device)
    opt = optim.Adam(pol.parameters(), lr=lr)

    # ensure training mode
    pol.train()

    c_ema = None  # EMA 상태 (torch.scalar)

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        # 에폭별 로컬 시드 (TRUE antithetic의 ± 및 Richardson coarse/fine에 동일 시드 변주)
        pair_seed = None if seed_train is None else int(seed_train) + ep

        # 동일 초기상태(공유 CRN) 준비: (W0, Y0, TmT)
        W0, Y0, TmT, _dt = sample_initial_states(batch_size, rng=make_generator(pair_seed))

        # 정책/마이오픽 유틸리티 (둘 다 같은 W0,Y0,τ와 같은 시드로 ALWAYS RUN)
        U_pol = simulate_run(pol,          batch_size, W0=W0, Y0=Y0, Tval=TmT, seed_local=pair_seed)
        with torch.no_grad():
            U_my  = simulate_run(base_policy, batch_size, W0=W0, Y0=Y0, Tval=TmT, seed_local=pair_seed)

        if use_cv:
            # 1) 계수 추정 (no-grad, detach는 여기서만)
            c_hat = _cv_coeff_only(U_pol, U_my, W0)

            # 2) EMA 갱신 (항상 no-grad 상태 유지)
            c_ema = c_hat.detach() if (c_ema is None) else ((1.0 - CV_EMA_DECAY) * c_ema + CV_EMA_DECAY * c_hat.detach())

            # 3) 손실에 들어갈 정규화 텐서 (grad 필요)
            scale   = W0.pow(1.0 - gamma)
            U_pol_n = U_pol / (scale + CV_EPS)             # 그래프 유지 → grad 흐름
            U_my_cn = (U_my - U_my.mean()) / (scale + CV_EPS)

            # 4) 최종 손실 (정규화 공간)
            loss = - (U_pol_n - c_ema * U_my_cn).mean()
        else:
            loss = - U_pol.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()

        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")

    # switch to eval mode before returning
    pol.eval()
    return pol

# --------------------------- Run (reuse common runner) ---------------------------
def main():
    # 공용 러너 재활용:
    #  - 학습: residual + CV(myopic) + simulate_run(항상 안티테틱+리처드슨)
    #  - 평가: run 모듈의 비교 함수 재사용 (CRN_SEED_EU, REPEATS/SUBBATCH 적용)
    run_common(
        train_fn=lambda seed_train=seed: train_residual_stage1_cv(seed_train=seed_train, use_cv=USE_CV),
        rmse_fn=print_policy_rmse_and_samples_run,  # ALWAYS RUN 평가 + 교사 재사용
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU, "repeats": REPEATS, "sub_batch": SUBBATCH},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "train_residual_stage1_cv",
]