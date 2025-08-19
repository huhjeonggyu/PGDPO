import math
import torch
import torch.nn as nn
import torch.optim as optim  # 스타일 맞춤 (직접 사용 X)

# ===== (1) BASE: configs, CF builder, sampler =====
from pgdpo_base_single import (
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, dt, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    sample_initial_states, build_closed_form_policy, epochs,
)

# ===== (3) Antithetic layer: simulate & costates & projector =====
from pgdpo_antithetic_single import (
    simulate, simulate_antithetic, estimate_costates_antithetic,
    # 아래 함수가 모듈에 존재한다고 가정 (없다면 antithetic 모듈에 추가 필요)
    ppgdpo_pi_antithetic,
)

# ===== (5) CV layer: Stage-1 trainer (+ print helpers 래핑 되어있을 것) =====
from pgdpo_cv_single import (
    train_residual_stage1_cv,
    compare_policy_functions as _compare_pf_v5,
    compare_expected_utility as _compare_eu_v5,
)

# --------------------------- Hyperparams & toggles ---------------------------
# Stage-1 (CV) 학습 세팅
EPOCHS_STAGE1        = epochs
LR_STAGE1            = 1e-3
USE_ANTITHETIC_TRAIN = True
SEED_TRAIN_BASE      = 13579   # epoch마다 +ep로 변주

# 평가용 공통 난수 시드
CRN_SEED_EU          = 202_409_01

# 리처드슨 데모 세팅
USE_ANTITHETIC_DEMO  = True
N_PATHS_DEMO         = 8192
CRN_SEED_RICH        = 202_409_02
DEMO_POLICY          = "closed_form"   # {"closed_form", "stage1", "ppgdpo"}

# --------------------------- Utilities ---------------------------
@torch.no_grad()
def _policy_rmse(policy: nn.Module, cf_policy: nn.Module, n_states: int = 200):
    W = torch.empty(n_states, 1, device=device).uniform_(*W0_range)
    Y = torch.empty(n_states, 1, device=device).uniform_(*Y0_range)
    TmT = torch.full_like(W, T)
    pi_p  = policy(W, TmT, Y)
    pi_cf = cf_policy(W, TmT, Y)
    return torch.sqrt(((pi_p - pi_cf) ** 2).mean()).item()

class ProjectedPolicyAntithetic(nn.Module):
    """
    P-PGDPO 사영 정책을 nn.Module 형태로 래핑.
    내부에서 ppgdpo_pi_antithetic(stage1, W, TmT, Y)를 호출한다.
    """
    def __init__(self, stage1_policy: nn.Module):
        super().__init__()
        self.s1 = stage1_policy
    def forward(self, W, TmT, Y):
        with torch.no_grad():
            pi = ppgdpo_pi_antithetic(self.s1, W, TmT, Y)
        return torch.clamp(pi, -pi_cap, pi_cap)

# --------------------------- Richardson core (coupled paths) ---------------------------
def _make_generator(seed: int | None):
    g = torch.Generator(device=device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    return g

def _correlated_normals(B: int, rho_val: float, gen: torch.Generator):
    u = torch.randn(B, 1, generator=gen, device=device)
    v = torch.randn(B, 1, generator=gen, device=device)
    zW = u
    zY = rho_val * u + math.sqrt(max(0.0, 1.0 - rho_val * rho_val)) * v
    return zW, zY

@torch.no_grad()
def _simulate_triplet(policy: nn.Module, B: int,
                      W0: torch.Tensor | None = None,
                      Y0: torch.Tensor | None = None,
                      Tval: float | torch.Tensor | None = None,
                      seed: int | None = None,
                      noise_sign: float = +1.0):
    """
    동일 브라운 경로를 공유하며 Δt, Δt/2, Δt/4를 동시에 굴린다.
    반환: (U_Δt, U_Δt/2, U_Δt/4)
    """
    if W0 is None or Y0 is None:
        W0, Y0 = sample_initial_states(B)

    if Tval is None:
        TmT = torch.full_like(W0, T, device=device)
    else:
        TmT = Tval.clone() if torch.is_tensor(Tval) else torch.full_like(W0, float(Tval), device=device)
        if TmT.shape != W0.shape:
            TmT = TmT.expand_as(W0) if TmT.numel() == 1 else TmT.view_as(W0)

    logW_c = W0.clamp(min=lb_W).log(); Y_c = Y0.clone(); Tc = TmT.clone()
    logW_f = logW_c.clone();            Y_f = Y_c.clone(); Tf = TmT.clone()
    logW_q = logW_c.clone();            Y_q = Y_c.clone(); Tq = TmT.clone()

    gen = _make_generator(seed)
    sqrt_dt = math.sqrt(dt); dt2, dt4 = 0.5*dt, 0.25*dt
    sqrt_dt2 = math.sqrt(dt2); sqrt_dt4 = math.sqrt(dt4)

    for _ in range(m):
        zW = []; zY = []
        for _k in range(4):
            a, b = _correlated_normals(B, rho, gen)
            zW.append(a); zY.append(b)
        zW1, zW2, zW3, zW4 = zW
        zY1, zY2, zY3, zY4 = zY

        zW_coarse = (zW1 + zW2 + zW3 + zW4) / 2.0
        zY_coarse = (zY1 + zY2 + zY3 + zY4) / 2.0

        zW_f1 = (zW1 + zW2) / math.sqrt(2.0); zY_f1 = (zY1 + zY2) / math.sqrt(2.0)
        zW_f2 = (zW3 + zW4) / math.sqrt(2.0); zY_f2 = (zY3 + zY4) / math.sqrt(2.0)

        dBW_c = noise_sign * (sqrt_dt   * zW_coarse); dBY_c = noise_sign * (sqrt_dt   * zY_coarse)
        dBW_f1 = noise_sign * (sqrt_dt2 * zW_f1);     dBY_f1 = noise_sign * (sqrt_dt2 * zY_f1)
        dBW_f2 = noise_sign * (sqrt_dt2 * zW_f2);     dBY_f2 = noise_sign * (sqrt_dt2 * zY_f2)
        dBW_q1 = noise_sign * (sqrt_dt4 * zW1);       dBY_q1 = noise_sign * (sqrt_dt4 * zY1)
        dBW_q2 = noise_sign * (sqrt_dt4 * zW2);       dBY_q2 = noise_sign * (sqrt_dt4 * zY2)
        dBW_q3 = noise_sign * (sqrt_dt4 * zW3);       dBY_q3 = noise_sign * (sqrt_dt4 * zY3)
        dBW_q4 = noise_sign * (sqrt_dt4 * zW4);       dBY_q4 = noise_sign * (sqrt_dt4 * zY4)

        # Δt
        pi_c = policy(logW_c.exp(), Tc, Y_c).clamp(-pi_cap, pi_cap)
        risk_prem_c = sigma * (alpha * Y_c); varW_c = (pi_c * sigma) ** 2
        logW_c = logW_c + (r + pi_c * risk_prem_c - 0.5*varW_c)*dt + (pi_c * sigma)*dBW_c
        Y_c    = Y_c    + kappaY*(thetaY - Y_c)*dt + sigmaY*dBY_c
        logW_c = logW_c.exp().clamp(min=lb_W).log(); Tc = Tc - dt

        # Δt/2 × 2
        pi_f = policy(logW_f.exp(), Tf, Y_f).clamp(-pi_cap, pi_cap)
        risk_prem_f = sigma * (alpha * Y_f); varW_f = (pi_f * sigma) ** 2
        logW_f = logW_f + (r + pi_f * risk_prem_f - 0.5*varW_f)*dt2 + (pi_f * sigma)*dBW_f1
        Y_f    = Y_f    + kappaY*(thetaY - Y_f)*dt2 + sigmaY*dBY_f1
        logW_f = logW_f.exp().clamp(min=lb_W).log(); Tf = Tf - dt2

        pi_f = policy(logW_f.exp(), Tf, Y_f).clamp(-pi_cap, pi_cap)
        risk_prem_f = sigma * (alpha * Y_f); varW_f = (pi_f * sigma) ** 2
        logW_f = logW_f + (r + pi_f * risk_prem_f - 0.5*varW_f)*dt2 + (pi_f * sigma)*dBW_f2
        Y_f    = Y_f    + kappaY*(thetaY - Y_f)*dt2 + sigmaY*dBY_f2
        logW_f = logW_f.exp().clamp(min=lb_W).log(); Tf = Tf - dt2

        # Δt/4 × 4
        for dBW_q, dBY_q in [(dBW_q1,dBY_q1),(dBW_q2,dBY_q2),(dBW_q3,dBY_q3),(dBW_q4,dBY_q4)]:
            pi_q = policy(logW_q.exp(), Tq, Y_q).clamp(-pi_cap, pi_cap)
            risk_prem_q = sigma * (alpha * Y_q); varW_q = (pi_q * sigma) ** 2
            logW_q = logW_q + (r + pi_q * risk_prem_q - 0.5*varW_q)*dt4 + (pi_q * sigma)*dBW_q
            Y_q    = Y_q    + kappaY*(thetaY - Y_q)*dt4 + sigmaY*dBY_q
            logW_q = logW_q.exp().clamp(min=lb_W).log(); Tq = Tq - dt4

    Wc_T, Wf_T, Wq_T = logW_c.exp(), logW_f.exp(), logW_q.exp()
    if abs(gamma - 1.0) < 1e-10:
        Uc = Wc_T.log(); Uf = Wf_T.log(); Uq = Wq_T.log()
    else:
        c = 1.0 - gamma
        Uc = Wc_T.pow(c)/c; Uf = Wf_T.pow(c)/c; Uq = Wq_T.pow(c)/c
    return Uc, Uf, Uq

@torch.no_grad()
def _simulate_antithetic_triplet(policy: nn.Module, B: int, seed: int | None = None):
    Uc_pos, Uf_pos, Uq_pos = _simulate_triplet(policy, B, seed=seed, noise_sign=+1.0)
    Uc_neg, Uf_neg, Uq_neg = _simulate_triplet(policy, B, seed=seed, noise_sign=-1.0)
    Uc = 0.5*(Uc_pos + Uc_neg); Uf = 0.5*(Uf_pos + Uf_neg); Uq = 0.5*(Uq_pos + Uq_neg)
    return Uc, Uf, Uq

@torch.no_grad()
def demo_richardson(policy: nn.Module, paths: int = N_PATHS_DEMO, seed: int = CRN_SEED_RICH, use_antithetic: bool = USE_ANTITHETIC_DEMO, title: str = ""):
    if use_antithetic:
        Uc, Uf, Uq = _simulate_antithetic_triplet(policy, paths, seed=seed)
    else:
        Uc, Uf, Uq = _simulate_triplet(policy, paths, seed=seed, noise_sign=+1.0)
    U_star = 2.0 * Uf - Uc

    mean = lambda x: float(x.mean().item())
    std  = lambda x: float(x.std(unbiased=True).item())

    mu_c, mu_f, mu_star, mu_q = mean(Uc), mean(Uf), mean(U_star), mean(Uq)
    sd_c, sd_f, sd_star, sd_q = std(Uc), std(Uf), std(U_star), std(Uq)

    tag = title if title else "policy"
    print(f"\n=== [RICHARDSON DEMO] {tag}, paths={paths}, antithetic={use_antithetic} ===")
    print(f"E[U]_Δt            = {mu_c:.8f}   (std {sd_c:.8f})")
    print(f"E[U]_Δt/2          = {mu_f:.8f}   (std {sd_f:.8f})")
    print(f"E[U]_Rich (2*Uf-Uc)= {mu_star:.8f}   (std {sd_star:.8f})")
    print(f"E[U]_Δt/4 (ref)    = {mu_q:.8f}   (std {sd_q:.8f})")
    print("Bias vs Δt/4 :  Δt={:+.8e}   Δt/2={:+.8e}   Rich={:+.8e}".format(mu_c-mu_q, mu_f-mu_q, mu_star-mu_q))

# --------------------------- Run ---------------------------
def main():
    # 1) Closed-form (기준선)
    cf_policy, _ = build_closed_form_policy(mode=CF_MODE)

    # 2) Stage-1 (CV) 학습
    stage1 = train_residual_stage1_cv(
        cf_policy,
        epochs=EPOCHS_STAGE1, lr=LR_STAGE1,
        use_antithetic=USE_ANTITHETIC_TRAIN, seed_base=SEED_TRAIN_BASE,
        use_cv=True,
    )
    with torch.no_grad():
        EU_s1 = simulate_antithetic(stage1, batch_size, train=False, seed=CRN_SEED_EU).mean().item()
    print(f"[After Train:S1] E[U] (antithetic eval) = {EU_s1:.6f}")

    # 3) P-PGDPO 사영 정책 만들기
    proj = ProjectedPolicyAntithetic(stage1)
    with torch.no_grad():
        EU_pp = simulate_antithetic(proj, batch_size, train=False, seed=CRN_SEED_EU).mean().item()
    print(f"[After Project] E[U] (P-PGDPO, antithetic eval) = {EU_pp:.6f}")

    # 4) 정책 함수 & EU 비교 (5번 파일의 출력 포맷 재사용)
    _compare_pf_v5(stage1, cf_policy)
    _compare_eu_v5(stage1, cf_policy)

    # 추가: P-PGDPO vs CF RMSE/EU 간단 숫자
    rmse_pp = _policy_rmse(proj, cf_policy)
    with torch.no_grad():
        EU_cf = simulate_antithetic(cf_policy, batch_size, train=False, seed=CRN_SEED_EU).mean().item()
    print(f"[PPGDPO vs CF] RMSE={rmse_pp:.6f}  E[U]_PP={EU_pp:.6f}  E[U]_CF={EU_cf:.6f}  Δ={EU_pp-EU_cf:.6f}")

    # 5) 리처드슨 외삽 데모 (정책 선택 가능)
    if DEMO_POLICY == "closed_form":
        demo_richardson(cf_policy, paths=N_PATHS_DEMO, seed=CRN_SEED_RICH, use_antithetic=USE_ANTITHETIC_DEMO, title="closed_form")
    elif DEMO_POLICY == "stage1":
        demo_richardson(stage1, paths=N_PATHS_DEMO, seed=CRN_SEED_RICH, use_antithetic=USE_ANTITHETIC_DEMO, title="stage1")
    elif DEMO_POLICY == "ppgdpo":
        demo_richardson(proj, paths=N_PATHS_DEMO, seed=CRN_SEED_RICH, use_antithetic=USE_ANTITHETIC_DEMO, title="ppgdpo")
    else:
        demo_richardson(cf_policy, paths=N_PATHS_DEMO, seed=CRN_SEED_RICH, use_antithetic=USE_ANTITHETIC_DEMO, title="closed_form")

if __name__ == "__main__":
    main()

__all__ = [
    "demo_richardson",
]
