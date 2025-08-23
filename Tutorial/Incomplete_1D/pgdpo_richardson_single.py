# pgdpo_richardson_single.py
import math
import torch
import torch.nn as nn
import torch.optim as optim

# ===== (1) BASE: configs, samplers, runner =====
from pgdpo_base_single import (
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, lb_W, batch_size, CF_MODE,
    make_generator, sample_initial_states, build_closed_form_policy,
    run_common, seed, CRN_SEED_EU, DirectPolicy, N_eval_states, epochs
)

# ===== (2) P-PGDPO: projection (repeats/sub_batch는 외부에서 명시) =====
from pgdpo_with_ppgdpo_single import project_pmp, REPEATS, SUBBATCH

# ===== (3) 4번의 마이오픽/레지듀얼 정책 재사용 =====
from pgdpo_residual_single import (
    MyopicPolicy, ResidualPolicy, SUBBATCH
)

SUBBATCH = SUBBATCH//2


# ---------------------------------------------------------------------
# Richardson runtime (domain-time + TRUE antithetic)
#   - Δt_i = τ_i/m, fine는 Δt_i/2
#   - coarse/fine Brownian-bridge coupling
# ---------------------------------------------------------------------
def _correlated_normals(B: int, rho_val: float, gen: torch.Generator | None):
    z1 = torch.randn(B, 1, device=device, generator=gen)
    z2 = torch.randn(B, 1, device=device, generator=gen)
    zW = z1
    zY = rho_val * z1 + math.sqrt(max(1.0 - rho_val * rho_val, 0.0)) * z2
    return zW, zY

def _simulate_richardson_core(
    policy: nn.Module,
    B: int,
    *,
    train: bool,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: torch.Tensor | None = None,     # [B,1] τ
    seed_local: int | None = None,
    noise_sign: float = 1.0,              # +1 or -1 (antithetic flip)
    m_steps: int = m,
):
    gen = make_generator(seed_local)

    # init states: domain-time (W0, Y0, TmT=τ, dt_vec=τ/m)
    if (W0 is None) or (Y0 is None) or (Tval is None):
        W0, Y0, TmT0, dt_vec = sample_initial_states(B, rng=gen)
    else:
        W0, Y0, TmT0 = W0, Y0, Tval
        dt_vec = TmT0 / float(m_steps)

    # clone for coarse/fine
    logW_c = W0.clamp(min=lb_W).log();  Y_c = Y0.clone(); T_c = TmT0.clone()
    logW_f = logW_c.clone();            Y_f = Y_c.clone(); T_f = T_c.clone()

    dt2_vec  = 0.5 * dt_vec
    sqrt_dt  = dt_vec.sqrt()
    sqrt_dt2 = dt2_vec.sqrt()

    # m coarse steps; each with 2 fine substeps
    for _ in range(int(m_steps)):
        a1, b1 = _correlated_normals(B, rho, gen)
        a2, b2 = _correlated_normals(B, rho, gen)
        # fine increments
        zW_f1, zY_f1 = a1, b1
        zW_f2, zY_f2 = a2, b2
        # coarse increment = (f1+f2)/sqrt(2)
        zW_c = (a1 + a2) / math.sqrt(2.0)
        zY_c = (b1 + b2) / math.sqrt(2.0)

        dBW_c  = noise_sign * (sqrt_dt  * zW_c);  dBY_c  = noise_sign * (sqrt_dt  * zY_c)
        dBW_f1 = noise_sign * (sqrt_dt2 * zW_f1); dBY_f1 = noise_sign * (sqrt_dt2 * zY_f1)
        dBW_f2 = noise_sign * (sqrt_dt2 * zW_f2); dBY_f2 = noise_sign * (sqrt_dt2 * zY_f2)

        # --- coarse: Δt_i ---
        with torch.set_grad_enabled(train):
            pi_c = policy(logW_c.exp(), T_c, Y_c)
        risk_c = sigma * (alpha * Y_c); varW_c = (pi_c * sigma) ** 2
        logW_c = logW_c + (r + pi_c * risk_c - 0.5 * varW_c) * dt_vec + (pi_c * sigma) * dBW_c
        Y_c    = Y_c    + kappaY * (thetaY - Y_c) * dt_vec + sigmaY * dBY_c
        logW_c = logW_c.exp().clamp(min=lb_W).log(); T_c = T_c - dt_vec

        # --- fine 1: Δt_i/2 ---
        with torch.set_grad_enabled(train):
            pi_f = policy(logW_f.exp(), T_f, Y_f)
        risk_f = sigma * (alpha * Y_f); varW_f = (pi_f * sigma) ** 2
        logW_f = logW_f + (r + pi_f * risk_f - 0.5 * varW_f) * dt2_vec + (pi_f * sigma) * dBW_f1
        Y_f    = Y_f    + kappaY * (thetaY - Y_f) * dt2_vec + sigmaY * dBY_f1
        logW_f = logW_f.exp().clamp(min=lb_W).log(); T_f = T_f - dt2_vec

        # --- fine 2: Δt_i/2 ---
        with torch.set_grad_enabled(train):
            pi_f = policy(logW_f.exp(), T_f, Y_f)
        risk_f = sigma * (alpha * Y_f); varW_f = (pi_f * sigma) ** 2
        logW_f = logW_f + (r + pi_f * risk_f - 0.5 * varW_f) * dt2_vec + (pi_f * sigma) * dBW_f2
        Y_f    = Y_f    + kappaY * (thetaY - Y_f) * dt2_vec + sigmaY * dBY_f2
        logW_f = logW_f.exp().clamp(min=lb_W).log(); T_f = T_f - dt2_vec

    # terminal utilities + Richardson
    Wc_T = logW_c.exp(); Wf_T = logW_f.exp()
    if abs(gamma - 1.0) < 1e-10:
        Uc = Wc_T.log(); Uf = Wf_T.log()
    else:
        c = 1.0 - gamma
        Uc = Wc_T.pow(c) / c
        Uf = Wf_T.pow(c) / c
    return 2.0 * Uf - Uc   # U* = 2Uf - Uc

def simulate_richardson_antithetic(
    policy_module: nn.Module,
    B: int, *,
    train: bool,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: torch.Tensor | None = None,
    seed_local: int | None = None,
):
    """TRUE antithetic 평균 of Richardson utility."""
    U_pos = _simulate_richardson_core(policy_module, B, train=train,
                                      W0=W0, Y0=Y0, Tval=Tval,
                                      seed_local=seed_local, noise_sign=+1.0)
    U_neg = _simulate_richardson_core(policy_module, B, train=train,
                                      W0=W0, Y0=Y0, Tval=Tval,
                                      seed_local=seed_local, noise_sign=-1.0)
    return 0.5 * (U_pos + U_neg)


# ---------------------------------------------------------------------
# Costates (Richardson utility; no CV in costates)
# ---------------------------------------------------------------------
def estimate_costates_richardson(
    policy_net: nn.Module,
    T0: torch.Tensor, W0: torch.Tensor, Y0: torch.Tensor,
    *, repeats: int, sub_batch: int,
    seed_eval: int | None = None,
):
    """
    Returns (λ=J_W, J_WW, J_WY) using TRUE antithetic paired Richardson utilities.
    """
    n = W0.size(0)
    W0g = W0.clone().requires_grad_(True)
    Y0g = Y0.clone().requires_grad_(True)

    lam_sum = torch.zeros_like(W0g)
    dW_sum  = torch.zeros_like(W0g)
    dY_sum  = torch.zeros_like(Y0g)

    # freeze params
    params = list(policy_net.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params: p.requires_grad_(False)

    try:
        seed_ctr = int(seed_eval) if (seed_eval is not None) else None
        done = 0
        while done < repeats:
            rpts = min(sub_batch, repeats - done)
            T_b = T0.repeat(rpts, 1)
            W_b = W0g.repeat(rpts, 1)
            Y_b = Y0g.repeat(rpts, 1)

            U = simulate_richardson_antithetic(
                policy_net, n * rpts, train=True,
                W0=W_b, Y0=Y_b, Tval=T_b, seed_local=seed_ctr
            )
            seed_ctr = None if seed_ctr is None else seed_ctr + 1

            U_avg = U.view(rpts, n, 1).mean(dim=0)            # [n,1]
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
    return lam_sum * inv, dW_sum * inv, dY_sum * inv

def ppgdpo_pi_richardson(
    policy_s1: nn.Module,
    W: torch.Tensor, TmT: torch.Tensor, Y: torch.Tensor,
    *, repeats: int, sub_batch: int,
    seed_eval: int | None = None,
):
    """PMP projection with costates estimated from Richardson utility."""
    with torch.enable_grad():
        lam, dlamW, dlamY = estimate_costates_richardson(
            policy_s1, TmT, W, Y, repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval
        )
        pi = project_pmp(lam, dlamW, dlamY, W, Y)
    return pi.detach()


# ---------------------------------------------------------------------
# Stage-1 (Residual + CV + Antithetic + Richardson) + RMSE (Rich teacher)
# ---------------------------------------------------------------------
# CV 하이퍼파라미터
USE_CV       = True
CV_BETA_CLIP = 5.0
CV_EMA_DECAY = 0.1
CV_EPS       = 1e-8

@torch.no_grad()
def _cv_coeff_only(u_pol: torch.Tensor, u_my: torch.Tensor, W0: torch.Tensor):
    """
    c_hat = Cov(U_pol_norm, U_my_centered_norm) / Var(U_my_centered_norm)
    (여기서만 detach/no-grad; 손실에 들어갈 U_pol_norm은 detach 금지)
    """
    scale       = W0.pow(1.0 - gamma)                    # [B,1]
    u_pol_n_det = u_pol.detach() / (scale + CV_EPS)
    u_my_cn     = (u_my - u_my.mean()) / (scale + CV_EPS)

    var_z = (u_my_cn * u_my_cn).mean() + CV_EPS
    cov_pz = ((u_pol_n_det - u_pol_n_det.mean()) * u_my_cn).mean()
    c_hat = cov_pz / var_z
    if CV_BETA_CLIP is not None:
        c_hat = torch.clamp(c_hat, -abs(CV_BETA_CLIP), abs(CV_BETA_CLIP))
    return c_hat

def train_residual_stage1_cv_richardson(
    *,
    epochs: int = epochs,
    lr: float = 1e-3,
    residual_cap: float = 1.0,
    seed_train: int | None = seed,
    use_cv: bool = USE_CV,
) -> nn.Module:
    """
    5번과 동일한 학습 구성(마이오픽 베이스 + Residual + CV + TRUE antithetic + 도메인-시간),
    단, 시뮬레이터만 Richardson-extrapolated 유틸리티를 사용.
    """
    base_policy = MyopicPolicy().to(device)
    pol = ResidualPolicy(base_policy, residual_cap=residual_cap).to(device)
    opt = optim.Adam(pol.parameters(), lr=lr)

    c_ema = None  # EMA 상태 (torch.scalar)

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        # 에폭별 로컬 시드 (TRUE antithetic은 ± 같은 시드)
        pair_seed = None if seed_train is None else int(seed_train) + ep
        gen = make_generator(pair_seed)

        # 동일 초기상태(공유 CRN): (W0, Y0, τ)
        W0, Y0, TmT, _dt = sample_initial_states(batch_size, rng=gen)

        # 정책/마이오픽 유틸리티 (둘 다 Richardson + TRUE antithetic, 동일 CRN)
        U_pol = simulate_richardson_antithetic(pol,         batch_size, train=True,
                                               W0=W0, Y0=Y0, Tval=TmT, seed_local=pair_seed)
        with torch.no_grad():
            U_my  = simulate_richardson_antithetic(base_policy, batch_size, train=False,
                                                   W0=W0, Y0=Y0, Tval=TmT, seed_local=pair_seed)

        if use_cv:
            # 1) 계수 추정 (no-grad)
            c_hat = _cv_coeff_only(U_pol, U_my, W0)
            # 2) EMA 갱신
            c_ema = c_hat.detach() if (c_ema is None) else ((1.0 - CV_EMA_DECAY) * c_ema + CV_EMA_DECAY * c_hat.detach())
            # 3) 손실용 정규화 텐서 (grad 필요)
            scale   = W0.pow(1.0 - gamma)
            U_pol_n = U_pol / (scale + CV_EPS)              # 그래프 유지
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

    return pol

@torch.no_grad()
def compare_policy_functions_richardson(
    stage1_policy: nn.Module,
    cf_policy: nn.Module,
    *,
    seed_eval: int | None = CRN_SEED_EU,
    repeats: int = REPEATS,
    sub_batch: int = SUBBATCH,
):
    """도메인-시간 샘플링으로 (W,Y,τ)을 뽑아 π_res, π_pp^rich, π_cf RMSE 비교."""
    gen = make_generator(seed_eval)
    W, Y, TmT, _dt = sample_initial_states(N_eval_states, rng=gen)

    pi_s1 = stage1_policy(W, TmT, Y)
    pi_pp = ppgdpo_pi_richardson(stage1_policy, W, TmT, Y,
                                 repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval)
    pi_cf = cf_policy(W, TmT, Y)

    rmse_s1_cf = torch.sqrt(((pi_s1 - pi_cf)**2).mean()).item()
    rmse_pp_cf = torch.sqrt(((pi_pp - pi_cf)**2).mean()).item()
    rmse_s1_pp = torch.sqrt(((pi_s1 - pi_pp)**2).mean()).item()

    print(f"[Policy RMSE] Stage-1(res+cv, rich) vs CF : {rmse_s1_cf:.6f}")
    print(f"[Policy RMSE] P-PGDPO(rich) vs CF        : {rmse_pp_cf:.6f}")
    print(f"[Policy RMSE] Stage-1 vs P-PGDPO         : {rmse_s1_pp:.6f}")

    idxs = [0, N_eval_states//2, N_eval_states-1]
    for i in idxs:
        print(f"  (W={W[i].item():.3f}, Y={Y[i].item():.3f}, τ={TmT[i].item():.2f})"
              f" -> (π_res={pi_s1[i,0].item():.4f}, π_pp(rich)={pi_pp[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})")


# ---------------------------------------------------------------------
# Run (reuse common runner)
# ---------------------------------------------------------------------
def main():
    run_common(
        train_fn=lambda seed_train=seed: train_residual_stage1_cv_richardson(seed_train=seed_train, use_cv=USE_CV),
        rmse_fn=compare_policy_functions_richardson,
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU, "repeats": REPEATS, "sub_batch": SUBBATCH},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "train_residual_stage1_cv_richardson",
    "compare_policy_functions_richardson",
]
