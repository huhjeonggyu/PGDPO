# pgdpo_run_single.py  
import math
import torch
import torch.nn as nn

# ===== (1) BASE: configs, samplers, runner =====
from pgdpo_base_single import (
    device, r, gamma, sigma, kappaY, thetaY, sigmaY, rho, alpha,
    T, m, batch_size, W0_range, Y0_range, lb_W, pi_cap, CF_MODE,
    N_eval_states, CRN_SEED_EU, epochs,
    make_generator, sample_initial_states,
    run_common, seed,
)

# ===== (2) WITH-PPGDPO: projector + repeats/subbatch =====
from pgdpo_with_ppgdpo_single import (
    project_pmp, REPEATS, SUBBATCH,
)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _crra_utility(WT: torch.Tensor, g: float) -> torch.Tensor:
    if abs(g - 1.0) < 1e-8:
        return torch.log(WT.clamp(min=1e-12))
    return (WT.clamp(min=1e-12).pow(1.0 - g) - 1.0) / (1.0 - g)

def _draw_correlated_normals(B: int, steps: int, gen: torch.Generator | None):
    z1 = torch.randn(B, steps, device=device, generator=gen)
    z2 = torch.randn(B, steps, device=device, generator=gen)
    zW = z1
    zY = rho*z1 + math.sqrt(max(0.0, 1.0 - rho*rho)) * z2
    return zW, zY

def _forward_path(policy: nn.Module, W0, TmT, Y0, ZW, ZY, steps):
    dt = (TmT / float(steps))         # [B,1]
    sqrt_dt = torch.sqrt(dt)

    logW = W0.clamp_min(lb_W).log()
    Y    = Y0.clone()
    for k in range(steps):
        # DirectPolicy(W, TmT, Y)  — inputs are [B,1] each
        pi = policy(logW.exp(), TmT - k*dt, Y)  # [B,1]
        if pi_cap is not None:
            pi = pi.clamp(min=-pi_cap, max=pi_cap)
        dBW = ZW[:, k:k+1] * sqrt_dt
        dBY = ZY[:, k:k+1] * sqrt_dt
        drift = r + pi * sigma * alpha * Y
        vol   = pi * sigma
        logW  = logW + (drift - 0.5 * vol*vol) * dt + vol * dBW
        Y     = Y + kappaY * (thetaY - Y) * dt + sigmaY * dBY
    return logW.exp().clamp_min(lb_W)

# ------------------------------------------------------------------
# ALWAYS RUN = TRUE Antithetic + Richardson (2*U_f - U_c)
# ------------------------------------------------------------------

def simulate_run(
    policy: nn.Module,
    B: int | None = None,
    *,
    W0: torch.Tensor | None = None,
    Y0: torch.Tensor | None = None,
    Tval: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    seed_local: int | None = None,
) -> torch.Tensor:
    """
    ALWAYS returns U_run = 2*U_fine - U_coarse for a batch.
    If (W0,Y0,Tval) are None, uses domain-time sampling for size B.
    """
    if rng is None and seed_local is not None:
        rng = make_generator(seed_local)

    if W0 is None or Y0 is None or Tval is None:
        assert B is not None, "Either B or (W0,Y0,Tval) must be provided."
        W0, Y0, TmT, _ = sample_initial_states(B, rng=rng)
    else:
        W0 = W0 if W0.ndim == 2 else W0.unsqueeze(1)
        Y0 = Y0 if Y0.ndim == 2 else Y0.unsqueeze(1)
        TmT = Tval if Tval.ndim == 2 else Tval.unsqueeze(1)
        B = W0.size(0)

    # coarse
    ZWc, ZYc = _draw_correlated_normals(B, m,   rng)
    WTc_p = _forward_path(policy, W0, TmT, Y0, +ZWc, +ZYc, m)
    WTc_m = _forward_path(policy, W0, TmT, Y0, -ZWc, -ZYc, m)
    Uc = 0.5 * (_crra_utility(WTc_p, gamma) + _crra_utility(WTc_m, gamma))

    # fine (decorrelated)
    rng_f = make_generator((seed_local or 0) + 8191) if seed_local is not None else None
    ZWf, ZYf = _draw_correlated_normals(B, 2*m, rng_f)
    WTf_p = _forward_path(policy, W0, TmT, Y0, +ZWf, +ZYf, 2*m)
    WTf_m = _forward_path(policy, W0, TmT, Y0, -ZWf, -ZYf, 2*m)
    Uf = 0.5 * (_crra_utility(WTf_p, gamma) + _crra_utility(WTf_m, gamma))

    return 2.0*Uf - Uc  # [B,1]

# ------------------------------------------------------------------
# Vectorized costate estimation (BATCH, RUN) with proper REPEATS/SUBBATCH
# + Freeze policy params during estimation
# ------------------------------------------------------------------

def estimate_costates_run(
    policy: nn.Module,
    T0: torch.Tensor, W0: torch.Tensor, Y0: torch.Tensor,   # [B] or [B,1]
    *,
    repeats: int = REPEATS,
    sub_batch: int = SUBBATCH,
    seed_eval: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Proper semantics:
      - repeats: total number of Monte Carlo paths per evaluation point.
      - sub_batch: memory-safe chunking; each chunk replicates (W0,Y0,T0) r_chunk times,
                   simulates on B*r_chunk paths, averages over r_chunk for each point,
                   then backpropagates through that averaged scalar.
    """
    # normalize shapes
    if W0.ndim == 1: W0 = W0.unsqueeze(1)
    if Y0.ndim == 1: Y0 = Y0.unsqueeze(1)
    if T0.ndim == 1: T0 = T0.unsqueeze(1)
    B = W0.size(0)

    # leafs
    W0_leaf = W0.detach().clone().requires_grad_(True)
    Y0_leaf = Y0.detach().clone().requires_grad_(True)
    T0_leaf = T0.detach().clone()

    # accumulators (graph-free)
    J_W_sum  = torch.zeros_like(W0_leaf)
    J_WW_sum = torch.zeros_like(W0_leaf)
    J_WY_sum = torch.zeros_like(Y0_leaf)

    # freeze policy params for memory/speed
    params = list(policy.parameters())
    req_bak = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)

    try:
        # repeat-by-chunk loop
        remain = int(repeats)
        sd = int(seed_eval) if seed_eval is not None else 1234567

        while remain > 0:
            r_chunk = min(int(sub_batch), remain)
            remain -= r_chunk

            # replicate states r_chunk times -> tensors of shape [B*r_chunk, 1]
            W_b = W0_leaf.repeat(r_chunk, 1)
            Y_b = Y0_leaf.repeat(r_chunk, 1)
            T_b = T0_leaf.repeat(r_chunk, 1)

            with torch.enable_grad():
                # one chunk utility using RUN simulator (antithetic + Richardson)
                U = simulate_run(policy, B * r_chunk, W0=W_b, Y0=Y_b, Tval=T_b, seed_local=sd)  # [B*r_chunk, 1]
                sd += 10007  # decorrelate chunks deterministically

                # per-point average over the r_chunk repeats
                U_bar = U.view(r_chunk, B, 1).mean(dim=0)   # [B,1]
                J = U_bar.mean()                             # scalar

                # first derivative w.r.t. W0 (need create_graph=True for 2nd derivs)
                J_W = torch.autograd.grad(
                    J, W0_leaf, retain_graph=True, create_graph=True
                )[0]  # [B,1]

                # second derivatives (graph freed here)
                J_WW, J_WY = torch.autograd.grad(
                    J_W.sum(), (W0_leaf, Y0_leaf),
                    retain_graph=False, create_graph=False, allow_unused=True
                )
                if J_WW is None: J_WW = torch.zeros_like(W0_leaf)
                if J_WY is None: J_WY = torch.zeros_like(Y0_leaf)

            # accumulate with r_chunk weights (since we averaged r_chunk inside this chunk)
            J_W_sum  += J_W.detach()  * r_chunk
            J_WW_sum += J_WW.detach() * r_chunk
            J_WY_sum += J_WY.detach() * r_chunk
    finally:
        # restore flags
        for p, r in zip(params, req_bak):
            p.requires_grad_(r)

    inv = 1.0 / float(repeats)
    return J_W_sum * inv, J_WW_sum * inv, J_WY_sum * inv

def ppgdpo_pi_run(
    policy: nn.Module,
    T0: torch.Tensor, W0: torch.Tensor, Y0: torch.Tensor,
    *,
    repeats: int = REPEATS,
    sub_batch: int = SUBBATCH,
    seed_eval: int | None = None,
) -> torch.Tensor:
    with torch.enable_grad():
        J_W, J_WW, J_WY = estimate_costates_run(
            policy, T0, W0, Y0, repeats=repeats, sub_batch=sub_batch, seed_eval=seed_eval
        )
    # project_pmp expects [B,1] shapes; broadcast-safe
    W_in = W0 if W0.ndim == 2 else W0.unsqueeze(1)
    Y_in = Y0 if Y0.ndim == 2 else Y0.unsqueeze(1)
    return project_pmp(J_W, J_WW, J_WY, W_in, Y_in)  # [B,1]

# ------------------------------------------------------------------
# Evaluation helpers (adaptive tiling)
# ------------------------------------------------------------------

def _divisors_desc(n: int):
    ds = []
    for d in range(1, n + 1):
        if n % d == 0:
            ds.append(d)
    ds.sort(reverse=True)
    return ds

# ------------------------------------------------------------------
# Stage-1 warm-up (ALWAYS RUN)
# ------------------------------------------------------------------

def train_stage1_run(epochs_override=None, lr_override=None, seed_train: int | None = None):
    """
    Stage-1 warm-up using ALWAYS RUN (antithetic + Richardson).
    Logs/hypers follow the base file.
    """
    from pgdpo_base_single import DirectPolicy, lr
    _epochs = epochs if epochs_override is None else int(epochs_override)
    _lr = lr if lr_override is None else float(lr_override)

    policy = DirectPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=_lr)
    gen = make_generator(seed_train)

    for ep in range(1, _epochs+1):
        opt.zero_grad()
        U = simulate_run(policy, batch_size, W0=None, Y0=None, Tval=None, rng=gen)  # RUN
        loss = -U.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if ep % 25 == 0 or ep == 1:
            print(f"[{ep:04d}] loss={loss.item():.6f}")
    return policy

# ------------------------------------------------------------------
# Evaluation (RUN; adaptive tiled)
# ------------------------------------------------------------------

@torch.no_grad()
def print_policy_rmse_and_samples_run(
    pol_s1: nn.Module,
    pol_cf: nn.Module,
    *,
    seed_eval: int | None = CRN_SEED_EU,
    repeats: int,
    sub_batch: int,
    tile: int | None = None,   # if None -> start at N_eval_states, then move to smaller divisors on OOM
) -> None:
    """
    Memory-safe evaluation with adaptive tiling:
      - Start with tile = N_eval_states (or provided tile snapped to nearest divisor ≤ tile).
      - On OOM: silently clear cache and fall back to the next smaller divisor of N_eval_states.
      - Only prints a short notice like: "[Eval] OOM; reducing tile -> 250".
    """
    # 1) shared evaluation states
    gen = make_generator(seed_eval)
    W, Y, TmT, _ = sample_initial_states(N_eval_states, rng=gen)  # [B,1] each

    # 2) Stage-1 vs CF (no_grad, full batch)
    pi_learn = pol_s1(W, TmT, Y)       # [B,1]
    pi_cf    = pol_cf(W, TmT, Y)       # [B,1]
    rmse_learn = torch.sqrt(((pi_learn - pi_cf)**2).mean()).item()
    print(f"[Policy RMSE] Stage-1 vs CF:         {rmse_learn:.6f} (over {N_eval_states})")

    # 3) Build divisor ladder
    divisors = _divisors_desc(N_eval_states)  # e.g., [1000, 500, 250, 200, 125, ... , 1]
    if tile is None:
        # use full batch start
        start_idx = 0
    else:
        # snap to the nearest divisor <= tile
        start_idx = 0
        for i, d in enumerate(divisors):
            if d <= int(tile):
                start_idx = i
                break

    # 4) Try tiles until success
    exc_msg_printed = False
    for idx in range(start_idx, len(divisors)):
        cur_tile = divisors[idx]
        try:
            pi_pp = torch.empty_like(pi_learn)  # [B,1]
            # tiled computation (OOM safe)
            B = W.size(0)
            for s in range(0, B, cur_tile):
                e = min(B, s + cur_tile)
                T0 = TmT[s:e, 0].detach()
                W0 = W[s:e, 0].detach()
                Y0 = Y[s:e, 0].detach()

                with torch.enable_grad():
                    # vary seed per tile to decorrelate across tiles
                    tile_seed = None if seed_eval is None else int(seed_eval) + s
                    pi_pp[s:e] = ppgdpo_pi_run(
                        pol_s1, T0, W0, Y0,
                        repeats=repeats,
                        sub_batch=sub_batch,
                        seed_eval=tile_seed,
                    )
            # success
            rmse_pp = torch.sqrt(((pi_pp - pi_cf)**2).mean()).item()
            print(f"[Policy RMSE] P-PGDPO(run) vs CF:    {rmse_pp:.6f} (over {N_eval_states}, tile={cur_tile})")
            # sample prints
            for i in [0, N_eval_states // 2, N_eval_states - 1]:
                print(
                    f"  (W={W[i,0].item():.3f}, Y={Y[i,0].item():.3f}, τ={TmT[i,0].item():.2f})"
                    f" -> (π_s1={pi_learn[i,0].item():.4f}, π_pp(run)={pi_pp[i,0].item():.4f}, π_cf={pi_cf[i,0].item():.4f})"
                )
            break  # done
        except RuntimeError as e:
            emsg = str(e).lower()
            if "out of memory" in emsg or "cuda" in emsg and "memory" in emsg:
                # clear CUDA cache and move to next smaller divisor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_idx = idx + 1
                if next_idx < len(divisors):
                    next_tile = divisors[next_idx]
                    print(f"[Eval] OOM; reducing tile -> {next_tile}")
                    continue
                else:
                    print("[Eval] OOM; could not reduce tile further (tile=1).")
                    # silently give up with current partial result structure
                    raise
            else:
                # non-OOM error: re-raise
                raise

# ========================= Run (reuse common runner) =========

def main():
    run_common(
        train_fn=lambda seed_train=seed: train_stage1_run(seed_train=seed_train),
        rmse_fn=print_policy_rmse_and_samples_run,
        seed_train=seed,
        rmse_kwargs={"seed_eval": CRN_SEED_EU, "repeats": REPEATS, "sub_batch": SUBBATCH},
    )

if __name__ == "__main__":
    main()

__all__ = [
    "simulate_run",
    "estimate_costates_run",
    "ppgdpo_pi_run",
    "train_stage1_run",
    "print_policy_rmse_and_samples_run",
]