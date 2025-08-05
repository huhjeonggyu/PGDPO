# utils.py
# 역할: 플로팅, 데이터 생성, 기타 공통 계산 등 보조 함수

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_uniform_domain(n, T_max, W_min, W_max, Y_min_vec, Y_max_vec, m, k, dev, seed=None):
    """
    학습을 위한 초기 상태(T, W, Y)를 샘플링합니다. [cite: 68, 71]
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    T = T_max * torch.rand([n, 1], device=dev) # [cite: 73]
    dt = T / m
    W = W_min + (W_max - W_min) * torch.rand([n, 1], device=dev) # [cite: 77]
    rand_vals = torch.rand([n, k], device=dev)
    Y = Y_min_vec.unsqueeze(0) + (Y_max_vec - Y_min_vec).unsqueeze(0) * rand_vals # [cite: 80]
    return T, W, Y, dt

def block_corr_matrix(Psi, Phi_Y, rho_Y):
    """
    전체 시스템의 (d+k) x (d+k) 상관관계 행렬을 만듭니다. [cite: 43]
    """
    top = torch.cat([Psi, rho_Y], dim=1)
    bot = torch.cat([rho_Y.T, Phi_Y], dim=1)
    block_corr = torch.cat([top, bot], dim=0)
    try:
        eigvals = torch.linalg.eigvalsh(block_corr)
        min_eigval = eigvals.min().item()
        if min_eigval < 1e-6: print(f"❌ Not Positive Definite (min eigval: {min_eigval:.2e})")
    except torch.linalg.LinAlgError as e:
        print(f"❌ Error during eigenvalue calculation: {e}")
    return block_corr

def compute_mu_of_Y(Y, alpha, sigma, r):
    """
    Y에 따른 자산의 기대수익률 벡터를 계산합니다.
    """
    if Y.ndim == 1:
        Y = Y.unsqueeze(0)
    alphaY = alpha @ Y.transpose(-1, -2)
    alphaY = alphaY * sigma.unsqueeze(-1)
    alphaY = alphaY + r
    return alphaY.transpose(-1, -2)

def get_optimal_pi(X, lam, dlam_dx, dlam_dY, Y, alpha, sigma, Sigma, sigma_Y, rho_Y, Psi, r, gamma, device=torch.device('cpu')):
    """
    추정된 co-state를 사용하여 PMP 기반의 최적 정책(P-PGDPO)을 계산합니다. [cite: 275, 356]
    문서 Listing 7의 `compute_projected_policy`에 해당합니다.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    lam_t = torch.as_tensor(lam, dtype=torch.float32, device=device)
    dlamdx_t = torch.as_tensor(dlam_dx, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
    dlam_dY_t = torch.as_tensor(dlam_dY, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    Sigma_t = torch.as_tensor(Sigma, dtype=torch.float32, device=device)
    sigma_Y_t = torch.as_tensor(sigma_Y, dtype=torch.float32, device=device)
    rho_Y_t = torch.as_tensor(rho_Y, dtype=torch.float32, device=device)
    Psi_t = torch.as_tensor(Psi, dtype=torch.float32, device=device)
    
    # 공통 계수 [cite: 365]
    scalar_coeff = -1.0 / (X_t * dlamdx_t)

    # Myopic 수요 [cite: 269, 358]
    mu_vec = compute_mu_of_Y(Y_t, alpha_t, sigma_t, r_t)
    myopic_raw = lam_t * (mu_vec - r_t)
    bracket_myopic = myopic_raw.transpose(-1, -2)
    solved_myopic = torch.linalg.solve(Sigma_t, bracket_myopic).transpose(-1, -2)
    myopic = scalar_coeff * solved_myopic

    # Intertemporal Hedging 수요 [cite: 270, 359]
    dlam_dY_sY = dlam_dY_t @ sigma_Y_t
    temp1 = dlam_dY_sY @ rho_Y_t.transpose(-1, -2)
    hedging_raw = sigma_t * temp1 
    bracket_hedging = hedging_raw.transpose(-1, -2)
    solved_hedging = torch.linalg.solve(Sigma_t, bracket_hedging).transpose(-1, -2)
    hedging = scalar_coeff * solved_hedging
    
    total = myopic + hedging
    return total, myopic, hedging

def analytic_EUM_vectorized(W0, T, pi_M, y_bar, alpha, sigma, Psi, r, gamma):
    """
    Control Variate를 위한 간단한 정책의 분석적 기대 효용을 계산합니다. [cite: 16]
    """
    B, pi_M_b = W0.shape[0], pi_M.expand(W0.shape[0], -1)
    risk = (alpha @ y_bar).expand(B, -1)
    mu_M = r + (pi_M_b * sigma * risk).sum(dim=1)
    sig2_M = ((pi_M_b * sigma) @ Psi @ (pi_M_b * sigma).T).diagonal()
    A = (1 - gamma) * (mu_M - 0.5 * gamma * sig2_M) * T.squeeze(-1)
    exp_term = torch.exp(A.clamp(max=14.0))
    EU = (W0.squeeze(-1).pow(1 - gamma) / (1 - gamma)) * exp_term
    return EU.detach().unsqueeze(-1)

def plot(iter, U, T, W, Y, kth, dt, lamb, dx_lamb, dy_lamb, net_pi,
         basis_folder, seed, interpolators, d, k,
         alpha, sigma, Sigma, sigma_Y, rho_Y, Psi, r, gamma, dev, use_residual_net):
    """
    평가 결과를 시각화하고 오차를 계산합니다. 문서 Appendix C.2에 해당합니다. [cite: 563]
    """
    folder = os.path.join(basis_folder, f'iter_{iter:05d}')
    os.makedirs(folder, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)

    def draw(t, y, u, filename, xlabel_text):
        """등고선 플롯(tricontourf)을 그리는 헬퍼 함수"""
        valid_mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(u)
        t, y, u = t[valid_mask], y[valid_mask], u[valid_mask]

        if len(u) < 4:
            print(f"⚠️  Skipping plot '{os.path.basename(filename)}': Not enough valid data points ({len(u)}).")
            return

        levels = 100
        plt.figure(figsize=(6.5, 5))
        try:
            contourf_plot = plt.tricontourf(y, t, u, levels=levels, cmap='jet')
        except Exception:
            contourf_plot = plt.tricontourf(y, t, u, cmap='jet')

        plt.xlabel(xlabel_text)
        plt.ylabel('T-t')
        plt.colorbar(contourf_plot, label=f'Policy Value')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    T_cpu, W_cpu, Y_cpu = T.cpu(), W.cpu(), Y.cpu()
    lamb_cpu, dx_lamb_cpu, dy_lamb_cpu = lamb.cpu(), dx_lamb.cpu(), dy_lamb.cpu()
    t_plot, ys_plot = T_cpu.numpy().squeeze(), Y_cpu.numpy()
    xlabel = f'Y_{kth}'

    with torch.no_grad():
        state_tensor = torch.cat([W_cpu, T_cpu, Y_cpu], dim=1).to(dev)
        a_pi = net_pi(state_tensor).cpu().numpy()

    _, _, pi1_lamb = get_optimal_pi(W_cpu, lamb_cpu, dx_lamb_cpu, dy_lamb_cpu, Y_cpu, alpha.cpu(), sigma.cpu(), Sigma.cpu(), sigma_Y.cpu(), rho_Y.cpu(), Psi.cpu(), r, gamma, device='cpu')
    pi1_lamb_np = pi1_lamb.numpy()
    
    err_pi, err_h_net, err_pi_lamb, err_pi0_lamb, err_pi1_lamb = 0., 0., 0., 0., 0.

    for dth in range(d):
        query_points = np.column_stack([t_plot, ys_plot[:, kth]])
        pi0_sol = interpolators[(dth, kth, 'Myopic')](query_points)
        pi1_sol = interpolators[(dth, kth, 'Hedging')](query_points)
        pi_sol = interpolators[(dth, kth, 'Total')](query_points)

        pi0_lamb_np_dth = pi0_sol
        pi_lamb_np_dth = pi0_sol + pi1_lamb_np[:, dth]

        valid_mask = ~np.isnan(pi_sol) & ~np.isnan(a_pi[:,dth])
        if np.sum(valid_mask) == 0: continue

        a_pi_valid, pi_sol_valid, pi0_sol_valid, pi1_sol_valid = a_pi[valid_mask, dth], pi_sol[valid_mask], pi0_sol[valid_mask], pi1_sol[valid_mask]
        pi_lamb_valid, pi0_lamb_valid, pi1_lamb_valid = pi_lamb_np_dth[valid_mask], pi0_lamb_np_dth[valid_mask], pi1_lamb_np[valid_mask, dth]

        err_pi += np.mean((a_pi_valid - pi_sol_valid)**2)
        err_h_net += np.mean(((a_pi_valid - pi0_sol_valid) - pi1_sol_valid)**2) if use_residual_net else 0.0
        err_pi_lamb += np.mean((pi_lamb_valid - pi_sol_valid)**2)
        err_pi0_lamb += np.mean((pi0_lamb_valid - pi0_sol_valid)**2)
        err_pi1_lamb += np.mean((pi1_lamb_valid - pi1_sol_valid)**2)

        if dth < 5 and kth == 0:
            draw(t_plot, ys_plot[:, kth], pi_sol, os.path.join(folder, f"pi_sol_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], a_pi[:, dth], os.path.join(folder, f"pi_net_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], a_pi[:, dth] - pi_sol, os.path.join(folder, f"pi_err_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], pi_lamb_np_dth, os.path.join(folder, f"pi_lamb_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], pi_lamb_np_dth - pi_sol, os.path.join(folder, f"pi_lamb_err_x_{dth}_y_{kth}.png"), xlabel)

    inv_d = 1.0 / d if d > 0 else 1.0
    err_pi, err_h_net = np.sqrt(err_pi * inv_d), np.sqrt(err_h_net * inv_d)
    err_pi_lamb, err_pi0_lamb, err_pi1_lamb = np.sqrt(err_pi_lamb*inv_d), np.sqrt(err_pi0_lamb*inv_d), np.sqrt(err_pi1_lamb*inv_d)
    
    res = (f"i:{iter:5d} kth:{kth:3d} | "
           f"U:{U:8.2e} | "
           f"err_pi:{err_pi:8.2e} err_h_net:{err_h_net:8.2e} | "
           f"err_pi_lamb:{err_pi_lamb:8.2e} err_pi1_lamb:{err_pi1_lamb:8.2e}")
    
    with open(os.path.join(folder, 'errs.txt'), 'wt' if kth == 0 else 'at') as f:
        print(res, file=f)
    print(res)