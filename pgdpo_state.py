#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd.functional import jacobian
from torch.distributions.multivariate_normal import MultivariateNormal

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.covariance import LedoitWolf

from scipy.stats import dirichlet
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

################################################################
# Command line arguments
################################################################
if len(sys.argv) < 4:
    print("Usage: python3 state.py <d> <k> <cuda_num> [--res] [--cv] [--no_richardson]")
    sys.exit(1)

d = int(sys.argv[1])
k = int(sys.argv[2])
cuda_num = int(sys.argv[3])
use_residual_net = '--res' in sys.argv
use_cv = '--cv' in sys.argv
use_richardson = '--no_richardson' not in sys.argv

torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True, precision=4, linewidth=100)

dev = f'cuda:{cuda_num}'
seed = 1

print(f"--- Configuration ---\n"
      f"Residual Network (--res): {use_residual_net}\n"
      f"Control Variate (--cv):  {use_cv}\n"
      f"Richardson Extrapolation: {use_richardson}\n"
      f"-----------------------")

def generate_multifactor_capm_structure_params(d, k, r=0.03, beta_corr_max=0.8, rho_max=0.2, seed=None):
    PD_tolerance = 1e-6
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    kappa_Y = torch.diag(torch.tensor([2.0 + i * 0.5 for i in range(k)]))
    theta_Y = torch.empty(k).uniform_(0.2, 0.4)
    sigma_Y = torch.diag(torch.empty(k).uniform_(0.3, 0.5))
    sigma = torch.empty(d).uniform_(0.1, 0.5)
    alpha = torch.tensor(dirichlet.rvs([1.0] * k, size=d), dtype=torch.float32)

    if beta_corr_max >= 1.0:
        raise ValueError("beta_corr_max must be less than 1 for Psi to be PD.")
    beta_corr = torch.empty(d).uniform_(-beta_corr_max, beta_corr_max)
    Psi = torch.outer(beta_corr, beta_corr)
    Psi.fill_diagonal_(1.0)

    factor_dim_k = max(1, k)
    Z_Y = torch.randn(k, factor_dim_k)
    corr_Y = Z_Y @ Z_Y.T
    diag_corr_Y = torch.diag(corr_Y)
    diag_corr_Y[diag_corr_Y <= 1e-8] = 1e-8
    d_inv_sqrt_Y = torch.diag(1.0 / torch.sqrt(diag_corr_Y))
    Phi_Y = d_inv_sqrt_Y @ corr_Y @ d_inv_sqrt_Y
    Phi_Y.fill_diagonal_(1.0)

    rho_Y = torch.empty(d, k).uniform_(-rho_max, rho_max)

    block_corr = torch.zeros((d + k, d + k), dtype=torch.float32)
    block_corr[:d, :d] = Psi
    block_corr[d:, d:] = Phi_Y
    block_corr[:d, d:] = rho_Y
    block_corr[d:, :d] = rho_Y.T

    try:
        eigvals = torch.linalg.eigvalsh(block_corr)
        min_eigval = eigvals.min().item()
        if min_eigval < PD_tolerance:
            adaptive_epsilon = max(0.0, -min_eigval) + 1e-4
            block_corr = block_corr + adaptive_epsilon * torch.eye(d + k)
            diag_B = torch.diag(block_corr)
            diag_B[diag_B <= 1e-8] = 1e-8
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(diag_B))
            block_corr = D_inv_sqrt @ block_corr @ D_inv_sqrt
            block_corr.fill_diagonal_(1.0)
            Psi = block_corr[:d, :d]
            Phi_Y = block_corr[d:, d:]
            rho_Y = block_corr[:d, d:]
    except torch.linalg.LinAlgError as e:
        print(f"Error during PD check: {e}")

    return {
        'r': r, 'kappa_Y': kappa_Y.float(), 'theta_Y': theta_Y.float(),
        'sigma_Y': sigma_Y.float(), 'sigma': sigma.float(), 'alpha': alpha.float(),
        'Phi_Y': Phi_Y.float(), 'Psi': Psi.float(), 'rho_Y': rho_Y.float()
    }

params = generate_multifactor_capm_structure_params(d=d, k=k, seed=42)

r = params['r']
kappa_Y = params['kappa_Y'].float().to(dev)
theta_Y = params['theta_Y'].float().to(dev)
sigma_Y = params['sigma_Y'].float().to(dev)
sigma = params['sigma'].float().to(dev)
alpha = params['alpha'].float().to(dev)
Phi_Y = params['Phi_Y'].float().to(dev)
Psi = params['Psi'].float().to(dev)
rho_Y = params['rho_Y'].float().to(dev)

Sigma = torch.diag(sigma) @ Psi @ torch.diag(sigma)
Sigma_inv = torch.linalg.inv(Sigma)

T_max = 1.5
W_min, W_max, lb_w = 1e-1, 3., 1e-1
W_cap = 1e10

Y_min_vec = theta_Y - 3 * torch.diag(sigma_Y)
Y_max_vec = theta_Y + 3 * torch.diag(sigma_Y)

n = 1000
m = 20
gamma = 2.

mu_Y_const = r + (sigma.unsqueeze(0) * (alpha @ theta_Y.unsqueeze(0).T).T)
pi_M_const = (1.0 / gamma) * (Sigma_inv @ (mu_Y_const - r).T).T

def block_corr_matrix(Psi, Phi_Y, rho_Y):
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
    if Y.ndim == 1:
        Y = Y.unsqueeze(0)
    alphaY = alpha @ Y.transpose(-1, -2)
    alphaY = alphaY * sigma.unsqueeze(-1)
    alphaY = alphaY + r
    return alphaY.transpose(-1, -2)

def get_optimal_pi(X, lam, dlam_dx, dlam_dY, Y, alpha, sigma, Sigma, sigma_Y, rho_Y, Psi, r, gamma, device=torch.device('cpu')):
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

    mu_vec = compute_mu_of_Y(Y_t, alpha_t, sigma_t, r_t)
    myopic_raw = lam_t * (mu_vec - r_t)
    if dlam_dY_t.ndim == 1:
        dlam_dY_t = dlam_dY_t.unsqueeze(0)
    dlam_dY_sY = dlam_dY_t @ sigma_Y_t
    temp1 = dlam_dY_sY @ rho_Y_t.transpose(-1, -2)
    #temp2 = torch.linalg.solve(Psi_t.transpose(-1, -2), temp1.transpose(-1, -2))
    #temp2 = temp2.transpose(-1, -2)
    #hedging_raw = sigma_t * temp2 # !!!
    hedging_raw = sigma_t * temp1 
    bracket_myopic = myopic_raw.transpose(-1, -2)
    bracket_hedging = hedging_raw.transpose(-1, -2)
    cholesky_Sigma = torch.linalg.cholesky(Sigma_t)
    solved_myopic = torch.cholesky_solve(bracket_myopic, cholesky_Sigma)
    solved_hedging = torch.cholesky_solve(bracket_hedging, cholesky_Sigma)
    solved_myopic = solved_myopic.transpose(-1, -2)
    solved_hedging = solved_hedging.transpose(-1, -2)
    scalar_coeff = -1.0 / (X_t * dlamdx_t)
    myopic = scalar_coeff * solved_myopic
    hedging = scalar_coeff * solved_hedging
    total = myopic + hedging
    return total, myopic, hedging


def analytic_EUM_vectorized(W0, T, pi_M, y_bar, alpha, sigma, Psi, r, gamma):
    B, pi_M_b = W0.shape[0], pi_M.expand(W0.shape[0], -1)
    risk = (alpha @ y_bar).expand(B, -1)
    mu_M = r + (pi_M_b * sigma * risk).sum(dim=1)
    sig2_M = ((pi_M_b * sigma) @ Psi @ (pi_M_b * sigma).T).diagonal()
    A = (1 - gamma) * (mu_M - 0.5 * gamma * sig2_M) * T.squeeze(-1)
    exp_term = torch.exp(A.clamp(max=14.0))
    EU = (W0.squeeze(-1).pow(1 - gamma) / (1 - gamma)) * exp_term
    return EU.detach().unsqueeze(-1)


# --- Network Architectures ---

class TradeNet(nn.Module):
    def __init__(self):
        super(TradeNet, self).__init__()
        self.linear1a = nn.Linear(2 + k, 200)
        self.linear2a = nn.Linear(200, 200)
        self.linear3a = nn.Linear(200, 200)
        self.linear4a = nn.Linear(200, d)
        self.F = nn.LeakyReLU()

    def forward(self, x0):
        xa = self.F(self.linear1a(x0))
        xa = self.F(self.linear2a(xa))
        xa = self.F(self.linear3a(xa))
        return self.linear4a(xa)

class MyopicLayer(nn.Module):
    def __init__(self, alpha, sigma, Sigma_inv, r, gamma):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.register_buffer("sigma", sigma)
        self.register_buffer("Sigma_inv", Sigma_inv)
        self.register_buffer("r", torch.tensor(r, device=alpha.device))
        self.gamma = gamma

    def forward(self, Y):
        mu_Y = self.r + (self.sigma.unsqueeze(0) * (self.alpha @ Y.T).T)
        return (1.0 / self.gamma) * (self.Sigma_inv @ (mu_Y - self.r).T).T

def create_sub_net(input_dim, output_dim):
    net = nn.Sequential(
        nn.Linear(input_dim, 120), nn.LeakyReLU(),
        nn.Linear(120, 120), nn.LeakyReLU(),
        nn.Linear(120, 120), nn.LeakyReLU(),
        nn.Linear(120, output_dim)
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net

class HedgeNet(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.long_term_net = create_sub_net(input_dim=k + 1, output_dim=d)
        self.short_term_net = create_sub_net(input_dim=k + 2, output_dim=d)
        self.lamb = nn.Parameter(torch.FloatTensor([0.]))

    def forward(self, W, TmT, Y):
        long_term_input = torch.cat([W, Y], dim=1)
        short_term_input = torch.cat([W, TmT, Y], dim=1)
        pi_long = self.long_term_net(long_term_input)
        pi_short = self.short_term_net(short_term_input) * torch.exp(-torch.exp(self.lamb)*TmT)
        return torch.tanh(pi_long) + torch.tanh(pi_short)

class PGDPOPolicy(nn.Module):
    def __init__(self, alpha, sigma, Sigma_inv, r, gamma, k, d):
        super().__init__()
        self.myopic_layer = MyopicLayer(alpha, sigma, Sigma_inv, r, gamma)
        self.hedging_net = HedgeNet(k, d)

    def forward(self, state):
        W, TmT, Y = state[:, :1], state[:, 1:2], state[:, 2:]
        pi_myopic = self.myopic_layer(Y)
        pi_hedging = self.hedging_net(W, TmT, Y)
        return pi_myopic + pi_hedging


def generate_uniform_domain(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    T = T_max * torch.rand([n, 1], device=dev)
    dt = T / m
    W = W_min + (W_max - W_min) * torch.rand([n, 1], device=dev)
    rand_vals = torch.rand([n, k], device=dev)
    Y = Y_min_vec.unsqueeze(0) + (Y_max_vec - Y_min_vec).unsqueeze(0) * rand_vals
    return T, W, Y, dt

def flatten_C_B(C, B):
    """C:(k,k), B:(k,)  →  z:(k²+k,)"""
    return np.concatenate([C.ravel(), B])

def unflatten_C_B(z, k):
    """역변환"""
    C = z[: k * k].reshape((k, k))
    B = z[k * k :]
    return C, B

def compute_beta_matrices(alpha, sigma, Sigma_inv, rho_Y):
    """
    β₀ = (σA)ᵀ Σ⁻¹ (σA)
    β₁ = (σA)ᵀ Σ⁻¹ (σρ σ_Y)
    β₂ = (σρ σ_Y)ᵀ Σ⁻¹ (σρ σ_Y)
    """
    sigma_diag = np.diag(sigma)            # σ (n×n)
    sigmaA     = sigma_diag @ alpha        # σA   (n×k)
    sigma_rho  = sigma_diag @ rho_Y        # σρ   (n×k)

    beta0 = sigmaA.T   @ Sigma_inv @ sigmaA
    beta1 = sigmaA.T   @ Sigma_inv @ sigma_rho
    beta2 = sigma_rho.T @ Sigma_inv @ sigma_rho
    return beta0, beta1, beta2

# --- main Riccati ODE ----------------------------------------------------
def ode_multifactor_ABC_backward_strict(
    t, z,
    gamma,
    kappa_Y,               # (k,k)
    sigma_Y, Phi_Y,        # (k,k), (k,k)
    alpha, Sigma_inv,      # (n,k), (n,n)
    sigma, rho_Y,          # (n,),  (n,k)
    theta_Y                # (k,) or None
):
    """
    Backward-in-time ODE   d/dt [C(t), B(t)] = RHS
    Return value is -dz   so that forward integrator (solve_ivp) runs backwards.
    """
    k = kappa_Y.shape[0]
    C, B = unflatten_C_B(z, k)

    beta0, beta1, beta2 = compute_beta_matrices(alpha, sigma, Sigma_inv, rho_Y)
    factor = (1.0 - gamma) / gamma

    # --- dC/dt -----------------------------------------------------------
    dC = (
        kappa_Y.T @ C + C @ kappa_Y
        - C @ sigma_Y @ Phi_Y @ sigma_Y.T @ C
        - factor * (
            beta0
            + beta1 @ sigma_Y @ C
            + C @ sigma_Y.T @ beta1.T
            + C @ sigma_Y.T @ beta2 @ sigma_Y @ C
        )
    )

    # --- dB/dt -----------------------------------------------------------
    dB = (
        kappa_Y.T @ B
        - C @ sigma_Y @ Phi_Y @ sigma_Y.T @ B
        - factor * (
            C @ sigma_Y.T @ beta2 @ sigma_Y @ B
            + beta1 @ sigma_Y @ B
        )
    )
    if theta_Y is not None:
        dB -= kappa_Y @ C @ theta_Y      # note: no transpose on κ_Y

    # --- package & flip sign for backward integration --------------------
    return -flatten_C_B(dC, dB)

def solve_multifactor_ABC_strict(gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv, sigma, rho_Y, theta_Y, T, rtol=1e-6, atol=1e-8, method='Radau'):
    k = kappa_Y.shape[0]
    C_T = np.zeros((k, k))
    B_T = np.zeros(k)
    zT = flatten_C_B(C_T, B_T)
    sol = solve_ivp(
        lambda t, z: ode_multifactor_ABC_backward_strict(t, z, gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv, sigma, rho_Y, theta_Y),
        [0, T], zT, method=method, rtol=rtol, atol=atol, dense_output=True
    )
    return sol

def get_weight_multifactor_strict(t, Y, sol, gamma, sigma, Sigma_inv, alpha, rho_Y, sigma_Y):
    smax = sol.t[-1]
    s = smax - t
    C_mat, B_vec = unflatten_C_B(sol.sol(s), len(Y))
    Sigma_Y = np.diag(sigma) @ rho_Y @ sigma_Y
    factor = 1.0 / gamma
    myopic = Sigma_inv @ (sigma * (alpha @ Y))
    hedging = Sigma_inv @ (Sigma_Y @ (B_vec + C_mat @ Y))
    pi_star = factor * (myopic + hedging)
    return pi_star, factor * myopic, factor * hedging

def demo_test_multifactor_strict(k, d, num_points=50):
    T = T_max
    kappa_Y = params['kappa_Y'].cpu().numpy()
    sigma_Y = params['sigma_Y'].cpu().numpy()
    Phi_Y = params['Phi_Y'].cpu().numpy()
    alpha = params['alpha'].cpu().numpy()
    sigma = params['sigma'].cpu().numpy()
    Psi = params['Psi'].cpu().numpy()
    rho_Y = params['rho_Y'].cpu().numpy()
    theta_Y = params['theta_Y'].cpu().numpy()
    Sigma = np.diag(sigma) @ Psi @ np.diag(sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    pi_ode_sol = solve_multifactor_ABC_strict(gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv, sigma, rho_Y, theta_Y, T)
    t_vals = np.linspace(0.0, T_max, num_points)
    Y_min_vec_np = Y_min_vec.cpu().numpy()
    Y_max_vec_np = Y_max_vec.cpu().numpy()
    interp_funcs = {}
    for asset_idx in range(d):
        for factor_idx in range(k):
            y_vals = np.linspace(Y_min_vec_np[factor_idx], Y_max_vec_np[factor_idx], num_points)
            Pi_map = np.zeros((num_points, num_points))
            Myopic_map = np.zeros((num_points, num_points))
            Hedging_map = np.zeros((num_points, num_points))
            for i, t_ in enumerate(t_vals):
                for j, y_ in enumerate(y_vals):
                    Y_current = theta_Y.copy()
                    Y_current[factor_idx] = y_
                    pi_star, myopic, hedging = get_weight_multifactor_strict(T_max - t_, Y_current, pi_ode_sol, gamma, sigma, Sigma_inv, alpha, rho_Y, sigma_Y)
                    Pi_map[i, j] = pi_star[asset_idx]
                    Myopic_map[i, j] = myopic[asset_idx]
                    Hedging_map[i, j] = hedging[asset_idx]
            interp_funcs[(asset_idx, factor_idx, 'Total')] = RegularGridInterpolator((t_vals, y_vals), Pi_map, method='linear', bounds_error=False)
            interp_funcs[(asset_idx, factor_idx, 'Myopic')] = RegularGridInterpolator((t_vals, y_vals), Myopic_map, method='linear', bounds_error=False)
            interp_funcs[(asset_idx, factor_idx, 'Hedging')] = RegularGridInterpolator((t_vals, y_vals), Hedging_map, method='linear', bounds_error=False)
    return interp_funcs
    
    interpolators = demo_test_multifactor_strict(k=k, d=d, num_points=50)
    basis_folder = 'pgdpo_%d_%d_%d_%d' % (m, n, d, k)
    if not os.path.exists(basis_folder):
        os.mkdir(basis_folder)
    

interpolators = demo_test_multifactor_strict(k=k, d=d, num_points=50)
res_suffix = '_res' if use_residual_net else ''
cv_suffix = '_cv' if use_cv else ''
basis_folder = f'multi_{m}_{n}_{d}_{k}{res_suffix}{cv_suffix}'
if not os.path.exists(basis_folder):
    os.mkdir(basis_folder)


# ===============================================================
# Section: Plotting Function (Restored and Improved)
# ===============================================================

def plot(iter, U, T, W, Y, kth, dt, lamb, dx_lamb, dy_lamb, net_pi,
         basis_folder, seed, interpolators, d, k,
         alpha, sigma, Sigma, sigma_Y, rho_Y, Psi, r, gamma, dev):
    """
    평가 결과를 시각화하고 오차를 계산하는 종합적인 함수입니다.
    (요청사항 반영: tricontourf 스타일, pi0_lamb = pi0_sol 가정)
    """
    folder = os.path.join(basis_folder, f'iter_{iter:05d}')
    os.makedirs(folder, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)

    # [STYLE CHANGE] 시각화 방식을 요청하신 tricontourf 스타일로 복원
    def draw(t, y, u, filename, xlabel_text):
        """등고선 플롯(tricontourf)을 그리는 헬퍼 함수"""
        valid_mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(u)
        t, y, u = t[valid_mask], y[valid_mask], u[valid_mask]

        if len(u) < 4:
            print(f"⚠️  Skipping plot '{os.path.basename(filename)}': Not enough valid data points ({len(u)}).")
            return

        tmp = np.sort(u)
        m = 100
        n = (len(tmp) - 1) // m if len(tmp) > m else 1
        levels = [tmp[i * n] for i in range(m + 1)] if n > 0 else 100
        if isinstance(levels, list): levels[-1] = tmp[-1]

        color_map = plt.cm.jet
        colors = color_map(np.linspace(0, 1, m + 1))
        
        plt.figure(figsize=(6.5, 5))
        plt.plot(y, t, 'o', markersize=0.5, alpha=0.2, color='grey')
        try:
            contourf_plot = plt.tricontourf(y, t, u, levels=levels, colors=colors)
        except:
            contourf_plot = plt.tricontourf(y, t, u, cmap='jet')

        plt.xlabel(xlabel_text)
        plt.ylabel('T-t')
        plt.colorbar(contourf_plot, label=f'Mean of {os.path.basename(filename).split("_")[0]}')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # --- 1. 계산을 위해 데이터 준비 ---
    T_cpu, W_cpu, Y_cpu = T.cpu(), W.cpu(), Y.cpu()
    lamb_cpu, dx_lamb_cpu, dy_lamb_cpu = lamb.cpu(), dx_lamb.cpu(), dy_lamb.cpu()
    t_plot, ys_plot = T_cpu.numpy().squeeze(), Y_cpu.numpy()
    xlabel = f'Y_{kth}'

    # --- 2. 각종 정책 계산 ---
    with torch.no_grad():
        state_tensor = torch.cat([W_cpu, T_cpu, Y_cpu], dim=1).to(dev)
        a_pi = net_pi(state_tensor).cpu().numpy()

    # PMP 기반 정책 계산 (pi1_lamb만 사용)
    _, _, pi1_lamb = get_optimal_pi(W_cpu, lamb_cpu, dx_lamb_cpu, dy_lamb_cpu, Y_cpu, alpha.cpu(), sigma.cpu(), Sigma.cpu(), sigma_Y.cpu(), rho_Y.cpu(), Psi.cpu(), r, gamma, device='cpu')
    pi1_lamb_np = pi1_lamb.numpy()
    
    # --- 3. 에러 계산 및 시각화 루프 ---
    err_pi, err_h_net, err_pi_lamb, err_pi0_lamb, err_pi1_lamb = 0., 0., 0., 0., 0.

    for dth in range(d):
        query_points = np.column_stack([t_plot, ys_plot[:, kth]])
        pi0_sol = interpolators[(dth, kth, 'Myopic')](query_points)
        pi1_sol = interpolators[(dth, kth, 'Hedging')](query_points)
        pi_sol = interpolators[(dth, kth, 'Total')](query_points)

        # [MODIFICATION] 요청에 따라 pi0_lamb를 pi0_sol과 동일하게 처리
        pi0_lamb_np_dth = pi0_sol
        # 새로운 PMP 정책은 pi0_sol과 PMP 헤징의 합으로 재구성
        pi_lamb_np_dth = pi0_sol + pi1_lamb_np[:, dth]

        valid_mask = ~np.isnan(pi_sol) & ~np.isnan(a_pi[:,dth])
        if np.sum(valid_mask) == 0: continue

        a_pi_valid, pi_sol_valid, pi0_sol_valid, pi1_sol_valid = a_pi[valid_mask, dth], pi_sol[valid_mask], pi0_sol[valid_mask], pi1_sol[valid_mask]
        pi_lamb_valid, pi0_lamb_valid, pi1_lamb_valid = pi_lamb_np_dth[valid_mask], pi0_lamb_np_dth[valid_mask], pi1_lamb_np[valid_mask, dth]

        err_pi += np.mean((a_pi_valid - pi_sol_valid)**2)
        err_h_net += np.mean(((a_pi_valid - pi0_sol_valid) - pi1_sol_valid)**2) if use_residual_net else 0.0
        err_pi_lamb += np.mean((pi_lamb_valid - pi_sol_valid)**2)
        err_pi0_lamb += np.mean((pi0_lamb_valid - pi0_sol_valid)**2) # 이 값은 정의에 의해 0에 가까움
        err_pi1_lamb += np.mean((pi1_lamb_valid - pi1_sol_valid)**2)

        if dth < 5 and kth == 0:
            # --- pi0 (Myopic Component) Plots ---
            draw(t_plot, ys_plot[:, kth], pi0_sol, os.path.join(folder, f"pi0_sol_x_{dth}_y_{kth}.png"), xlabel)
            # pi0_lamb는 pi0_sol과 동일하므로 플롯 생략
            
            # --- pi1 (Hedging Component) Plots ---
            draw(t_plot, ys_plot[:, kth], pi1_sol, os.path.join(folder, f"pi1_sol_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], pi1_lamb_np[:, dth], os.path.join(folder, f"pi1_lamb_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], pi1_lamb_np[:, dth] - pi1_sol, os.path.join(folder, f"pi1_lamb_err_x_{dth}_y_{kth}.png"), xlabel)
            
            # --- Total Policy Plots ---
            draw(t_plot, ys_plot[:, kth], pi_sol, os.path.join(folder, f"pi_sol_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], a_pi[:, dth], os.path.join(folder, f"pi_net_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], a_pi[:, dth] - pi_sol, os.path.join(folder, f"pi_err_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], pi_lamb_np_dth, os.path.join(folder, f"pi_lamb_x_{dth}_y_{kth}.png"), xlabel)
            draw(t_plot, ys_plot[:, kth], pi_lamb_np_dth - pi_sol, os.path.join(folder, f"pi_lamb_err_x_{dth}_y_{kth}.png"), xlabel)

    # --- 4. 최종 오차 계산 및 출력 ---
    inv_d = 1.0 / d if d > 0 else 1.0
    err_pi, err_h_net = np.sqrt(err_pi * inv_d), np.sqrt(err_h_net * inv_d)
    err_pi_lamb, err_pi0_lamb, err_pi1_lamb = np.sqrt(err_pi_lamb*inv_d), np.sqrt(err_pi0_lamb*inv_d), np.sqrt(err_pi1_lamb*inv_d)
    
    res = (f"i:{iter:5d} kth:{kth:3d} | "
           f"U:{U:8.2e} | "
           f"err_pi:{err_pi:8.2e} err_h_net:{err_h_net:8.2e} | "
           # pi0_lamb_err는 0이므로, pi1_lamb_err가 더 중요
           f"err_pi_lamb:{err_pi_lamb:8.2e} err_pi1_lamb:{err_pi1_lamb:8.2e}")
    
    with open(os.path.join(folder, 'errs.txt'), 'wt' if kth == 0 else 'at') as f:
        print(res, file=f)
    print(res)

block_corr = block_corr_matrix(Psi, Phi_Y, rho_Y).to(dev)

def sim(net_pi, pi_M, T, W, Y, dt, anti=1, seed=None, train=True, use_richardson=True):
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    batch_size, dev = len(W), W.device
    logW = W.log()
    logW_M = W.log() if train and use_cv else None

    risk_premium_M_const = torch.einsum('dk,k->d', alpha, theta_Y)
    drift_term_M_const = r + (pi_M.squeeze() * sigma * risk_premium_M_const).sum()
    var_term_M_const = ((pi_M * sigma) @ Psi @ (pi_M * sigma).T).squeeze()
    logW_M_drift = (drift_term_M_const - 0.5 * var_term_M_const)

    for kstep in range(m):
        t = kstep * dt
        
        if use_richardson:
            dt_coarse_sqrt, dt_half_sqrt = dt.sqrt(), (dt * 0.5).sqrt()
            mvn = MultivariateNormal(torch.zeros(d + k, device=dev), covariance_matrix=block_corr)
            Z_base, Z_noise = anti*mvn.sample((batch_size,)), anti*mvn.sample((batch_size,))
            sqrt2_inv = 1.0 / np.sqrt(2.0)
            Z_half1, Z_half2 = (Z_base + Z_noise) * sqrt2_inv, (Z_base - Z_noise) * sqrt2_inv
            dW_X_coarse, dW_Y_coarse = Z_base[:, :d] * dt_coarse_sqrt, Z_base[:, d:] * dt_coarse_sqrt
            dW_X_half1, dW_Y_half1 = Z_half1[:, :d] * dt_half_sqrt, Z_half1[:, d:] * dt_half_sqrt
            dW_X_half2, dW_Y_half2 = Z_half2[:, :d] * dt_half_sqrt, Z_half2[:, d:] * dt_half_sqrt

            state_t = torch.cat([logW.exp(), T - t, Y], dim=1)
            pi_t = net_pi(state_t) if train else net_pi(state_t).detach()
            risk_premium_t = torch.einsum('dk,bk->bd', alpha, Y)
            Y_drift_t = -(Y - theta_Y) @ kappa_Y.T
            drift_term_t = r + (pi_t * sigma * risk_premium_t).sum(1, keepdim=True)
            var_term_t = torch.einsum('bd,bd->b', (pi_t * sigma) @ Psi, (pi_t * sigma)).view(-1, 1)
            logW_drift_coarse = (drift_term_t - 0.5 * var_term_t)
            logW_coarse = logW + logW_drift_coarse*dt + (pi_t*sigma*dW_X_coarse).sum(1, keepdim=True)
            Y_coarse = Y + Y_drift_t*dt + (sigma_Y @ dW_Y_coarse.T).T

            dt_half = dt * 0.5
            logW_half = logW + logW_drift_coarse*dt_half + (pi_t*sigma*dW_X_half1).sum(1, keepdim=True)
            Y_half = Y + Y_drift_t*dt_half + (sigma_Y @ dW_Y_half1.T).T
            state_half = torch.cat([logW_half.exp(), T - (t + dt_half), Y_half], dim=1)
            pi_half = net_pi(state_half) if train else net_pi(state_half).detach()
            risk_premium_half = torch.einsum('dk,bk->bd', alpha, Y_half)
            Y_drift_half = -(Y_half - theta_Y) @ kappa_Y.T
            drift_term_half = r + (pi_half * sigma * risk_premium_half).sum(1, keepdim=True)
            var_term_half = torch.einsum('bd,bd->b', (pi_half*sigma)@Psi, (pi_half*sigma)).view(-1, 1)
            logW_drift_fine = (drift_term_half - 0.5 * var_term_half)
            logW_fine = logW_half + logW_drift_fine*dt_half + (pi_half*sigma*dW_X_half2).sum(1, keepdim=True)
            Y_fine = Y_half + Y_drift_half*dt_half + (sigma_Y @ dW_Y_half2.T).T
            logW, Y = 2.0*logW_fine - logW_coarse, 2.0*Y_fine - Y_coarse

            if train and use_cv:
                logW_M_coarse = logW_M + logW_M_drift*dt + (pi_M*sigma*dW_X_coarse).sum(1, keepdim=True)
                logW_M_half = logW_M + logW_M_drift*dt_half + (pi_M*sigma*dW_X_half1).sum(1, keepdim=True)
                logW_M_fine = logW_M_half + logW_M_drift*dt_half + (pi_M*sigma*dW_X_half2).sum(1, keepdim=True)
                logW_M = 2.0 * logW_M_fine - logW_M_coarse
        else: # Simple Euler-Maruyama
            mvn = MultivariateNormal(torch.zeros(d+k, device=dev), covariance_matrix=block_corr)
            Z = anti * mvn.sample((batch_size,))
            dW_X, dW_Y = Z[:, :d]*dt.sqrt(), Z[:, d:]*dt.sqrt()
            state_t = torch.cat([logW.exp(), T-t, Y], dim=1)
            pi_t = net_pi(state_t) if train else net_pi(state_t).detach()
            Y_drift = -(Y - theta_Y) @ kappa_Y.T
            Y = Y + Y_drift*dt + (sigma_Y @ dW_Y.T).T
            risk_premium = torch.einsum('dk,bk->bd', alpha, Y)
            drift_term = r + (pi_t * sigma * risk_premium).sum(1, keepdim=True)
            var_term = torch.einsum('bd,bd->b', (pi_t*sigma)@Psi, (pi_t*sigma)).view(-1, 1)
            logW_drift, logW_diffusion = drift_term-0.5*var_term, (pi_t*sigma*dW_X).sum(1, keepdim=True)
            logW = logW + logW_drift*dt + logW_diffusion
            if train and use_cv:
                logW_M = logW_M + logW_M_drift*dt + (pi_M*sigma*dW_X).sum(1, keepdim=True)

        logW = logW.exp().clamp(min=lb_w, max=W_cap).log()
        if train and use_cv: logW_M = logW_M.exp().clamp(min=lb_w, max=W_cap).log()

    W_final = logW.exp()
    U_theta = W_final**(1.0 - gamma) / (1.0 - gamma)
    if not train: return U_theta
    if use_cv:
        W_M_final = logW_M.exp()
        U_M = W_M_final**(1.0 - gamma) / (1.0 - gamma)
        return U_theta, U_M
    else:
        return U_theta, None

# ======================= TRAINING LOOP =======================
torch.manual_seed(seed); np.random.seed(seed)
if use_residual_net:
    net_pi = PGDPOPolicy(alpha, sigma, Sigma_inv, r, gamma, k, d).to(dev)
else:
    net_pi = TradeNet().to(dev)
opt_pi = torch.optim.Adam(net_pi.parameters(), lr=1e-5)
total_epoch = 20000
if use_cv:
    c_hat_ema, ema_beta = torch.tensor(0.0, device=dev), 0.99

for i in range(total_epoch):
    opt_pi.zero_grad()
    T0, W0, Y0, dt0 = generate_uniform_domain(n, i)
    
    U_theta_pos, U_M_pos = sim(net_pi, pi_M_const, T0.detach(), W0.detach(), Y0.detach(), dt0.detach(), +1, i, train=True, use_richardson=use_richardson)
    U_theta_neg, U_M_neg = sim(net_pi, pi_M_const, T0.detach(), W0.detach(), Y0.detach(), dt0.detach(), -1, i, train=True, use_richardson=use_richardson)
    U_theta_sim = 0.5 * (U_theta_pos + U_theta_neg)
    
    if use_cv:
        U_M_simulated = (0.5 * (U_M_pos + U_M_neg)).detach()
        with torch.no_grad():
            EU_M_analytic = analytic_EUM_vectorized(W0, T0, pi_M_const, theta_Y, alpha, sigma, Psi, r, gamma)
            scale = W0.pow(1 - gamma).detach()
            U_M_centered_norm = (U_M_simulated - EU_M_analytic) / scale
        U_theta_norm = U_theta_sim / scale
        with torch.no_grad():
            valid_indices = ~torch.isnan(U_theta_norm) & ~torch.isnan(U_M_centered_norm)
            if valid_indices.sum() > 1:
                U_theta_valid, U_M_valid = U_theta_norm[valid_indices], U_M_centered_norm[valid_indices]
                cov = torch.nanmean((U_theta_valid - U_theta_valid.nanmean()) * (U_M_valid - U_M_valid.nanmean()))
                var = torch.var(U_M_valid, unbiased=False) + 1e-6
                c_hat_batch = cov / var
                if torch.isfinite(c_hat_batch) and 0.0 <= c_hat_batch <= 10.0:
                    c_hat_ema = ema_beta * c_hat_ema + (1 - ema_beta) * c_hat_batch
            c_hat_final = c_hat_ema
        finite_mask = torch.isfinite(U_theta_norm) & torch.isfinite(U_M_centered_norm)
        U = - (U_theta_norm[finite_mask] - c_hat_final * U_M_centered_norm[finite_mask]).nanmean() if finite_mask.sum() > 0 else torch.tensor(float('nan'))
    else:
        U = -U_theta_sim.nanmean()

    if torch.isfinite(U):
        U.backward()
        torch.nn.utils.clip_grad_norm_(net_pi.parameters(), 1.0)
        opt_pi.step()

    # ===============================================================
    # Costate estimator — 수정된 버전
    # ===============================================================
    def estimate_costates(net_pi, pi_M, T0, W0, Y0, dt0,
                          repeats: int = 2000,
                          sub_batch_size: int = 100,
                          use_richardson: bool = True):
        """
        메모리 효율적인 서브-배치 방식으로 λ, ∂λ/∂W, ∂λ/∂Y를 추정합니다.
        각 시작 포인트별로 유틸리티를 평균 낸 뒤, 그래디언트를 계산하고 누적합니다.
        """
        device = W0.device
        n_eval = W0.shape[0]
    
        # 그래디언트 계산을 위해 requires_grad=True 설정
        W0_grad = W0.detach().clone().requires_grad_(True)
        Y0_grad = Y0.detach().clone().requires_grad_(True)
    
        # 결과를 누적할 텐서 초기화
        U_accum_sum = torch.zeros(n_eval, 1, device=device)
        lamb_accum_sum = torch.zeros_like(W0_grad)
        dx_lamb_accum_sum = torch.zeros_like(W0_grad)
        dy_lamb_accum_sum = torch.zeros_like(Y0_grad)
        
        total_repeats_done = 0
    
        # 전체 repeats를 sub_batch_size 만큼 나누어 처리
        for i in range(0, repeats, sub_batch_size):
            current_repeats = min(sub_batch_size, repeats - i)
            
            # 현재 서브-배치에 맞게 텐서 복제
            T_batch = T0.repeat(current_repeats, 1)
            W_batch = W0_grad.repeat(current_repeats, 1)
            Y_batch = Y0_grad.repeat(current_repeats, 1)
            dt_batch = dt0.repeat(current_repeats, 1)
    
            # Antithetic-pair로 시뮬레이션 실행
            U_pos = sim(net_pi, pi_M, T_batch, W_batch, Y_batch, dt_batch, +1, seed=i, train=False, use_richardson=use_richardson)
            U_neg = sim(net_pi, pi_M, T_batch, W_batch, Y_batch, dt_batch, -1, seed=i, train=False, use_richardson=use_richardson)
            
            U_pairs = 0.5 * (U_pos + U_neg)
            
            # [수정된 로직] 각 시작점별로 평균 유틸리티 계산
            # [current_repeats * n_eval, 1] -> [current_repeats, n_eval] -> [n_eval]
            U_mean_per_point = U_pairs.view(current_repeats, n_eval).mean(dim=0)
            
            # 각 시작점별 평균 유틸리티의 합에 대해 그래디언트 계산
            # 이래야 W0_grad의 각 원소에 대한 그래디언트가 나옴
            lamb_batch, = torch.autograd.grad(U_mean_per_point.sum(), W0_grad, create_graph=True, retain_graph=True)
            dx_lamb_batch, dy_lamb_batch = torch.autograd.grad(lamb_batch.sum(), (W0_grad, Y0_grad))
            
            # 계산된 값들을 누적
            U_accum_sum += U_mean_per_point.unsqueeze(1).detach() * current_repeats
            lamb_accum_sum += lamb_batch.detach() * current_repeats
            dx_lamb_accum_sum += dx_lamb_batch.detach() * current_repeats
            dy_lamb_accum_sum += dy_lamb_batch.detach() * current_repeats
            total_repeats_done += current_repeats
    
        # 최종 평균 계산
        if total_repeats_done > 0:
            inv_N = 1.0 / total_repeats_done
            U_final_mean = (U_accum_sum * inv_N).mean().item()
            return (lamb_accum_sum * inv_N,
                    dx_lamb_accum_sum * inv_N,
                    dy_lamb_accum_sum * inv_N,
                    U_final_mean)
        else:
            # 시뮬레이션이 한 번도 실행되지 않은 경우
            return (torch.full_like(W0, float('nan')),
                    torch.full_like(W0, float('nan')),
                    torch.full_like(Y0, float('nan')),
                    float('nan'))
    
    # ===============================================================
    # TRAINING LOOP — 평가(플롯) 구간 (수정된 호출부)
    # ===============================================================
    if i % 200 == 199 or i == 0:
        n_eval = 1000
        # 평가를 위한 기본 T, W, Y 샘플 생성
        T_org, W_org_base, Y_org_samples, dt_org = generate_uniform_domain(n_eval, 0)
        W_org = torch.ones_like(W_org_base) # W=1 고정
    
        # --- 플로팅 루프 ---
        for kth in range(k):
            if k > 1 and kth != 0:
                continue  # k>1이면 k==0일 때만 그림
    
            # [MODIFICATION] 요청사항 반영: Y 벡터 재구성
            # Y_kth는 랜덤 샘플을 그대로 사용하고, 나머지 Y_j (j!=k)는 theta_Y[j]로 고정
            Y_plot = Y_org_samples.clone().detach()
            for j_fix in range(k):
                if j_fix != kth:
                    Y_plot[:, j_fix] = theta_Y[j_fix]
    
            # --- Co-state 추정 함수 호출 ---
            # 수정된 Y_plot을 사용하여 co-state 추정
            lamb_exact, dx_lamb_exact, dy_lamb_exact, U_test = estimate_costates(
                net_pi, pi_M_const, T_org, W_org, Y_plot, dt_org,
                repeats=2000, sub_batch_size=100, use_richardson=use_richardson)
            
            # --- 플로팅 함수 호출 ---
            # 수정된 Y_plot을 plot 함수에 전달
            plot(i, U_test,
                 T_org.detach(), W_org.detach(), Y_plot.detach(), kth, dt_org.detach(),
                 lamb_exact.detach(), dx_lamb_exact.detach(), dy_lamb_exact.detach(),
                 net_pi, basis_folder, seed, interpolators,
                 d, k, alpha, sigma, Sigma, sigma_Y, rho_Y, Psi, r, gamma, dev)

        folder = os.path.join(basis_folder, f'iter_{i:05d}')
        torch.save(net_pi.state_dict(), "%s/net_pi.pth" % folder)
        torch.save(opt_pi.state_dict(), "%s/opt_pi.pth" % folder)    
        
    if (i+1) % 10 == 0:
        print(f'i {i}', end=' ', flush=True)

    if i > 0 and (i+1) % 5000 == 0:
        folder = os.path.join(basis_folder, f'iter_{i:05d}')
        os.makedirs(folder, exist_ok=True)
        torch.save(net_pi.state_dict(), f"{folder}/net_pi.pth")
        torch.save(opt_pi.state_dict(), f"{folder}/opt_pi.pth")

print('\ncompleted')