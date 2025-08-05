# analytical_solver.py
# 역할: Riccati ODE를 풀어 분석적 벤치마크 해를 계산 (문서 Appendix D) 

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

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
    β₀, β₁, β₂ 행렬을 계산합니다. [cite: 623]
    """
    sigma_diag = np.diag(sigma)
    sigmaA = sigma_diag @ alpha
    sigma_rho = sigma_diag @ rho_Y

    beta0 = sigmaA.T @ Sigma_inv @ sigmaA
    beta1 = sigmaA.T @ Sigma_inv @ sigma_rho
    beta2 = sigma_rho.T @ Sigma_inv @ sigma_rho
    return beta0, beta1, beta2

def ode_multifactor_ABC_backward_strict(t, z, gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv, sigma, rho_Y, theta_Y):
    """
    Riccati ODE 시스템을 정의합니다. [cite: 621, 622]
    """
    k = kappa_Y.shape[0]
    C, B = unflatten_C_B(z, k)

    beta0, beta1, beta2 = compute_beta_matrices(alpha, sigma, Sigma_inv, rho_Y)
    factor = (1.0 - gamma) / gamma

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

    dB = (
        kappa_Y.T @ B
        - C @ sigma_Y @ Phi_Y @ sigma_Y.T @ B
        - factor * (
            C @ sigma_Y.T @ beta2 @ sigma_Y @ B
            + beta1 @ sigma_Y @ B
        )
    )
    if theta_Y is not None:
        dB -= kappa_Y @ C @ theta_Y

    return -flatten_C_B(dC, dB)

def solve_multifactor_ABC_strict(gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv, sigma, rho_Y, theta_Y, T, rtol=1e-6, atol=1e-8, method='Radau'):
    """
    scipy.integrate.solve_ivp를 사용하여 Riccati ODE를 수치적으로 풉니다. [cite: 626, 627]
    """
    k = kappa_Y.shape[0]
    C_T = np.zeros((k, k)) # [cite: 633]
    B_T = np.zeros(k)      # [cite: 634]
    zT = flatten_C_B(C_T, B_T) # [cite: 635]
    sol = solve_ivp(
        lambda t, z: ode_multifactor_ABC_backward_strict(t, z, gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv, sigma, rho_Y, theta_Y),
        [0, T], zT, method=method, rtol=rtol, atol=atol, dense_output=True # [cite: 641]
    )
    return sol

def get_weight_multifactor_strict(t, Y, sol, gamma, sigma, Sigma_inv, alpha, rho_Y, sigma_Y):
    """
    ODE 해로부터 특정 시점(t)과 상태(Y)에서의 최적 정책을 계산합니다. [cite: 645]
    """
    smax = sol.t[-1]
    s = smax - t
    C_mat, B_vec = unflatten_C_B(sol.sol(s), len(Y))
    Sigma_Y = np.diag(sigma) @ rho_Y @ sigma_Y
    factor = 1.0 / gamma
    myopic = Sigma_inv @ (sigma * (alpha @ Y))
    hedging = Sigma_inv @ (Sigma_Y @ (B_vec + C_mat @ Y))
    pi_star = factor * (myopic + hedging)
    return pi_star, factor * myopic, factor * hedging

def demo_test_multifactor_strict(params, T_max, Y_min_vec, Y_max_vec, gamma, k, d, num_points=50):
    """
    분석적 벤치마크를 생성하기 위해 전체 과정을 조율하고, 최종적으로 보간기 객체를 반환합니다. [cite: 643, 651]
    """
    kappa_Y = params['kappa_Y'].cpu().numpy()
    sigma_Y = params['sigma_Y'].cpu().numpy()
    Phi_Y = params['Phi_Y'].cpu().numpy()
    alpha = params['alpha'].cpu().numpy()
    sigma = params['sigma'].cpu().numpy()
    Psi = params['Psi'].cpu().numpy()
    rho_Y = params['rho_Y'].cpu().numpy()
    theta_Y = params['theta_Y'].cpu().numpy()
    Sigma_np = np.diag(sigma) @ Psi @ np.diag(sigma)
    Sigma_inv_np = np.linalg.inv(Sigma_np)
    
    # 1. Riccati ODE 풀이
    pi_ode_sol = solve_multifactor_ABC_strict(gamma, kappa_Y, sigma_Y, Phi_Y, alpha, Sigma_inv_np, sigma, rho_Y, theta_Y, T_max) # [cite: 654]
    
    t_vals = np.linspace(0.0, T_max, num_points)
    Y_min_vec_np = Y_min_vec.cpu().numpy()
    Y_max_vec_np = Y_max_vec.cpu().numpy()
    interp_funcs = {}

    # 2. 그리드 위에서 정책 값을 미리 계산하고 보간기 생성
    for asset_idx in range(d):
        for factor_idx in range(k):
            y_vals = np.linspace(Y_min_vec_np[factor_idx], Y_max_vec_np[factor_idx], num_points) # [cite: 656]
            Pi_map = np.zeros((num_points, num_points))
            Myopic_map = np.zeros((num_points, num_points))
            Hedging_map = np.zeros((num_points, num_points))
            
            for i, t_ in enumerate(t_vals):
                for j, y_ in enumerate(y_vals):
                    Y_current = theta_Y.copy()
                    Y_current[factor_idx] = y_
                    pi_star, myopic, hedging = get_weight_multifactor_strict(T_max - t_, Y_current, pi_ode_sol, gamma, sigma, Sigma_inv_np, alpha, rho_Y, sigma_Y) # [cite: 663]
                    Pi_map[i, j] = pi_star[asset_idx]
                    Myopic_map[i, j] = myopic[asset_idx]
                    Hedging_map[i, j] = hedging[asset_idx]
            
            # 3. 보간기 객체 생성
            interp_funcs[(asset_idx, factor_idx, 'Total')] = RegularGridInterpolator((t_vals, y_vals), Pi_map, method='linear', bounds_error=False) # [cite: 649, 668]
            interp_funcs[(asset_idx, factor_idx, 'Myopic')] = RegularGridInterpolator((t_vals, y_vals), Myopic_map, method='linear', bounds_error=False)
            interp_funcs[(asset_idx, factor_idx, 'Hedging')] = RegularGridInterpolator((t_vals, y_vals), Hedging_map, method='linear', bounds_error=False)
            
    return interp_funcs