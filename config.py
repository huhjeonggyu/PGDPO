# config.py
# 역할: config.json에서 모든 설정을 로드하고, 전역 파라미터를 생성합니다.

import os
import sys
import json
import torch
import numpy as np
from scipy.stats import dirichlet

################################################################
# 1. JSON 설정 파일 로드
################################################################
CONFIG_FILE = 'config.json'
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"설정 파일 '{CONFIG_FILE}'을 찾을 수 없습니다. 프로젝트 루트에 파일을 생성해주세요.")

with open(CONFIG_FILE, 'r') as f:
    cfg = json.load(f)

# 각 섹션에서 파라미터 추출
run_settings = cfg['run_settings']
fin_params = cfg['financial_params']
sim_params = cfg['simulation_params']
train_params = cfg['training_params']
eval_params = cfg['evaluation_params']

################################################################
# 2. 전역 변수 및 상수 설정 (모두 JSON에서 로드)
################################################################
# 실행 관련 설정
d = run_settings['d']
k = run_settings['k']
cuda_num = run_settings['cuda_num']
use_residual_net = run_settings['use_residual_net']
use_cv = run_settings['use_cv']
use_richardson = run_settings['use_richardson']

# 시스템 및 시드 설정
torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True, precision=4, linewidth=100)
dev = f'cuda:{cuda_num}'
seed = 1

# JSON에서 로드한 변수들
r = fin_params['r']
gamma = fin_params['gamma']
T_max = sim_params['T_max']
m = sim_params['m']
W_min, W_max, lb_w = sim_params['W_min'], sim_params['W_max'], sim_params['lb_w']
W_cap = sim_params['W_cap']
n = train_params['n']
total_epoch = train_params['total_epoch']
learning_rate = train_params['learning_rate']
n_eval = eval_params['n_eval']
repeats = eval_params['repeats']
sub_batch_size = eval_params['sub_batch_size']

print(f"--- Configuration (Loaded from {CONFIG_FILE}) ---\n"
      f"Assets (d): {d}, Factors (k): {k}\n"
      f"Residual Network: {use_residual_net}\n"
      f"Control Variate:  {use_cv}\n"
      f"Richardson Extrapolation: {use_richardson}\n"
      f"----------------------------------------------------")

################################################################
# 3. 파라미터 생성 함수 및 실행
################################################################
def generate_multifactor_capm_structure_params(d, k, r, beta_corr_max, rho_max, seed=None):
    """
    [cite_start]시장 파라미터를 생성합니다. [cite: 384, 392]
    [cite_start]이 함수는 문서의 Appendix A에 설명된 절차를 따릅니다. [cite: 383]
    """
    PD_tolerance = 1e-6
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # [cite_start]팩터 동역학 파라미터 생성 [cite: 394]
    kappa_Y = torch.diag(torch.tensor([2.0 + i * 0.5 for i in range(k)]))
    theta_Y = torch.empty(k).uniform_(0.2, 0.4)
    sigma_Y = torch.diag(torch.empty(k).uniform_(0.3, 0.5))
    
    # [cite_start]자산별 파라미터 생성 [cite: 398]
    sigma = torch.empty(d).uniform_(0.1, 0.5)
    alpha = torch.tensor(dirichlet.rvs([1.0] * k, size=d), dtype=torch.float32)

    # [cite_start]상관관계 구조 생성 [cite: 402]
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

    # [cite_start]블록 행렬 조립 및 PSD 보정 [cite: 406]
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
            
            # [cite_start]보정된 행렬로부터 각 부분을 다시 추출하여 일관성 보장 [cite: 412]
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

# 파라미터 생성
params = generate_multifactor_capm_structure_params(d=d, k=k, r=r, 
    beta_corr_max=fin_params['beta_corr_max'], rho_max=fin_params['rho_max'], seed=seed)

# GPU로 보낼 전역 변수 설정
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

Y_min_vec = theta_Y - 3 * torch.diag(sigma_Y)
Y_max_vec = theta_Y + 3 * torch.diag(sigma_Y)

# Myopic 정책 및 Control Variate를 위한 상수
mu_Y_const = r + (sigma.unsqueeze(0) * (alpha @ theta_Y.unsqueeze(0).T).T)
pi_M_const = (1.0 / gamma) * (Sigma_inv @ (mu_Y_const - r).T).T

# 결과 저장을 위한 폴더 설정
res_suffix = '_res' if use_residual_net else ''
cv_suffix = '_cv' if use_cv else ''
basis_folder = f'multi_{m}_{n}_{d}_{k}{res_suffix}{cv_suffix}'
if not os.path.exists(basis_folder):
    os.makedirs(basis_folder)