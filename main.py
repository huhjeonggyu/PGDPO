# main.py
# 역할: 메인 학습 루프 및 전체 프로세스 조율

import torch
import numpy as np
import os
from torch.distributions.multivariate_normal import MultivariateNormal

# 분리된 모듈에서 필요한 변수, 함수, 클래스 가져오기
from config import (
    d, k, dev, seed, use_residual_net, use_cv, use_richardson, params,
    r, kappa_Y, theta_Y, sigma_Y, sigma, alpha, Phi_Y, Psi, rho_Y,
    Sigma, Sigma_inv, pi_M_const, m, n, gamma, basis_folder, T_max,
    W_min, W_max, Y_min_vec, Y_max_vec, lb_w, W_cap, n_eval, repeats, sub_batch_size
)
from models import PGDPOPolicy, TradeNet
from analytical_solver import demo_test_multifactor_strict
from utils import (
    generate_uniform_domain, plot, block_corr_matrix, analytic_EUM_vectorized,
    get_optimal_pi
)

# 상관관계 행렬 생성
block_corr = block_corr_matrix(Psi, Phi_Y, rho_Y).to(dev)

def sim(net_pi, pi_M, T, W, Y, dt, anti=1, seed=None, train=True, use_richardson=True, use_cv=True):
    """
    메인 정책과 제어 정책의 궤적을 동시에 시뮬레이션합니다.
    Richardson Extrapolation을 사용하여 수치 정확도를 높입니다.
    문서 Listing 5에 해당합니다.
    """
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    batch_size, device = len(W), W.device
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
            mvn = MultivariateNormal(torch.zeros(d + k, device=device), covariance_matrix=block_corr)
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
            
            # Coarse Step
            logW_coarse = logW + logW_drift_coarse*dt + (pi_t*sigma*dW_X_coarse).sum(1, keepdim=True)
            Y_coarse = Y + Y_drift_t*dt + (sigma_Y @ dW_Y_coarse.T).T
            if train and use_cv: logW_M_coarse = logW_M + logW_M_drift*dt + (pi_M*sigma*dW_X_coarse).sum(1, keepdim=True)

            # Fine Steps
            dt_half = dt * 0.5
            logW_half = logW + logW_drift_coarse*dt_half + (pi_t*sigma*dW_X_half1).sum(1, keepdim=True)
            Y_half = Y + Y_drift_t*dt_half + (sigma_Y @ dW_Y_half1.T).T
            if train and use_cv: logW_M_half = logW_M + logW_M_drift*dt_half + (pi_M*sigma*dW_X_half1).sum(1, keepdim=True)

            state_half = torch.cat([logW_half.exp(), T - (t + dt_half), Y_half], dim=1)
            pi_half = net_pi(state_half) if train else net_pi(state_half).detach()
            risk_premium_half = torch.einsum('dk,bk->bd', alpha, Y_half)
            Y_drift_half = -(Y_half - theta_Y) @ kappa_Y.T
            drift_term_half = r + (pi_half * sigma * risk_premium_half).sum(1, keepdim=True)
            var_term_half = torch.einsum('bd,bd->b', (pi_half*sigma)@Psi, (pi_half*sigma)).view(-1, 1)
            logW_drift_fine = (drift_term_half - 0.5 * var_term_half)
            
            logW_fine = logW_half + logW_drift_fine*dt_half + (pi_half*sigma*dW_X_half2).sum(1, keepdim=True)
            Y_fine = Y_half + Y_drift_half*dt_half + (sigma_Y @ dW_Y_half2.T).T
            if train and use_cv: logW_M_fine = logW_M_half + logW_M_drift*dt_half + (pi_M*sigma*dW_X_half2).sum(1, keepdim=True)
            
            # Richardson Extrapolation
            logW, Y = 2.0*logW_fine - logW_coarse, 2.0*Y_fine - Y_coarse
            if train and use_cv: logW_M = 2.0 * logW_M_fine - logW_M_coarse
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

def estimate_costates(net_pi, pi_M, T0, W0, Y0, dt0,
                      repeats: int = 2000,
                      sub_batch_size: int = 100,
                      use_richardson: bool = True):
    """
    BPTT와 몬테카를로 시뮬레이션을 통해 PMP co-state를 추정합니다.
    문서 Listing 6에 해당합니다.
    """
    device = W0.device
    n_eval = W0.shape[0]

    W0_grad = W0.detach().clone().requires_grad_(True)
    Y0_grad = Y0.detach().clone().requires_grad_(True)

    U_accum_sum = torch.zeros(n_eval, 1, device=device)
    lamb_accum_sum = torch.zeros_like(W0_grad)
    dx_lamb_accum_sum = torch.zeros_like(W0_grad)
    dy_lamb_accum_sum = torch.zeros_like(Y0_grad)
    
    total_repeats_done = 0

    for i in range(0, repeats, sub_batch_size):
        current_repeats = min(sub_batch_size, repeats - i)
        
        T_batch = T0.repeat(current_repeats, 1)
        W_batch = W0_grad.repeat(current_repeats, 1)
        Y_batch = Y0_grad.repeat(current_repeats, 1)
        dt_batch = dt0.repeat(current_repeats, 1)

        U_pos = sim(net_pi, pi_M, T_batch, W_batch, Y_batch, dt_batch, +1, seed=i, train=False, use_richardson=use_richardson, use_cv=False)
        U_neg = sim(net_pi, pi_M, T_batch, W_batch, Y_batch, dt_batch, -1, seed=i, train=False, use_richardson=use_richardson, use_cv=False)
        
        U_pairs = 0.5 * (U_pos + U_neg)
        
        U_mean_per_point = U_pairs.view(current_repeats, n_eval).mean(dim=0)
        
        lamb_batch, = torch.autograd.grad(U_mean_per_point.sum(), W0_grad, create_graph=True, retain_graph=True)
        dx_lamb_batch, dy_lamb_batch = torch.autograd.grad(lamb_batch.sum(), (W0_grad, Y0_grad))
        
        U_accum_sum += U_mean_per_point.unsqueeze(1).detach() * current_repeats
        lamb_accum_sum += lamb_batch.detach() * current_repeats
        dx_lamb_accum_sum += dx_lamb_batch.detach() * current_repeats
        dy_lamb_accum_sum += dy_lamb_batch.detach() * current_repeats
        total_repeats_done += current_repeats

    if total_repeats_done > 0:
        inv_N = 1.0 / total_repeats_done
        U_final_mean = (U_accum_sum * inv_N).mean().item()
        return (lamb_accum_sum * inv_N,
                dx_lamb_accum_sum * inv_N,
                dy_lamb_accum_sum * inv_N,
                U_final_mean)
    else:
        return (torch.full_like(W0, float('nan')),
                torch.full_like(W0, float('nan')),
                torch.full_like(Y0, float('nan')),
                float('nan'))

if __name__ == '__main__':
    # 분석적 벤치마크 생성
    interpolators = demo_test_multifactor_strict(params, T_max, Y_min_vec, Y_max_vec, gamma, k, d, num_points=50)

    # 모델 및 옵티마이저 초기화
    torch.manual_seed(seed); np.random.seed(seed)
    if use_residual_net:
        net_pi = PGDPOPolicy(alpha, sigma, Sigma_inv, r, gamma, k, d).to(dev)
    else:
        # [수정된 부분] TradeNet 생성 시 d와 k를 전달
        net_pi = TradeNet(d, k).to(dev)
    opt_pi = torch.optim.Adam(net_pi.parameters(), lr=1e-5)
    total_epoch = 20000
    
    if use_cv:
        c_hat_ema, ema_beta = torch.tensor(0.0, device=dev), 0.99

    # ======================= TRAINING LOOP =======================
    for i in range(total_epoch):
        opt_pi.zero_grad()
        T0, W0, Y0, dt0 = generate_uniform_domain(n, T_max, W_min, W_max, Y_min_vec, Y_max_vec, m, k, dev, i)
        
        # Antithetic Sampling
        U_theta_pos, U_M_pos = sim(net_pi, pi_M_const, T0.detach(), W0.detach(), Y0.detach(), dt0.detach(), +1, i, train=True, use_richardson=use_richardson, use_cv=use_cv)
        U_theta_neg, U_M_neg = sim(net_pi, pi_M_const, T0.detach(), W0.detach(), Y0.detach(), dt0.detach(), -1, i, train=True, use_richardson=use_richardson, use_cv=use_cv)
        U_theta_sim = 0.5 * (U_theta_pos + U_theta_neg)
        
        if use_cv:
            # Control Variate Loss Calculation
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
            loss = - (U_theta_norm[finite_mask] - c_hat_final * U_M_centered_norm[finite_mask]).nanmean() if finite_mask.sum() > 0 else torch.tensor(float('nan'))
        else:
            loss = -U_theta_sim.nanmean()

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_pi.parameters(), 1.0)
            opt_pi.step()

        # ======================= EVALUATION & PLOTTING =======================
        if i % 200 == 199 or i == 0:
            print()
            
            T_org, W_org_base, Y_org_samples, dt_org = generate_uniform_domain(n_eval, T_max, W_min, W_max, Y_min_vec, Y_max_vec, m, k, dev, 0)
            W_org = torch.ones_like(W_org_base)
        
            for kth in range(k):
                if k > 1 and kth != 0:
                    continue

                Y_plot = Y_org_samples.clone().detach()
                for j_fix in range(k):
                    if j_fix != kth:
                        Y_plot[:, j_fix] = theta_Y[j_fix]
        
                lamb_exact, dx_lamb_exact, dy_lamb_exact, U_test = estimate_costates(
                    net_pi, pi_M_const, T_org, W_org, Y_plot, dt_org,
                    repeats=repeats, sub_batch_size=sub_batch_size, use_richardson=use_richardson)

                plot(i, U_test,
                     T_org.detach(), W_org.detach(), Y_plot.detach(), kth, dt_org.detach(),
                     lamb_exact.detach(), dx_lamb_exact.detach(), dy_lamb_exact.detach(),
                     net_pi, basis_folder, seed, interpolators,
                     d, k, alpha, sigma, Sigma, sigma_Y, rho_Y, Psi, r, gamma, dev, use_residual_net)

            folder = os.path.join(basis_folder, f'iter_{i:05d}')
            torch.save(net_pi.state_dict(), "%s/net_pi.pth" % folder)
            torch.save(opt_pi.state_dict(), "%s/opt_pi.pth" % folder)    
            
        if i % 10 == 9 or i == 0 :
            print(f'i {i}', end=' ', flush=True)

        if i % 200 == 199 :
            folder = os.path.join(basis_folder, f'iter_{i:05d}')
            os.makedirs(folder, exist_ok=True)
            torch.save(net_pi.state_dict(), f"{folder}/net_pi.pth")
            torch.save(opt_pi.state_dict(), f"{folder}/opt_pi.pth")

    print('\ncompleted')