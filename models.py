# models.py
# 역할: PyTorch 신경망 모델 아키텍처 정의 (문서 Appendix B)

import torch
import torch.nn as nn

class TradeNet(nn.Module):
    # [수정된 부분] __init__ 함수가 d와 k를 인자로 받도록 변경
    def __init__(self, d, k):
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
    """
    고정된 비학습 레이어로, 분석적인 Myopic 정책을 계산합니다.
    """
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
    """
    서브 네트워크를 생성하고 마지막 레이어의 가중치와 편향을 0으로 초기화합니다.
    """
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
    """
    학습 가능한 헤징 컴포넌트로, 장기 및 단기 효과를 모델링합니다.
    """
    def __init__(self, k, d):
        super().__init__()
        self.long_term_net = create_sub_net(input_dim=k + 1, output_dim=d)
        self.short_term_net = create_sub_net(input_dim=k + 2, output_dim=d)
        self.lamb = nn.Parameter(torch.FloatTensor([0.]))

    def forward(self, W, TmT, Y):
        long_term_input = torch.cat([W, Y], dim=1)
        short_term_input = torch.cat([W, TmT, Y], dim=1)
        pi_long = self.long_term_net(long_term_input)
        # 시간 T-t가 커짐에 따라 단기 수요가 감소
        decay = torch.exp(-torch.exp(self.lamb) * TmT)
        pi_short = self.short_term_net(short_term_input) * decay
        return torch.tanh(pi_long) + torch.tanh(pi_short)

class PGDPOPolicy(nn.Module):
    """
    최종 정책 네트워크. Myopic 부분과 Hedging 부분을 결합한 Residual Learning 구조.
    """
    def __init__(self, alpha, sigma, Sigma_inv, r, gamma, k, d):
        super().__init__()
        self.myopic_layer = MyopicLayer(alpha, sigma, Sigma_inv, r, gamma)
        self.hedging_net = HedgeNet(k, d)

    def forward(self, state):
        W, TmT, Y = state[:, :1], state[:, 1:2], state[:, 2:]
        pi_myopic = self.myopic_layer(Y)
        pi_hedging = self.hedging_net(W, TmT, Y)
        return pi_myopic + pi_hedging