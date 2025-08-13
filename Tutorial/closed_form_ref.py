# closed_form_ref.py
# Closed-form policy for CRRA investor with one OU factor (Kim–Omberg style)
# Provides: precompute_BC(..., mode="Y"|"X") and ClosedFormPolicy(...)
import math, numpy as np, torch
import torch.nn as nn

__all__ = ["precompute_BC", "ClosedFormPolicy"]

# ---------- Linear interp ----------
@torch.no_grad()
def _interp_BC(tau, taus, Btab, Ctab, T):
    tau = tau.to(dtype=torch.float64)
    tau = torch.clamp(tau, min=0.0, max=float(T))
    s = (tau / float(T)) * (taus.shape[0]-1)
    idx0 = torch.floor(s).long().clamp(max=taus.shape[0]-2)
    frac = (s - idx0.to(s.dtype))
    B0 = Btab[idx0]; B1 = Btab[idx0+1]
    C0 = Ctab[idx0]; C1 = Ctab[idx0+1]
    B = (1.0-frac)*B0 + frac*B1
    C = (1.0-frac)*C0 + frac*C1
    return B.to(torch.float32).view(-1,1), C.to(torch.float32).view(-1,1)

# ---------- RK4 helper ----------
def _rk4_step(B, C, h, fB, fC):
    k1C = fC(C);           k1B = fB(B, C)
    k2C = fC(C + 0.5*h*k1C)
    k2B = fB(B + 0.5*h*k1B, C + 0.5*h*k1C)
    k3C = fC(C + 0.5*h*k2C)
    k3B = fB(B + 0.5*h*k2B, C + 0.5*h*k2C)
    k4C = fC(C + h*k3C)
    k4B = fB(B + h*k3B, C + h*k3C)
    C_next = C + (h/6.0)*(k1C + 2*k2C + 2*k3C + k4C)
    B_next = B + (h/6.0)*(k1B + 2*k2B + 2*k3B + k4B)
    return B_next, C_next

# ---------- Main API ----------
def precompute_BC(
    T: float,
    kappaY: float,
    thetaY: float,
    sigmaY: float,
    gamma: float,
    rho: float,
    alpha: float,
    n_grid: int = 2048,
    mode: str = "Y",          # "Y" (default) or "X"
    device: torch.device | str = "cpu",
):
    """
    Returns (taus, Btab, Ctab) where B,C are in the **Y-based** policy form:
        pi_cf(W, τ, Y) = (1/γ)*[(α/σ)Y + (ρ σ_Y / σ)(B(τ) + C(τ) Y)]
    If mode="X", ODE is solved in X := αY coordinates and then converted to Y-based B,C.
    """
    taus = np.linspace(0.0, T, n_grid+1, dtype=np.float64)

    if mode.upper() == "Y":
        # Y-based ODE (your base file style)
        delta = (1.0 - gamma) / gamma

        def C_rhs(C):
            return 2.0*kappaY*C - (sigmaY**2)*(C**2) - delta * (alpha + sigmaY*rho*C)**2

        def B_rhs(B, C):
            return kappaY*B - kappaY*thetaY*C - (sigmaY**2)*B*C - delta * B * (sigmaY*rho) * (alpha + sigmaY*rho*C)

        B = 0.0; C = 0.0
        Btab = np.zeros_like(taus); Ctab = np.zeros_like(taus)
        for i in range(n_grid):
            h = taus[i+1] - taus[i]
            B, C = _rk4_step(B, C, h, B_rhs, C_rhs)
            Btab[i+1] = B; Ctab[i+1] = C

    else:
        # X-based ODE (Kim–Omberg; then convert to Y-based via B_Y=α B_X, C_Y=α^2 C_X)
        delta = (1.0 - gamma) / gamma
        kX    = kappaY
        sigX  = alpha * sigmaY
        xbar  = alpha * thetaY
        a = delta
        b = 2.0 * (delta * rho * sigX - kX)
        c = (sigX**2) * (1.0 + delta * (rho**2))

        def C_rhs(C):          # dC/dτ = 0.5 c C^2 + b C + a
            return 0.5*c*(C**2) + b*C + a

        def B_rhs(B, C):       # dB/dτ = c B C + b B + kX xbar C
            return c*B*C + b*B + kX*xbar*C

        Bx = 0.0; Cx = 0.0
        Bxtab = np.zeros_like(taus); Cxtab = np.zeros_like(taus)
        for i in range(n_grid):
            h = taus[i+1] - taus[i]
            Bx, Cx = _rk4_step(Bx, Cx, h, B_rhs, C_rhs)
            Bxtab[i+1] = Bx; Cxtab[i+1] = Cx
        # convert to Y-based tables
        Btab = alpha * Bxtab
        Ctab = (alpha**2) * Cxtab

    taus_t = torch.tensor(taus, device=device, dtype=torch.float64)
    Btab_t = torch.tensor(Btab, device=device, dtype=torch.float64)
    Ctab_t = torch.tensor(Ctab, device=device, dtype=torch.float64)
    return taus_t, Btab_t, Ctab_t


class ClosedFormPolicy(nn.Module):
    """
    Y-based closed-form policy module that works for both modes:
      pi_cf = (1/γ)[ (α/σ) Y + (ρ σ_Y / σ) (B(τ) + C(τ) Y) ]
    B,C must be Y-based (precompute_BC returns Y-based in both modes).
    """
    def __init__(self, taus, Btab, Ctab, sigma, sigmaY, alpha, rho, gamma, T, pi_cap=2.0):
        super().__init__()
        self.register_buffer("taus", taus)
        self.register_buffer("Btab", Btab)
        self.register_buffer("Ctab", Ctab)
        self.sigma  = float(sigma)
        self.sigmaY = float(sigmaY)
        self.alpha  = float(alpha)
        self.rho    = float(rho)
        self.gamma  = float(gamma)
        self.T      = float(T)
        self.pi_cap = float(pi_cap)

    def forward(self, W, TmT, Y):
        B, C = _interp_BC(TmT, self.taus, self.Btab, self.Ctab, self.T)
        pi = (1.0/self.gamma) * (
            (self.alpha/self.sigma)*Y + (self.rho*self.sigmaY/self.sigma)*(B + C*Y)
        )
        return torch.clamp(pi, -self.pi_cap, self.pi_cap)