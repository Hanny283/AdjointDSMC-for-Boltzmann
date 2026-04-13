"""
Shape gradient  dL/dC  for the adjoint DSMC method.

Given the forward SimulationHistory and the adjoint arrays (betas, alphas)
from SimulationHistory.backward_pass(), this module computes the gradient of
the objective L with respect to the Fourier boundary coefficients C.

The formula comes from differentiating through every specular-reflection event:

    dL/dC = Σ_{k=0}^{M-1}  Σ_{i: boundary hit at step k}
              [ β_{k+1,i}^T (∂ṽ_{k,i}/∂C)  +  α_{k+1,i}^T (∂x̃_{k,i}/∂C) ]

where β_{k+1,i} and α_{k+1,i} are the velocity and position adjoints at step
k+1 computed by the backward pass.

The boundary maps depend on C via:
    θ_inter(C)  →  n(θ_inter, C)  →  ṽ = v' − 2⟨n,v'⟩n
                                      x̃ = x' − 2(⟨n,x'⟩−c)n

Public API
----------
dr_dC(theta, C)                 -- ∂r/∂C at fixed θ         (2N+1,)
drtheta_dC(theta, C)            -- ∂r_θ/∂C at fixed θ       (2N+1,)
dtheta_inter_dC(theta, v, C)    -- ∂θ_inter/∂C              (2N+1,)
dv_reflected_dC(theta, v, C)    -- ∂ṽ/∂C                   (2, 2N+1)
dx_reflected_dC(theta, xp, v, C)-- ∂x̃/∂C                  (2, 2N+1)
shape_gradient(history, b, a)   -- full dL/dC               (2N+1,)
perimeter(C, n_quad)            -- P(C) = ∫||γ'|| dθ        scalar
perimeter_gradient(C, n_quad)   -- dP/dC                    (2N+1,)
project_step_perimeter_cap(...) -- hard cap P(C) ≤ p_max on a grad step
"""

import numpy as np

from .boundary_geometry import (
    radius_r,
    radius_r_theta,
    f_unnormalized,
    normal_n,
    compute_c_inter,
    gamma_prime,
)
from .adjoint_jacobians import dn_dtheta, dc_dtheta_scalar


# ---------------------------------------------------------------------------
# Partial derivatives of r and r_θ w.r.t. C at fixed θ
# ---------------------------------------------------------------------------

def dr_dC(theta: float, C) -> np.ndarray:
    """
    ∂r(θ;C)/∂C at fixed θ.  Shape (2N+1,).

        [1,  cos(2πθ), cos(4πθ), …, cos(2πNθ),
             sin(2πθ), sin(4πθ), …, sin(2πNθ)]
    """
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    k = np.arange(1, N + 1)
    result = np.empty(len(C))
    result[0]        = 1.0
    result[1:N + 1]  = np.cos(2 * np.pi * k * theta)
    result[N + 1:]   = np.sin(2 * np.pi * k * theta)
    return result


def drtheta_dC(theta: float, C) -> np.ndarray:
    """
    ∂r_θ(θ;C)/∂C at fixed θ.  Shape (2N+1,).

        [0,  −2πk sin(2πkθ) for k=1…N,
              2πk cos(2πkθ) for k=1…N]
    """
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    k = np.arange(1, N + 1)
    result = np.zeros(len(C))
    result[1:N + 1]  = -2 * np.pi * k * np.sin(2 * np.pi * k * theta)
    result[N + 1:]   =  2 * np.pi * k * np.cos(2 * np.pi * k * theta)
    return result


# ---------------------------------------------------------------------------
# Partial derivative of the unnormalized normal f w.r.t. C at fixed θ
# ---------------------------------------------------------------------------

def _df_dC_fixed(theta: float, C) -> np.ndarray:
    """
    (∂f/∂C)_{θ fixed}.  Shape (2, 2N+1).

    f₁ = r_θ sin(2πθ) + 2π r cos(2πθ)
    f₂ = −r_θ cos(2πθ) + 2π r sin(2πθ)
    """
    c2 = np.cos(2 * np.pi * theta)
    s2 = np.sin(2 * np.pi * theta)
    dr  = dr_dC(theta, C)
    drt = drtheta_dC(theta, C)
    row1 = drt * s2 + 2 * np.pi * dr * c2    # ∂f₁/∂C
    row2 = -drt * c2 + 2 * np.pi * dr * s2   # ∂f₂/∂C
    return np.vstack([row1, row2])             # (2, 2N+1)


# ---------------------------------------------------------------------------
# Partial derivative of n w.r.t. C at fixed θ
# ---------------------------------------------------------------------------

def _dn_dC_fixed(theta: float, C) -> np.ndarray:
    """
    (∂n/∂C)_{θ fixed} = (1/‖f‖)(I − nnᵀ)(∂f/∂C).  Shape (2, 2N+1).
    """
    f      = f_unnormalized(theta, C)
    f_norm = np.linalg.norm(f)
    n      = f / f_norm
    dfdC   = _df_dC_fixed(theta, C)              # (2, 2N+1)
    proj   = dfdC - np.outer(n, n @ dfdC)        # (2, 2N+1)
    return proj / f_norm


# ---------------------------------------------------------------------------
# Total ∂θ_inter/∂C  via implicit differentiation on eq (10)
# ---------------------------------------------------------------------------

def dtheta_inter_dC(theta_inter: float, v_prime, C) -> np.ndarray:
    """
    Total ∂θ_inter/∂C from implicit differentiation of F(θ;C) = 0.

    Shape (2N+1,).

    F(θ;C) = r(θ;C)(vʸ cos2πθ − vˣ sin2πθ) − (vʸ xˣ − vˣ xʸ) = 0

    ∂θ/∂Cⱼ = −(∂F/∂Cⱼ) / (∂F/∂θ)  =  −drⱼ · (vʸ cos2πθ − vˣ sin2πθ) / denom

    where denom = ∂F/∂θ is the same denominator used in dtheta_dv_prime.
    """
    v_prime = np.asarray(v_prime, dtype=float)
    vx, vy  = v_prime
    c2      = np.cos(2 * np.pi * theta_inter)
    s2      = np.sin(2 * np.pi * theta_inter)
    r       = radius_r(theta_inter, C)
    r_th    = radius_r_theta(theta_inter, C)

    factor  = vy * c2 - vx * s2
    denom   = r_th * factor - 2 * np.pi * r * (vy * s2 + vx * c2)

    if abs(denom) < 1e-14:
        return np.zeros(len(C))

    return -dr_dC(theta_inter, C) * factor / denom   # (2N+1,)


# ---------------------------------------------------------------------------
# Total ∂n/∂C  (including chain rule through θ_inter)
# ---------------------------------------------------------------------------

def _dn_dC_total(theta_inter: float, v_prime, C) -> np.ndarray:
    """
    Total ∂n/∂C = (∂n/∂θ)(∂θ/∂C) + (∂n/∂C)_{θ fixed}.  Shape (2, 2N+1).
    """
    dth_dC = dtheta_inter_dC(theta_inter, v_prime, C)   # (2N+1,)
    dn_dth = dn_dtheta(theta_inter, C)                   # (2,)
    dn_fix = _dn_dC_fixed(theta_inter, C)                # (2, 2N+1)
    return np.outer(dn_dth, dth_dC) + dn_fix             # (2, 2N+1)


# ---------------------------------------------------------------------------
# Total ∂c̃/∂C
# ---------------------------------------------------------------------------

def _dc_dC_total(theta_inter: float, v_prime, C) -> np.ndarray:
    """
    Total ∂c̃/∂C where c̃ = r̃⟨ñ, e_r⟩,  e_r = (cos2πθ, sin2πθ).

    Shape (2N+1,).

    ∂c̃/∂C = F_scalar · ∂θ/∂C  +  (∂c̃/∂C)_{θ fixed}

    (∂c̃/∂C)_{θ fixed} = ⟨n, e_r⟩ ∂r/∂C  +  r · eᵣᵀ (∂n/∂C)_{θ fixed}
    """
    theta = theta_inter
    c2, s2 = np.cos(2 * np.pi * theta), np.sin(2 * np.pi * theta)
    e_r    = np.array([c2, s2])
    r      = radius_r(theta, C)
    n      = normal_n(theta, C)

    F_scalar = dc_dtheta_scalar(theta, C)          # scalar ∂c̃/∂θ̃
    dth_dC   = dtheta_inter_dC(theta, v_prime, C)  # (2N+1,)
    dr_fix   = dr_dC(theta, C)                     # (2N+1,)
    dn_fix   = _dn_dC_fixed(theta, C)              # (2, 2N+1)

    c_fixed = np.dot(n, e_r) * dr_fix + r * (e_r @ dn_fix)   # (2N+1,)
    return F_scalar * dth_dC + c_fixed


# ---------------------------------------------------------------------------
# ∂ṽ/∂C  and  ∂x̃/∂C
# ---------------------------------------------------------------------------

def dv_reflected_dC(theta_inter: float, v_prime, C) -> np.ndarray:
    """
    ∂ṽ/∂C  where  ṽ = v' − 2⟨n, v'⟩n.  Shape (2, 2N+1).

    ∂ṽ/∂C = −2 n(v'ᵀ ∂n/∂C) − 2⟨n, v'⟩ ∂n/∂C
    """
    v_prime  = np.asarray(v_prime, dtype=float)
    n        = normal_n(theta_inter, C)
    n_dot_v  = np.dot(n, v_prime)
    dndC     = _dn_dC_total(theta_inter, v_prime, C)   # (2, 2N+1)

    v_dot_dndC = v_prime @ dndC                        # (2N+1,)
    return -2 * np.outer(n, v_dot_dndC) - 2 * n_dot_v * dndC   # (2, 2N+1)


def dx_reflected_dC(theta_inter: float, x_prime, v_prime, C) -> np.ndarray:
    """
    ∂x̃/∂C  where  x̃ = x' − 2(⟨n, x'⟩ − c̃)n.  Shape (2, 2N+1).

    ∂x̃/∂C = −2 n(x'ᵀ ∂n/∂C − ∂c̃/∂C) − 2(⟨n, x'⟩ − c̃) ∂n/∂C
    """
    x_prime  = np.asarray(x_prime, dtype=float)
    v_prime  = np.asarray(v_prime, dtype=float)
    n        = normal_n(theta_inter, C)
    c        = compute_c_inter(theta_inter, C)
    n_dot_x  = np.dot(n, x_prime)

    dndC   = _dn_dC_total(theta_inter, v_prime, C)    # (2, 2N+1)
    dcdC   = _dc_dC_total(theta_inter, v_prime, C)    # (2N+1,)

    x_dot_dndC = x_prime @ dndC                        # (2N+1,)
    row_vec    = x_dot_dndC - dcdC                     # (2N+1,)

    return -2 * np.outer(n, row_vec) - 2 * (n_dot_x - c) * dndC   # (2, 2N+1)


# ---------------------------------------------------------------------------
# Main shape-gradient entry point
# ---------------------------------------------------------------------------

def shape_gradient(history, betas: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Gradient of the objective L w.r.t. the Fourier coefficients C.

    dL/dC = Σ_{k,i: boundary hit}
              [ β_{k+1,i}ᵀ (∂ṽ_{k,i}/∂C)  +  α_{k+1,i}ᵀ (∂x̃_{k,i}/∂C) ]

    Parameters
    ----------
    history : SimulationHistory
        Recorded forward trace from ForwardSimulation.run().
    betas : ndarray, shape (M+1, N, 2)
        Velocity adjoints from SimulationHistory.backward_pass().
    alphas : ndarray, shape (M+1, N, 2)
        Position adjoints from SimulationHistory.backward_pass().

    Returns
    -------
    ndarray, shape (2N+1,)
        Gradient dL/dC.
    """
    C    = history.C
    grad = np.zeros_like(C)

    for k, step in enumerate(history.steps):
        beta_k1  = betas[k + 1]    # (N, 2)
        alpha_k1 = alphas[k + 1]   # (N, 2)

        for rec in step.boundary:
            if rec.in_domain:
                continue   # no C-dependence for particles that don't hit boundary

            i     = rec.idx
            theta = rec.theta_inter

            dv_dC = dv_reflected_dC(theta, rec.v_prime, C)            # (2, 2N+1)
            dx_dC = dx_reflected_dC(theta, rec.x_prime, rec.v_prime, C)  # (2, 2N+1)

            grad += beta_k1[i]  @ dv_dC    # (2,)·(2,2N+1) → (2N+1,)
            grad += alpha_k1[i] @ dx_dC

    return grad


# ---------------------------------------------------------------------------
# Perimeter and its gradient
# ---------------------------------------------------------------------------

def perimeter(C, n_quad: int = 400) -> float:
    """
    Approximate perimeter  P(C) = ∫₀¹ ‖γ'(θ;C)‖ dθ  via uniform quadrature.
    """
    thetas = np.linspace(0.0, 1.0, n_quad, endpoint=False)
    norms  = np.array([np.linalg.norm(gamma_prime(t, C)) for t in thetas])
    return float(norms.mean())


def perimeter_gradient(C, n_quad: int = 400) -> np.ndarray:
    """
    Gradient  dP/dC.  Shape (2N+1,).

    dP/dCⱼ = ∫₀¹ (γ'(θ)ᵀ ∂γ'(θ)/∂Cⱼ) / ‖γ'(θ)‖ dθ
    """
    C      = np.asarray(C, dtype=float)
    thetas = np.linspace(0.0, 1.0, n_quad, endpoint=False)
    grad   = np.zeros_like(C)

    for theta in thetas:
        gp   = gamma_prime(theta, C)
        gp_n = np.linalg.norm(gp)
        if gp_n < 1e-14:
            continue

        c2, s2  = np.cos(2 * np.pi * theta), np.sin(2 * np.pi * theta)
        dr      = dr_dC(theta, C)         # (2N+1,)
        drt     = drtheta_dC(theta, C)    # (2N+1,)

        # ∂γ'₁/∂C = drt·c2 − 2π dr·s2
        # ∂γ'₂/∂C = drt·s2 + 2π dr·c2
        dgp1 = drt * c2 - 2 * np.pi * dr * s2
        dgp2 = drt * s2 + 2 * np.pi * dr * c2

        grad += (gp[0] * dgp1 + gp[1] * dgp2) / gp_n

    return grad / n_quad


def project_step_perimeter_cap(
    C,
    direction,
    lr_max: float,
    project_fn,
    p_max: float,
    *,
    n_grid: int = 48,
    tol: float = 1e-9,
):
    """
    Apply a bounded gradient-style update with a hard perimeter (material) cap.

        C_new = project_fn(C − α direction),   α ∈ [0, lr_max],

    choosing the **largest** α on a uniform grid such that
    perimeter(C_new) ≤ p_max.  If the full step is feasible, α = lr_max.

    This models a fusion-style budget: boundary material scales with perimeter
    and cannot exceed p_max.  Assumes α = 0 is feasible (typically
    perimeter(project_fn(C)) ≤ p_max).

    Parameters
    ----------
    C : ndarray
        Current Fourier coefficients.
    direction : ndarray
        Same shape as C; the raw descent direction (not yet scaled by lr).
    lr_max : float
        Maximum step size α.
    project_fn : callable
        box / amplitude projector, e.g. lambda Z: _project_C(Z, ...).
    p_max : float
        Upper limit on perimeter(C_new).
    n_grid : int
        Number of grid points for α (including 0 and lr_max); larger → finer cap.
    tol : float
        Numerical slack on the inequality.

    Returns
    -------
    ndarray
        Feasible projected coefficients C_new.
    """
    if lr_max <= 0:
        return np.asarray(project_fn(np.asarray(C, dtype=float)), dtype=float)

    C = np.asarray(C, dtype=float)
    direction = np.asarray(direction, dtype=float)

    for j in range(n_grid + 1):
        alpha = lr_max * (1.0 - j / max(n_grid, 1))
        C_try = np.asarray(project_fn(C - alpha * direction), dtype=float)
        if perimeter(C_try) <= p_max + tol:
            return C_try

    return np.asarray(project_fn(C), dtype=float)
