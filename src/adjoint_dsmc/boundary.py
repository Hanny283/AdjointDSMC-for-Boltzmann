"""
Boundary adjoint utilities.

adjoint_through_reflection()
    Apply the specular-reflection operator to a batch of adjoint vectors.
    The operator is self-adjoint (R = Rᵀ) so the same formula is used in
    both forward and backward passes.

boundary_normal_gradient()
    Analytical gradient ∂n̂/∂c_m of the outward unit normal at a point on the
    star-shaped boundary, as a function of the boundary angle θ and the
    Fourier coefficients c.

    The star-shape r(θ; c) parameterisation:
        r(θ) = c[0] + Σ_{m=1}^{M}  c[m]       cos(m θ)
                     + Σ_{m=1}^{M}  c[m + M]   sin(m θ)

    Tangent vector ∂(x,y)/∂θ:
        x(θ) = r(θ) cosθ,  y(θ) = r(θ) sinθ
        ∂x/∂θ = r'(θ) cosθ − r(θ) sinθ
        ∂y/∂θ = r'(θ) sinθ + r(θ) cosθ

    Un-normalised outward normal  (90° clockwise rotation of tangent):
        n_raw = (∂y/∂θ, −∂x/∂θ)
              = ( r cosθ + r' sinθ,   r sinθ − r' cosθ )

    Normalisation:
        L = |n_raw|,   n̂ = n_raw / L
        ∂n̂/∂c_m = (∂n_raw/∂c_m − (n̂ · ∂n_raw/∂c_m) n̂) / L

    Derivatives w.r.t. individual Fourier coefficients:
        m = 0  (DC term):
            ∂r/∂c_0 = 1,  ∂r'/∂c_0 = 0
            ∂n_raw / ∂c_0 = (cosθ, sinθ)

        m = 1..M  (cosine modes, index m):
            ∂r/∂c_m  = cos(mθ)
            ∂r'/∂c_m = −m sin(mθ)
            ∂n_raw_x / ∂c_m = cos(mθ) cosθ − m sin(mθ) sinθ
            ∂n_raw_y / ∂c_m = cos(mθ) sinθ + m sin(mθ) cosθ

        m = 1..M  (sine modes, index m + M):
            ∂r/∂c_{m+M}  = sin(mθ)
            ∂r'/∂c_{m+M} = m cos(mθ)
            ∂n_raw_x / ∂c_{m+M} = sin(mθ) cosθ + m cos(mθ) sinθ
            ∂n_raw_y / ∂c_{m+M} = sin(mθ) sinθ − m cos(mθ) cosθ
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Specular reflection adjoint
# ---------------------------------------------------------------------------

def adjoint_through_reflection(
    gamma: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Apply the (self-adjoint) specular reflection operator to γ.

    R(n̂) v = v − 2 (v · n̂) n̂

    Since R is symmetric and orthogonal, Rᵀ = R, so the adjoint step is
    identical to the forward reflection.

    Parameters
    ----------
    gamma : (n, 2) float array
        Adjoint vectors to reflect.
    normals : (n, 2) float array
        Outward unit normals at the boundary.

    Returns
    -------
    (n, 2) float array  — reflected adjoint vectors.
    """
    gdn = np.einsum("ij,ij->i", gamma, normals)   # (n,)
    return gamma - 2.0 * gdn[:, np.newaxis] * normals


# ---------------------------------------------------------------------------
# Analytical boundary-normal gradient
# ---------------------------------------------------------------------------

def boundary_normal_gradient(
    theta: np.ndarray,
    fourier_coefficients: np.ndarray,
) -> np.ndarray:
    """Compute ∂n̂/∂c for all Fourier modes at given boundary angles θ.

    Parameters
    ----------
    theta : (n_hit,) float array
        Boundary parameter angle (radians) of each reflection event.
        Typically obtained as ``np.arctan2(closest_y, closest_x)``.
    fourier_coefficients : (2M+1,) float array
        Current Fourier coefficients [c_0, c_1, ..., c_M, s_1, ..., s_M].

    Returns
    -------
    dn_dc : (n_hit, 2M+1, 2) float array
        dn_dc[h, m, :] = ∂n̂/∂c_m  at hit h.
        Axis 1 layout: index 0 → c_0 (DC), indices 1..M → cosine modes,
        indices M+1..2M → sine modes.
    """
    theta = np.asarray(theta, dtype=float)
    c = np.asarray(fourier_coefficients, dtype=float)

    n_coeff = len(c)
    n_modes_full = n_coeff - 1    # 2M
    assert n_modes_full % 2 == 0, (
        "fourier_coefficients must have length 2M+1 (odd number of elements)"
    )
    M = n_modes_full // 2         # number of harmonic modes

    n_hit = len(theta)
    costh = np.cos(theta)         # (n_hit,)
    sinth = np.sin(theta)         # (n_hit,)

    # ------------------------------------------------------------------
    # 1.  Evaluate r(θ) and r'(θ) for the given fourier_coefficients
    # ------------------------------------------------------------------
    # r(θ) = c[0] + Σ c[m] cos(mθ) + Σ c[M+m] sin(mθ)
    r = np.full(n_hit, c[0])
    r_prime = np.zeros(n_hit)
    for m in range(1, M + 1):
        cosmth = np.cos(m * theta)
        sinmth = np.sin(m * theta)
        r       +=      c[m]   * cosmth + c[M + m] * sinmth
        r_prime += -m * c[m]   * sinmth + m * c[M + m] * cosmth

    # ------------------------------------------------------------------
    # 2.  Un-normalised normal and its magnitude L
    # ------------------------------------------------------------------
    n_raw_x = r * costh + r_prime * sinth   # (n_hit,)
    n_raw_y = r * sinth - r_prime * costh

    L = np.sqrt(n_raw_x ** 2 + n_raw_y ** 2)   # (n_hit,)
    safe_L = np.where(L > 1e-30, L, 1e-30)
    n_hat_x = n_raw_x / safe_L
    n_hat_y = n_raw_y / safe_L

    # ------------------------------------------------------------------
    # 3.  Derivative of n_raw w.r.t. each Fourier coefficient
    # ------------------------------------------------------------------
    # dn_dc[h, m, :] = ∂n_raw/∂c_m  (before normalisation)
    dn_raw_dc = np.zeros((n_hit, n_coeff, 2))

    # DC mode  (m=0)
    dn_raw_dc[:, 0, 0] = costh
    dn_raw_dc[:, 0, 1] = sinth

    for m in range(1, M + 1):
        cosmth = np.cos(m * theta)   # (n_hit,)
        sinmth = np.sin(m * theta)

        # Cosine mode (index m)
        dn_raw_dc[:, m, 0] = cosmth * costh - m * sinmth * sinth
        dn_raw_dc[:, m, 1] = cosmth * sinth + m * sinmth * costh

        # Sine mode (index M+m)
        dn_raw_dc[:, M + m, 0] = sinmth * costh + m * cosmth * sinth
        dn_raw_dc[:, M + m, 1] = sinmth * sinth - m * cosmth * costh

    # ------------------------------------------------------------------
    # 4.  Normalise: ∂n̂/∂c_m = (∂n_raw/∂c_m − (n̂·∂n_raw/∂c_m) n̂) / L
    # ------------------------------------------------------------------
    n_hat = np.stack([n_hat_x, n_hat_y], axis=-1)    # (n_hit, 2)

    # dot product n̂ · ∂n_raw/∂c_m : shape (n_hit, n_coeff)
    proj = np.einsum("hi,hmi->hm", n_hat, dn_raw_dc)  # (n_hit, n_coeff)

    # Subtract projection: (n_hit, n_coeff, 2)
    dn_dc = (
        dn_raw_dc
        - proj[:, :, np.newaxis] * n_hat[:, np.newaxis, :]
    )
    dn_dc /= safe_L[:, np.newaxis, np.newaxis]

    return dn_dc   # (n_hit, n_coeff, 2)
