"""
Boundary geometry functions for the Fourier-parameterised domain.

The boundary ∂Ω is described by eqs (1)-(2):
    γ(θ; C) = r(θ; C) · (cos 2πθ, sin 2πθ)
    r(θ; C) = c0 + Σ_{k=1}^N (a_k cos 2πkθ + b_k sin 2πkθ)

with coefficient vector
    C = [c0, a1, …, aN, b1, …, bN]^T   (length 2N+1).

References
----------
Paper sections 1–2, eqs (1), (2), (7), (10), (11).
"""

import numpy as np
from scipy.optimize import brentq


# -------------------------------------------------------------------------
# Radius function and its θ-derivatives
# -------------------------------------------------------------------------

def _radius_r_vec(thetas: np.ndarray, C) -> np.ndarray:
    """
    Vectorized radius r(θ; C) for an array of θ values.

    Parameters
    ----------
    thetas : ndarray, shape (K,)
        Array of angular parameters.
    C : array-like, shape (2N+1,)
        Fourier coefficients.

    Returns
    -------
    ndarray, shape (K,)
    """
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    c0 = C[0]
    a = C[1:N + 1]
    b = C[N + 1:2 * N + 1]
    k = np.arange(1, N + 1)
    phases = 2 * np.pi * np.outer(thetas, k)   # (K, N)
    return c0 + (a * np.cos(phases)).sum(axis=1) + (b * np.sin(phases)).sum(axis=1)


def radius_r(theta: float, C) -> float:
    """
    Radius r(θ; C) from eq (2).

        r = c0 + Σ_{k=1}^N [a_k cos(2πkθ) + b_k sin(2πkθ)]

    Parameters
    ----------
    theta : float
        Angular parameter θ ∈ [0, 1).
    C : array-like, shape (2N+1,)
        Fourier coefficients [c0, a1, …, aN, b1, …, bN].

    Returns
    -------
    float
    """
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    c0 = C[0]
    a = C[1:N + 1]
    b = C[N + 1:2 * N + 1]
    k = np.arange(1, N + 1)
    return c0 + np.dot(a, np.cos(2 * np.pi * k * theta)) \
              + np.dot(b, np.sin(2 * np.pi * k * theta))


def radius_r_theta(theta: float, C) -> float:
    """
    First θ-derivative ∂r/∂θ.

        ∂r/∂θ = Σ_{k=1}^N [-2πk a_k sin(2πkθ) + 2πk b_k cos(2πkθ)]

    Parameters
    ----------
    theta : float
    C : array-like, shape (2N+1,)

    Returns
    -------
    float
    """
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    a = C[1:N + 1]
    b = C[N + 1:2 * N + 1]
    k = np.arange(1, N + 1)
    return (np.dot(a, -2 * np.pi * k * np.sin(2 * np.pi * k * theta)) +
            np.dot(b,  2 * np.pi * k * np.cos(2 * np.pi * k * theta)))


def radius_r_theta_theta(theta: float, C) -> float:
    """
    Second θ-derivative ∂²r/∂θ².

        ∂²r/∂θ² = Σ_{k=1}^N [-(2πk)² a_k cos(2πkθ) - (2πk)² b_k sin(2πkθ)]

    Parameters
    ----------
    theta : float
    C : array-like, shape (2N+1,)

    Returns
    -------
    float
    """
    C = np.asarray(C, dtype=float)
    N = (len(C) - 1) // 2
    a = C[1:N + 1]
    b = C[N + 1:2 * N + 1]
    k = np.arange(1, N + 1)
    return (np.dot(a, -(2 * np.pi * k) ** 2 * np.cos(2 * np.pi * k * theta)) +
            np.dot(b, -(2 * np.pi * k) ** 2 * np.sin(2 * np.pi * k * theta)))


# -------------------------------------------------------------------------
# Boundary curve γ and its tangent γ'
# -------------------------------------------------------------------------

def gamma(theta: float, C) -> np.ndarray:
    """
    Boundary point γ(θ; C) = r(θ; C) · (cos 2πθ, sin 2πθ).

    Parameters
    ----------
    theta : float
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    r = radius_r(theta, C)
    return r * np.array([np.cos(2 * np.pi * theta), np.sin(2 * np.pi * theta)])


def gamma_prime(theta: float, C) -> np.ndarray:
    """
    Tangent vector γ'(θ; C).

        γ' = (rθ cos 2πθ − 2πr sin 2πθ,  rθ sin 2πθ + 2πr cos 2πθ)

    Parameters
    ----------
    theta : float
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    r    = radius_r(theta, C)
    r_th = radius_r_theta(theta, C)
    c    = np.cos(2 * np.pi * theta)
    s    = np.sin(2 * np.pi * theta)
    return np.array([r_th * c - 2 * np.pi * r * s,
                     r_th * s + 2 * np.pi * r * c])


# -------------------------------------------------------------------------
# Unnormalized normal vector f and its θ-derivative
# -------------------------------------------------------------------------

def f_unnormalized(theta: float, C) -> np.ndarray:
    """
    Unnormalized outward-normal vector f(θ) = ||γ'|| · n(θ, C).

        f = (rθ sin 2πθ + 2πr cos 2πθ,  −rθ cos 2πθ + 2πr sin 2πθ)

    Normalising f gives the unit outward normal n (eq 7).

    Parameters
    ----------
    theta : float
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    r    = radius_r(theta, C)
    r_th = radius_r_theta(theta, C)
    c    = np.cos(2 * np.pi * theta)
    s    = np.sin(2 * np.pi * theta)
    return np.array([ r_th * s + 2 * np.pi * r * c,
                     -r_th * c + 2 * np.pi * r * s])


def f_unnormalized_prime(theta: float, C) -> np.ndarray:
    """
    Derivative ∂f/∂θ of the unnormalized normal vector.

    Used in Lemma 2.6 to build ∂ñ/∂θ̃.

    Derivation (component-wise, let tw = 2πθ):
        f₁ = rθ sin tw + 2πr cos tw
        ∂f₁/∂θ = rθθ sin tw + 4π rθ cos tw − (2π)² r sin tw

        f₂ = −rθ cos tw + 2πr sin tw
        ∂f₂/∂θ = −rθθ cos tw + 4π rθ sin tw + (2π)² r cos tw

    Parameters
    ----------
    theta : float
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
    """
    r      = radius_r(theta, C)
    r_th   = radius_r_theta(theta, C)
    r_thth = radius_r_theta_theta(theta, C)
    c      = np.cos(2 * np.pi * theta)
    s      = np.sin(2 * np.pi * theta)
    two_pi = 2 * np.pi

    df1 = r_thth * s + 4 * np.pi * r_th * c - two_pi ** 2 * r * s
    df2 = -r_thth * c + 4 * np.pi * r_th * s + two_pi ** 2 * r * c
    return np.array([df1, df2])


# -------------------------------------------------------------------------
# Outward unit normal  n(θ, C)  — eq (7)
# -------------------------------------------------------------------------

def normal_n(theta: float, C) -> np.ndarray:
    """
    Outward unit normal n(θ, C) from eq (7).

        n = f(θ) / ||f(θ)||

    where f is the unnormalized normal from f_unnormalized().

    Parameters
    ----------
    theta : float
        Angular parameter (typically θ_inter after calling solve_theta_inter).
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray, shape (2,)
        Unit outward normal.
    """
    f = f_unnormalized(theta, C)
    return f / np.linalg.norm(f)


# -------------------------------------------------------------------------
# Intersection angle  θ_inter  — eq (10)
# -------------------------------------------------------------------------

def solve_theta_inter(x_k, v_k, C, n_grid: int = 400):
    """
    Solve eq (10) for the intersection angle θ_inter.

    The line from x_k in direction v_k intersects the boundary where

        r(θ; C) · (v^y cos 2πθ − v^x sin 2πθ) = v^y x^x − v^x x^y     (10)

    Rearranging as F(θ) = 0 and scanning for sign changes, then refining
    each root with Brent's method.

    The returned root is the one with the smallest positive angular
    displacement from the polar angle of x_k (i.e. the first intersection
    in the forward direction of travel).

    Parameters
    ----------
    x_k : array-like, shape (2,)
        Particle position (x^x, x^y) inside Ω before the step.
    v_k : array-like, shape (2,)
        Particle velocity (v^x, v^y).
    C : array-like, shape (2N+1,)
        Fourier coefficients.
    n_grid : int, optional
        Number of equispaced grid points used for the initial scan (default 400).
        Increase for highly non-convex boundaries.

    Returns
    -------
    theta_inter : float or None
        Intersection angle in [0, 1), or None if no intersection is found.
    """
    x_k = np.asarray(x_k, dtype=float)
    v_k = np.asarray(v_k, dtype=float)
    vx, vy = v_k[0], v_k[1]
    xx, xy = x_k[0], x_k[1]
    rhs = vy * xx - vx * xy

    def F(theta):
        r = radius_r(theta, C)
        return r * (vy * np.cos(2 * np.pi * theta) - vx * np.sin(2 * np.pi * theta)) - rhs

    thetas = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    dth    = thetas[1] - thetas[0]
    Fvals  = np.vectorize(F)(thetas)

    roots = []
    for i in range(len(thetas) - 1):
        if Fvals[i] * Fvals[i + 1] < 0.0:
            try:
                root = brentq(F, thetas[i], thetas[i + 1], xtol=1e-12, rtol=1e-12)
                roots.append(root)
            except ValueError:
                pass

    # Check wrap-around interval [thetas[-1], 1]
    if Fvals[-1] * Fvals[0] < 0.0:
        try:
            root = brentq(lambda t: F(t % 1.0),
                          thetas[-1], thetas[-1] + dth,
                          xtol=1e-12, rtol=1e-12)
            roots.append(root % 1.0)
        except ValueError:
            pass

    if not roots:
        return None

    # Select the root whose boundary point γ(θ) lies in the FORWARD direction
    # from x_k along v_k, i.e. (γ(θ)−x_k)·v_k > 0 and t is minimised.
    # This fixes the backwards-root bug that caused particle escape.
    v_sq = vx ** 2 + vy ** 2
    candidates = []
    for root in roots:
        gx = radius_r(root, C) * np.cos(2 * np.pi * root)
        gy = radius_r(root, C) * np.sin(2 * np.pi * root)
        t  = ((gx - xx) * vx + (gy - xy) * vy) / (v_sq + 1e-30)
        if t > 1e-8:                      # strictly forward
            candidates.append((t, root))

    if candidates:
        return min(candidates, key=lambda x: x[0])[1]

    # Fallback (particle nearly tangential or already on boundary):
    # use the root with smallest |t| regardless of sign.
    t_abs = []
    for root in roots:
        gx = radius_r(root, C) * np.cos(2 * np.pi * root)
        gy = radius_r(root, C) * np.sin(2 * np.pi * root)
        t  = ((gx - xx) * vx + (gy - xy) * vy) / (v_sq + 1e-30)
        t_abs.append((abs(t), root))
    return min(t_abs, key=lambda x: x[0])[1]


# -------------------------------------------------------------------------
# Batch intersection solver (vectorized over N particles)
# -------------------------------------------------------------------------

def solve_theta_inter_batch(
    x: np.ndarray,
    v: np.ndarray,
    C,
    n_grid: int = 400,
    n_bisect: int = 35,
) -> np.ndarray:
    """
    Batch version of solve_theta_inter for K particles simultaneously.

    Replaces K sequential calls to solve_theta_inter (each doing a 400-point
    grid scan + Brent's method) with:
      1. One vectorized grid scan:  F_matrix (K, n_grid) via NumPy broadcasting.
      2. Per-particle best-interval selection using masked argmin on t-values.
      3. K simultaneous bisection steps (n_bisect pure-NumPy iterations).

    Precision: n_grid=400, n_bisect=35 gives ≈ (1/400)·2⁻³⁵ ≈ 7e-13 — on
    par with brentq's xtol=1e-12.

    Parameters
    ----------
    x : ndarray, shape (K, 2)
        Particle positions (only the subset that exited the domain).
    v : ndarray, shape (K, 2)
        Particle velocities.
    C : array-like, shape (2N+1,)
        Fourier coefficients.
    n_grid : int
        Number of equispaced scan points (default 400).
    n_bisect : int
        Number of bisection iterations (default 35).

    Returns
    -------
    theta_inter : ndarray, shape (K,)
        Intersection angles in [0, 1).  NaN for particles where no crossing
        was found (rare edge case: particle nearly tangential to boundary).
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    C = np.asarray(C, dtype=float)
    K = x.shape[0]

    vx, vy = v[:, 0], v[:, 1]          # (K,)
    xx, xy = x[:, 0], x[:, 1]          # (K,)
    rhs  = vy * xx - vx * xy           # (K,)
    v_sq = vx ** 2 + vy ** 2           # (K,)

    # ------------------------------------------------------------------
    # Grid scan — compute F_matrix (K, n_grid) in one shot
    # ------------------------------------------------------------------
    thetas = np.linspace(0.0, 1.0, n_grid, endpoint=False)  # (n_grid,)
    dth    = thetas[1] - thetas[0]
    cos_t  = np.cos(2 * np.pi * thetas)                     # (n_grid,)
    sin_t  = np.sin(2 * np.pi * thetas)                     # (n_grid,)
    r_grid = _radius_r_vec(thetas, C)                        # (n_grid,)

    # F_matrix[i, j] = F(thetas[j]) for particle i
    F_matrix = (r_grid[None, :] * (vy[:, None] * cos_t[None, :]
                                    - vx[:, None] * sin_t[None, :])
                - rhs[:, None])                              # (K, n_grid)

    # ------------------------------------------------------------------
    # Detect sign changes: (n_grid-1) main intervals + 1 wrap-around
    # ------------------------------------------------------------------
    sc_main = F_matrix[:, :-1] * F_matrix[:, 1:] < 0        # (K, n_grid-1)
    sc_wrap = (F_matrix[:, -1] * F_matrix[:, 0] < 0)        # (K,)

    # Combined arrays — append wrap column last
    sc_all = np.concatenate([sc_main, sc_wrap[:, None]], axis=1)  # (K, n_grid)

    has_any = np.any(sc_all, axis=1)   # (K,) — at least one crossing

    # Interval left/right endpoints (wrap-around stored as > 1)
    lo_all = np.append(thetas[:-1], thetas[-1])              # (n_grid,)
    hi_all = np.append(thetas[1:],  thetas[-1] + dth)        # (n_grid,)

    # ------------------------------------------------------------------
    # Forward-direction selection — pick interval with smallest positive t
    # ------------------------------------------------------------------
    # Evaluate F at interval midpoints to compute the t parameter
    theta_mid = 0.5 * (lo_all + hi_all)                      # (n_grid,)
    r_mid = _radius_r_vec(theta_mid % 1.0, C)                # (n_grid,)
    gx_mid = r_mid * np.cos(2 * np.pi * theta_mid)           # (n_grid,)
    gy_mid = r_mid * np.sin(2 * np.pi * theta_mid)           # (n_grid,)

    # t_all[i, j] = signed travel time to interval-j midpoint for particle i
    t_all = (((gx_mid[None, :] - xx[:, None]) * vx[:, None]
               + (gy_mid[None, :] - xy[:, None]) * vy[:, None])
             / (v_sq[:, None] + 1e-30))                      # (K, n_grid)

    # Best forward interval
    valid_fwd = sc_all & (t_all > 1e-8)                      # (K, n_grid)
    has_fwd   = np.any(valid_fwd, axis=1)                    # (K,)

    t_fwd   = np.where(valid_fwd, t_all, np.inf)             # (K, n_grid)
    t_fall  = np.where(sc_all, np.abs(t_all), np.inf)        # (K, n_grid)

    best_j_fwd  = np.argmin(t_fwd,  axis=1)                  # (K,)
    best_j_fall = np.argmin(t_fall, axis=1)                  # (K,)
    best_j = np.where(has_fwd, best_j_fwd, best_j_fall)      # (K,)

    # ------------------------------------------------------------------
    # Batch bisection — K intervals refined simultaneously
    # ------------------------------------------------------------------
    lo = lo_all[best_j]                                       # (K,)
    hi = hi_all[best_j]                                       # (K,)

    # F at left endpoint — index into F_matrix (last column is wrap-around lo)
    # For best_j == n_grid-1: lo = thetas[-1], F_lo = F_matrix[:, n_grid-1]
    F_lo = F_matrix[np.arange(K), np.minimum(best_j, n_grid - 1)]  # (K,)

    for _ in range(n_bisect):
        mid   = 0.5 * (lo + hi)
        r_mid2 = _radius_r_vec(mid % 1.0, C)
        F_mid = (r_mid2 * (vy * np.cos(2 * np.pi * mid)
                           - vx * np.sin(2 * np.pi * mid)) - rhs)  # (K,)
        move_lo = F_lo * F_mid > 0                           # (K,) bool
        lo   = np.where(move_lo, mid, lo)
        F_lo = np.where(move_lo, F_mid, F_lo)
        hi   = np.where(move_lo, hi,  mid)

    theta_result = (0.5 * (lo + hi)) % 1.0                   # (K,) in [0, 1)
    return np.where(has_any, theta_result, np.nan)


# -------------------------------------------------------------------------
# Reflection constant  c_inter  — eq (11)
# -------------------------------------------------------------------------

def compute_c_inter(theta_inter: float, C) -> float:
    """
    Constant c_inter from eq (11).

        c_inter = r(θ_inter, C) · ⟨n(θ_inter, C), (cos 2πθ_inter, sin 2πθ_inter)⟩

    This is the signed offset of the tangent-plane reflection line
    L = { x : ⟨ñ, x⟩ = c̃ } at the intersection point γ(θ_inter, C).

    Parameters
    ----------
    theta_inter : float
        Intersection angle from solve_theta_inter().
    C : array-like, shape (2N+1,)

    Returns
    -------
    float
    """
    r   = radius_r(theta_inter, C)
    n   = normal_n(theta_inter, C)
    e_r = np.array([np.cos(2 * np.pi * theta_inter),
                    np.sin(2 * np.pi * theta_inter)])
    return r * np.dot(n, e_r)
