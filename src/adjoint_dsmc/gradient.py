"""
Gradient accumulation and finite-difference verification.

compute_gradient_wrt_fourier()
    Accumulate ∂J/∂c_m from boundary-reflection events across all time steps.

    The gradient formula (chain-rule through the specular BC):

        ∂J/∂c_m = Σ_{k=0}^{M-1}  Σ_{i ∈ hits_k}
                      γ[k+1, i] · ∂v'_{k,i} / ∂c_m

    where  v' = v − 2(v·n̂)n̂  (post-reflection velocity),

        ∂v'/∂c_m = −2 (v · ∂n̂/∂c_m) n̂ − 2 (v · n̂) ∂n̂/∂c_m,

    and γ[k+1, i] is the adjoint variable at time k+1 (BEFORE the BC adjoint
    Step A in the backward pass), which corresponds to the stored gamma[k+1]
    slice from backward_adjoint_pass().

    v here is the pre-reflection velocity (bc_record.v_pre).

    Assumption: ∂v⁰/∂c = 0 (initial conditions do not depend on c).

check_gradient_fd()
    Compare adjoint gradient to central finite differences for each Fourier
    mode m.  Returns the relative error per mode and prints a summary table.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .records import ForwardHistory
from .boundary import boundary_normal_gradient


# ---------------------------------------------------------------------------
# Main gradient accumulation
# ---------------------------------------------------------------------------

def compute_gradient_wrt_fourier(
    gamma: np.ndarray,
    history: ForwardHistory,
) -> np.ndarray:
    """Compute ∂J/∂c from boundary-collision events in the forward history.

    Parameters
    ----------
    gamma : (M+1, N, 2) float array
        Output of backward_adjoint_pass().
    history : ForwardHistory
        Output of forward_pass_with_history().

    Returns
    -------
    grad_c : (n_coeff,) float array
        Gradient of the objective w.r.t. every Fourier coefficient.
    """
    c = history.fourier_coefficients
    n_coeff = len(c)
    M = len(history.bc)

    grad_c = np.zeros(n_coeff)

    for k in range(M):
        bc_rec = history.bc[k]
        if bc_rec.hit_indices.size == 0:
            continue

        idx     = bc_rec.hit_indices      # (n_hit,)
        n_hat   = bc_rec.normals          # (n_hit, 2)
        theta   = bc_rec.boundary_angles  # (n_hit,)
        v_pre   = bc_rec.v_pre            # (n_hit, 2)

        # Adjoint at time k+1 for the hit particles (BEFORE Step A undo)
        gam_kp1 = gamma[k + 1][idx]       # (n_hit, 2)

        # ∂n̂/∂c_m for all modes at all hit points: (n_hit, n_coeff, 2)
        dn_dc = boundary_normal_gradient(theta, c)

        # v · n̂  (scalar per hit)
        vdn = np.einsum("hi,hi->h", v_pre, n_hat)           # (n_hit,)

        # ∂v'/∂c_m for each mode:
        #   ∂v'/∂c_m = −2 (v · dn_dc_m) n̂ − 2 (v·n̂) dn_dc_m
        # where dn_dc_m = dn_dc[:, m, :]  (n_hit, 2)

        # (v · dn_dc_m): (n_hit, n_coeff)
        v_dot_dndcm = np.einsum("hi,hmi->hm", v_pre, dn_dc)    # (n_hit, n_coeff)

        # term1 = −2 (v·dn_dc_m) n̂: (n_hit, n_coeff, 2)
        term1 = -2.0 * v_dot_dndcm[:, :, np.newaxis] * n_hat[:, np.newaxis, :]

        # term2 = −2 (v·n̂) dn_dc_m: (n_hit, n_coeff, 2)
        term2 = -2.0 * vdn[:, np.newaxis, np.newaxis] * dn_dc

        dv_prime_dcm = term1 + term2   # (n_hit, n_coeff, 2)

        # γ[k+1,i] · ∂v'/∂c_m : (n_hit, n_coeff)
        contrib = np.einsum("hi,hmi->hm", gam_kp1, dv_prime_dcm)

        # Sum over hits
        grad_c += contrib.sum(axis=0)

    return grad_c


# ---------------------------------------------------------------------------
# Finite-difference gradient check
# ---------------------------------------------------------------------------

def check_gradient_fd(
    c_base: np.ndarray,
    obj_fn: Callable[[np.ndarray], float],
    h: float = 1e-4,
    verbose: bool = True,
) -> np.ndarray:
    """Compare the analytic gradient (via adjoint) to central finite differences.

    Parameters
    ----------
    c_base : (n_coeff,) float array
        Fourier coefficients at which to evaluate the gradient.
    obj_fn : callable (n_coeff,) → float
        Black-box objective function.  Internally calls forward_pass +
        compute gradient; only the scalar objective is used here for FD.
    h : float
        Finite-difference step size (default 1e-4).
    verbose : bool
        If True, print a table of adjoint vs FD gradient and relative errors.

    Returns
    -------
    rel_errors : (n_coeff,) float array
        |grad_adj[m] − grad_fd[m]| / (|grad_fd[m]| + 1e-15) per mode.
    """
    n_coeff = len(c_base)
    grad_fd = np.zeros(n_coeff)

    for m in range(n_coeff):
        c_plus  = c_base.copy(); c_plus[m]  += h
        c_minus = c_base.copy(); c_minus[m] -= h
        grad_fd[m] = (obj_fn(c_plus) - obj_fn(c_minus)) / (2.0 * h)

    return grad_fd


def compare_gradients(
    grad_adj: np.ndarray,
    grad_fd: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Compute and optionally display relative errors between two gradients.

    Parameters
    ----------
    grad_adj : (n_coeff,) float array
        Adjoint gradient.
    grad_fd  : (n_coeff,) float array
        Finite-difference gradient.
    verbose : bool
        If True, print a summary table.

    Returns
    -------
    rel_errors : (n_coeff,) float array
    """
    rel_errors = np.abs(grad_adj - grad_fd) / (np.abs(grad_fd) + 1e-15)

    if verbose:
        header = f"{'Mode':>6}  {'Adjoint':>14}  {'FD':>14}  {'Rel. error':>12}"
        print(header)
        print("-" * len(header))
        for m, (ga, gf, re) in enumerate(zip(grad_adj, grad_fd, rel_errors)):
            print(f"{m:>6}  {ga:>14.6e}  {gf:>14.6e}  {re:>12.4e}")
        print(f"\nMax relative error: {rel_errors.max():.4e}")
        print(f"Mean relative error: {rel_errors.mean():.4e}")

    return rel_errors
