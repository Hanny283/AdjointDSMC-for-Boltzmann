"""
Backward adjoint pass for the discrete adjoint DSMC.

Implements Algorithm 2 from Yang, Silantyev, Caflisch (2023) for general
collision kernels, reduced to Algorithm 3 from Caflisch, Silantyev, Yang
(2021) when ``kernel`` is MaxwellKernel (η ≡ 0).

Time indexing
-------------
Forward pass produces snapshots at t = 0, 1, ..., M.
  velocities[k] = velocity of all particles AFTER step k-1 (k=0 is the IC).

The backward pass propagates the adjoint variable γ[M] → γ[0]:
  γ[k] is the adjoint at time k.

At each backward step from k+1 → k we apply (in order):

  Step A — adjoint through specular BC (self-adjoint reflection):
      γ[k, i] ← R(n̂_i)ᵀ γ[k+1, i]  for each hit particle i
              = γ[k+1, i] − 2 (γ[k+1, i] · n̂_i) n̂_i
      (R is orthogonal and symmetric, so Rᵀ = R.)

      Note: the gradient contribution ∂J/∂c uses γ[k+1] BEFORE Step A.
      This is handled in gradient.py.

  Step B — adjoint through collision (Paper 2, eq. 36b):
      For j=2 (real, accepted) pairs:
          [γ_i, γ_j] ← B(σ,α) [γ_i, γ_j] + [η_i, η_j]
          η_i = ρ/N · (dlog_q_dv_i) · (φ_i + φ_j)
          η_j = −η_i   (antisymmetry)
      For j=3 (virtual-not-real, rejected) pairs:
          η_i = ρ/N · (dlog_eps_minus_q_dv_i) · (φ_i + φ_j)
          η_j = −η_i
          γ_i ← γ_i + η_i,  γ_j ← γ_j + η_j   (D = I for j=3)

phi_final[i] = φ(v^M_i) for ALL particles.  It is provided in ForwardHistory
and used as φ_i = phi_final[i].

Final condition (Paper 2, eq. 36a):
    γ[M, i] = (ρ/N) · ∇_v φ(v^M_i)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .records import ForwardHistory
from .kernels import CollisionKernel, MaxwellKernel


def initialize_adjoint(
    history: ForwardHistory,
    grad_phi: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Set the final condition for the adjoint variable.

    γ[M, i] = (ρ/N) · ∇_v φ(v^M_i)

    Parameters
    ----------
    history : ForwardHistory
        Output of forward_pass_with_history().
    grad_phi : callable (N, 2) → (N, 2)
        Gradient of the per-particle objective w.r.t. velocity.
        Example: ``lambda v: 2*v`` for kinetic-energy objective φ = |v|².

    Returns
    -------
    gamma : (M+1, N, 2) float array
        Adjoint variable array, with γ[M] initialised and all other
        time slices zero-filled (to be computed by backward_adjoint_pass).
    """
    M_plus_1, N, _ = history.velocities.shape
    gamma = np.zeros((M_plus_1, N, 2), dtype=float)

    v_final = history.velocities[-1]               # (N, 2)
    grad_at_final = grad_phi(v_final)              # (N, 2)
    gamma[-1] = history.rho_over_N * grad_at_final
    return gamma


def backward_adjoint_pass(
    history: ForwardHistory,
    grad_phi: Callable[[np.ndarray], np.ndarray],
    kernel: CollisionKernel,
) -> np.ndarray:
    """Run the full backward adjoint sweep.

    Parameters
    ----------
    history : ForwardHistory
        Output of forward_pass_with_history().
    grad_phi : callable (N, 2) → (N, 2)
        Gradient of the per-particle objective w.r.t. velocity.
    kernel : CollisionKernel
        Must be the same kernel used in the forward pass.

    Returns
    -------
    gamma : (M+1, N, 2) float array
        gamma[k] is the adjoint variable at time k, for k=0..M.
        gradient.compute_gradient_wrt_fourier() uses gamma[k+1] (before Step A)
        and bc_history[k] (normals, v_pre) to accumulate ∂J/∂c.
    """
    gamma = initialize_adjoint(history, grad_phi)

    M = len(history.collision)          # number of time steps
    phi_final = history.phi_final       # (N,)
    rho_over_N = history.rho_over_N

    # Backward sweep: k = M−1, M−2, ..., 0
    for k in range(M - 1, -1, -1):
        # Initialise γ[k] from γ[k+1]
        gamma[k] = gamma[k + 1].copy()

        # ---- Step A: adjoint through specular BC -------------------------
        bc_rec = history.bc[k]
        if bc_rec.hit_indices.size > 0:
            idx = bc_rec.hit_indices       # (n_hit,)
            n_hat = bc_rec.normals         # (n_hit, 2)
            # γ_i ← γ_i − 2 (γ_i · n̂) n̂   (reflection = its own adjoint)
            gam_hit = gamma[k][idx]        # (n_hit, 2)
            gdn = np.einsum("ij,ij->i", gam_hit, n_hat)  # (n_hit,)
            gamma[k][idx] = gam_hit - 2.0 * gdn[:, np.newaxis] * n_hat

        # ---- Step B: adjoint through all virtual collision pairs ----------
        col_rec = history.collision[k]
        if col_rec.pair_i.size == 0:
            continue

        pi = col_rec.pair_i      # (P,)
        pj = col_rec.pair_j      # (P,)
        v_i = col_rec.v_i_pre    # (P, 2)  pre-collision velocities at step k
        v_j = col_rec.v_j_pre    # (P, 2)
        accepted = col_rec.accepted   # (P,) bool
        q_vals   = col_rec.q_values   # (P,)
        epsilon  = col_rec.epsilon    # (P,)

        # phi_sum = φ_i + φ_j  (Paper 2, appears in both η terms)
        phi_sum = phi_final[pi] + phi_final[pj]   # (P,)

        # Current adjoint values for involved particles
        # (will be accumulated below)
        delta_i = np.zeros((len(pi), 2))
        delta_j = np.zeros((len(pj), 2))

        # ---- j=2: real collisions ----------------------------------------
        real_mask = accepted
        if real_mask.any():
            ri = pi[real_mask]
            rj = pj[real_mask]
            sig  = col_rec.sigma[real_mask]    # (n_real, 2)
            alph = col_rec.alpha[real_mask]    # (n_real, 2)

            # Apply D = B(σ,α) to the current adjoint
            gi = gamma[k][ri].copy()           # (n_real, 2)
            gj = gamma[k][rj].copy()           # (n_real, 2)
            gi_new, gj_new = kernel.apply_B(gi, gj, sig, alph)

            # Score function η_i (Paper 2, eq. 36b, case j=2)
            dlog_q_i = kernel.dlog_q_dv_i(
                v_i[real_mask], v_j[real_mask]
            )                                  # (n_real, 2)
            phi_sum_r = phi_sum[real_mask]
            eta_i = rho_over_N * dlog_q_i * phi_sum_r[:, np.newaxis]
            # Antisymmetry: η_j = −η_i  (Paper 2 just after eq. 36b)
            eta_j = -eta_i

            gi_new += eta_i
            gj_new += eta_j

            # Accumulate into gamma[k] via scatter-add to handle repeated GIDs
            np.add.at(gamma[k], ri, gi_new - gi)
            np.add.at(gamma[k], rj, gj_new - gj)

        # ---- j=3: virtual-not-real (rejected) ----------------------------
        virtual_mask = ~accepted
        if virtual_mask.any():
            vi_v = pi[virtual_mask]
            vj_v = pj[virtual_mask]

            # D = I  (identity), so no Jacobian term
            dlog_em_q_i = kernel.dlog_eps_minus_q_dv_i(
                v_i[virtual_mask],
                v_j[virtual_mask],
                epsilon[virtual_mask],
            )                                  # (n_virt, 2)
            phi_sum_v = phi_sum[virtual_mask]
            eta_i_v = rho_over_N * dlog_em_q_i * phi_sum_v[:, np.newaxis]
            eta_j_v = -eta_i_v

            np.add.at(gamma[k], vi_v, eta_i_v)
            np.add.at(gamma[k], vj_v, eta_j_v)

    return gamma
