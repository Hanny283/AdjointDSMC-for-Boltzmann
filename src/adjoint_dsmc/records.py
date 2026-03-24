"""
Dataclasses that carry the full forward-pass history needed by the backward
adjoint pass.

Collision outcome taxonomy (Paper 2, eq. 37):
  j=1  no virtual collision  -- pairs never selected; not recorded here
  j=2  real collision        -- accepted=True
  j=3  virtual-not-real      -- accepted=False

Both j=2 and j=3 pairs are recorded in CollisionStepRecord because the
score function η is non-zero for both outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class CollisionStepRecord:
    """All virtual collision pairs at one forward time step.

    Arrays are indexed [0 .. n_pairs-1].  Pairs where accepted=True are real
    collisions (j=2); pairs where accepted=False are virtual-but-not-real
    (j=3).  j=1 pairs (never selected) are absent.

    Attributes
    ----------
    pair_i : (n_pairs,) int
        Global particle index of the first member of each pair.
    pair_j : (n_pairs,) int
        Global particle index of the second member of each pair.
    v_i_pre : (n_pairs, 2) float
        Velocity of particle i *before* collision.
    v_j_pre : (n_pairs, 2) float
        Velocity of particle j *before* collision.
    sigma : (n_pairs, 2) float
        Collision direction σ (unit vector in 2-D).
        Zero-filled for j=3 pairs (sigma not sampled for rejected pairs).
    alpha : (n_pairs, 2) float
        Unit relative velocity α = (v_i − v_j) / |v_i − v_j|.
    accepted : (n_pairs,) bool
        True  → j=2, real collision.
        False → j=3, virtual-not-real.
    q_values : (n_pairs,) float
        Raw collision kernel value q^k for each pair before acceptance test.
        For VHS: q = |v_i − v_j|^beta.
    epsilon : (n_pairs,) float
        Per-pair upper bound ε used in the rejection-sampling acceptance test.
        Different cells may have different ε values so this is stored per pair.
    """

    pair_i:   np.ndarray
    pair_j:   np.ndarray
    v_i_pre:  np.ndarray
    v_j_pre:  np.ndarray
    sigma:    np.ndarray
    alpha:    np.ndarray
    accepted: np.ndarray
    q_values: np.ndarray
    epsilon:  np.ndarray


@dataclass
class BCStepRecord:
    """Boundary-reflection events at one forward time step.

    Attributes
    ----------
    hit_indices : (n_hit,) int
        Global particle indices that hit (and were reflected by) the boundary.
    normals : (n_hit, 2) float
        Outward unit normals n̂ at the closest boundary point for each hit.
    boundary_angles : (n_hit,) float
        Parameter θ = atan2(cy, cx) of the closest boundary point, used to
        evaluate ∂n̂/∂c_m analytically in boundary_normal_gradient().
    v_pre : (n_hit, 2) float
        Particle velocities *before* reflection (pre-collision in the BC step).
    """

    hit_indices:     np.ndarray
    normals:         np.ndarray
    boundary_angles: np.ndarray
    v_pre:           np.ndarray


@dataclass
class ForwardHistory:
    """Complete history from one forward pass.

    Carries everything needed by backward_adjoint_pass() and
    compute_gradient_wrt_fourier().

    Attributes
    ----------
    collision : List[CollisionStepRecord], length M
        One record per time step; may contain zero pairs if no virtual
        collisions occurred (empty arrays).
    bc : List[BCStepRecord], length M
        One record per time step; may contain zero hits (empty arrays).
    velocities : (M+1, N, 2) float
        Velocity snapshot *after* each time step (index 0 = initial state).
    positions : (M+1, N, 2) float
        Position snapshot *after* each time step (index 0 = initial state).
    phi_final : (N,) float
        Objective function evaluated at each particle's final-time velocity:
        phi_final[i] = phi(v^M_i).  Required for the score function η.
    fourier_coefficients : (n_modes,) float
        Fourier coefficients c that parameterize the boundary shape.
    rho_over_N : float
        Density normalisation ρ/N from Paper 2 eq. (36a).  With ρ = 1
        (probability-density convention) this equals 1/N.
    """

    collision:            List[CollisionStepRecord]
    bc:                   List[BCStepRecord]
    velocities:           np.ndarray
    positions:            np.ndarray
    phi_final:            np.ndarray
    fourier_coefficients: np.ndarray
    rho_over_N:           float
