"""
Collision kernel abstractions and concrete implementations.

Each kernel provides:
  q(v_i, v_j)                      -- collision rate
  dlog_q_dv_i(v_i, v_j)            -- ∂ log q / ∂v_i  (score for j=2 pairs)
  dlog_eps_minus_q_dv_i(v_i, v_j, epsilon)
                                   -- ∂ log(ε−q) / ∂v_i  (score for j=3 pairs)
  apply_B(gamma_i, gamma_j, sigma, alpha)
                                   -- vectorised B(σ,α) applied to adjoint pairs

The 2-D backward B(σ,α) matrix (Paper 1, eq. 5; Paper 2, eq. 45) is:
    B(σ,α) = ½ [[I + ασᵀ,  I − ασᵀ],
                 [I − ασᵀ,  I + ασᵀ]]
where I is the 2×2 identity.  The action on a pair (γ_i, γ_j) simplifies to:
    g_cm   = ½(γ_i + γ_j)
    s_dot  = σ · (γ_i − γ_j)   (scalar per pair)
    new γ_i = g_cm + ½ α s_dot
    new γ_j = g_cm − ½ α s_dot

For VHS (angle-independent) kernels running with Algorithm-3-style forward
sampling (θ sampled independently and σ derived afterwards), the B̃ correction
term from Paper 2 eq. (47–48) averages to zero and D = B(σ,α) is correct
(Paper 2 Section 3.4.2 and Fig. 1 numerical evidence).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CollisionKernel(ABC):
    """Abstract base class for DSMC collision kernels."""

    @abstractmethod
    def q(self, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
        """Collision kernel value q(v_i, v_j).

        Parameters
        ----------
        v_i, v_j : (n, 2) arrays
            Pre-collision velocities.

        Returns
        -------
        (n,) array of non-negative kernel values.
        """

    @abstractmethod
    def dlog_q_dv_i(self, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
        """Gradient ∂ log q / ∂v_i.

        Used in the score-function term η for j=2 (real collision) pairs.

        Parameters
        ----------
        v_i, v_j : (n, 2) arrays

        Returns
        -------
        (n, 2) array.  Zero for constant kernels (Maxwell molecules).
        """

    @abstractmethod
    def dlog_eps_minus_q_dv_i(
        self,
        v_i: np.ndarray,
        v_j: np.ndarray,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        """Gradient ∂ log(ε − q) / ∂v_i.

        Used in the score-function term η for j=3 (virtual-not-real) pairs.

        Parameters
        ----------
        v_i, v_j : (n, 2) arrays
        epsilon  : (n,) array  upper bound values per pair

        Returns
        -------
        (n, 2) array.  Zero for constant kernels (Maxwell molecules).
        """

    @property
    @abstractmethod
    def is_angle_independent(self) -> bool:
        """True when the kernel does not depend on the scattering angle θ."""

    # ------------------------------------------------------------------
    # Shared helper: vectorised B(σ,α) application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_B(
        gamma_i: np.ndarray,
        gamma_j: np.ndarray,
        sigma: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the adjoint Jacobian B(σ,α) to a batch of adjoint pairs.

        Exploits the factored form (Paper 1, eq. 34):
            g_cm     = ½(γ_i + γ_j)
            s_dot    = σ · (γ_i − γ_j)
            γ_i_new = g_cm + ½ α s_dot
            γ_j_new = g_cm − ½ α s_dot

        Parameters
        ----------
        gamma_i, gamma_j : (n, 2) arrays
            Adjoint variables at time k+1.
        sigma, alpha : (n, 2) arrays
            Collision direction and unit relative velocity.

        Returns
        -------
        gamma_i_new, gamma_j_new : (n, 2) arrays
        """
        g_cm = 0.5 * (gamma_i + gamma_j)
        diff = gamma_i - gamma_j
        s_dot = np.einsum("ij,ij->i", sigma, diff)     # (n,)
        half_alpha_s = 0.5 * alpha * s_dot[:, np.newaxis]  # (n, 2)
        return g_cm + half_alpha_s, g_cm - half_alpha_s


# ---------------------------------------------------------------------------
# Maxwell molecules  (Paper 1 Algorithm 3 exact reduction)
# ---------------------------------------------------------------------------

class MaxwellKernel(CollisionKernel):
    """Constant collision kernel: q = const.

    All virtual collisions are real (no rejection sampling needed).
    Both score-function methods return zero, so η = 0 for all pairs and the
    adjoint DSMC reduces exactly to Algorithm 3 from Caflisch et al. (2021).
    """

    def __init__(self, q_const: float = 1.0) -> None:
        self._q_const = float(q_const)

    def q(self, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
        n = v_i.shape[0]
        return np.full(n, self._q_const)

    def dlog_q_dv_i(self, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
        return np.zeros_like(v_i)

    def dlog_eps_minus_q_dv_i(
        self,
        v_i: np.ndarray,
        v_j: np.ndarray,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        return np.zeros_like(v_i)

    @property
    def is_angle_independent(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"MaxwellKernel(q_const={self._q_const})"


# ---------------------------------------------------------------------------
# Variable Hard Sphere  (angle-independent, D = B(σ,α))
# ---------------------------------------------------------------------------

class VHSKernel(CollisionKernel):
    """Variable Hard Sphere collision kernel: q = C_beta * |v_i − v_j|^beta.

    This matches the kernel implicitly used by `ArraySigma_VHS` in
    ``universal_sim_helpers.py`` (Constant=1, alpha=1 ↔ beta=1).

    For VHS, σ is angle-independent, so D = B(σ,α) is correct and the B̃
    correction term from Paper 2 eq. (47–48) vanishes (Section 3.4.2).

    Score function terms:
      ∂ log q / ∂v_i = beta * (v_i − v_j) / |v_i − v_j|²
      ∂ log(ε−q) / ∂v_i = −(q / (ε−q)) · ∂ log q / ∂v_i
                        = −beta * (v_i − v_j) * q / ((ε−q) |v_i − v_j|²)

    Parameters
    ----------
    beta : float
        Velocity exponent.  Default 1.0 matches universal_sim_helpers.
    C_beta : float
        Pre-factor.  Default 1.0 matches universal_sim_helpers.
    """

    def __init__(self, beta: float = 1.0, C_beta: float = 1.0) -> None:
        self.beta = float(beta)
        self.C_beta = float(C_beta)

    def q(self, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
        v_rel_mag = np.linalg.norm(v_i - v_j, axis=1)   # (n,)
        return self.C_beta * v_rel_mag ** self.beta

    def dlog_q_dv_i(self, v_i: np.ndarray, v_j: np.ndarray) -> np.ndarray:
        """∂ log q / ∂v_i = beta * (v_i − v_j) / |v_i − v_j|²."""
        diff = v_i - v_j                                     # (n, 2)
        mag_sq = np.einsum("ij,ij->i", diff, diff)          # (n,)
        # Guard against zero relative velocity
        safe_mag_sq = np.where(mag_sq > 1e-30, mag_sq, 1.0)
        grad = self.beta * diff / safe_mag_sq[:, np.newaxis]
        # Zero out where relative velocity is truly zero
        return np.where(mag_sq[:, np.newaxis] > 1e-30, grad, 0.0)

    def dlog_eps_minus_q_dv_i(
        self,
        v_i: np.ndarray,
        v_j: np.ndarray,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        """∂ log(ε−q) / ∂v_i = −(q/(ε−q)) · ∂ log q / ∂v_i."""
        q_vals = self.q(v_i, v_j)                            # (n,)
        eps_minus_q = epsilon - q_vals                       # (n,)
        # Guard against ε ≈ q (denominator near zero)
        safe_denom = np.where(np.abs(eps_minus_q) > 1e-30, eps_minus_q, 1e-30)
        ratio = q_vals / safe_denom                          # (n,)
        dlog_q = self.dlog_q_dv_i(v_i, v_j)                 # (n, 2)
        result = -ratio[:, np.newaxis] * dlog_q
        # Zero out where denominator was unsafe
        valid = np.abs(eps_minus_q) > 1e-30
        return np.where(valid[:, np.newaxis], result, 0.0)

    @property
    def is_angle_independent(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"VHSKernel(beta={self.beta}, C_beta={self.C_beta})"
