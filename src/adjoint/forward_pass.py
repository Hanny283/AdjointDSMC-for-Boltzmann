"""
Forward pass wrapper for adjoint DSMC simulation.

Runs the Nanbu-Babovsky DSMC simulation for M time steps, recording every
collision and boundary-reflection event in a structured history object.
The history object exposes a ``backward_pass`` method that walks the trace
in reverse to produce the velocity adjoints β and position adjoints α.

Data model
----------
At each step k the simulation does two sub-steps in order:

    1. Collision sub-step  (one Nanbu-Babovsky pair per collision record)
    2. Free-flight + boundary sub-step  (per particle)

Records mirror this order; the backward pass reverses it.

Classes
-------
CollisionRecord
    Stores indices, pre/post velocities, and the random unit vector ω for
    one collision pair at one step.

BoundaryRecord
    Stores the particle index, pre/post positions and velocities, the
    intersection angle θ_inter (or None if the particle stayed inside),
    and a boolean indicating whether a reflection occurred.

StepRecord
    Groups all CollisionRecord and BoundaryRecord objects for one step k,
    plus a snapshot of positions/velocities at the start and end of the step.

SimulationHistory
    Immutable container holding the full forward trace.  Provides
    ``backward_pass(terminal_beta_fn, terminal_alpha_fn)`` which returns
    β and α arrays of shape (M+1, N, 2).

ForwardSimulation
    Driver class.  ``run(positions, velocities, n_steps)`` executes the
    forward simulation and returns a SimulationHistory.

References
----------
Propositions 2.1 and 2.2 of the paper; Lemmas 2.2, 2.3, 2.9.
"""

from __future__ import annotations

import os
import sys

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — give access to src/ and src/2d/Arbitrary Shape/ so we can
# reuse the same cell, mesh, and rebinning infrastructure as the standalone
# arbitrary-shape DSMC simulation.
# ---------------------------------------------------------------------------
_here    = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_here)                            # …/src/
_arb_dir = os.path.join(_src_dir, "2d", "Arbitrary Shape")  # …/src/2d/Arbitrary Shape/
for _p in (_src_dir, _arb_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygmsh                                   # mesh generation
from scipy.spatial import cKDTree               # fast nearest-centroid lookup
import cell_class as _ct                        # cell_triangle
import universal_sim_helpers as _uh             # ArraySigma_VHS, Iround
import arbitrary_helpers as _ah                 # mesh build, rebinning helpers

from .boundary_geometry import (
    radius_r,
    normal_n,
    solve_theta_inter_batch,
    compute_c_inter,
    _radius_r_vec,
)
from .adjoint_jacobians import (
    dv_reflected_dv,
    compute_N_ki,
    collision_jacobian_transpose,
    apply_proposition_22,
)


# ---------------------------------------------------------------------------
# Small scalar helpers
# ---------------------------------------------------------------------------

def _iround(x: float, rng: np.random.Generator) -> int:
    """Probabilistic rounding: floor(x) + 1 with probability frac(x)."""
    lo   = int(np.floor(x))
    frac = x - lo
    return lo + int(rng.random() < frac)


# ---------------------------------------------------------------------------
# Helper: check whether a point is inside the star-shaped domain
# ---------------------------------------------------------------------------

def _is_inside(x: np.ndarray, C) -> bool:
    """Return True if point x lies strictly inside the Fourier boundary."""
    r_x = np.linalg.norm(x)
    if r_x == 0.0:
        return True
    theta = np.arctan2(x[1], x[0]) / (2.0 * np.pi) % 1.0
    return r_x < radius_r(theta, C)


def _is_inside_batch(x: np.ndarray, C) -> np.ndarray:
    """
    Vectorized inside check for N points.

    Parameters
    ----------
    x : ndarray, shape (N, 2)
    C : array-like, shape (2N+1,)

    Returns
    -------
    ndarray of bool, shape (N,)
    """
    r_x = np.linalg.norm(x, axis=1)                              # (N,)
    thetas = (np.arctan2(x[:, 1], x[:, 0]) / (2.0 * np.pi)) % 1.0  # (N,)
    r_boundary = _radius_r_vec(thetas, C)                         # (N,)
    return r_x < r_boundary


# ---------------------------------------------------------------------------
# Helper: specular reflection
# ---------------------------------------------------------------------------

def _reflect(x_prime: np.ndarray, v_prime: np.ndarray, theta_inter: float, C):
    """
    Apply the specular reflection map at the boundary.

    Returns
    -------
    x_tilde : ndarray, shape (2,)
    v_tilde : ndarray, shape (2,)
    """
    n = normal_n(theta_inter, C)
    c = compute_c_inter(theta_inter, C)

    # ṽ = v' − 2⟨n, v'⟩ n
    v_tilde = v_prime - 2.0 * np.dot(n, v_prime) * n

    # x̃ = x' − 2(⟨n, x'⟩ − c) n
    x_tilde = x_prime - 2.0 * (np.dot(n, x_prime) - c) * n

    return x_tilde, v_tilde


# ---------------------------------------------------------------------------
# Record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CollisionRecord:
    """
    One Nanbu-Babovsky collision event.

    Attributes
    ----------
    idx_i, idx_i1 : int
        Indices of the two colliding particles (i and i').
    v_i, v_i1 : ndarray, shape (2,)
        Pre-collision velocities.
    v_prime_i, v_prime_i1 : ndarray, shape (2,)
        Post-collision velocities.
    omega : ndarray, shape (2,)
        Random unit vector ω drawn during the collision.
    """
    idx_i:      int
    idx_i1:     int
    v_i:        np.ndarray
    v_i1:       np.ndarray
    v_prime_i:  np.ndarray
    v_prime_i1: np.ndarray
    omega:      np.ndarray


@dataclass
class BoundaryRecord:
    """
    One free-flight + (optional) boundary-reflection event for a particle.

    Attributes
    ----------
    idx : int
        Particle index.
    x_k : ndarray, shape (2,)
        Position at the start of the free-flight (before moving).
    v_prime : ndarray, shape (2,)
        Velocity used for free-flight (post-collision velocity).
    x_prime : ndarray, shape (2,)
        Position after free-flight (x_k + dt * v_prime), before reflection.
    in_domain : bool
        True if x_prime is inside the domain (no reflection needed).
    theta_inter : float or None
        Intersection angle θ_inter; None when in_domain is True.
    x_tilde : ndarray, shape (2,)
        Final position (= x_prime if in_domain, reflected otherwise).
    v_tilde : ndarray, shape (2,)
        Final velocity (= v_prime if in_domain, reflected otherwise).
    """
    idx:         int
    x_k:         np.ndarray
    v_prime:     np.ndarray
    x_prime:     np.ndarray
    in_domain:   bool
    theta_inter: Optional[float]
    x_tilde:     np.ndarray
    v_tilde:     np.ndarray


@dataclass
class StepRecord:
    """
    All events that occurred during step k.

    Attributes
    ----------
    k : int
        Time step index (0-based).
    positions_start : ndarray, shape (N, 2)
        Particle positions at the beginning of step k.
    velocities_start : ndarray, shape (N, 2)
        Particle velocities at the beginning of step k.
    collisions : list of CollisionRecord
        Collision events (may be empty if n_coll_pairs == 0).
    boundary : list of BoundaryRecord
        One record per particle for the free-flight / reflection sub-step.
    positions_end : ndarray, shape (N, 2)
        Particle positions at the end of step k.
    velocities_end : ndarray, shape (N, 2)
        Particle velocities at the end of step k.
    """
    k:                int
    positions_start:  np.ndarray
    velocities_start: np.ndarray
    collisions:       List[CollisionRecord] = field(default_factory=list)
    boundary:         List[BoundaryRecord]  = field(default_factory=list)
    positions_end:    np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    velocities_end:   np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


# ---------------------------------------------------------------------------
# SimulationHistory
# ---------------------------------------------------------------------------

@dataclass
class SimulationHistory:
    """
    Immutable record of a complete forward simulation.

    Attributes
    ----------
    C : ndarray, shape (2N+1,)
        Fourier boundary coefficients used during the run.
    dt : float
        Time step size.
    steps : list of StepRecord
        One entry per time step k = 0, …, M-1.

    Properties
    ----------
    n_steps : int         (= M)
    n_particles : int     (= N)
    final_positions : ndarray, shape (N, 2)
    final_velocities : ndarray, shape (N, 2)

    Methods
    -------
    backward_pass(terminal_beta_fn, terminal_alpha_fn) → (betas, alphas)
        Walk the trace backwards to compute adjoints.
    """
    C:     np.ndarray
    dt:    float
    steps: List[StepRecord]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_particles(self) -> int:
        if not self.steps:
            return 0
        return self.steps[0].positions_start.shape[0]

    @property
    def final_positions(self) -> np.ndarray:
        if not self.steps:
            raise ValueError("No steps recorded.")
        return self.steps[-1].positions_end

    @property
    def final_velocities(self) -> np.ndarray:
        if not self.steps:
            raise ValueError("No steps recorded.")
        return self.steps[-1].velocities_end

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward_pass(
        self,
        terminal_beta_fn:  Callable[[np.ndarray, np.ndarray], np.ndarray],
        terminal_alpha_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        """
        Compute velocity adjoints β and position adjoints α by walking
        the recorded forward trace in reverse.

        Parameters
        ----------
        terminal_beta_fn : callable (velocities, positions) → ndarray (N, 2)
            Terminal condition β_M = ∂L/∂v_M per particle.
        terminal_alpha_fn : callable (velocities, positions) → ndarray (N, 2)
            Terminal condition α_M = ∂L/∂x_M per particle.

        Returns
        -------
        betas : ndarray, shape (M+1, N, 2)
            β_{k, i, :} for k = 0, …, M.
        alphas : ndarray, shape (M+1, N, 2)
            α_{k, i, :} for k = 0, …, M.

        Notes
        -----
        The backward recurrence at step k (running from M-1 down to 0):

        α backward (Proposition 2.2, per particle, no coupling):
            α_{k,i} = G_{k,i}^T α_{k+1,i}   if boundary hit at step k
            α_{k,i} = α_{k+1,i}              otherwise

        β backward (Proposition 2.1, two sub-steps reversed):

          Sub-step A — boundary / free-flight (applied before collision
          in the backward direction):
            For each particle i:
              if in_domain:   rhs_i = β_{k+1,i} + dt * α_{k+1,i}
              if boundary hit: rhs_i = M_{k,i}^T β_{k+1,i}
                                       + N_{k,i}^T α_{k+1,i}

          Sub-step B — collision Jacobian^T (couples pair (i, i')):
            For each CollisionRecord with pre-collision v_i, v_i1, ω:
              J_T = J_coll(v_i, v_i1, ω)^T
              [β_{k,i}; β_{k,i1}] = J_T * [rhs_i; rhs_i1]
            Non-collided particles: β_{k,i} = rhs_i
        """
        M = self.n_steps
        N = self.n_particles
        C = self.C
        dt = self.dt

        # Allocate output arrays
        betas  = np.zeros((M + 1, N, 2))
        alphas = np.zeros((M + 1, N, 2))

        # Terminal conditions at step M
        x_M = self.final_positions
        v_M = self.final_velocities
        betas[M]  = np.asarray(terminal_beta_fn(v_M, x_M),  dtype=float)
        alphas[M] = np.asarray(terminal_alpha_fn(v_M, x_M), dtype=float)

        # Walk backwards k = M-1, …, 0
        for k in reversed(range(M)):
            step = self.steps[k]
            beta_k1  = betas[k + 1]   # shape (N, 2)
            alpha_k1 = alphas[k + 1]  # shape (N, 2)

            # Build a lookup from particle index to its BoundaryRecord
            bdry_by_idx: dict[int, BoundaryRecord] = {
                rec.idx: rec for rec in step.boundary
            }

            # ----------------------------------------------------------
            # α backward step (Proposition 2.2): per-particle, independent
            # ----------------------------------------------------------
            alpha_k = np.empty((N, 2))
            for i in range(N):
                rec = bdry_by_idx.get(i)
                if rec is None:
                    alpha_k[i] = alpha_k1[i]
                else:
                    alpha_k[i] = apply_proposition_22(
                        alpha_k1=alpha_k1[i],
                        theta_inter=rec.theta_inter,
                        x_prime=rec.x_prime,
                        v_k=rec.v_prime,
                        C=C,
                        in_domain=rec.in_domain,
                    )
            alphas[k] = alpha_k

            # ----------------------------------------------------------
            # β backward step — Sub-step A:
            # Apply boundary / free-flight Jacobians to get rhs per particle
            # ----------------------------------------------------------
            rhs = np.empty((N, 2))
            for i in range(N):
                rec = bdry_by_idx.get(i)
                if rec is None or rec.in_domain:
                    # No boundary hit: identity velocity map, free-flight α coupling
                    rhs[i] = beta_k1[i] + dt * alpha_k1[i]
                else:
                    # Boundary hit: M^T β + N^T α
                    Mmat = dv_reflected_dv(rec.theta_inter, rec.x_k,
                                          rec.v_prime, C)
                    Nmat = compute_N_ki(rec.theta_inter, rec.x_prime,
                                       rec.x_k, rec.v_prime, C, dt)
                    rhs[i] = Mmat.T @ beta_k1[i] + Nmat.T @ alpha_k1[i]

            # ----------------------------------------------------------
            # β backward step — Sub-step B:
            # Propagate rhs through J_coll^T for each collision pair.
            # rhs already incorporates the boundary / free-flight Jacobians;
            # we only need to apply J_coll^T (v_i, v_i1, ω from forward).
            # ----------------------------------------------------------
            beta_k = rhs.copy()  # non-collided particles keep their rhs

            for crec in step.collisions:
                i  = crec.idx_i
                i1 = crec.idx_i1
                J_T = collision_jacobian_transpose(crec.v_i, crec.v_i1, crec.omega)
                stacked = J_T @ np.concatenate([rhs[i], rhs[i1]])   # shape (4,)
                beta_k[i]  = stacked[:2]
                beta_k[i1] = stacked[2:]

            betas[k] = beta_k

        return betas, alphas


# ---------------------------------------------------------------------------
# ForwardSimulation
# ---------------------------------------------------------------------------

class ForwardSimulation:
    """
    Driver for the forward Nanbu-Babovsky DSMC simulation.

    Uses a pygmsh triangle mesh (matching the arbitrary-shape standalone DSMC)
    so that collision cells conform to the Fourier boundary and only exist
    inside the simulation domain.

    Parameters
    ----------
    C : array-like, shape (2N+1,)
        Fourier boundary coefficients.
    dt : float
        Time step size.
    n_coll_pairs : int or None
        Kept for API compatibility but no longer used.  The collision rate is
        now determined per-cell by Bird's formula (N²·σ·dt / 2·e·A_cell).
    seed : int or None
        Seed for the internal NumPy random generator.
    e : float
        Bird's-formula parameter — effective number of real particles per
        simulated particle.  Default 1.0.
    num_boundary_points : int
        Number of sample points around the Fourier boundary used to build
        the pygmsh mesh.  Default 80.
    mesh_size : float
        pygmsh characteristic edge length.  Larger → fewer, coarser cells.
        Default 0.3 gives ~3–6 particles per cell for N≈150.

    Methods
    -------
    run(positions, velocities, n_steps) → SimulationHistory
    """

    def __init__(
        self,
        C,
        dt: float,
        n_coll_pairs: Optional[int] = None,   # kept for API compat, unused
        seed: Optional[int] = None,
        e: float = 1.0,
        num_boundary_points: int = 80,
        mesh_size: float = 0.3,
    ):
        self.C  = np.asarray(C, dtype=float)
        self.dt = float(dt)
        self._e = float(e)
        self._rng = np.random.default_rng(seed)

        # Build pygmsh triangle mesh from the Fourier boundary.
        # This matches the pattern in arbitrary_parameterized.py:
        #   boundary_pts = hf.sample_star_shape(C, num_boundary_points)
        #   mesh = hf.create_arbitrary_shape_mesh_2d(N, boundary_pts, mesh_size)
        #   cell_list, edge_to_cells = hf.create_cell_list_and_adjacency_lists(mesh)
        boundary_pts = _ah.sample_star_shape(self.C, num_boundary_points)
        mesh = _ah.create_arbitrary_shape_mesh_2d(0, boundary_pts, mesh_size=mesh_size)
        self._cell_list, self._edge_to_cells = \
            _ah.create_cell_list_and_adjacency_lists(mesh)

        # Pre-build a KD-tree on cell centroids for fast nearest-cell lookup
        # during rebinning (same role as find_nearest_centroid_cell_kdtree).
        centroids = np.array([c.center for c in self._cell_list])
        self._centroid_kdtree = cKDTree(centroids)
        # Reverse map: cell object → integer index (used for cell_assignment array)
        self._cell_index = {id(c): i for i, c in enumerate(self._cell_list)}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        positions:  np.ndarray,
        velocities: np.ndarray,
        n_steps:    int,
    ) -> SimulationHistory:
        """
        Run the simulation for ``n_steps`` steps.

        Parameters
        ----------
        positions : ndarray, shape (N, 2)
        velocities : ndarray, shape (N, 2)
        n_steps : int

        Returns
        -------
        SimulationHistory
        """
        x = np.asarray(positions,  dtype=float).copy()
        v = np.asarray(velocities, dtype=float).copy()

        # Initial cell assignment — same as the initial particle-to-cell
        # assignment in arbitrary_parameterized.py (KD-tree + triangle-following).
        cell_asgn = self._assign_cells(x)   # int array, shape (N,)

        steps = []

        for k in range(n_steps):
            x_start = x.copy()
            v_start = v.copy()

            # -- Collision sub-step ------------------------------------
            # Uses current cell_asgn (positions before free-flight).
            # Matches the per-cell collision loop in arbitrary_parameterized.py.
            collision_records = self._collision_step(v, cell_asgn)

            # -- Free-flight + boundary sub-step ----------------------
            boundary_records = self._boundary_step(x, v)

            # -- Rebin -------------------------------------------------
            # After free-flight some particles may have crossed cell boundaries.
            # Detect them and reassign using KD-tree + find_containing_cell,
            # matching Step 6 in arbitrary_parameterized.py.
            cell_asgn = self._rebin(x, cell_asgn)

            steps.append(StepRecord(
                k=k,
                positions_start=x_start,
                velocities_start=v_start,
                collisions=collision_records,
                boundary=boundary_records,
                positions_end=x.copy(),
                velocities_end=v.copy(),
            ))

        return SimulationHistory(C=self.C.copy(), dt=self.dt, steps=steps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_cells(self, x: np.ndarray) -> np.ndarray:
        """
        Assign every particle to a triangle cell using KD-tree + triangle-
        following, matching find_nearest_centroid_cell_kdtree +
        find_containing_cell in arbitrary_helpers.py.

        Returns
        -------
        cell_asgn : ndarray of int, shape (N,)
            Index into self._cell_list for each particle.
        """
        N = x.shape[0]
        _, nearest_idx = self._centroid_kdtree.query(x)   # shape (N,)
        cell_asgn = np.empty(N, dtype=int)
        for i in range(N):
            start = self._cell_list[nearest_idx[i]]
            cell  = _ah.find_containing_cell(x[i], start, self._edge_to_cells)
            cell_asgn[i] = self._cell_index[id(cell)]
        return cell_asgn

    def _rebin(self, x: np.ndarray, cell_asgn: np.ndarray) -> np.ndarray:
        """
        Rebin particles that have left their assigned cell after free-flight.

        Matches Step 6 of arbitrary_parameterized.py:
          - detect particles violating cell.is_inside
          - find correct cell via KD-tree + find_containing_cell
          - update assignment array (no particle data moved — global arrays
            remain the source of truth for the adjoint)

        Returns updated cell_asgn (same array, mutated in place, also returned
        for clarity).
        """
        cell_list = self._cell_list
        for i, (pos, cid) in enumerate(zip(x, cell_asgn)):
            if not cell_list[cid].is_inside(pos[0], pos[1]):
                # Particle has left its cell — find the correct one
                _, ni = self._centroid_kdtree.query(pos.reshape(1, 2))
                start = cell_list[int(ni[0])]
                cell  = _ah.find_containing_cell(pos, start, self._edge_to_cells)
                cell_asgn[i] = self._cell_index[id(cell)]
        return cell_asgn

    def _collision_step(
        self, v: np.ndarray, cell_asgn: np.ndarray
    ) -> List[CollisionRecord]:
        """
        Cell-based Nanbu-Babovsky collision step using triangle mesh cells.

        Matches the per-cell collision loop in arbitrary_parameterized.py:
          for cell in cell_list:
              num_collisions = cell.num_collisions(dt, e)   ← Bird's formula
              ... select pairs (permutation, split) ...
              ... VHS acceptance-rejection ...
              ... collide_and_update_particles ...

        Differences from the standalone code:
        - Particles live in global arrays v[N,2]; cell_asgn[N] is the index
          mapping.  This is necessary so CollisionRecord stores global indices
          that the backward pass can look up.
        - Free-flight is handled separately (_boundary_step), not inside
          collide_and_update_particles.
        - rho_cell is computed dynamically as n_cell / cell.area() because
          cell.rho_cell is set to 0 at __init__ and never updated when
          particles are added (would always return 0 collision pairs otherwise).
        - self._rng is used instead of np.random for reproducibility.

        Mutates ``v`` in-place and returns CollisionRecord list.
        """
        N = v.shape[0]
        if N < 2:
            return []

        records: List[CollisionRecord] = []

        for c_idx, cell in enumerate(self._cell_list):
            # Global indices of particles assigned to this cell
            mask = np.nonzero(cell_asgn == c_idx)[0]
            n_cell = len(mask)
            if n_cell < 2:
                continue

            v_cell = v[mask]   # shape (n_cell, 2)

            # ub_sigma: upper bound on the VHS cross-section for this cell.
            # Same formula as cell_triangle.upper_bound_cross_section():
            #   2 * max(|v_i - v_mean|)
            v_mean = v_cell.mean(axis=0)
            delta_v = np.linalg.norm(v_cell - v_mean, axis=1).max()
            ub_sigma = 2.0 * delta_v
            if ub_sigma == 0.0:
                continue

            # Bird's formula — same as cell_triangle.num_collisions(dt, e).
            # rho_cell computed dynamically (cell.rho_cell is stale at 0).
            rho_cell = n_cell / cell.area()
            expected = (n_cell * rho_cell * ub_sigma * self.dt) / (2.0 * self._e)
            n_select = min(_iround(expected, self._rng), n_cell)
            if n_select < 2:
                continue

            # Select n_select candidate particles (permutation, then split),
            # matching arbitrary_parameterized.py:
            #   perm = np.random.permutation(indices)[:num_collisions]
            #   indices_i = perm[:num_collisions // 2]
            #   indices_j = perm[num_collisions // 2:]
            local_perm = self._rng.permutation(n_cell)[:n_select]
            half       = n_select // 2
            i_local    = local_perm[:half]
            j_local    = local_perm[half : 2 * half]

            # VHS acceptance-rejection — same as arbitrary_parameterized.py:
            #   sigma_ij = ArraySigma_VHS(v_rel_mag)   (= v_rel_mag for alpha=1)
            #   u_rand   = rand() * ub_sigma
            #   accept   = u_rand < sigma_ij
            v_i_cand   = v[mask[i_local]]
            v_j_cand   = v[mask[j_local]]
            v_rel_mag  = np.linalg.norm(v_i_cand - v_j_cand, axis=1)
            sigma_ij   = _uh.ArraySigma_VHS(v_rel_mag).reshape(-1)
            u_rand     = self._rng.random(half) * ub_sigma
            accept     = u_rand < sigma_ij

            i_acc = i_local[accept]
            j_acc = j_local[accept]

            # Apply Nanbu-Babovsky rule to accepted pairs and record
            for li, lj in zip(i_acc, j_acc):
                idx_i  = int(mask[li])
                idx_j  = int(mask[lj])

                v_i  = v[idx_i].copy()
                v_j  = v[idx_j].copy()

                v_cm      = 0.5 * (v_i + v_j)
                v_rel_mag_pair = np.linalg.norm(v_i - v_j)

                angle  = self._rng.uniform(0.0, 2.0 * np.pi)
                omega  = np.array([np.cos(angle), np.sin(angle)])

                v_prime_i = v_cm + 0.5 * v_rel_mag_pair * omega
                v_prime_j = v_cm - 0.5 * v_rel_mag_pair * omega

                v[idx_i] = v_prime_i
                v[idx_j] = v_prime_j

                records.append(CollisionRecord(
                    idx_i=idx_i,
                    idx_i1=idx_j,
                    v_i=v_i,
                    v_i1=v_j,
                    v_prime_i=v_prime_i.copy(),
                    v_prime_i1=v_prime_j.copy(),
                    omega=omega.copy(),
                ))

        return records

    def _boundary_step(
        self, x: np.ndarray, v: np.ndarray
    ) -> List[BoundaryRecord]:
        """
        Advance each particle by ``dt``, apply specular reflection if needed,
        mutate ``x`` and ``v`` in-place, and return BoundaryRecords.

        Vectorization strategy
        ----------------------
        1. Free-flight:       x_prime = x + dt·v                (one broadcast)
        2. Inside check:      _is_inside_batch(x_prime, C)      (one vectorized pass)
        3. θ solve:           solve_theta_inter_batch for all    (K×n_grid broadcast
                              outside particles at once           + K simultaneous bisections)
        4. Reflection + records: per-particle loop (unavoidable due to branching,
                              but is now small — only touches outside particles for
                              the theta solve).
        """
        C  = self.C
        dt = self.dt
        N  = x.shape[0]

        # Step 1: vectorized free-flight
        x_start   = x.copy()                           # pre-flight snapshot for records
        x_prime_all = x_start + dt * v                 # (N, 2)

        # Step 2: vectorized inside check
        inside_all = _is_inside_batch(x_prime_all, C)  # (N,) bool

        # Step 3: batch θ solve for all outside particles
        outside_mask = ~inside_all                      # (N,) bool
        outside_idx  = np.nonzero(outside_mask)[0]     # (K,)

        theta_inter_arr = np.full(N, np.nan)           # (N,) NaN = no crossing
        if outside_idx.size > 0:
            theta_inter_arr[outside_idx] = solve_theta_inter_batch(
                x_start[outside_idx],
                v[outside_idx],
                C,
            )

        # Step 4: apply reflections and build records
        records: List[BoundaryRecord] = []
        for i in range(N):
            x_k     = x_start[i]
            v_prime = v[i].copy()
            x_prime = x_prime_all[i]

            if inside_all[i]:
                x[i] = x_prime
                records.append(BoundaryRecord(
                    idx=i, x_k=x_k, v_prime=v_prime, x_prime=x_prime,
                    in_domain=True, theta_inter=None,
                    x_tilde=x_prime, v_tilde=v_prime,
                ))
            else:
                th = theta_inter_arr[i]
                if np.isnan(th):
                    # No intersection found — keep particle at x_k (rare edge case)
                    x[i] = x_k
                    records.append(BoundaryRecord(
                        idx=i, x_k=x_k, v_prime=v_prime, x_prime=x_prime,
                        in_domain=True, theta_inter=None,
                        x_tilde=x_k, v_tilde=v_prime,
                    ))
                else:
                    theta_inter = float(th)
                    x_tilde, v_tilde = _reflect(x_prime, v_prime, theta_inter, C)
                    # Safety clamp: nearly-tangential trajectories
                    if not _is_inside(x_tilde, C):
                        r_b = radius_r(theta_inter, C)
                        x_tilde = (r_b * 0.9999) * np.array([
                            np.cos(2 * np.pi * theta_inter),
                            np.sin(2 * np.pi * theta_inter),
                        ])
                    x[i] = x_tilde
                    v[i] = v_tilde
                    records.append(BoundaryRecord(
                        idx=i, x_k=x_k, v_prime=v_prime, x_prime=x_prime,
                        in_domain=False, theta_inter=theta_inter,
                        x_tilde=x_tilde, v_tilde=v_tilde,
                    ))

        return records
