"""
Forward DSMC pass with full history recording.

This module replicates the time loop of ``Arbitrary_Shape_Parameterized``
step-by-step, adding:

  * A global particle-ID (GID) array so that every particle can be tracked
    through cell reassignments (rebinning).
  * Per-step CollisionStepRecord capturing ALL virtual pairs (both accepted
    and rejected) with their pre-collision velocities, σ, α, q, ε.
  * Per-step BCStepRecord capturing hit particles' normals, boundary angles,
    and pre-reflection velocities.
  * Velocity/position snapshots after every step.
  * phi_final computed from caller-supplied phi_fn after the last step.

Design notes
------------
- Global arrays ``global_pos[N,2]`` and ``global_vel[N,2]`` are indexed by
  GID (0..N-1) throughout.  Cell storage (cell.particle_positions etc.) is
  NOT used; only cell geometry methods (is_inside, area, center) are called.
- The position update (pos += vel*dt) is performed for ALL particles at each
  step, independent of whether that particle collided.  This fixes the bug in
  the original code where positions were only updated inside
  collide_and_update_particles (and therefore never updated when Nc=0).
- rho_cell is computed dynamically (n_in_cell / cell_area) so that collisions
  actually occur even after rebinning changes cell occupancy.
- epsilon (upper bound ε) is per-cell (and therefore stored per-pair in the
  record), matching the actual rejection-sampling implementation.
"""

from __future__ import annotations

import sys
import os
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

# ------------------------------------------------------------------
# Path setup so we can import the existing project helpers without
# modifying them.
# ------------------------------------------------------------------
_this_dir   = os.path.dirname(os.path.abspath(__file__))   # src/adjoint_dsmc/
_src_dir    = os.path.dirname(_this_dir)                    # src/
_arb_dir    = os.path.join(_src_dir, "2d", "Arbitrary Shape")
_2d_dir     = os.path.join(_src_dir, "2d")

for _p in [_src_dir, _arb_dir, _2d_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arbitrary_helpers as hf
from edge_class import Edge
import cell_class as ct
import general_helpers as gh

from .records import CollisionStepRecord, BCStepRecord, ForwardHistory
from .kernels import CollisionKernel, VHSKernel


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _probabilistic_round(x: float) -> int:
    """Round x to the nearest integer with fractional-part probability."""
    lower = int(np.floor(x))
    return lower + (1 if np.random.rand() < (x - lower) else 0)


def _upper_bound_cross_section(velocities: np.ndarray) -> float:
    """ε = 2 * max |v_i − v_mean|  (same formula as cell_triangle)."""
    if len(velocities) == 0:
        return 0.0
    v_mean = velocities.mean(axis=0)
    return 2.0 * float(np.linalg.norm(velocities - v_mean, axis=1).max())


def _build_edge_to_cells(cell_list, tri_conn, pts2d):
    """Rebuild edge_to_cells dict keyed by Edge objects."""
    edge_to_cells: dict = {}
    for i, tri in enumerate(tri_conn):
        verts = pts2d[tri]
        edges = [
            Edge(verts[0], verts[1]),
            Edge(verts[1], verts[2]),
            Edge(verts[2], verts[0]),
        ]
        for edge in edges:
            edge_to_cells.setdefault(edge, []).append(cell_list[i])
    return edge_to_cells


def _find_containing_cell(position, start_cell, edge_to_cells):
    """Walk the mesh to find the cell containing position."""
    return hf.find_containing_cell(position, start_cell, edge_to_cells)


# ------------------------------------------------------------------
# Public function
# ------------------------------------------------------------------

def forward_pass_with_history(
    N: int,
    fourier_coefficients,
    num_boundary_points: int,
    T_x0: float,
    T_y0: float,
    dt: float,
    n_tot: int,
    e: float,
    phi_fn: Callable[[np.ndarray], np.ndarray],
    kernel: Optional[CollisionKernel] = None,
    mesh_size: float = 0.1,
    rho: float = 1.0,
    verbose: bool = False,
) -> ForwardHistory:
    """Run the forward DSMC simulation and return a complete ForwardHistory.

    Parameters
    ----------
    N : int
        Number of simulation particles.
    fourier_coefficients : array-like of shape (2M+1,)
        Fourier coefficients defining the star-shaped boundary.
    num_boundary_points : int
        Number of points sampled along the boundary polygon.
    T_x0, T_y0 : float
        Initial temperatures in x and y directions.
    dt : float
        Time step size.
    n_tot : int
        Total number of time steps (M in the papers).
    e : float
        Denominator factor in the collision-rate formula Nc = N*ρ*ε*dt/(2e).
    phi_fn : callable (N,2) → (N,)
        Objective function applied to each particle's velocity vector.
        Example: ``lambda v: np.sum(v**2, axis=1)``  for kinetic energy.
    kernel : CollisionKernel, optional
        Collision kernel object.  Defaults to VHSKernel(beta=1, C_beta=1)
        which matches ``ArraySigma_VHS`` in ``universal_sim_helpers``.
    mesh_size : float
        Triangle mesh characteristic length passed to ``pygmsh``.
    rho : float
        Physical density ρ.  Used only to compute rho_over_N = ρ/N.
        For a probability-density normalisation use rho=1.
    verbose : bool
        Print progress messages.

    Returns
    -------
    ForwardHistory
        Complete history object ready for backward_adjoint_pass().
    """
    if kernel is None:
        kernel = VHSKernel(beta=1.0, C_beta=1.0)

    fourier_coefficients = np.asarray(fourier_coefficients, dtype=float)

    # ------------------------------------------------------------------
    # 1. Build mesh and initialise particles
    # ------------------------------------------------------------------
    boundary_points = hf.sample_star_shape(fourier_coefficients, num_boundary_points)
    mesh = hf.create_arbitrary_shape_mesh_2d(N, boundary_points, mesh_size=mesh_size)

    init_positions = hf.assign_positions_arbitrary_2d(N, mesh)
    init_velocities = gh.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)

    # Build cell list and edge-to-cells using existing helpers
    cell_list, edge_to_cells = hf.create_cell_list_and_adjacency_lists(mesh)
    n_cells = len(cell_list)
    cell_to_idx = {id(c): i for i, c in enumerate(cell_list)}

    # Build KD-tree on cell centroids for fast nearest-cell lookup
    cell_centers = np.array([c.center for c in cell_list])
    cell_kdtree = cKDTree(cell_centers)

    # Cached boundary for BC step
    cached_boundary = hf.CachedBoundary(boundary_points)

    # ------------------------------------------------------------------
    # 2. Global arrays and GID tracking
    # ------------------------------------------------------------------
    # global_pos[gid] / global_vel[gid] — the master state arrays
    global_pos = init_positions.copy()    # (N, 2)
    global_vel = init_velocities.copy()   # (N, 2)

    # cell_gids[cell_idx] = 1-D int array of GIDs currently in that cell
    cell_gids: list[np.ndarray] = [np.empty(0, dtype=int) for _ in range(n_cells)]

    # Assign initial GIDs: place each particle in its containing cell
    unassigned: list[int] = []
    for gid in range(N):
        pos = global_pos[gid]
        assigned = False
        for c_idx, cell in enumerate(cell_list):
            if cell.is_inside(pos[0], pos[1]):
                cell_gids[c_idx] = np.append(cell_gids[c_idx], gid)
                assigned = True
                break
        if not assigned:
            unassigned.append(gid)

    # Fall back: nearest centroid for any particle not inside any cell
    if unassigned:
        u_pos = global_pos[unassigned]
        _, nn_idx = cell_kdtree.query(u_pos)
        for local_i, gid in enumerate(unassigned):
            c_idx = int(nn_idx[local_i])
            cell_gids[c_idx] = np.append(cell_gids[c_idx], gid)

    # ------------------------------------------------------------------
    # 3. Pre-allocate snapshot arrays
    # ------------------------------------------------------------------
    M = n_tot
    all_velocities = np.zeros((M + 1, N, 2))
    all_positions  = np.zeros((M + 1, N, 2))
    all_velocities[0] = global_vel.copy()
    all_positions[0]  = global_pos.copy()

    collision_history: list[CollisionStepRecord] = []
    bc_history:        list[BCStepRecord]        = []

    # ------------------------------------------------------------------
    # 4. Main time loop
    # ------------------------------------------------------------------
    for step in range(n_tot):

        # ---- 4a. Collision step per cell --------------------------------
        step_pair_i:   list[int]           = []
        step_pair_j:   list[int]           = []
        step_v_i_pre:  list[np.ndarray]    = []
        step_v_j_pre:  list[np.ndarray]    = []
        step_sigma:    list[np.ndarray]    = []
        step_alpha:    list[np.ndarray]    = []
        step_accepted: list[bool]          = []
        step_q:        list[float]         = []
        step_eps:      list[float]         = []

        for c_idx, cell in enumerate(cell_list):
            gids = cell_gids[c_idx]
            n_in = len(gids)
            if n_in < 2:
                continue

            vel_in_cell = global_vel[gids]   # (n_in, 2)
            epsilon = _upper_bound_cross_section(vel_in_cell)
            if epsilon < 1e-30:
                continue

            cell_area = cell.area()
            if cell_area < 1e-30:
                continue

            rho_cell_dyn = n_in / cell_area
            ub_sigma = epsilon   # ε already is the upper bound for q
            Nc_float = (n_in * rho_cell_dyn * ub_sigma * dt) / (2.0 * e) if e != 0 else 0.0
            Nc = int(min(_probabilistic_round(Nc_float), n_in // 2))
            if Nc == 0:
                continue

            # Select virtual collision pairs (without replacement)
            perm = np.random.permutation(n_in)[: 2 * Nc]
            local_i_all = perm[:Nc]
            local_j_all = perm[Nc:]

            gid_i_all = gids[local_i_all]   # (Nc,) global IDs
            gid_j_all = gids[local_j_all]

            v_i_all = global_vel[gid_i_all]  # (Nc, 2) pre-collision velocities
            v_j_all = global_vel[gid_j_all]

            # Relative velocity and kernel evaluation
            v_rel     = v_i_all - v_j_all                              # (Nc, 2)
            v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)   # (Nc, 1)
            v_rel_mag_1d = v_rel_mag.ravel()                           # (Nc,)

            q_vals = kernel.q(v_i_all, v_j_all)                        # (Nc,)

            # α = unit relative velocity
            safe_mag = np.where(v_rel_mag_1d > 1e-30, v_rel_mag_1d, 1.0)
            alpha_all = v_rel / safe_mag[:, np.newaxis]                 # (Nc, 2)

            # Rejection sampling: accept if ξ·ε < q
            xi = np.random.rand(Nc) * epsilon
            accept_mask = xi < q_vals                                   # (Nc,) bool

            # Sample σ for accepted pairs only
            theta_accepted = np.random.uniform(0.0, 2.0 * np.pi, size=accept_mask.sum())
            sigma_all = np.zeros((Nc, 2))
            sigma_all[accept_mask, 0] = np.cos(theta_accepted)
            sigma_all[accept_mask, 1] = np.sin(theta_accepted)

            # Apply real collisions (update global_vel for accepted pairs)
            if accept_mask.any():
                ai = gid_i_all[accept_mask]
                aj = gid_j_all[accept_mask]
                sig = sigma_all[accept_mask]           # (n_acc, 2)
                vi  = v_i_all[accept_mask]
                vj  = v_j_all[accept_mask]
                v_cm = 0.5 * (vi + vj)
                rel_m = v_rel_mag[accept_mask]         # (n_acc, 1)
                global_vel[ai] = v_cm + 0.5 * rel_m * sig
                global_vel[aj] = v_cm - 0.5 * rel_m * sig

            # Record ALL virtual pairs (both j=2 and j=3)
            eps_arr = np.full(Nc, epsilon)
            for p in range(Nc):
                step_pair_i.append(int(gid_i_all[p]))
                step_pair_j.append(int(gid_j_all[p]))
                step_v_i_pre.append(v_i_all[p])
                step_v_j_pre.append(v_j_all[p])
                step_sigma.append(sigma_all[p])
                step_alpha.append(alpha_all[p])
                step_accepted.append(bool(accept_mask[p]))
                step_q.append(float(q_vals[p]))
                step_eps.append(float(epsilon))

        # Build CollisionStepRecord (may have zero pairs)
        if step_pair_i:
            col_rec = CollisionStepRecord(
                pair_i   = np.array(step_pair_i,   dtype=int),
                pair_j   = np.array(step_pair_j,   dtype=int),
                v_i_pre  = np.array(step_v_i_pre,  dtype=float),
                v_j_pre  = np.array(step_v_j_pre,  dtype=float),
                sigma    = np.array(step_sigma,     dtype=float),
                alpha    = np.array(step_alpha,     dtype=float),
                accepted = np.array(step_accepted,  dtype=bool),
                q_values = np.array(step_q,         dtype=float),
                epsilon  = np.array(step_eps,       dtype=float),
            )
        else:
            col_rec = CollisionStepRecord(
                pair_i   = np.empty(0, dtype=int),
                pair_j   = np.empty(0, dtype=int),
                v_i_pre  = np.empty((0, 2), dtype=float),
                v_j_pre  = np.empty((0, 2), dtype=float),
                sigma    = np.empty((0, 2), dtype=float),
                alpha    = np.empty((0, 2), dtype=float),
                accepted = np.empty(0, dtype=bool),
                q_values = np.empty(0, dtype=float),
                epsilon  = np.empty(0, dtype=float),
            )
        collision_history.append(col_rec)

        # ---- 4b. Transport: advance ALL particles by dt -----------------
        global_pos += global_vel * dt

        # ---- 4c. Boundary-condition step --------------------------------
        inside_mask   = hf.points_in_polygon_vectorized(
            global_pos, cached_boundary.boundary_points
        )
        outside_mask  = ~inside_mask
        outside_gids  = np.where(outside_mask)[0]   # GIDs of particles outside

        if outside_gids.size > 0:
            out_pos = global_pos[outside_gids]    # (n_out, 2)
            out_vel = global_vel[outside_gids]    # (n_out, 2)  v_pre

            _, _, closest_pts, normals = cached_boundary.get_closest_edge_info(out_pos)

            # Specular reflection
            overshoot         = out_pos - closest_pts
            vdn               = np.sum(out_vel * normals,   axis=1, keepdims=True)
            odn               = np.sum(overshoot * normals, axis=1, keepdims=True)
            reflected_vel     = out_vel  - 2.0 * vdn * normals
            reflected_pos     = closest_pts + (overshoot - 2.0 * odn * normals)

            # Boundary angles for ∂n̂/∂c_m
            bnd_angles = np.arctan2(closest_pts[:, 1], closest_pts[:, 0])

            # Build BCStepRecord
            bc_rec = BCStepRecord(
                hit_indices     = outside_gids.copy(),
                normals         = normals.copy(),
                boundary_angles = bnd_angles,
                v_pre           = out_vel.copy(),    # velocity BEFORE reflection
            )

            # Write reflected state back
            global_pos[outside_gids] = reflected_pos
            global_vel[outside_gids] = reflected_vel
        else:
            bc_rec = BCStepRecord(
                hit_indices     = np.empty(0, dtype=int),
                normals         = np.empty((0, 2), dtype=float),
                boundary_angles = np.empty(0, dtype=float),
                v_pre           = np.empty((0, 2), dtype=float),
            )
        bc_history.append(bc_rec)

        # ---- 4d. Rebin: reassign particles to their correct cells --------
        # Collect particles that left their current cell
        particles_to_move: list[tuple[int, int]] = []   # (from_c_idx, gid)
        gids_to_remove: list[list[int]] = [[] for _ in range(n_cells)]

        for c_idx, cell in enumerate(cell_list):
            gids = cell_gids[c_idx]
            if len(gids) == 0:
                continue
            for local_idx, gid in enumerate(gids):
                pos = global_pos[gid]
                if not cell.is_inside(pos[0], pos[1]):
                    particles_to_move.append((c_idx, gid))
                    gids_to_remove[c_idx].append(local_idx)

        # Remove from their old cells
        for c_idx in range(n_cells):
            if gids_to_remove[c_idx]:
                cell_gids[c_idx] = np.delete(
                    cell_gids[c_idx], gids_to_remove[c_idx]
                )

        # Reassign to correct cells
        if particles_to_move:
            move_positions = np.array(
                [global_pos[gid] for _, gid in particles_to_move]
            )
            _, nn_idx = cell_kdtree.query(move_positions)
            for local_i, (_, gid) in enumerate(particles_to_move):
                start_cell = cell_list[int(nn_idx[local_i])]
                target_cell = hf.find_containing_cell(
                    global_pos[gid], start_cell, edge_to_cells
                )
                t_idx = cell_to_idx[id(target_cell)]
                cell_gids[t_idx] = np.append(cell_gids[t_idx], gid)

        # ---- 4e. Snapshot -----------------------------------------------
        all_velocities[step + 1] = global_vel.copy()
        all_positions[step + 1]  = global_pos.copy()

        if verbose and ((step + 1) % 10 == 0 or step + 1 == n_tot):
            n_hits = bc_rec.hit_indices.size
            n_cols = col_rec.pair_i.size
            print(f"  Step {step+1}/{n_tot}  BC hits={n_hits}  virtual pairs={n_cols}")

    # ------------------------------------------------------------------
    # 5. Compute phi_final from the last-step velocities
    # ------------------------------------------------------------------
    phi_final = phi_fn(all_velocities[M])   # (N,)

    rho_over_N = rho / N

    return ForwardHistory(
        collision            = collision_history,
        bc                   = bc_history,
        velocities           = all_velocities,
        positions            = all_positions,
        phi_final            = phi_final,
        fourier_coefficients = fourier_coefficients,
        rho_over_N           = rho_over_N,
    )
