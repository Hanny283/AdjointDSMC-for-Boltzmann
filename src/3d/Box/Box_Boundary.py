import numpy as np
import sys
import os

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_3d_dir = os.path.dirname(_current_dir)
_src_dir = os.path.dirname(_3d_dir)

# Add directories to path
if _3d_dir not in sys.path:
    sys.path.insert(0, _3d_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import from same directory (Box/)
import Box_bc as gbc

# Import from 3d directory
import general_helpers_3d as gh3

# Import from src directory
import universal_sim_helpers as uh


def Nanbu_Babovsky_3D_Periodic(N, dt, n_tot, e , mu, alpha, Lx, Ly, Lz, ncx, ncy, ncz, S, dx, T_x0=1.0, T_y0=1.0, T_z0=1.0, bc="pc"):
    positions = gh3.assign_positions_3d(N, Lx, Ly, Lz)
    velocities = gh3.sample_velocities_from_maxwellian_3d(T_x0, T_y0, T_z0, N)

    #spatial discretization
    cell_width = Lx / ncx
    cell_height = Ly / ncy
    cell_depth = Lz / ncz

    ix = np.floor(positions[:,0] / cell_width).astype(int)
    iy = np.floor(positions[:,1] / cell_height).astype(int)
    iz = np.floor(positions[:,2] / cell_depth).astype(int)

    # assign particles to cells
    particle_cell_indices = np.column_stack((ix, iy, iz))

    particle_cell_indices[particle_cell_indices[:,0] == ncx, 0] = ncx - 1
    particle_cell_indices[particle_cell_indices[:,1] == ncy, 1] = ncy - 1
    particle_cell_indices[particle_cell_indices[:,2] == ncz, 2] = ncz - 1

    #count number of particles in each cell 
    particles_per_cell = np.zeros((ncx, ncy, ncz), dtype=int)
    np.add.at(particles_per_cell, (ix, iy, iz), 1)

    cell_velocities = [
    velocities[(particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j) & (particle_cell_indices[:,2] == k)]
    for i in range(ncx) for j in range(ncy) for k in range(ncz)
    ]

    cell_velocities = np.array(cell_velocities, dtype=object).reshape(ncx, ncy, ncz)

    cglobal_indices = [
    np.flatnonzero((particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j) & (particle_cell_indices[:,2] == k))
    for i in range(ncx) for j in range(ncy) for k in range(ncz)
    ]

    cglobal_indices = np.array(cglobal_indices, dtype=object).reshape(ncx, ncy, ncz)

    temperature_history = np.zeros(n_tot)

    for n in range(n_tot):

        #compute upper bound cross section for each cell 

        upper_bound_cross_sections = np.array([
            uh.compute_upper_bound_cross_section(cell) if len(cell) else 0.0
            for cell in cell_velocities.ravel()
            ]).reshape(ncx, ncy, ncz)

        #physical number density per cell
        rho_cell = particles_per_cell / (cell_width * cell_height * cell_depth)

        # Expected pairs to collide per cell

        Nc = np.minimum(
            uh.Iround((particles_per_cell * rho_cell * upper_bound_cross_sections * dt)/ (2*e)), 
            particles_per_cell // 2
            )

        #sample and pair indices per cell 

        sampled_indices = uh.sample_particle_indices_to_collide_grid(Nc, cell_velocities) 

        indices_i_global = []
        indices_j_global = []

        for cx in range(ncx):
            for cy in range(ncy):
                for cz in range(ncz):
                    idx_local = sampled_indices[cx,cy,cz]

                    if len(idx_local) == 0 :
                        continue

                    pairs_loc = gh3.pair_particle_indices_3d(idx_local)

                    indices_particles_i = pairs_loc[:,0]
                    indices_particles_j = pairs_loc[:,1]

                    idx_map = cglobal_indices[cx,cy,cz]
                    indices_i_global.append(idx_map[indices_particles_i])
                    indices_j_global.append(idx_map[indices_particles_j])

        indices_ij = (np.concatenate(indices_i_global) if indices_i_global else np.array([], dtype=int)).ravel()
        indices_kl = (np.concatenate(indices_j_global) if indices_j_global else np.array([], dtype=int)).ravel()

        if len(indices_ij) > 0 and len(indices_kl) > 0:
            indices_ij = np.asarray(indices_ij).reshape(-1)
            indices_kl = np.asarray(indices_kl).reshape(-1)

            v_rel = velocities[indices_ij] - velocities[indices_kl]
            v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)

            sigma_ijkl = uh.ArraySigma_VHS(v_rel_mag).reshape(-1)

            Sigma_pairs_of_pairs = upper_bound_cross_sections[particle_cell_indices[indices_ij][:, 0], particle_cell_indices[indices_ij][:, 1], particle_cell_indices[indices_ij][:, 2]]

            u_rand = np.random.rand(len(indices_ij)) * Sigma_pairs_of_pairs

            accept_condition  = (u_rand < sigma_ijkl)

            indices_ij = indices_ij[accept_condition]
            indices_kl = indices_kl[accept_condition]

            if len(indices_ij) > 0:
                velocities = gh3.collide_particles_3d(velocities, indices_ij, indices_kl)

            positions = gh3.update_positions_3d(positions, velocities, dt)

            # Apply boundary conditions based on bc parameter
            if bc == "pc":  # periodic boundary condition
                positions = gbc.periodic_BC_3d(positions, Lx, Ly, Lz)
            elif bc == "rf":  # reflecting boundary condition
                velocities, positions = gbc.reflecting_BC_3d(velocities, positions, Lx, Ly, Lz)
            elif bc == "mx":  # Maxwell boundary condition
                velocities, positions = gbc.maxwell_bc_3d(positions, velocities, Lx, Ly, Lz, alpha, T_x0, T_y0, T_z0)
            else:
                raise ValueError(f"Invalid boundary condition '{bc}'. Must be 'pc', 'rf', or 'mx'.")

            #Re-bin particles

            particle_cell_indices[particle_cell_indices[:,0] == ncx, 0] = ncx - 1
            particle_cell_indices[particle_cell_indices[:,1] == ncy, 1] = ncy - 1
            particle_cell_indices[particle_cell_indices[:,2] == ncz, 2] = ncz - 1

            #count number of particles in each cell 
            particles_per_cell = np.zeros((ncx, ncy, ncz), dtype=int)
            np.add.at(particles_per_cell, (ix, iy, iz), 1)

            cell_velocities = [
            velocities[(particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j) & (particle_cell_indices[:,2] == k)]
            for i in range(ncx) for j in range(ncy) for k in range(ncz)
            ]
            cell_velocities = np.array(cell_velocities, dtype=object).reshape(ncx, ncy, ncz)

            cglobal_indices = [
            np.flatnonzero((particle_cell_indices[:,0] == i) & (particle_cell_indices[:,1] == j) & (particle_cell_indices[:,2] == k))
            for i in range(ncx) for j in range(ncy) for k in range(ncz)
            ]

            cglobal_indices = np.array(cglobal_indices, dtype=object).reshape(ncx, ncy, ncz)

        temperature_history[n]  = np.sum(velocities**2) / velocities.shape[0]

    return positions, velocities, temperature_history
    
    
    
    




