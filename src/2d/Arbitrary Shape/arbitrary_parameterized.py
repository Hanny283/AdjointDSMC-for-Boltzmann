import pygmsh 
import numpy as np
import time
import sys
import os
from scipy.spatial import cKDTree

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_2d_dir = os.path.dirname(_current_dir)
_src_dir = os.path.dirname(_2d_dir)

# Add directories to path
if _2d_dir not in sys.path:
    sys.path.insert(0, _2d_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import from same directory (Arbitrary Shape/)
import arbitrary_helpers as hf
import arbitrary_bc as bc

# Import from 2d directory
import general_helpers as gh

# Import from src directory
import universal_sim_helpers as uh
import cell_class as ct
from edge_class import Edge


def Arbitrary_Shape_Parameterized(N, fourier_coefficients, num_boundary_points, T_x0, T_y0, dt, n_tot, e, mu, alpha, mesh_size=0.1, T_wall_x=None, T_wall_y=None, accommodation_coefficient=0.0):
    """
    Run DSMC simulation on arbitrary parameterized shape.
    
    Workflow:
    1. Create mesh from fourier coefficients
    2. Generate positions (area-weighted) and velocities
    3. Create cell list and edge-to-cells mapping
    4. Run algorithm (collide + update positions)
    5. Apply boundary condition
    6. Rebin particles that violate cell invariant
    
    Parameters
    ----------
    N : int
        Number of particles
    fourier_coefficients : array-like
        Fourier coefficients defining the shape
    num_boundary_points : int
        Number of points to sample along boundary
    T_x0, T_y0 : float
        Initial temperatures in x and y directions
    dt : float
        Time step
    n_tot : int
        Total number of time steps
    e : float
        Parameter for collision rate
    mu : float
        Viscosity parameter (unused currently)
    alpha : float
        VHS model parameter (unused currently)
    mesh_size : float, optional
        Characteristic mesh size. Larger values = fewer, larger cells.
        Default is 0.1. Recommended: 0.05 (fine) to 0.5 (coarse).
    T_wall_x : float, optional
        Wall temperature in x-direction for thermal (Maxwell) BC.
        If None or accommodation_coefficient==0.0, uses specular reflection.
    T_wall_y : float, optional
        Wall temperature in y-direction for thermal BC.
    accommodation_coefficient : float, optional
        Thermal accommodation coefficient α ∈ [0, 1].
        0.0 = fully specular (default), 1.0 = fully diffuse thermal.
    """
    

    boundary_points = hf.sample_star_shape(fourier_coefficients, num_boundary_points)
    mesh = hf.create_arbitrary_shape_mesh_2d(N, boundary_points, mesh_size=mesh_size)
    
    positions = hf.assign_positions_arbitrary_2d(N, mesh)
    velocities = gh.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)
    
    cell_list, edge_to_cells = hf.create_cell_list_and_adjacency_lists(mesh)
    
    # Create cached boundary structure once (reused for all time steps)
    cached_boundary = hf.CachedBoundary(boundary_points)

    for position, velocity in zip(positions, velocities):
        for cell in cell_list:
            if cell.is_inside(position[0], position[1]):
                cell.add_particle(position, velocity)
                break
    
    temperature_history = np.zeros(n_tot)
    
    total_start_time = time.time()
    
    for n in range(n_tot):
        step_start_time = time.time()
        
        # Step 4: Run collision algorithm
        collision_start = time.time()
        for cell in cell_list: 
            num_collisions = cell.num_collisions(dt, e)

            indices_particles = np.arange(len(cell.particle_positions))

            indices_particles_to_collide = np.random.permutation(indices_particles)[:num_collisions]

            indices_i = indices_particles_to_collide[:num_collisions // 2]
            indices_j = indices_particles_to_collide[num_collisions // 2:]

            indices_pairs = np.column_stack((indices_i, indices_j))
            
            if len(indices_i) > 0 and len(indices_j) > 0:

                v_rel = cell.particle_velocities[indices_i] - cell.particle_velocities[indices_j]
                v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)

                sigma_ij = uh.ArraySigma_VHS(v_rel_mag).reshape(-1)

                Upper_bound_cross_sections = cell.upper_bound_cross_section()
                
                u_rand = np.random.rand(len(indices_i)) * Upper_bound_cross_sections

                accept_condition = (u_rand < sigma_ij)

                indices_i = indices_i[accept_condition]
                indices_j = indices_j[accept_condition]

                if len(indices_i) > 0:
                    cell.collide_and_update_particles(dt, indices_i, indices_j)
        collision_time = time.time() - collision_start

        bc_start = time.time()
        total_kinetic_energy = 0.0
        total_particles = 0
        
        # OPTIMIZED: Collect all particles globally for boundary condition processing
        all_positions = []
        all_velocities = []
        particle_to_cell = []  # Track which cell each particle belongs to
        
        for cell_idx, cell in enumerate(cell_list):
            if len(cell.particle_positions) > 0:
                num_particles = len(cell.particle_positions)
                all_positions.append(cell.particle_positions)
                all_velocities.append(cell.particle_velocities)
                particle_to_cell.extend([cell_idx] * num_particles)
        
        # Process all particles globally with cached boundary
        if len(all_positions) > 0:
            global_positions = np.vstack(all_positions)
            global_velocities = np.vstack(all_velocities)
            
            # Apply boundary condition globally
            _use_thermal = (accommodation_coefficient > 0.0
                            and T_wall_x is not None
                            and T_wall_y is not None)
            if _use_thermal:
                global_velocities, global_positions = bc.thermal_reflection(
                    global_velocities, global_positions, boundary_points,
                    T_wall_x, T_wall_y, accommodation_coefficient,
                    cached_boundary=cached_boundary
                )
            else:
                global_velocities, global_positions = bc.reflecting_BC_arbitrary_shape(
                    global_velocities, global_positions, boundary_points,
                    cached_boundary=cached_boundary
                )
            
            # Redistribute particles back to cells
            particle_idx = 0
            for cell_idx, cell in enumerate(cell_list):
                if len(cell.particle_positions) > 0:
                    num_particles = len(cell.particle_positions)
                    cell.particle_positions = global_positions[particle_idx:particle_idx + num_particles]
                    cell.particle_velocities = global_velocities[particle_idx:particle_idx + num_particles]
                    particle_idx += num_particles
                    
                    # Track temperature (sum of squared velocities)
                    total_kinetic_energy += np.sum(cell.particle_velocities**2)
                    total_particles += len(cell.particle_velocities)
        
        # Track temperature
        temperature_history[n] = total_kinetic_energy / total_particles if total_particles > 0 else 0.0
        bc_time = time.time() - bc_start

        # Step 6: Rebin particles that violate cell invariant
        rebin_start = time.time()
        particles_to_move = []  # List of (old_cell, particle_idx, position, velocity)
        
        # Loop through cells and detect particles that don't respect the invariant
        for cell in cell_list:
            indices_to_remove = []
            for i, position in enumerate(cell.particle_positions):
                if not cell.is_inside(position[0], position[1]):
                    # Particle violates invariant - mark for removal and rebinning
                    particles_to_move.append((cell, i, position, cell.particle_velocities[i]))
                    indices_to_remove.append(i)
            
            # Remove violating particles from this cell (in reverse order to preserve indices)
            for i in reversed(indices_to_remove):
                cell.remove_particle(i)
        
        # Now reassign the violating particles to their correct cells
        if len(particles_to_move) > 0:
            # Extract all positions for nearest centroid lookup
            positions_to_rebin = np.array([position for _, _, position, _ in particles_to_move])
            
            # Find nearest centroid cells for all particles at once using KD-tree
            nearest_cells = hf.find_nearest_centroid_cell_kdtree(positions_to_rebin, cell_list)
            
            # Now iterate through and find containing cells using triangle following
            for (old_cell, _, position, velocity), nearest_cell in zip(particles_to_move, nearest_cells):
                # Use triangle_to_follow to iteratively find the containing cell
                containing_cell = hf.find_containing_cell(position, nearest_cell, edge_to_cells)
                
                # Add particle to the containing cell
                containing_cell.add_particle(position, velocity)
        
        rebin_time = time.time() - rebin_start
        
        step_time = time.time() - step_start_time
        
        # Print progress every 10 steps or on last step
        if (n + 1) % 10 == 0 or (n + 1) == n_tot:
            print(f"{n+1}/{n_tot} ", end="", flush=True)
            if (n + 1) == n_tot or (n + 1) % 50 == 0:
                num_moved = len(particles_to_move)
                print(f"\n  Step {n+1} timing: collisions={collision_time:.3f}s, "
                      f"BC={bc_time:.3f}s, rebin={rebin_time:.3f}s, total={step_time:.3f}s")
                print(f"  Rebinned {num_moved} particles", flush=True)
    
    # Reconstruct final global arrays from cells
    all_positions = []
    all_velocities = []
    for cell in cell_list:
        if len(cell.particle_positions) > 0:
            all_positions.append(cell.particle_positions)
            all_velocities.append(cell.particle_velocities)
    
    if all_positions:
        final_positions = np.vstack(all_positions)
        final_velocities = np.vstack(all_velocities)
    else:
        final_positions = np.empty((0, 2))
        final_velocities = np.empty((0, 2))
    
    total_time = time.time() - total_start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per step: {total_time/n_tot:.3f} seconds")
    
    return final_positions, final_velocities, temperature_history, cell_list, boundary_points






