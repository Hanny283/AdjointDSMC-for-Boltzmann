import numpy as np
import sys
import os
import pygmsh

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_basic_shapes_dir = os.path.dirname(_current_dir)
_2d_dir = os.path.dirname(_basic_shapes_dir)
_src_dir = os.path.dirname(_2d_dir)

# Add directories to path
if _2d_dir not in sys.path:
    sys.path.insert(0, _2d_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import circular_helpers as hf
import circular_bc as bc
import general_helpers as gh
import universal_sim_helpers as uh
import cell_class as ct

def Circular_Boundary(N, R, positions, radius, T_x0, T_y0, dt, n_tot, e, mu, alpha, buckets_x, buckets_y):

    positions = hf.assign_positions_circular(N, R)
    velocities = gh.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)


    #spatial discretization
    cell_width = radius / buckets_x
    cell_height = radius / buckets_y

    bucket_dict = {(i, j): set() for i in range(buckets_x) for j in range(buckets_y)}



    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle([0.0, 0.0, 0.0], radius=1.0, mesh_size=0.1)
        mesh = geom.generate_mesh()

    tri_conn = mesh.get_cells_type("triangle")
    # 2D coords (drop z); points shape is (n_pts, 3)
    pts2d = mesh.points[:, :2]

    cells = []
    for tri in tri_conn:
        verts = pts2d[tri]          # shape (3, 2): the triangle's three (x,y) vertices
        cells.append(ct.cell_triangle(verts))

    for position, velocity in zip(positions, velocities):
        for cell in cells:
            if cell.is_inside(position[0], position[1]):
                cell.add_particle(position, velocity)
                break

    for bucket in bucket_dict:
        for cell in cells:
            if (cell.bounding_box[0] < bucket[0] * cell_width + cell_width and 
                cell.bounding_box[1] > bucket[0] * cell_width and 
                cell.bounding_box[2] < bucket[1] * cell_height + cell_height and 
                cell.bounding_box[3] > bucket[1] * cell_height):
                bucket_dict[bucket].add(cell)
                break

    temperature_history = np.zeros(n_tot)

    for n in range(n_tot):
        for cell in cells:
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

        # Apply circular reflecting BC on global arrays
        velocities, positions = bc.reflecting_BC_circular(velocities, positions, R)

        # Track temperature
        temperature_history[n] = np.sum(velocities**2) / velocities.shape[0]

        # Re-binning (if desired; left as-is)
        for cell in cells:
            for particle in cell.particle_positions:
                if not(cell.is_inside(particle[0], particle[1])):
                    cell.remove_particle(particle)
                    x_coord = particle[0]
                    y_coord = particle[1]
                    ix = np.floor(x_coord / cell_width).astype(int)
                    iy = np.floor(y_coord / cell_height).astype(int)
                    corresponding_bucket = (ix, iy)

                    for triangle in bucket_dict[corresponding_bucket]:
                        if triangle.is_inside(particle[0], particle[1]):
                            triangle.add_particle(particle, cell.particle_velocities[particle])
                            break
    return positions, velocities, temperature_history
        