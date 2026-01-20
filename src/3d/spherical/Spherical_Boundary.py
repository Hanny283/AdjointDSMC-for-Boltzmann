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

# Import from same directory (Spherical/)
import spherical_helpers as hf
import spherical_bc as bc

# Import from 3d directory
import general_helpers_3d as gh3

# Import from src directory
import universal_sim_helpers as uh

# Optional pygmsh import - will create simple mesh if not available
HAS_PYGMSH = False
try:
    import pygmsh
    HAS_PYGMSH = True
except (ImportError, OSError):
    HAS_PYGMSH = False
    print("Warning: pygmsh not available. Using simple mesh generation.")

class cell_tetrahedron:
    def __init__(self, vertices):
        """
        vertices: a list or array of 4 coordinate pairs [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
        """
        self.vertices = np.array(vertices)
        self.center = np.mean(self.vertices, axis=0)  # centroid
        self.particle_positions = np.empty((0, 3))
        self.particle_velocities = np.empty((0, 3))
        self.rho_cell = len(self.particle_positions) / self.volume()

        #make bounding box 
        minx = min(self.vertices[:,0])
        maxx = max(self.vertices[:,0])
        miny = min(self.vertices[:,1])
        maxy = max(self.vertices[:,1])
        minz = min(self.vertices[:,2])
        maxz = max(self.vertices[:,2])
        self.bounding_box = (minx, maxx, miny, maxy, minz, maxz)

    def volume(self):
        # Use a robust 3D shoelace/cross-product formula for tetrahedron volume
        x1, y1, z1 = self.vertices[0]
        x2, y2, z2 = self.vertices[1]
        x3, y3, z3 = self.vertices[2]
        x4, y4, z4 = self.vertices[3]
        return 0.5 * np.abs((x2 - x1) * (y3 - y1) * (z4 - z1) + (x3 - x1) * (y4 - y1) * (z2 - z1) + (x4 - x1) * (y2 - y1) * (z3 - z1) - (x3 - x1) * (y2 - y1) * (z4 - z1) - (x4 - x1) * (y3 - y1) * (z2 - z1) - (x2 - x1) * (y4 - y1) * (z3 - z1))

    def is_inside(self, x, y, z):
        """
        Return True if point (x, y, z) lies inside or on the edges of the tetrahedron.
        Uses barycentric coordinates.   

        """
        x1, y1, z1 = self.vertices[0]
        x2, y2, z2 = self.vertices[1]
        x3, y3, z3 = self.vertices[2]
        x4, y4, z4 = self.vertices[3]

        denom = (y2 - y3)*(z1 - z4) + (y3 - y4)*(z1 - z2) + (y4 - y2)*(z1 - z3)
        a = ((y2 - y3)*(z - z4) + (y3 - y4)*(z - z2) + (y4 - y2)*(z - z3)) / denom
        b = ((y3 - y1)*(z - z4) + (y1 - y4)*(z - z3) + (y4 - y3)*(z - z1)) / denom
        c = ((y1 - y2)*(z - z4) + (y2 - y4)*(z - z2) + (y4 - y1)*(z - z2)) / denom
        d = 1 - a - b - c

        return (a >= 0) and (b >= 0) and (c >= 0) and (d >= 0)

    def add_particle(self, position, velocity):
        position = np.asarray(position).reshape(1, 3)
        velocity = np.asarray(velocity).reshape(1, 3)
        self.particle_positions = np.vstack((self.particle_positions, position))
        self.particle_velocities = np.vstack((self.particle_velocities, velocity))

    def remove_particle(self, index):
        self.particle_positions = np.delete(self.particle_positions, index, axis=0)
        self.particle_velocities = np.delete(self.particle_velocities, index, axis=0)
        
    def get_particle_positions(self):
        return self.particle_positions

    def get_particle_velocities(self):
        return self.particle_velocities
    
    def upper_bound_cross_section(self):
        if len(self.particle_velocities) == 0:
            return 0.0
        
        v_mean = self.particle_velocities.mean(axis= 0)
        delta_v = np.linalg.norm(self.particle_velocities - v_mean, axis=1).max()
        return 2.0 * delta_v

    def num_collisions(self, dt, e):
        num_particles = len(self.particle_positions)
        if num_particles < 2:
            return 0
        ub_sigma = self.upper_bound_cross_section()
        expected_pairs = (num_particles * self.rho_cell * ub_sigma * dt) / (2 * e) if e != 0 else 0.0
        expected_pairs_int = uh.Iround(expected_pairs)
        max_pairs = num_particles // 2
        return int(min(expected_pairs_int, max_pairs))
    
    def collide_and_update_particles(self, dt, indices_i, indices_j):   
        v_i = self.particle_velocities[indices_i]
        v_j = self.particle_velocities[indices_j]

        v_rel = v_i - v_j                      
        v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  

        theta = np.arccos(2*np.random.rand( len(indices_i) ) - 1)
        phi = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))

        omega = np.column_stack((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)))


        v_cm = 0.5 * (v_i + v_j)

        v_i_prime = v_cm + 0.5 * v_rel_mag * omega
    
        v_j_prime = v_cm - 0.5 * v_rel_mag * omega

        self.particle_velocities[indices_i] = v_i_prime
        self.particle_velocities[indices_j] = v_j_prime

        self.particle_positions = self.particle_positions + self.particle_velocities * dt

        return self.particle_positions, self.particle_velocities
    
    
def Spherical_Boundary(N, R, positions, radius, T_x0, T_y0, T_z0, dt, n_tot, e, mu, alpha, buckets_x, buckets_y, buckets_z):
    print("Starting spherical boundary simulation...")
    print(f"Parameters: N={N}, R={R}, n_tot={n_tot}")
    
    positions = hf.assign_positions_spherical(N, R)
    print("Generated initial positions")
    
    velocities = gh3.sample_velocities_from_maxwellian_3d(T_x0, T_y0, T_z0, N)
    print("Generated initial velocities")

    cell_width = radius / buckets_x
    cell_height = radius / buckets_y
    cell_depth = radius / buckets_z

    bucket_dict = {(i, j, k): set() for i in range(buckets_x) for j in range(buckets_y) for k in range(buckets_z)}
    
    # Generate mesh - use pygmsh if available, otherwise create simple mesh
    print("Generating mesh...")
    if HAS_PYGMSH:
        print("Using pygmsh for mesh generation")
        with pygmsh.geo.Geometry() as geom:
            sphere = geom.add_ball([0.0, 0.0, 0.0], radius=1.0, mesh_size=0.1)
            mesh = geom.generate_mesh()
        
        # Check what cell types are available
        print("Available cell types:", list(mesh.cells_dict.keys()))
        
        # Try to get tetrahedra, fall back to other cell types if not available
        if "tetra" in mesh.cells_dict:
            tet_conn = mesh.get_cells_type("tetra")
            print(f"Found {len(tet_conn)} tetrahedra")
        elif "tetrahedron" in mesh.cells_dict:
            tet_conn = mesh.get_cells_type("tetrahedron")
            print(f"Found {len(tet_conn)} tetrahedra")
        elif "triangle" in mesh.cells_dict:
            # If only triangles available, we need to create a simple tetrahedral mesh
            print("Warning: Only triangles available, creating simple tetrahedral mesh")
            tet_conn = []
        else:
            print("Warning: No suitable cell types found, creating simple mesh")
            tet_conn = []
        
        pts3d = mesh.points[:, :3]
        print(f"Generated {len(pts3d)} mesh points")
    else:
        # Create a simple cubic mesh as fallback
        n_cells_per_dim = 4  # Simple 4x4x4 grid
        cell_size = 2.0 / n_cells_per_dim
        pts3d = []
        tet_conn = []
        
        # Generate points
        for i in range(n_cells_per_dim + 1):
            for j in range(n_cells_per_dim + 1):
                for k in range(n_cells_per_dim + 1):
                    x = -1.0 + i * cell_size
                    y = -1.0 + j * cell_size
                    z = -1.0 + k * cell_size
                    pts3d.append([x, y, z])
        
        pts3d = np.array(pts3d)
        
        # Generate tetrahedra (simplified - just use cubes split into tetrahedra)
        for i in range(n_cells_per_dim):
            for j in range(n_cells_per_dim):
                for k in range(n_cells_per_dim):
                    # Get the 8 vertices of the cube
                    v0 = i * (n_cells_per_dim + 1) * (n_cells_per_dim + 1) + j * (n_cells_per_dim + 1) + k
                    v1 = v0 + 1
                    v2 = v0 + (n_cells_per_dim + 1)
                    v3 = v2 + 1
                    v4 = v0 + (n_cells_per_dim + 1) * (n_cells_per_dim + 1)
                    v5 = v4 + 1
                    v6 = v4 + (n_cells_per_dim + 1)
                    v7 = v6 + 1
                    
                    # Split cube into 5 tetrahedra (simplified approach)
                    tet_conn.append([v0, v1, v2, v4])
                    tet_conn.append([v1, v3, v2, v5])
                    tet_conn.append([v2, v3, v6, v5])
                    tet_conn.append([v1, v5, v2, v4])
                    tet_conn.append([v2, v5, v6, v4])
        
        tet_conn = np.array(tet_conn)
   
    cells = []
    for tet in tet_conn:
        verts = pts3d[tet]          # shape (4, 3): the tetrahedron's four (x,y,z) vertices
        cells.append(cell_tetrahedron(verts))  
    
    for position, velocity in zip(positions, velocities):
        for cell in cells:
            if cell.is_inside(position[0], position[1], position[2]):
                cell.add_particle(position, velocity)
                break

    for bucket in bucket_dict:
        for cell in cells:
            if cell.bounding_box[0] <= bucket[0] * cell_width and cell.bounding_box[1] >= bucket[0] * cell_width and cell.bounding_box[2] <= bucket[1] * cell_height and cell.bounding_box[3] >= bucket[1] * cell_height and cell.bounding_box[4] <= bucket[2] * cell_depth and cell.bounding_box[5] >= bucket[2] * cell_depth:
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

        # Apply spherical reflecting BC on global arrays
        velocities, positions = bc.reflecting_BC_spherical(velocities, positions, R)

        temperature_history[n] = np.sum(velocities**2) / velocities.shape[0]

        # Re-binning (if desired; left as-is)
        for cell in cells:
            for i, particle in enumerate(cell.particle_positions):
                if not(cell.is_inside(particle[0], particle[1], particle[2])):
                    cell.remove_particle(i)
                    x_coord = particle[0]
                    y_coord = particle[1]
                    z_coord = particle[2]
                    ix = np.floor(x_coord / cell_width).astype(int)
                    iy = np.floor(y_coord / cell_height).astype(int)
                    iz = np.floor(z_coord / cell_depth).astype(int)
                    corresponding_bucket = (ix, iy, iz)

                    for tetrahedron in bucket_dict[corresponding_bucket]:
                        if tetrahedron.is_inside(particle[0], particle[1], particle[2]):
                            tetrahedron.add_particle(particle, cell.particle_velocities[i])
                            break
                    
    return positions, velocities, temperature_history