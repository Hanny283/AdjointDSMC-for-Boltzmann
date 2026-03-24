import numpy as np
import universal_sim_helpers as hf
import sys
import os

# Add root directory to path to import edge_class
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from edge_class import Edge


class cell_triangle:
    def __init__(self, vertices):
        """
        vertices: a list or array of 3 coordinate pairs [(x1, y1), (x2, y2), (x3, y3)]
        """
        self.vertices = np.array(vertices)
        self.center = np.mean(self.vertices, axis=0)  # centroid
        self.edges = [Edge(self.vertices[0], self.vertices[1]), Edge(self.vertices[1], self.vertices[2]), Edge(self.vertices[2], self.vertices[0])]
        self.particle_positions = np.empty((0, 2))
        self.particle_velocities = np.empty((0, 2))
        self.rho_cell = len(self.particle_positions) / self.area()

        # make bounding box
        minx = min(self.vertices[:,0])
        maxx = max(self.vertices[:,0])
        miny = min(self.vertices[:,1])
        maxy = max(self.vertices[:,1])
        self.bounding_box = (minx, maxx, miny, maxy)

    
        
    def area(self):
        # Use a robust 2D shoelace/cross-product formula for triangle area
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]
        return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def is_inside(self, x, y):
        """
        Return True if point (x, y) lies inside or on the edges of the triangle.
        Uses barycentric coordinates.
        """
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]

        denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
        c = 1 - a - b

        return (a >= 0) and (b >= 0) and (c >= 0)

    def add_particle(self, position, velocity):
        position = np.asarray(position).reshape(1, 2)
        velocity = np.asarray(velocity).reshape(1, 2)
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
        expected_pairs_int = hf.Iround(expected_pairs)
        max_pairs = num_particles // 2
        return int(min(expected_pairs_int, max_pairs))

    def collide_and_update_particles(self, dt, indices_i, indices_j):
        v_i = self.particle_velocities[indices_i]
        v_j = self.particle_velocities[indices_j]

        v_rel = v_i - v_j                      
        v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  

        theta = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))
        omega = np.column_stack((np.cos(theta), np.sin(theta))) 

        v_cm = 0.5 * (v_i + v_j)

        v_i_prime = v_cm + 0.5 * v_rel_mag * omega
        v_j_prime = v_cm - 0.5 * v_rel_mag * omega

        self.particle_velocities[indices_i] = v_i_prime
        self.particle_velocities[indices_j] = v_j_prime

        self.particle_positions = self.particle_positions + self.particle_velocities * dt

        return self.particle_positions, self.particle_velocities

