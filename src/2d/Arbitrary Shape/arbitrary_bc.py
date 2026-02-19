import numpy as np
import arbitrary_helpers as hf
import general_helpers as gh
import random

# Global cache for boundary structures (keyed by boundary_points id)
_boundary_cache = {}

def _get_cached_boundary(boundary_points):
    """Get or create cached boundary structure."""
    # Use id() as key since boundary_points might be different objects
    # In practice, we'll pass the cached boundary directly, but this is a fallback
    boundary_id = id(boundary_points)
    if boundary_id not in _boundary_cache:
        _boundary_cache[boundary_id] = hf.CachedBoundary(boundary_points)
    return _boundary_cache[boundary_id]


def reflecting_BC_arbitrary_shape(velocities, positions, boundary_points, cached_boundary=None):
    """
    Apply reflecting boundary condition for arbitrary 2D shape (Algorithm 6.2).
    
    OPTIMIZED VERSION: Uses vectorized point-in-polygon and KD-tree for closest edge lookup.
    
    Uses overshoot mirroring: if particle overshoots boundary, mirror the 
    overshoot distance back using the reflected velocity direction.
    
    Parameters:
    velocities: array of shape (N, 2) - particle velocities
    positions: array of shape (N, 2) - particle positions (already updated by dt)
    boundary_points: list of (x, y) tuples defining the shape boundary
    cached_boundary: CachedBoundary object (optional, will create if None)
    
    Returns:
    velocities, positions: updated arrays after reflection
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    
    if len(positions) == 0:
        return velocities, positions
    
    # Get or create cached boundary
    if cached_boundary is None:
        cached_boundary = _get_cached_boundary(boundary_points)
    
    # Vectorized point-in-polygon check
    inside_mask = hf.points_in_polygon_vectorized(positions, cached_boundary.boundary_points)
    outside_mask = ~inside_mask
    
    if not np.any(outside_mask):
        return velocities, positions
    
    # Get positions and velocities of outside particles
    outside_positions = positions[outside_mask]
    outside_velocities = velocities[outside_mask]
    outside_indices = np.where(outside_mask)[0]
    
    # Find closest edge info for all outside particles at once
    edge_indices, distances, closest_points, normals = cached_boundary.get_closest_edge_info(outside_positions)
    
    # Vectorized reflection computation
    # Overshoot vectors
    overshoot = outside_positions - closest_points
    
    # Reflect velocities: v' = v - 2(v·n)n
    vel_dot_normal = np.sum(outside_velocities * normals, axis=1, keepdims=True)
    vel_reflected = outside_velocities - 2 * vel_dot_normal * normals
    
    # Reflect overshoot: overshoot' = overshoot - 2(overshoot·n)n
    overshoot_dot_normal = np.sum(overshoot * normals, axis=1, keepdims=True)
    overshoot_reflected = overshoot - 2 * overshoot_dot_normal * normals
    
    # New positions: boundary + reflected overshoot
    new_positions = closest_points + overshoot_reflected
    
    # Update arrays
    positions[outside_indices] = new_positions
    velocities[outside_indices] = vel_reflected
    
    return velocities, positions

def thermal_reflection(velocities, positions, boundary_points, Tx, Ty, accommodation_coefficient, cached_boundary=None):
    """
    Apply Maxwell thermal boundary condition with diffuse reflection for arbitrary 2D shape.
    
    OPTIMIZED VERSION: Uses vectorized point-in-polygon and KD-tree for closest edge lookup.
    
    With probability (1-α): specular reflection
    With probability α: diffuse thermal emission following Lambert's cosine law
    
    Parameters:
    velocities: array of shape (N, 2) - particle velocities
    positions: array of shape (N, 2) - particle positions (already updated by dt)
    boundary_points: list of (x, y) tuples defining the shape boundary
    Tx, Ty: float - wall temperatures (for Maxwellian velocity distribution)
    accommodation_coefficient: float - thermal accommodation coefficient α ∈ [0,1]
    cached_boundary: CachedBoundary object (optional, will create if None)
    
    Returns:
    velocities, positions: updated arrays after reflection
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    
    if len(positions) == 0:
        return velocities, positions
    
    # Get or create cached boundary
    if cached_boundary is None:
        cached_boundary = _get_cached_boundary(boundary_points)
    
    # Vectorized point-in-polygon check
    inside_mask = hf.points_in_polygon_vectorized(positions, cached_boundary.boundary_points)
    outside_mask = ~inside_mask
    
    if not np.any(outside_mask):
        return velocities, positions
    
    # Get positions and velocities of outside particles
    outside_positions = positions[outside_mask]
    outside_velocities = velocities[outside_mask]
    outside_indices = np.where(outside_mask)[0]
    
    # Find closest edge info for all outside particles
    edge_indices, distances, closest_points, normals = cached_boundary.get_closest_edge_info(outside_positions)
    
    # Process each outside particle (thermal reflection requires per-particle randomness)
    for i, idx in enumerate(outside_indices):
        pos = outside_positions[i]
        vel = outside_velocities[i]
        normal = normals[i]
        closest_point = closest_points[i]
        
        # For thermal reflection, we need the INWARD normal
        to_particle = pos - closest_point
        if np.dot(to_particle, normal) < 0:
            normal = -normal
        inward_normal = -normal
        
        # Decide reflection type
        prob = random.random()
        
        if prob <= (1.0 - accommodation_coefficient):
            # SPECULAR REFLECTION
            vel_dot_normal = np.dot(vel, normal)
            vel_new = vel - 2 * vel_dot_normal * normal
        else:
            # DIFFUSE THERMAL REFLECTION
            vel_sampled = gh.sample_velocities_from_maxwellian_2d(Tx, Ty, 1)[0]
            speed = np.linalg.norm(vel_sampled)
            
            u = random.random()
            theta = np.arcsin(2.0 * u - 1.0)
            
            tangent = np.array([-inward_normal[1], inward_normal[0]])
            direction = np.cos(theta) * inward_normal + np.sin(theta) * tangent
            direction = direction / np.linalg.norm(direction)
            
            vel_new = speed * direction
            
            if np.dot(vel_new, inward_normal) < 0:
                vel_new = -vel_new
        
        # Update position and velocity
        positions[idx] = closest_point
        velocities[idx] = vel_new
    
    return velocities, positions

