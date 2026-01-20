import numpy as np
import arbitrary_helpers as hf
import general_helpers as gh
import random

def reflecting_BC_arbitrary_shape(velocities, positions, boundary_points):
    """
    Apply reflecting boundary condition for arbitrary 2D shape (Algorithm 6.2).
    
    Uses overshoot mirroring: if particle overshoots boundary, mirror the 
    overshoot distance back using the reflected velocity direction.
    
    Parameters:
    velocities: array of shape (N, 2) - particle velocities
    positions: array of shape (N, 2) - particle positions (already updated by dt)
    boundary_points: list of (x, y) tuples defining the shape boundary
                    (should be actual B-spline boundary points, not control points)
    
    Returns:
    velocities, positions: updated arrays after reflection
    """
    # Get boundary edges
    edges = hf.get_boundary_edges(boundary_points)
    
    # Check which particles are outside the domain
    outside_mask = np.zeros(len(positions), dtype=bool)
    for i, pos in enumerate(positions):
        # Use point-in-polygon test to determine if particle is outside
        if not hf.point_in_polygon(pos[0], pos[1], boundary_points):
            outside_mask[i] = True
    
    if not np.any(outside_mask):
        return velocities, positions
    
    # Process particles that are outside
    for idx in np.where(outside_mask)[0]:
        pos = positions[idx]
        vel = velocities[idx]
        
        # Find the closest edge and reflect
        min_dist = float('inf')
        closest_edge = None
        closest_point = None
        
        for edge in edges:
            x1, y1, x2, y2 = edge
            dist, closest_pt = hf.point_to_line_distance(pos[0], pos[1], x1, y1, x2, y2)
            
            if dist < min_dist:
                min_dist = dist
                closest_edge = edge
                closest_point = closest_pt
        
        if closest_edge is not None:
            x1, y1, x2, y2 = closest_edge
            
            # Get normal vector to the edge
            normal_x, normal_y = hf.get_edge_normal(x1, y1, x2, y2)
            normal = np.array([normal_x, normal_y])
            
            # Algorithm 6.2 approach: Mirror the overshoot distance
            # Boundary crossing point
            boundary_point = np.array(closest_point)
            
            # Overshoot vector (how far particle went past boundary)
            overshoot = pos - boundary_point
            
            # Reflect velocity: v' = v - 2(v·n)n
            vel_dot_normal = np.dot(vel, normal)
            vel_reflected = vel - 2 * vel_dot_normal * normal
            
            # Reflect the overshoot: project overshoot onto reflected velocity direction
            # This mirrors the overshoot distance back from boundary
            overshoot_reflected = overshoot - 2 * np.dot(overshoot, normal) * normal
            
            # New position: boundary + reflected overshoot (Algorithm 6.2 formula)
            positions[idx] = boundary_point + overshoot_reflected
            velocities[idx] = vel_reflected
    
    return velocities, positions

def thermal_reflection(velocities, positions, boundary_points, Tx, Ty, accommodation_coefficient):
    """
    Apply Maxwell thermal boundary condition with diffuse reflection for arbitrary 2D shape.
    
    With probability (1-α): specular reflection
    With probability α: diffuse thermal emission following Lambert's cosine law
    
    In diffuse (thermal) reflection:
    - Particles are absorbed and re-emitted by the wall
    - New velocity magnitude is drawn from Maxwellian at wall temperature
    - Direction follows Lambert's cosine law over the inward-pointing hemisphere
    - Most probable direction is along inward normal, but all inward angles are possible
    
    Parameters:
    velocities: array of shape (N, 2) - particle velocities
    positions: array of shape (N, 2) - particle positions (already updated by dt)
    boundary_points: list of (x, y) tuples defining the shape boundary
    Tx, Ty: float - wall temperatures (for Maxwellian velocity distribution)
    accommodation_coefficient: float - thermal accommodation coefficient α ∈ [0,1]
                              α=0: fully specular, α=1: fully diffuse (thermal)
    
    Returns:
    velocities, positions: updated arrays after reflection
    """
    edges = hf.get_boundary_edges(boundary_points)
    
    # Check which particles are outside the domain
    outside_mask = np.zeros(len(positions), dtype=bool)
    for i, pos in enumerate(positions):
        if not hf.point_in_polygon(pos[0], pos[1], boundary_points):
            outside_mask[i] = True
    
    if not np.any(outside_mask):
        return velocities, positions
    
    # Process particles that are outside
    for idx in np.where(outside_mask)[0]:
        pos = positions[idx]
        vel = velocities[idx]
        
        # Find the closest edge (nearest point on boundary)
        min_dist = float('inf')
        closest_edge = None
        closest_point = None
        
        for edge in edges:
            x1, y1, x2, y2 = edge
            dist, closest_pt = hf.point_to_line_distance(pos[0], pos[1], x1, y1, x2, y2)
            
            if dist < min_dist:
                min_dist = dist
                closest_edge = edge
                closest_point = closest_pt
        
        if closest_edge is not None:
            x1, y1, x2, y2 = closest_edge
            
            # Get normal vector to the edge (pointing outward from domain)
            normal_x, normal_y = hf.get_edge_normal(x1, y1, x2, y2)
            normal = np.array([normal_x, normal_y])
            
            # For thermal reflection, we need the INWARD normal (into the domain)
            # Check if normal points outward by seeing if particle is in direction of normal
            to_particle = pos - np.array(closest_point)
            if np.dot(to_particle, normal) < 0:
                # Normal points away from particle (inward), flip it to point outward
                normal = -normal
            
            # Now normal points outward, so inward normal is -normal
            inward_normal = -normal
            
            # Boundary crossing point (place particle at boundary)
            boundary_point = np.array(closest_point)
            
            # Decide reflection type based on accommodation coefficient
            prob = random.random()
            
            if prob <= (1.0 - accommodation_coefficient):
                # SPECULAR REFLECTION (probability 1-α)
                # Reflect velocity: v' = v - 2(v·n)n where n is outward normal
                vel_dot_normal = np.dot(vel, normal)
                vel_new = vel - 2 * vel_dot_normal * normal
                
            else:
                # DIFFUSE THERMAL REFLECTION (probability α)
                # Following Lambert's cosine law and Maxwellian distribution
                
                # Step 1: Sample speed from 2D Maxwellian distribution at wall temperature
                # This gives us v = (vx, vy) ~ Maxwellian(T)
                vel_sampled = gh.sample_velocities_from_maxwellian_2d(Tx, Ty, 1)[0]
                speed = np.linalg.norm(vel_sampled)
                
                # Step 2: Sample direction using Lambert's cosine law
                # In 2D, we sample angle θ from inward normal
                # Lambert's law: pdf(θ) ∝ cos(θ) for θ ∈ [-π/2, π/2]
                # CDF(θ) = (sin(θ) + 1)/2
                # Inverse sampling: θ = arcsin(2u - 1) where u ~ U(0,1)
                u = random.random()
                theta = np.arcsin(2.0 * u - 1.0)  # Angle from inward normal
                
                # Step 3: Construct velocity in global coordinates
                # Get tangent vector (perpendicular to inward normal)
                tangent = np.array([-inward_normal[1], inward_normal[0]])
                
                # Direction unit vector: d = cos(θ)*n_in + sin(θ)*t
                direction = np.cos(theta) * inward_normal + np.sin(theta) * tangent
                direction = direction / np.linalg.norm(direction)  # Normalize
                
                # New velocity: speed * direction
                vel_new = speed * direction
                
                # Safety check: ensure velocity points inward (v·n_in > 0)
                if np.dot(vel_new, inward_normal) < 0:
                    # Flip if pointing outward (shouldn't happen with correct sampling)
                    vel_new = -vel_new
            
            # Update position and velocity
            # Place particle at boundary (no time integration for simplicity)
            positions[idx] = boundary_point
            velocities[idx] = vel_new
    
    return velocities, positions

