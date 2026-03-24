import numpy as np

def sample_velocities_from_maxwellian_3d(T_x0, T_y0, T_z0, N):
    """
    Sample from a 3D Maxwellian distribution.

    Parameters
    ----------
    T_x0 : float
    The x-component of the temperature.

    T_y0 : float
    The y-component of the temperature.

    T_z0 : float
    The z-component of the temperature.

    N : int 
    The number of velocities to draw
    Returns a numpy array of shape (N, 3) containing the sampled velocities.
    """

    v_x_samples = np.random.normal(0, np.sqrt(T_x0), N)
    v_y_samples = np.random.normal(0, np.sqrt(T_y0), N)
    v_z_samples = np.random.normal(0, np.sqrt(T_z0), N)

    return np.column_stack((v_x_samples, v_y_samples, v_z_samples))

def assign_positions_3d(N, Lx, Ly, Lz):
    """
    Assign positions to particles on a 3-dimensional spatial domain.

    Parameters
    ----------
    N : int
        Number of particles
    Lx : float 
        The length of the x spatial domain
    Ly : float 
        The length of the y spatial domain
    Lz : float 
        The length of the z spatial domain

    Returns
    -------
    positions : numpy array of shape (N, 3)
        The positions of the particles.
    """
    positions_x = np.random.uniform(0, Lx, N)
    positions_y = np.random.uniform(0, Ly, N)
    positions_z = np.random.uniform(0, Lz, N)
    positions = np.column_stack((positions_x, positions_y, positions_z))
    return positions


def collide_particles_3d(velocities, indices_i, indices_j):
    """
    Collides M particles with velocities given by the arrays v_i and v_j.

    Parameters
    ----------
    velocities : numpy array of shape (N, 3)
    indices_i : numpy array of shape (M,)
    indices_j : numpy array of shape (M,)
    """
    v_i = velocities[indices_i]
    v_j = velocities[indices_j]

    v_rel = v_i - v_j                      
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  
    
    theta = np.arccos(2*np.random.rand( len(indices_i) ) - 1)
    phi = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))

    omega = np.column_stack((np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)))

    v_cm = 0.5 * (v_i + v_j)

    v_i_prime = v_cm + 0.5 * v_rel_mag * omega
    v_j_prime = v_cm - 0.5 * v_rel_mag * omega

    velocities[indices_i] = v_i_prime
    velocities[indices_j] = v_j_prime

    return velocities


def pair_particle_indices_3d(sampled_indices):
    """
    input: list of particle indices in a cell, returns pairs for collision
    """
    shuffled = sampled_indices[:]           # copy so original isn't changed
    np.random.shuffle(shuffled)   # shuffle order

    # group into consecutive pairs
    return np.array([[shuffled[i], shuffled[i+1]] for i in range(0, len(shuffled), 2)])

def update_positions_3d(positions, velocities, dt):
    """
    Update particle positions using velocities and time step.
    """
    positions = positions + velocities * dt
    return positions
