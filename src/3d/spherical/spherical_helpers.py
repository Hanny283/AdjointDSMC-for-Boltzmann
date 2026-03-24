import numpy as np


def assign_positions_spherical(N, R):
    """
    Assign positions to particles uniformly distributed inside a sphere.
    """
    # Generate random points in unit sphere using rejection sampling
    positions = np.zeros((N, 3))
    count = 0
    while count < N:
        # Generate random point in cube [-1, 1]^3
        x = 2 * np.random.rand() - 1
        y = 2 * np.random.rand() - 1
        z = 2 * np.random.rand() - 1
        
        # Check if point is inside unit sphere
        if x*x + y*y + z*z <= 1:
            positions[count] = [x, y, z]
            count += 1
    
    # Scale to radius R
    positions *= R
    return positions
