import numpy as np

def assign_positions_circular(N, R):
    """
    Assign positions to particles on a circular domain.
    """
    positions_x = R * np.cos(2 * np.pi * np.random.rand(N))
    positions_y = R * np.sin(2 * np.pi * np.random.rand(N))
    positions = np.column_stack((positions_x, positions_y))
    return positions
