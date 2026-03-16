"""
Configuration parameters for star-shape boundary optimization.

This module contains all configuration parameters for the optimization process,
including optimization settings and DSMC simulation parameters.
"""

import numpy as np

# Optimization configuration
OPTIMIZATION_CONFIG = {
    'M': 4,                    # Number of Fourier modes (total params = 2M+1 = 9)
    
    # Geometric constraints
    'a': 0.5,                  # Half-side length of inscribed square (square: [-a,a]×[-a,a])
    'L': 5.0,                  # Half-width of bounding box (box: [-L,L]×[-L,L])
    'r_min': 0.1,              # Minimum radius to ensure star-shape validity
    
    # Area constraint
    'area_tolerance': 0.05,    # Relative tolerance for area constraint (5%)
    'area_target': None,       # Will be computed from initial guess (set at runtime)
    
    # Regularization
    'lambda_reg': 0.01,        # Spectral regularization weight (penalizes high-frequency modes)
    
    # Constraint sampling
    'K_angles': 100,           # Number of angular samples for boundary constraints
    
    # Optimizer settings
    'maxiter': 20,             # Maximum optimization iterations
    'workers': -1,             # Number of parallel workers (1=sequential, -1=all cores)
    'popsize': 5,              # Population size multiplier → effective pop = popsize*(2M+1) = 45
    'tol': 0.01,               # Convergence tolerance
    'atol': 0.0,               # Absolute tolerance
    'strategy': 'best1bin',    # DE strategy
    'mutation': (0.5, 1),      # DE mutation factor range
    'recombination': 0.7,      # DE crossover probability
    'seed': None,              # Random seed (None for random)
    'polish': False,           # L-BFGS-B polish is unreliable on stochastic objectives
    
    # Visualization settings
    'viz_interval': 5,         # Visualize every N iterations
    'save_frames': True,       # Save visualization frames
    'output_dir': 'optimization_results',  # Output directory for results
}

# DSMC simulation configuration
# These parameters balance physical fidelity with speed for the 30-run batch.
# For final high-quality verification, use: N=3000, n_tot=100, mesh_size=0.25
SIMULATION_CONFIG = {
    # Particle settings
    'N': 1500,                 # Number of particles (3000 for high-quality)

    # Time integration
    'dt': 0.01,                # Time step
    'n_tot': 50,               # Total time steps (100 for high-quality)

    # Initial conditions
    'T_x0': 1.0,               # Initial temperature in x-direction
    'T_y0': 1.0,               # Initial temperature in y-direction

    # Collision parameters
    'e': 1.0,                  # Energy parameter for collision rate
    'mu': 1.0,                 # Viscosity parameter
    'alpha': 1.0,              # VHS model parameter

    # Mesh settings
    'num_boundary_points': 100,  # Boundary sample points (200 for high-quality)
    'mesh_size': 0.4,          # Mesh cell size — larger = coarser = faster (0.25 for high-quality)
}

# Initial guess for optimization
# Start with a circle of radius R=2.0, which satisfies all constraints
INITIAL_GUESS = {
    'R': 2.0,                  # Initial circle radius
    # This generates: c = [R, 0, 0, ..., 0] for M modes
    # i.e., r(θ) = R (a perfect circle)
}

# Parameter bounds for optimization
def get_parameter_bounds(M):
    """
    Get parameter bounds for optimization.
    
    Parameters
    ----------
    M : int
        Number of Fourier modes
        
    Returns
    -------
    list of tuples
        Bounds for each parameter [(lower, upper), ...]
        First bound is for c0 (base radius)
        Next M bounds are for cosine coefficients
        Last M bounds are for sine coefficients
    """
    bounds = [
        (0.5, 4.0),  # c0: base radius (must be positive and reasonable)
    ]
    # Fourier coefficients: allow moderate variation
    for _ in range(2 * M):  # M cosine + M sine terms
        bounds.append((-0.5, 0.5))
    
    return bounds

# Constraint weights (for penalty methods, if needed)
CONSTRAINT_WEIGHTS = {
    'square_inscribed': 1000.0,   # High penalty for violating square constraint
    'box_boundary': 1000.0,        # High penalty for exceeding box
    'area_preservation': 100.0,    # Penalty for area deviation
    'positivity': 10000.0,         # Very high penalty for negative radius
}

# Print configuration summary
def print_config_summary():
    """Print a summary of the optimization configuration."""
    print("=" * 70)
    print("OPTIMIZATION CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nFourier Parameterization:")
    print(f"  Number of modes (M): {OPTIMIZATION_CONFIG['M']}")
    print(f"  Total parameters: {2 * OPTIMIZATION_CONFIG['M'] + 1}")
    print(f"\nRegion of Interest:")
    print(f"  Inscribed square half-side (a): {OPTIMIZATION_CONFIG['a']}")
    print(f"  Square domain: [-{OPTIMIZATION_CONFIG['a']}, {OPTIMIZATION_CONFIG['a']}] × "
          f"[-{OPTIMIZATION_CONFIG['a']}, {OPTIMIZATION_CONFIG['a']}]")
    print(f"  Square side length: {2 * OPTIMIZATION_CONFIG['a']}")
    print(f"\nBoundary Constraints:")
    print(f"  Bounding box half-width (L): {OPTIMIZATION_CONFIG['L']}")
    print(f"  Bounding box: [-{OPTIMIZATION_CONFIG['L']}, {OPTIMIZATION_CONFIG['L']}] × "
          f"[-{OPTIMIZATION_CONFIG['L']}, {OPTIMIZATION_CONFIG['L']}]")
    print(f"  Minimum radius (r_min): {OPTIMIZATION_CONFIG['r_min']}")
    print(f"  Area tolerance: {OPTIMIZATION_CONFIG['area_tolerance'] * 100}%")
    print(f"\nOptimizer Settings:")
    print(f"  Max iterations: {OPTIMIZATION_CONFIG['maxiter']}")
    print(f"  Population size: {OPTIMIZATION_CONFIG['popsize']}")
    print(f"  Workers: {OPTIMIZATION_CONFIG['workers']}")
    print(f"  Regularization weight (λ): {OPTIMIZATION_CONFIG['lambda_reg']}")
    print(f"\nSimulation Settings:")
    print(f"  Particles (N): {SIMULATION_CONFIG['N']}")
    print(f"  Time steps (n_tot): {SIMULATION_CONFIG['n_tot']}")
    print(f"  Time step (dt): {SIMULATION_CONFIG['dt']}")
    print(f"  Mesh size: {SIMULATION_CONFIG['mesh_size']}")
    print(f"\nInitial Guess:")
    print(f"  Circle radius: {INITIAL_GUESS['R']}")
    print("=" * 70)

if __name__ == "__main__":
    # Print configuration when run as script
    print_config_summary()
    
    # Show parameter bounds
    M = OPTIMIZATION_CONFIG['M']
    bounds = get_parameter_bounds(M)
    print(f"\nParameter bounds (M={M}):")
    print(f"  c0 (base radius): {bounds[0]}")
    for i in range(1, M + 1):
        print(f"  c{i} (cos {i}θ): {bounds[i]}")
    for i in range(M + 1, 2 * M + 1):
        print(f"  c{i} (sin {i-M}θ): {bounds[i]}")
