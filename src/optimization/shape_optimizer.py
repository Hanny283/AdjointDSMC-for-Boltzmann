"""
Core optimization functions for star-shaped boundary optimization.

This module contains the objective function and helper functions for computing
geometric properties and particle kinetic energy.
"""

import numpy as np
import sys
import os

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
_2d_arbitrary_dir = os.path.join(_src_dir, '2d', 'Arbitrary Shape')

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _2d_arbitrary_dir not in sys.path:
    sys.path.insert(0, _2d_arbitrary_dir)

# Import DSMC simulation function
from arbitrary_parameterized import Arbitrary_Shape_Parameterized


def compute_radius(theta, c):
    """
    Compute radius r(θ; c) for star-shaped boundary.
    
    Formula: r(θ; c) = c₀ + Σ[c_m cos(mθ) + c_{m+M} sin(mθ)] for m=1..M
    
    Parameters
    ----------
    theta : float or array-like
        Angle(s) in radians [0, 2π]
    c : array-like of shape (2M+1,)
        Fourier coefficients [c0, c1..c_M, c_{M+1}..c_{2M}]
        
    Returns
    -------
    r : float or array
        Radius at angle(s) theta
    """
    c = np.asarray(c)
    theta = np.asarray(theta)
    M = (len(c) - 1) // 2  # Number of Fourier modes
    
    # Base radius
    r = c[0] * np.ones_like(theta)
    
    # Add Fourier terms
    for m in range(1, M + 1):
        r += c[m] * np.cos(m * theta) + c[m + M] * np.sin(m * theta)
    
    return r


def compute_area(c, num_samples=1000):
    """
    Compute area enclosed by star-shaped boundary.
    
    Formula: Area = ½ ∫₀^{2π} r(θ)² dθ
    
    Uses trapezoidal rule for numerical integration.
    
    Parameters
    ----------
    c : array-like of shape (2M+1,)
        Fourier coefficients
    num_samples : int, optional
        Number of samples for numerical integration (default: 1000)
        
    Returns
    -------
    area : float
        Area enclosed by the boundary
    """
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=True)
    r = compute_radius(theta, c)
    
    # Area = ½ ∫ r² dθ
    area = 0.5 * np.trapz(r**2, theta)
    
    return area


def particles_in_square(positions, a):
    """
    Filter particles inside the square region [-a, a] × [-a, a].
    
    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        Particle positions (x, y)
    a : float
        Half-side length of the square
        
    Returns
    -------
    mask : ndarray of shape (N,) dtype bool
        Boolean mask: True for particles inside square
    indices : ndarray of shape (K,) dtype int
        Indices of particles inside square
    """
    if len(positions) == 0:
        return np.array([], dtype=bool), np.array([], dtype=int)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    # Check if inside square: |x| ≤ a AND |y| ≤ a
    mask = (np.abs(x) <= a) & (np.abs(y) <= a)
    indices = np.where(mask)[0]
    
    return mask, indices


def compute_kinetic_energy(velocities):
    """
    Compute total kinetic energy (sum of squared velocities).
    
    KE = Σ(vₓ² + vᵧ²) for all particles
    
    Parameters
    ----------
    velocities : ndarray of shape (N, 2)
        Particle velocities (vx, vy)
        
    Returns
    -------
    ke : float
        Total kinetic energy
    """
    if len(velocities) == 0:
        return 0.0
    
    # Sum of squared velocities
    ke = np.sum(velocities**2)
    
    return ke


def compute_regularization(c, lambda_reg):
    """
    Compute spectral regularization term.
    
    Regularization = λ Σ_{m=1}^M m⁴(c_m² + c_{m+M}²)
    
    This penalizes high-frequency modes to encourage smooth boundaries.
    
    Parameters
    ----------
    c : array-like of shape (2M+1,)
        Fourier coefficients
    lambda_reg : float
        Regularization weight
        
    Returns
    -------
    reg : float
        Regularization penalty
    """
    c = np.asarray(c)
    M = (len(c) - 1) // 2
    
    reg = 0.0
    for m in range(1, M + 1):
        # Weight by m⁴ to heavily penalize high frequencies
        weight = m**4
        reg += weight * (c[m]**2 + c[m + M]**2)
    
    reg *= lambda_reg
    
    return reg


def objective_function(c, sim_params, opt_config, verbose=False):
    """
    Objective function to minimize: kinetic energy in square + regularization.
    
    Steps:
    1. Sample boundary from Fourier coefficients
    2. Run DSMC simulation
    3. Identify particles inside square region Q
    4. Compute kinetic energy of particles in Q
    5. Add spectral regularization
    6. Return total objective value
    
    Parameters
    ----------
    c : array-like of shape (2M+1,)
        Fourier coefficients to optimize
    sim_params : dict
        DSMC simulation parameters (N, dt, n_tot, etc.)
    opt_config : dict
        Optimization configuration (a, lambda_reg, etc.)
    verbose : bool, optional
        Print detailed information (default: False)
        
    Returns
    -------
    obj : float
        Objective value to minimize
    """
    c = np.asarray(c)
    
    # Extract configuration
    a = opt_config['a']
    lambda_reg = opt_config['lambda_reg']
    num_boundary_points = sim_params['num_boundary_points']
    
    try:
        # Run DSMC simulation
        if verbose:
            print(f"\n  Running simulation with c = {c}")
        
        positions, velocities, temperature_history, cell_list, boundary_points = \
            Arbitrary_Shape_Parameterized(
                N=sim_params['N'],
                fourier_coefficients=c,
                num_boundary_points=num_boundary_points,
                T_x0=sim_params['T_x0'],
                T_y0=sim_params['T_y0'],
                dt=sim_params['dt'],
                n_tot=sim_params['n_tot'],
                e=sim_params['e'],
                mu=sim_params['mu'],
                alpha=sim_params['alpha'],
                mesh_size=sim_params['mesh_size']
            )
        
        # Filter particles in square region
        mask, indices = particles_in_square(positions, a)
        
        if len(indices) == 0:
            if verbose:
                print(f"  WARNING: No particles in square region!")
            # Return high penalty if no particles in square
            return 1e10
        
        # Compute kinetic energy in square
        velocities_in_square = velocities[indices]
        ke_in_square = compute_kinetic_energy(velocities_in_square)
        
        # Normalize by number of particles to make objective scale-invariant
        ke_per_particle = ke_in_square / len(indices) if len(indices) > 0 else 1e10
        
        # Compute regularization
        reg = compute_regularization(c, lambda_reg)
        
        # Total objective
        obj = ke_per_particle + reg
        
        if verbose:
            print(f"  Particles in square: {len(indices)} / {len(positions)}")
            print(f"  KE in square: {ke_in_square:.6f}")
            print(f"  KE per particle: {ke_per_particle:.6f}")
            print(f"  Regularization: {reg:.6f}")
            print(f"  Total objective: {obj:.6f}")
        
        return obj
        
    except Exception as e:
        print(f"  ERROR in objective function: {str(e)}")
        # Return high penalty on failure
        return 1e10


def evaluate_with_details(c, sim_params, opt_config):
    """
    Evaluate objective function and return detailed breakdown.
    
    This is useful for analysis and visualization.
    
    Parameters
    ----------
    c : array-like of shape (2M+1,)
        Fourier coefficients
    sim_params : dict
        DSMC simulation parameters
    opt_config : dict
        Optimization configuration
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'objective': total objective value
        - 'ke_total': total kinetic energy
        - 'ke_in_square': kinetic energy in square
        - 'ke_per_particle': normalized KE
        - 'regularization': regularization term
        - 'num_particles_in_square': particle count in square
        - 'num_particles_total': total particle count
        - 'positions': final particle positions
        - 'velocities': final particle velocities
        - 'boundary_points': boundary point coordinates
        - 'temperature_history': temperature evolution
        - 'area': enclosed area
    """
    c = np.asarray(c)
    a = opt_config['a']
    lambda_reg = opt_config['lambda_reg']
    num_boundary_points = sim_params['num_boundary_points']
    
    try:
        # Run simulation
        positions, velocities, temperature_history, cell_list, boundary_points = \
            Arbitrary_Shape_Parameterized(
                N=sim_params['N'],
                fourier_coefficients=c,
                num_boundary_points=num_boundary_points,
                T_x0=sim_params['T_x0'],
                T_y0=sim_params['T_y0'],
                dt=sim_params['dt'],
                n_tot=sim_params['n_tot'],
                e=sim_params['e'],
                mu=sim_params['mu'],
                alpha=sim_params['alpha'],
                mesh_size=sim_params['mesh_size']
            )
        
        # Particle analysis
        mask, indices = particles_in_square(positions, a)
        velocities_in_square = velocities[indices] if len(indices) > 0 else np.array([])
        
        # Compute metrics
        ke_total = compute_kinetic_energy(velocities)
        ke_in_square = compute_kinetic_energy(velocities_in_square)
        ke_per_particle = ke_in_square / len(indices) if len(indices) > 0 else 0.0
        reg = compute_regularization(c, lambda_reg)
        area = compute_area(c)
        
        obj = ke_per_particle + reg
        
        return {
            'objective': obj,
            'ke_total': ke_total,
            'ke_in_square': ke_in_square,
            'ke_per_particle': ke_per_particle,
            'regularization': reg,
            'num_particles_in_square': len(indices),
            'num_particles_total': len(positions),
            'positions': positions,
            'velocities': velocities,
            'boundary_points': boundary_points,
            'temperature_history': temperature_history,
            'area': area,
            'cell_list': cell_list
        }
        
    except Exception as e:
        print(f"ERROR in evaluate_with_details: {str(e)}")
        return None


if __name__ == "__main__":
    # Test the functions
    print("Testing shape_optimizer.py functions...")
    
    # Test compute_radius
    M = 4
    c_test = np.array([2.0, 0.1, 0.2, -0.1, 0.15, 0.05, -0.08, 0.12, -0.05])
    theta_test = np.array([0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2])
    r_test = compute_radius(theta_test, c_test)
    print(f"\nRadius test:")
    print(f"  θ = {theta_test}")
    print(f"  r(θ) = {r_test}")
    
    # Test compute_area
    area_test = compute_area(c_test)
    print(f"\nArea test:")
    print(f"  c = {c_test}")
    print(f"  Area = {area_test:.4f}")
    
    # Test particles_in_square
    positions_test = np.array([
        [0.2, 0.3],
        [0.6, 0.4],
        [-0.3, -0.2],
        [0.8, 0.8],
    ])
    mask_test, indices_test = particles_in_square(positions_test, a=0.5)
    print(f"\nParticles in square test (a=0.5):")
    print(f"  Positions: {positions_test}")
    print(f"  Mask: {mask_test}")
    print(f"  Indices: {indices_test}")
    
    # Test compute_kinetic_energy
    velocities_test = np.array([
        [1.0, 0.5],
        [0.3, -0.8],
        [-0.5, 0.2],
    ])
    ke_test = compute_kinetic_energy(velocities_test)
    print(f"\nKinetic energy test:")
    print(f"  Velocities: {velocities_test}")
    print(f"  Total KE: {ke_test:.4f}")
    
    # Test compute_regularization
    reg_test = compute_regularization(c_test, lambda_reg=0.01)
    print(f"\nRegularization test:")
    print(f"  c = {c_test}")
    print(f"  λ = 0.01")
    print(f"  Regularization: {reg_test:.6f}")
    
    print("\nAll tests completed successfully!")
