"""
Constraint functions for star-shaped boundary optimization.

This module provides constraint builders and evaluation functions for scipy.optimize,
including square inscribed, bounding box, area preservation, and positivity constraints.
"""

import numpy as np
from .shape_optimizer import compute_radius, compute_area


def rho_square(theta, a):
    """
    Compute radial distance to square boundary along angle θ.
    
    For a square [-a, a] × [-a, a], the distance from origin to boundary
    along angle θ is: ρ_Q(θ) = a / max(|cos(θ)|, |sin(θ)|)
    
    Parameters
    ----------
    theta : float or array-like
        Angle(s) in radians
    a : float
        Half-side length of square
        
    Returns
    -------
    rho : float or array
        Radial distance(s) to square boundary
    """
    theta = np.asarray(theta)
    
    # Avoid division by zero
    cos_theta = np.abs(np.cos(theta))
    sin_theta = np.abs(np.sin(theta))
    
    # Small epsilon to avoid division by zero
    eps = 1e-12
    cos_theta = np.maximum(cos_theta, eps)
    sin_theta = np.maximum(sin_theta, eps)
    
    rho = a / np.maximum(cos_theta, sin_theta)
    
    return rho


def rho_box(theta, L):
    """
    Compute radial distance to bounding box along angle θ.
    
    For a box [-L, L] × [-L, L], the distance from origin to boundary
    along angle θ is: ρ_B(θ) = L / max(|cos(θ)|, |sin(θ)|)
    
    Parameters
    ----------
    theta : float or array-like
        Angle(s) in radians
    L : float
        Half-width of bounding box
        
    Returns
    -------
    rho : float or array
        Radial distance(s) to box boundary
    """
    theta = np.asarray(theta)
    
    # Avoid division by zero
    cos_theta = np.abs(np.cos(theta))
    sin_theta = np.abs(np.sin(theta))
    
    # Small epsilon to avoid division by zero
    eps = 1e-12
    cos_theta = np.maximum(cos_theta, eps)
    sin_theta = np.maximum(sin_theta, eps)
    
    rho = L / np.maximum(cos_theta, sin_theta)
    
    return rho


def square_inscribed_constraint(c, opt_config, theta_samples=None):
    """
    Constraint: square must be inside the star-shaped domain.
    
    Requires: r(θ; c) ≥ ρ_Q(θ) for all θ
    
    Returns minimum margin (should be ≥ 0 for feasibility).
    
    Parameters
    ----------
    c : array-like
        Fourier coefficients
    opt_config : dict
        Configuration with 'a' and 'K_angles'
    theta_samples : array-like, optional
        Angular samples to check (if None, uses K_angles from config)
        
    Returns
    -------
    margin : float
        Minimum margin (positive = constraint satisfied)
    """
    a = opt_config['a']
    
    if theta_samples is None:
        K = opt_config['K_angles']
        theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    # Compute radius at sample points
    r = compute_radius(theta_samples, c)
    
    # Compute required distance to contain square
    rho_q = rho_square(theta_samples, a)
    
    # Constraint: r(θ) - ρ_Q(θ) ≥ 0
    margins = r - rho_q
    
    # Return minimum margin (most violated constraint)
    return np.min(margins)


def box_constraint(c, opt_config, theta_samples=None):
    """
    Constraint: star-shaped domain must stay inside bounding box.
    
    Requires: r(θ; c) ≤ ρ_B(θ) for all θ
    
    Returns minimum margin (should be ≥ 0 for feasibility).
    
    Parameters
    ----------
    c : array-like
        Fourier coefficients
    opt_config : dict
        Configuration with 'L' and 'K_angles'
    theta_samples : array-like, optional
        Angular samples to check
        
    Returns
    -------
    margin : float
        Minimum margin (positive = constraint satisfied)
    """
    L = opt_config['L']
    
    if theta_samples is None:
        K = opt_config['K_angles']
        theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    # Compute radius at sample points
    r = compute_radius(theta_samples, c)
    
    # Compute maximum allowed distance
    rho_b = rho_box(theta_samples, L)
    
    # Constraint: ρ_B(θ) - r(θ) ≥ 0
    margins = rho_b - r
    
    # Return minimum margin
    return np.min(margins)


def area_constraint(c, opt_config):
    """
    Constraint: preserve area of the domain.
    
    Requires: |Area(c) - Area_target| ≤ tolerance * Area_target
    
    Returns constraint value (should be ≥ 0 for feasibility).
    
    Parameters
    ----------
    c : array-like
        Fourier coefficients
    opt_config : dict
        Configuration with 'area_target' and 'area_tolerance'
        
    Returns
    -------
    margin : float
        Margin from constraint violation (positive = satisfied)
    """
    area_target = opt_config['area_target']
    tolerance = opt_config['area_tolerance']
    
    # Compute current area
    area = compute_area(c)
    
    # Relative deviation
    rel_deviation = np.abs(area - area_target) / area_target
    
    # Constraint: rel_deviation ≤ tolerance
    # Reformulate as: tolerance - rel_deviation ≥ 0
    margin = tolerance - rel_deviation
    
    return margin


def positivity_constraint(c, opt_config, theta_samples=None):
    """
    Constraint: radius must be positive to ensure star-shape validity.
    
    Requires: r(θ; c) ≥ r_min for all θ
    
    Parameters
    ----------
    c : array-like
        Fourier coefficients
    opt_config : dict
        Configuration with 'r_min' and 'K_angles'
    theta_samples : array-like, optional
        Angular samples to check
        
    Returns
    -------
    margin : float
        Minimum margin (positive = constraint satisfied)
    """
    r_min = opt_config['r_min']
    
    if theta_samples is None:
        K = opt_config['K_angles']
        theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    # Compute radius at sample points
    r = compute_radius(theta_samples, c)
    
    # Constraint: r(θ) - r_min ≥ 0
    margins = r - r_min
    
    # Return minimum margin
    return np.min(margins)


def build_constraints(opt_config):
    """
    Build constraint objects for scipy.optimize.
    
    Creates constraint functions compatible with scipy.optimize.differential_evolution
    using NonlinearConstraint objects.
    
    Parameters
    ----------
    opt_config : dict
        Optimization configuration containing constraint parameters
        
    Returns
    -------
    constraints : list of NonlinearConstraint
        List of NonlinearConstraint objects for differential_evolution
        
        For inequality constraints: constraint(x) >= 0
    """
    from scipy.optimize import NonlinearConstraint
    
    # Pre-compute angular samples (shared across constraints)
    K = opt_config['K_angles']
    theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    constraints = []
    
    # 1. Square inscribed constraint: r(θ) >= ρ_Q(θ)
    # Returns minimum margin (should be >= 0)
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: square_inscribed_constraint(c, opt_config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    # 2. Bounding box constraint: r(θ) <= ρ_B(θ)
    # Returns minimum margin (should be >= 0)
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: box_constraint(c, opt_config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    # 3. Area preservation constraint: |Area - Area_target| <= tolerance * Area_target
    # Returns margin (should be >= 0)
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: area_constraint(c, opt_config),
            lb=0.0,
            ub=np.inf
        )
    )
    
    # 4. Positivity constraint: r(θ) >= r_min
    # Returns minimum margin (should be >= 0)
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: positivity_constraint(c, opt_config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    return constraints


def check_all_constraints(c, opt_config, verbose=True):
    """
    Check all constraints and report violations.
    
    Useful for debugging and validation.
    
    Parameters
    ----------
    c : array-like
        Fourier coefficients
    opt_config : dict
        Optimization configuration
    verbose : bool, optional
        Print detailed information (default: True)
        
    Returns
    -------
    feasible : bool
        True if all constraints satisfied
    violations : dict
        Dictionary of constraint values
    """
    # Pre-compute samples
    K = opt_config['K_angles']
    theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    # Evaluate all constraints
    square_margin = square_inscribed_constraint(c, opt_config, theta_samples)
    box_margin = box_constraint(c, opt_config, theta_samples)
    area_margin = area_constraint(c, opt_config)
    pos_margin = positivity_constraint(c, opt_config, theta_samples)
    
    violations = {
        'square_inscribed': square_margin,
        'box_boundary': box_margin,
        'area_preservation': area_margin,
        'positivity': pos_margin
    }
    
    # Check feasibility (all margins should be >= 0)
    feasible = all(v >= -1e-6 for v in violations.values())  # Small tolerance for numerical errors
    
    if verbose:
        print("=" * 60)
        print("CONSTRAINT CHECK")
        print("=" * 60)
        for name, value in violations.items():
            status = "✓ OK" if value >= -1e-6 else "✗ VIOLATED"
            print(f"  {name:25s}: {value:10.6f}  {status}")
        print("-" * 60)
        print(f"  Overall: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
        print("=" * 60)
    
    return feasible, violations


if __name__ == "__main__":
    # Test the constraint functions
    print("Testing constraints.py functions...")
    
    # Test configuration
    test_config = {
        'M': 4,
        'a': 0.5,
        'L': 5.0,
        'r_min': 0.1,
        'K_angles': 100,
        'area_target': 12.566,  # π * 2² for circle of radius 2
        'area_tolerance': 0.05
    }
    
    # Test case 1: Circle of radius 2 (should be feasible)
    print("\n" + "=" * 60)
    print("Test Case 1: Circle of radius 2.0")
    print("=" * 60)
    c_circle = np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0])
    feasible, violations = check_all_constraints(c_circle, test_config, verbose=True)
    
    # Test case 2: Very small circle (violates square constraint)
    print("\n" + "=" * 60)
    print("Test Case 2: Circle of radius 0.3 (too small for square)")
    print("=" * 60)
    c_small = np.array([0.3, 0, 0, 0, 0, 0, 0, 0, 0])
    feasible, violations = check_all_constraints(c_small, test_config, verbose=True)
    
    # Test case 3: Very large circle (violates box constraint)
    print("\n" + "=" * 60)
    print("Test Case 3: Circle of radius 6.0 (exceeds bounding box)")
    print("=" * 60)
    c_large = np.array([6.0, 0, 0, 0, 0, 0, 0, 0, 0])
    feasible, violations = check_all_constraints(c_large, test_config, verbose=True)
    
    # Test case 4: Shape with Fourier coefficients
    print("\n" + "=" * 60)
    print("Test Case 4: Star-shape with Fourier modes")
    print("=" * 60)
    c_star = np.array([2.0, 0.1, -0.05, 0.08, -0.03, 0.05, 0.02, -0.04, 0.03])
    feasible, violations = check_all_constraints(c_star, test_config, verbose=True)
    
    # Test rho functions
    print("\n" + "=" * 60)
    print("Test radial distance functions")
    print("=" * 60)
    theta_test = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    rho_s = rho_square(theta_test, a=0.5)
    rho_b = rho_box(theta_test, L=5.0)
    print(f"θ:          {theta_test}")
    print(f"ρ_square:   {rho_s}")
    print(f"ρ_box:      {rho_b}")
    
    print("\nAll constraint tests completed!")
