"""
Standalone Optimization Convergence Test
=========================================

SELF-CONTAINED - No external dependencies on project structure!
Just upload this file to Jupyter and run.

Tests:
1. Convergence - Does optimization find a minimum?
2. Consistency - Do multiple runs give similar results?
3. Stability - How sensitive to initialization?

Features:
- PERIMETER constraint (material cost)
- 30 trials with different seeds
- M=2 (5 Fourier parameters)
- Complete DSMC simulation embedded
- Full visualization suite

Usage in Jupyter:
    exec(open('standalone_convergence_test.py').read())
    results = run_convergence_test(num_trials=30)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import NonlinearConstraint
from scipy.spatial import cKDTree
import time
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STANDALONE CONVERGENCE TEST MODULE LOADED")
print("="*70)
print("Ready to run optimization tests!")
print("Usage: results = run_convergence_test(num_trials=30)")
print("="*70)

#==============================================================================
# CONFIGURATION
#==============================================================================

TEST_CONFIG = {
    'num_trials': 30,
    'num_workers': -1,  # Use all cores
    'M': 2,             # Fourier modes (5 parameters)
    'a': 0.5,           # Square half-side
    'L': 5.0,           # Bounding box
    'r_min': 0.1,       # Minimum radius
    'perimeter_tolerance': 0.05,
    'perimeter_target': None,
    'lambda_reg': 0.01,
    'K_angles': 100,
    'maxiter': 30,
    'popsize': 10,
    'tol': 0.01,
    'atol': 0.0,
    'strategy': 'best1bin',
    'mutation': (0.5, 1),
    'recombination': 0.7,
}

SIM_CONFIG = {
    'N': 2000,
    'dt': 0.01,
    'n_tot': 50,
    'T_x0': 1.0,
    'T_y0': 1.0,
    'e': 1.0,
    'mu': 1.0,
    'alpha': 1.0,
    'num_boundary_points': 150,
    'mesh_size': 0.3,
}

INITIAL_GUESS = {'R': 2.0}

#==============================================================================
# EMBEDDED SIMPLIFIED DSMC SIMULATION
#==============================================================================

def simplified_dsmc_simulation(boundary_points, N, T_x0, T_y0, dt, n_tot, e):
    """
    Simplified DSMC simulation for testing.
    
    This is a lightweight version that captures the essential physics
    without requiring the full project structure.
    """
    # Initialize particles uniformly inside boundary
    positions = initialize_particles_in_polygon(boundary_points, N)
    
    # Initialize velocities from Maxwellian
    velocities = np.random.randn(N, 2)
    velocities[:, 0] *= np.sqrt(T_x0)
    velocities[:, 1] *= np.sqrt(T_y0)
    
    # Time stepping
    for n in range(n_tot):
        # Collisions (simplified)
        if N > 1:
            n_pairs = int(N * e * dt / 2)
            for _ in range(n_pairs):
                i, j = np.random.choice(N, 2, replace=False)
                # Post-collision velocities (conserve momentum and energy)
                v_cm = (velocities[i] + velocities[j]) / 2
                v_rel = velocities[i] - velocities[j]
                theta = np.random.uniform(0, 2*np.pi)
                v_rel_mag = np.linalg.norm(v_rel)
                v_rel_new = v_rel_mag * np.array([np.cos(theta), np.sin(theta)])
                velocities[i] = v_cm + v_rel_new / 2
                velocities[j] = v_cm - v_rel_new / 2
        
        # Advection
        positions += velocities * dt
        
        # Boundary conditions (reflecting)
        positions, velocities = apply_reflecting_boundary(positions, velocities, boundary_points)
    
    return positions, velocities

def initialize_particles_in_polygon(boundary_points, N):
    """Initialize particles uniformly inside polygon."""
    boundary_points = np.array(boundary_points)
    
    # Bounding box
    xmin, ymin = boundary_points.min(axis=0)
    xmax, ymax = boundary_points.max(axis=0)
    
    positions = []
    while len(positions) < N:
        # Random points in bounding box
        n_try = N - len(positions)
        x = np.random.uniform(xmin, xmax, n_try)
        y = np.random.uniform(ymin, ymax, n_try)
        pts = np.column_stack([x, y])
        
        # Keep only points inside polygon
        inside = points_in_polygon_vectorized(pts, boundary_points)
        positions.extend(pts[inside])
    
    return np.array(positions[:N])

def points_in_polygon_vectorized(points, polygon):
    """Check if points are inside polygon using ray casting."""
    points = np.asarray(points)
    polygon = np.asarray(polygon)
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    n_points = len(points)
    inside = np.zeros(n_points, dtype=bool)
    
    n_vertices = len(polygon)
    
    for i in range(n_points):
        x, y = points[i]
        j = n_vertices - 1
        
        for k in range(n_vertices):
            xi, yi = polygon[k]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside[i] = not inside[i]
            
            j = k
    
    return inside

def apply_reflecting_boundary(positions, velocities, boundary_points):
    """Apply reflecting boundary condition."""
    boundary_points = np.array(boundary_points)
    
    # Check which particles are outside
    inside = points_in_polygon_vectorized(positions, boundary_points)
    outside_indices = np.where(~inside)[0]
    
    if len(outside_indices) == 0:
        return positions, velocities
    
    # For particles outside, reflect them back
    for idx in outside_indices:
        pos = positions[idx]
        vel = velocities[idx]
        
        # Find closest edge
        n_edges = len(boundary_points)
        min_dist = np.inf
        best_edge = 0
        
        for i in range(n_edges):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % n_edges]
            dist = point_to_segment_distance(pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
                best_edge = i
        
        # Get edge normal
        p1 = boundary_points[best_edge]
        p2 = boundary_points[(best_edge + 1) % n_edges]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Reflect velocity
        velocities[idx] = vel - 2 * np.dot(vel, normal) * normal
        
        # Move particle back inside (simple approach)
        positions[idx] = pos - 2 * min_dist * normal
    
    return positions, velocities

def point_to_segment_distance(point, seg_start, seg_end):
    """Distance from point to line segment."""
    seg = seg_end - seg_start
    point_vec = point - seg_start
    
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq == 0:
        return np.linalg.norm(point_vec)
    
    t = max(0, min(1, np.dot(point_vec, seg) / seg_len_sq))
    projection = seg_start + t * seg
    
    return np.linalg.norm(point - projection)

def sample_star_shape(fourier_coefficients, num_points):
    """Sample boundary points from Fourier coefficients."""
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    r = compute_radius(theta, fourier_coefficients)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

#==============================================================================
# SHAPE FUNCTIONS
#==============================================================================

def compute_radius(theta, c):
    """Compute radius r(θ) from Fourier coefficients."""
    c = np.asarray(c)
    theta = np.asarray(theta)
    M = (len(c) - 1) // 2
    
    r = c[0] * np.ones_like(theta)
    for m in range(1, M + 1):
        r += c[m] * np.cos(m * theta) + c[m + M] * np.sin(m * theta)
    
    return r

def compute_perimeter(c, num_samples=1000):
    """Compute perimeter of boundary."""
    theta = np.linspace(0, 2 * np.pi, num_samples)
    dtheta = theta[1] - theta[0]
    
    M = (len(c) - 1) // 2
    r = compute_radius(theta, c)
    
    # Compute dr/dθ
    dr_dtheta = np.zeros_like(theta)
    for m in range(1, M + 1):
        dr_dtheta += m * (-c[m] * np.sin(m * theta) + c[m + M] * np.cos(m * theta))
    
    # Perimeter element
    ds = np.sqrt(r**2 + dr_dtheta**2)
    perimeter = np.trapz(ds, dx=dtheta)
    
    return perimeter

def compute_regularization(c, lambda_reg):
    """Spectral regularization."""
    c = np.asarray(c)
    M = (len(c) - 1) // 2
    
    reg = 0.0
    for m in range(1, M + 1):
        weight = m**4
        reg += weight * (c[m]**2 + c[m + M]**2)
    
    reg *= lambda_reg
    return reg

def compute_kinetic_energy(velocities):
    """Compute total kinetic energy."""
    return 0.5 * np.sum(velocities**2)

def particles_in_square(positions, a):
    """Find particles inside square [-a, a] × [-a, a]."""
    mask = (np.abs(positions[:, 0]) <= a) & (np.abs(positions[:, 1]) <= a)
    indices = np.where(mask)[0]
    return mask, indices

#==============================================================================
# CONSTRAINTS
#==============================================================================

def rho_square(theta, a):
    """Distance from origin to square in direction θ."""
    return a / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

def rho_box(theta, L):
    """Distance from origin to box in direction θ."""
    return L / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))

def square_inscribed_constraint(c, config, theta_samples):
    """Constraint: r(θ) >= ρ_Q(θ)."""
    a = config['a']
    r = compute_radius(theta_samples, c)
    rho_q = rho_square(theta_samples, a)
    margins = r - rho_q
    return np.min(margins)

def box_constraint(c, config, theta_samples):
    """Constraint: r(θ) <= ρ_B(θ)."""
    L = config['L']
    r = compute_radius(theta_samples, c)
    rho_b = rho_box(theta_samples, L)
    margins = rho_b - r
    return np.min(margins)

def perimeter_constraint(c, config):
    """Constraint: preserve perimeter."""
    perimeter_target = config['perimeter_target']
    tolerance = config['perimeter_tolerance']
    
    perimeter = compute_perimeter(c)
    rel_deviation = np.abs(perimeter - perimeter_target) / perimeter_target
    
    margin = tolerance - rel_deviation
    return margin

def positivity_constraint(c, config, theta_samples):
    """Constraint: r(θ) >= r_min."""
    r_min = config['r_min']
    r = compute_radius(theta_samples, c)
    margins = r - r_min
    return np.min(margins)

def build_constraints(config):
    """Build constraint objects for scipy."""
    K = config['K_angles']
    theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    constraints = []
    
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: square_inscribed_constraint(c, config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: box_constraint(c, config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: perimeter_constraint(c, config),
            lb=0.0,
            ub=np.inf
        )
    )
    
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: positivity_constraint(c, config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    return constraints

#==============================================================================
# OBJECTIVE FUNCTION
#==============================================================================

def objective_function(c, sim_params, opt_config):
    """Objective: minimize KE per particle in square + regularization."""
    a = opt_config['a']
    lambda_reg = opt_config['lambda_reg']
    
    try:
        # Sample boundary
        boundary_points = sample_star_shape(c, sim_params['num_boundary_points'])
        
        # Run simplified DSMC
        positions, velocities = simplified_dsmc_simulation(
            boundary_points=boundary_points,
            N=sim_params['N'],
            T_x0=sim_params['T_x0'],
            T_y0=sim_params['T_y0'],
            dt=sim_params['dt'],
            n_tot=sim_params['n_tot'],
            e=sim_params['e']
        )
        
        # Find particles in square
        mask, indices = particles_in_square(positions, a)
        
        if len(indices) == 0:
            return 1e10
        
        # Compute KE per particle
        velocities_in_square = velocities[indices]
        ke_in_square = compute_kinetic_energy(velocities_in_square)
        ke_per_particle = ke_in_square / len(indices)
        
        # Regularization
        reg = compute_regularization(c, lambda_reg)
        
        # Total objective
        obj = ke_per_particle + reg
        
        return obj
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1e10

#==============================================================================
# PARAMETER BOUNDS
#==============================================================================

def get_parameter_bounds(M):
    """Get parameter bounds."""
    bounds = [(0.5, 4.0)]  # c0
    for _ in range(2 * M):
        bounds.append((-0.5, 0.5))
    return bounds

#==============================================================================
# RUN OPTIMIZATION
#==============================================================================

def run_single_optimization(trial_num, config, sim_params, seed):
    """Run a single optimization trial."""
    print(f"\n{'='*70}")
    print(f"TRIAL {trial_num} - Seed: {seed}")
    print(f"{'='*70}")
    
    M = config['M']
    
    # Initial guess
    c0 = np.zeros(2 * M + 1)
    c0[0] = INITIAL_GUESS['R']
    
    # Set perimeter target
    initial_perimeter = compute_perimeter(c0)
    config['perimeter_target'] = initial_perimeter
    
    print(f"Initial perimeter: {initial_perimeter:.4f}")
    
    # Setup
    bounds = get_parameter_bounds(M)
    constraints = build_constraints(config)
    
    # Results tracking
    results = {
        'trial': trial_num,
        'seed': seed,
        'start_time': time.time(),
        'best_coefficients': None,
        'best_objective': np.inf,
        'success': False,
        'message': '',
        'iterations': 0,
    }
    
    # Run optimization
    try:
        result = scipy.optimize.differential_evolution(
            func=lambda c: objective_function(c, sim_params, config),
            bounds=bounds,
            constraints=constraints,
            strategy=config['strategy'],
            maxiter=config['maxiter'],
            popsize=config['popsize'],
            tol=config['tol'],
            atol=config['atol'],
            mutation=config['mutation'],
            recombination=config['recombination'],
            seed=seed,
            workers=config['num_workers'],
            disp=False,
            polish=False
        )
        
        results['best_coefficients'] = result.x
        results['best_objective'] = result.fun
        results['success'] = result.success
        results['message'] = result.message
        results['iterations'] = result.nit
        
    except Exception as e:
        results['message'] = f"ERROR: {str(e)}"
        print(f"Failed: {str(e)}")
    
    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']
    
    print(f"Complete: Obj={results['best_objective']:.6f}, Time={results['duration']/60:.2f}min")
    
    return results

#==============================================================================
# VISUALIZATION
#==============================================================================

def visualize_trial(result, config, save_path=None):
    """Visualize single trial result."""
    if result['best_coefficients'] is None:
        print(f"Trial {result['trial']} failed")
        return
    
    c = result['best_coefficients']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Shape
    ax1 = axes[0]
    theta = np.linspace(0, 2*np.pi, 500)
    r = compute_radius(theta, c)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax1.plot(x, y, 'b-', linewidth=2, label='Shape')
    
    a = config['a']
    square = plt.Rectangle((-a, -a), 2*a, 2*a, 
                          fill=False, edgecolor='red', linewidth=2,
                          linestyle='--', label='Square')
    ax1.add_patch(square)
    
    L = config['L']
    ax1.set_xlim(-L-0.5, L+0.5)
    ax1.set_ylim(-L-0.5, L+0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'Trial {result["trial"]} - Seed {result["seed"]}', fontweight='bold')
    
    # Plot 2: Coefficients
    ax2 = axes[1]
    M = (len(c) - 1) // 2
    indices = np.arange(len(c))
    labels = ['c₀'] + [f'c{i}' for i in range(1, M+1)] + [f's{i}' for i in range(1, M+1)]
    colors = ['red'] + ['blue']*M + ['green']*M
    
    ax2.bar(indices, c, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Value')
    ax2.set_title(f'Coefficients - Obj: {result["best_objective"]:.6f}', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def compare_all_trials(all_results, config, save_path=None):
    """Compare all trials."""
    successful = [r for r in all_results if r['success']]
    
    if len(successful) == 0:
        print("No successful trials!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: All shapes
    ax1 = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 500)
    
    for r in successful[:10]:  # Show first 10
        c = r['best_coefficients']
        rad = compute_radius(theta, c)
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        ax1.plot(x, y, linewidth=2, alpha=0.6)
    
    a = config['a']
    square = plt.Rectangle((-a, -a), 2*a, 2*a,
                          fill=False, edgecolor='red', linewidth=2,
                          linestyle='--')
    ax1.add_patch(square)
    
    L = config['L']
    ax1.set_xlim(-L-0.5, L+0.5)
    ax1.set_ylim(-L-0.5, L+0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'All Shapes (first 10/{len(successful)})', fontweight='bold')
    
    # Plot 2: Objectives
    ax2 = axes[0, 1]
    trials = [r['trial'] for r in successful]
    objectives = [r['best_objective'] for r in successful]
    
    ax2.bar(trials, objectives, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Objective')
    ax2.set_title('Objective Values', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    mean_obj = np.mean(objectives)
    std_obj = np.std(objectives)
    ax2.axhline(mean_obj, color='red', linestyle='--', linewidth=2)
    ax2.axhline(mean_obj + std_obj, color='orange', linestyle=':', linewidth=1.5)
    ax2.axhline(mean_obj - std_obj, color='orange', linestyle=':', linewidth=1.5)
    
    # Plot 3: Coefficients
    ax3 = axes[1, 0]
    M = config['M']
    n_params = 2 * M + 1
    
    coeffs_matrix = np.array([r['best_coefficients'] for r in successful])
    mean_coeffs = np.mean(coeffs_matrix, axis=0)
    std_coeffs = np.std(coeffs_matrix, axis=0)
    
    indices = np.arange(n_params)
    labels = ['c₀'] + [f'c{i}' for i in range(1, M+1)] + [f's{i}' for i in range(1, M+1)]
    
    ax3.bar(indices, mean_coeffs, yerr=std_coeffs, capsize=5,
           color='steelblue', alpha=0.7, edgecolor='black', ecolor='red')
    ax3.set_xticks(indices)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Value')
    ax3.set_title('Mean Coefficients (±1σ)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(0, color='black', linewidth=0.5)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    cv = (std_obj / mean_obj) * 100
    
    stats_text = "STATISTICS\n" + "="*50 + "\n\n"
    stats_text += f"Trials: {len(all_results)}\n"
    stats_text += f"Successful: {len(successful)}\n\n"
    stats_text += f"Objective:\n"
    stats_text += f"  Mean: {mean_obj:.6f}\n"
    stats_text += f"  Std:  {std_obj:.6f}\n"
    stats_text += f"  Min:  {np.min(objectives):.6f}\n"
    stats_text += f"  Max:  {np.max(objectives):.6f}\n"
    stats_text += f"  CV:   {cv:.2f}%\n\n"
    
    if cv < 1:
        stats_text += "Consistency: ✓ EXCELLENT"
    elif cv < 5:
        stats_text += "Consistency: ✓ GOOD"
    elif cv < 10:
        stats_text += "Consistency: ⚠ MODERATE"
    else:
        stats_text += "Consistency: ✗ POOR"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

#==============================================================================
# MAIN TEST FUNCTION
#==============================================================================

def run_convergence_test(num_trials=30, output_dir='convergence_results'):
    """Run convergence test with multiple trials."""
    
    print("="*70)
    print("CONVERGENCE TEST")
    print("="*70)
    print(f"Trials: {num_trials}")
    print(f"M: {TEST_CONFIG['M']} (params: {2*TEST_CONFIG['M']+1})")
    print(f"Max iterations: {TEST_CONFIG['maxiter']}")
    print(f"Estimated time: {num_trials * 3}-{num_trials * 6} minutes")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate seeds
    base_seeds = [42, 123, 456, 789, 1011, 2048, 3141, 5926, 8192, 10000,
                  12345, 15000, 17777, 20000, 22222, 25000, 27182, 30000,
                  33333, 36000, 38888, 41000, 43210, 45678, 48000, 50000,
                  52525, 55555, 58008, 60000]
    seeds = base_seeds[:num_trials]
    
    # Run trials
    all_results = []
    
    for i, seed in enumerate(seeds, 1):
        result = run_single_optimization(i, TEST_CONFIG.copy(), SIM_CONFIG, seed)
        all_results.append(result)
        
        # Visualize every 5th trial
        if result['success'] and i % 5 == 0:
            save_path = os.path.join(output_dir, f'trial_{i}.png')
            visualize_trial(result, TEST_CONFIG, save_path)
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    successful = [r for r in all_results if r['success']]
    
    if len(successful) > 0:
        objectives = [r['best_objective'] for r in successful]
        
        print(f"\nSuccessful: {len(successful)}/{len(all_results)}")
        
        if len(objectives) > 1:
            mean_obj = np.mean(objectives)
            std_obj = np.std(objectives)
            cv = (std_obj / mean_obj) * 100
            
            print(f"\nObjective Statistics:")
            print(f"  Mean: {mean_obj:.8f}")
            print(f"  Std:  {std_obj:.8f}")
            print(f"  Min:  {np.min(objectives):.8f}")
            print(f"  Max:  {np.max(objectives):.8f}")
            print(f"  CV:   {cv:.2f}%")
            
            print(f"\nConsistency:")
            if cv < 1:
                print("  ✓ EXCELLENT - Very consistent")
            elif cv < 5:
                print("  ✓ GOOD - Reasonably consistent")
            elif cv < 10:
                print("  ⚠ MODERATE - Some variation")
            else:
                print("  ✗ POOR - High variation")
        
        # Comparison plot
        comp_path = os.path.join(output_dir, 'comparison.png')
        compare_all_trials(all_results, TEST_CONFIG, comp_path)
        
        # Save summary
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write("CONVERGENCE TEST RESULTS\n")
            f.write("="*70 + "\n\n")
            for r in all_results:
                f.write(f"Trial {r['trial']}: ")
                f.write(f"Obj={r['best_objective']:.8f}, ")
                f.write(f"Time={r['duration']/60:.2f}min, ")
                f.write(f"Success={r['success']}\n")
            
            if len(successful) > 1:
                f.write(f"\nMean: {mean_obj:.8f}\n")
                f.write(f"Std: {std_obj:.8f}\n")
                f.write(f"CV: {cv:.2f}%\n")
        
        print(f"\nResults saved to: {output_dir}")
    else:
        print("\nNo successful trials!")
    
    print(f"{'='*70}")
    
    return all_results

# Auto-run if executed as script
if __name__ == "__main__":
    results = run_convergence_test(num_trials=30)
