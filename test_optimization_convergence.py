"""
Test Optimization Convergence - Multiple Run Analysis

This script runs the shape optimization multiple times to test:
1. Consistency - Do repeated runs find similar solutions?
2. Convergence - Does the algorithm find a minimum?
3. Stability - How sensitive is it to random initialization?

Modified for Jupyter notebook / GPU execution.
Uses PERIMETER constraint instead of area (material cost).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import NonlinearConstraint
import time
from datetime import datetime
import os
import sys

# Add paths for imports (adjust if needed for your setup)
sys.path.insert(0, '/path/to/src')  # Update this path!
sys.path.insert(0, '/path/to/src/2d/Arbitrary Shape')  # Update this path!

# Import DSMC simulation
from arbitrary_parameterized import Arbitrary_Shape_Parameterized

#==============================================================================
# CONFIGURATION - Simplified for Testing
#==============================================================================

TEST_CONFIG = {
    # Test parameters
    'num_trials': 30,          # Number of optimization runs
    'num_workers': -1,         # Use all available cores (32 threads on your GPU workstation!)
    
    # Fourier parameterization (simplified)
    'M': 2,                    # Fewer modes for faster testing (2M+1 = 5 params)
    
    # Geometric constraints
    'a': 0.5,                  # Inscribed square half-side
    'L': 5.0,                  # Bounding box
    'r_min': 0.1,              # Minimum radius
    
    # PERIMETER constraint (material cost)
    'perimeter_tolerance': 0.05,  # 5% tolerance
    'perimeter_target': None,      # Will be computed from initial guess
    
    # Regularization
    'lambda_reg': 0.01,
    
    # Constraint sampling
    'K_angles': 100,
    
    # Optimizer settings (reduced for faster testing)
    'maxiter': 30,             # Fewer iterations for testing
    'popsize': 10,             # Smaller population for speed
    'tol': 0.01,
    'atol': 0.0,
    'strategy': 'best1bin',
    'mutation': (0.5, 1),
    'recombination': 0.7,
}

# DSMC simulation config (lightweight for testing)
SIM_CONFIG = {
    'N': 2000,                 # Fewer particles for speed
    'dt': 0.01,
    'n_tot': 50,               # Fewer timesteps for speed
    'T_x0': 1.0,
    'T_y0': 1.0,
    'e': 1.0,
    'mu': 1.0,
    'alpha': 1.0,
    'num_boundary_points': 150,
    'mesh_size': 0.3,          # Coarser mesh for speed
}

INITIAL_GUESS = {
    'R': 2.0,  # Start with circle
}

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
    """
    Compute perimeter of star-shaped boundary.
    
    Perimeter = ∫ √(r² + (dr/dθ)²) dθ
    """
    theta = np.linspace(0, 2 * np.pi, num_samples)
    dtheta = theta[1] - theta[0]
    
    M = (len(c) - 1) // 2
    
    # Compute r(θ)
    r = compute_radius(theta, c)
    
    # Compute dr/dθ analytically
    dr_dtheta = np.zeros_like(theta)
    for m in range(1, M + 1):
        dr_dtheta += m * (-c[m] * np.sin(m * theta) + c[m + M] * np.cos(m * theta))
    
    # Perimeter element: √(r² + (dr/dθ)²)
    ds = np.sqrt(r**2 + dr_dtheta**2)
    
    # Integrate using trapezoidal rule
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
    """Compute total kinetic energy (sum of v²/2)."""
    return 0.5 * np.sum(velocities**2)

def particles_in_square(positions, a):
    """Find particles inside square [-a, a] × [-a, a]."""
    mask = (np.abs(positions[:, 0]) <= a) & (np.abs(positions[:, 1]) <= a)
    indices = np.where(mask)[0]
    return mask, indices

#==============================================================================
# CONSTRAINT FUNCTIONS
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
    """
    Constraint: preserve perimeter (material cost).
    
    |Perimeter(c) - Perimeter_target| <= tolerance * Perimeter_target
    """
    perimeter_target = config['perimeter_target']
    tolerance = config['perimeter_tolerance']
    
    perimeter = compute_perimeter(c)
    rel_deviation = np.abs(perimeter - perimeter_target) / perimeter_target
    
    # Constraint: tolerance - rel_deviation >= 0
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
    from scipy.optimize import NonlinearConstraint
    
    K = config['K_angles']
    theta_samples = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    constraints = []
    
    # 1. Square inscribed
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: square_inscribed_constraint(c, config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    # 2. Bounding box
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: box_constraint(c, config, theta_samples),
            lb=0.0,
            ub=np.inf
        )
    )
    
    # 3. PERIMETER preservation (material cost)
    constraints.append(
        NonlinearConstraint(
            fun=lambda c: perimeter_constraint(c, config),
            lb=0.0,
            ub=np.inf
        )
    )
    
    # 4. Positivity
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
        # Run DSMC simulation
        positions, velocities, _, _, _ = Arbitrary_Shape_Parameterized(
            N=sim_params['N'],
            fourier_coefficients=c,
            num_boundary_points=sim_params['num_boundary_points'],
            T_x0=sim_params['T_x0'],
            T_y0=sim_params['T_y0'],
            dt=sim_params['dt'],
            n_tot=sim_params['n_tot'],
            e=sim_params['e'],
            mu=sim_params['mu'],
            alpha=sim_params['alpha'],
            mesh_size=sim_params['mesh_size']
        )
        
        # Find particles in square
        mask, indices = particles_in_square(positions, a)
        
        if len(indices) == 0:
            return 1e10  # Penalty
        
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
        print(f"ERROR in simulation: {str(e)}")
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
# RUN SINGLE OPTIMIZATION
#==============================================================================

def run_single_optimization(trial_num, config, sim_params, seed):
    """Run a single optimization trial."""
    print(f"\n{'='*70}")
    print(f"TRIAL {trial_num} - Seed: {seed}")
    print(f"{'='*70}")
    
    M = config['M']
    
    # Initial guess: circle
    c0 = np.zeros(2 * M + 1)
    c0[0] = INITIAL_GUESS['R']
    
    # Set perimeter target from initial guess
    initial_perimeter = compute_perimeter(c0)
    config['perimeter_target'] = initial_perimeter
    
    print(f"Initial perimeter: {initial_perimeter:.4f}")
    print(f"Perimeter constraint: {initial_perimeter:.4f} ± {config['perimeter_tolerance']*100}%")
    
    # Setup
    bounds = get_parameter_bounds(M)
    constraints = build_constraints(config)
    
    # Track results
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
            disp=True,
            polish=False
        )
        
        results['best_coefficients'] = result.x
        results['best_objective'] = result.fun
        results['success'] = result.success
        results['message'] = result.message
        results['iterations'] = result.nit
        
    except Exception as e:
        results['message'] = f"ERROR: {str(e)}"
        print(f"Optimization failed: {str(e)}")
    
    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']
    
    print(f"\nTrial {trial_num} Complete:")
    print(f"  Objective: {results['best_objective']:.6f}")
    print(f"  Success: {results['success']}")
    print(f"  Duration: {results['duration']/60:.2f} minutes")
    
    return results

#==============================================================================
# VISUALIZATION
#==============================================================================

def visualize_trial_result(result, config, sim_params, save_path=None):
    """Visualize a single trial result."""
    c = result['best_coefficients']
    if c is None:
        print(f"Trial {result['trial']} failed, skipping visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Shape
    ax1 = axes[0]
    theta = np.linspace(0, 2*np.pi, 500)
    r = compute_radius(theta, c)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax1.plot(x, y, 'b-', linewidth=2, label='Optimized Shape')
    
    # Plot inscribed square
    a = config['a']
    square = plt.Rectangle((-a, -a), 2*a, 2*a, 
                          fill=False, edgecolor='red', linewidth=2, 
                          linestyle='--', label='Target Square')
    ax1.add_patch(square)
    
    # Plot bounding box
    L = config['L']
    box = plt.Rectangle((-L, -L), 2*L, 2*L,
                       fill=False, edgecolor='gray', linewidth=1,
                       linestyle=':', label='Bounding Box')
    ax1.add_patch(box)
    
    ax1.set_xlim(-L-0.5, L+0.5)
    ax1.set_ylim(-L-0.5, L+0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title(f'Trial {result["trial"]} - Optimized Shape\nSeed: {result["seed"]}', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot 2: Coefficients
    ax2 = axes[1]
    M = (len(c) - 1) // 2
    
    indices = np.arange(len(c))
    labels = ['c₀'] + [f'c{i}' for i in range(1, M+1)] + [f's{i}' for i in range(1, M+1)]
    
    colors = ['red'] + ['blue']*(M) + ['green']*(M)
    ax2.bar(indices, c, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title(f'Fourier Coefficients\nObjective: {result["best_objective"]:.6f}',
                  fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', linewidth=0.5)
    
    # Add text info
    perimeter = compute_perimeter(c)
    reg = compute_regularization(c, config['lambda_reg'])
    textstr = f"Objective: {result['best_objective']:.6f}\n"
    textstr += f"Perimeter: {perimeter:.4f}\n"
    textstr += f"Regularization: {reg:.6f}\n"
    textstr += f"Iterations: {result['iterations']}\n"
    textstr += f"Time: {result['duration']/60:.2f} min"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return fig

def compare_all_trials(all_results, config, save_path=None):
    """Compare all trial results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    trials = [r['trial'] for r in all_results if r['success']]
    objectives = [r['best_objective'] for r in all_results if r['success']]
    durations = [r['duration']/60 for r in all_results if r['success']]
    
    if len(trials) == 0:
        print("No successful trials to compare!")
        return
    
    # Plot 1: All shapes overlaid
    ax1 = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 500)
    
    for i, result in enumerate(all_results):
        if result['success'] and result['best_coefficients'] is not None:
            c = result['best_coefficients']
            r = compute_radius(theta, c)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax1.plot(x, y, linewidth=2, alpha=0.7, 
                    label=f"Trial {result['trial']} (obj={result['best_objective']:.4f})")
    
    # Add square
    a = config['a']
    square = plt.Rectangle((-a, -a), 2*a, 2*a,
                          fill=False, edgecolor='red', linewidth=2,
                          linestyle='--', label='Target Square')
    ax1.add_patch(square)
    
    L = config['L']
    ax1.set_xlim(-L-0.5, L+0.5)
    ax1.set_ylim(-L-0.5, L+0.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title('All Optimized Shapes Overlaid', fontweight='bold', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot 2: Objective values
    ax2 = axes[0, 1]
    ax2.bar(trials, objectives, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Final Objective Value')
    ax2.set_title('Objective Values Across Trials', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_obj = np.mean(objectives)
    std_obj = np.std(objectives)
    ax2.axhline(mean_obj, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_obj:.4f}')
    ax2.axhline(mean_obj + std_obj, color='orange', linestyle=':', linewidth=1.5, label=f'±σ: {std_obj:.4f}')
    ax2.axhline(mean_obj - std_obj, color='orange', linestyle=':', linewidth=1.5)
    ax2.legend()
    
    # Plot 3: Coefficient comparison
    ax3 = axes[1, 0]
    M = config['M']
    n_params = 2 * M + 1
    indices = np.arange(n_params)
    width = 0.8 / len(trials)
    
    for i, result in enumerate(all_results):
        if result['success'] and result['best_coefficients'] is not None:
            offset = (i - len(trials)/2) * width
            ax3.bar(indices + offset, result['best_coefficients'], width,
                   alpha=0.7, label=f"Trial {result['trial']}")
    
    labels = ['c₀'] + [f'c{i}' for i in range(1, M+1)] + [f's{i}' for i in range(1, M+1)]
    ax3.set_xticks(indices)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Coefficient Comparison', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(0, color='black', linewidth=0.5)
    
    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute statistics
    stats_text = "CONVERGENCE ANALYSIS\n" + "="*50 + "\n\n"
    stats_text += f"Number of trials: {len(all_results)}\n"
    stats_text += f"Successful trials: {len(trials)}\n"
    stats_text += f"Failed trials: {len(all_results) - len(trials)}\n\n"
    
    if len(objectives) > 1:
        stats_text += "OBJECTIVE VALUES:\n"
        stats_text += f"  Mean: {mean_obj:.6f}\n"
        stats_text += f"  Std Dev: {std_obj:.6f}\n"
        stats_text += f"  Min: {np.min(objectives):.6f}\n"
        stats_text += f"  Max: {np.max(objectives):.6f}\n"
        stats_text += f"  Coefficient of Variation: {(std_obj/mean_obj)*100:.2f}%\n\n"
        
        stats_text += "CONSISTENCY CHECK:\n"
        cv = (std_obj / mean_obj) * 100
        if cv < 1:
            stats_text += "  ✓ EXCELLENT - Very consistent results\n"
        elif cv < 5:
            stats_text += "  ✓ GOOD - Reasonably consistent\n"
        elif cv < 10:
            stats_text += "  ⚠ MODERATE - Some variation\n"
        else:
            stats_text += "  ✗ POOR - High variation between runs\n"
        
        stats_text += f"\n\nAVERAGE TIME PER TRIAL:\n"
        stats_text += f"  {np.mean(durations):.2f} ± {np.std(durations):.2f} minutes\n"
        
        # Pairwise differences
        if len(objectives) >= 2:
            diffs = []
            for i in range(len(objectives)):
                for j in range(i+1, len(objectives)):
                    diffs.append(abs(objectives[i] - objectives[j]))
            max_diff = np.max(diffs)
            stats_text += f"\nMAX PAIRWISE DIFFERENCE:\n"
            stats_text += f"  {max_diff:.6f} ({(max_diff/mean_obj)*100:.2f}%)\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison: {save_path}")
    
    plt.show()
    
    return fig

#==============================================================================
# MAIN EXECUTION
#==============================================================================

def run_convergence_test(num_trials=5, output_dir='convergence_test_results'):
    """Run multiple optimization trials and analyze convergence."""
    
    print("="*70)
    print("OPTIMIZATION CONVERGENCE TEST")
    print("="*70)
    print(f"Number of trials: {num_trials}")
    print(f"Fourier modes (M): {TEST_CONFIG['M']}")
    print(f"Parameters: {2*TEST_CONFIG['M']+1}")
    print(f"Max iterations: {TEST_CONFIG['maxiter']}")
    print(f"Population size: {TEST_CONFIG['popsize'] * (2*TEST_CONFIG['M']+1)}")
    print(f"Constraint: PERIMETER (material cost)")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run trials with different seeds
    all_results = []
    # Generate diverse seeds for 30 trials
    base_seeds = [42, 123, 456, 789, 1011, 2048, 3141, 5926, 8192, 10000,
                  12345, 15000, 17777, 20000, 22222, 25000, 27182, 30000,
                  33333, 36000, 38888, 41000, 43210, 45678, 48000, 50000,
                  52525, 55555, 58008, 60000]
    seeds = base_seeds[:num_trials]  # Use as many as needed
    
    for i, seed in enumerate(seeds, 1):
        result = run_single_optimization(i, TEST_CONFIG.copy(), SIM_CONFIG, seed)
        all_results.append(result)
        
        # Visualize individual trial
        if result['success']:
            save_path = os.path.join(output_dir, f'trial_{i}_seed_{seed}.png')
            visualize_trial_result(result, TEST_CONFIG, SIM_CONFIG, save_path)
    
    # Compare all trials
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON PLOTS")
    print(f"{'='*70}")
    
    comparison_path = os.path.join(output_dir, 'all_trials_comparison.png')
    compare_all_trials(all_results, TEST_CONFIG, comparison_path)
    
    # Save results to file
    results_file = os.path.join(output_dir, 'results_summary.txt')
    with open(results_file, 'w') as f:
        f.write("OPTIMIZATION CONVERGENCE TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for result in all_results:
            f.write(f"Trial {result['trial']} (Seed: {result['seed']}):\n")
            f.write(f"  Success: {result['success']}\n")
            f.write(f"  Objective: {result['best_objective']:.8f}\n")
            f.write(f"  Iterations: {result['iterations']}\n")
            f.write(f"  Duration: {result['duration']/60:.2f} minutes\n")
            if result['best_coefficients'] is not None:
                f.write(f"  Coefficients: {result['best_coefficients']}\n")
            f.write(f"  Message: {result['message']}\n")
            f.write("\n")
        
        # Statistics
        successful = [r for r in all_results if r['success']]
        if len(successful) > 0:
            objs = [r['best_objective'] for r in successful]
            f.write("\nSTATISTICS:\n")
            f.write(f"  Mean objective: {np.mean(objs):.8f}\n")
            f.write(f"  Std deviation: {np.std(objs):.8f}\n")
            f.write(f"  Min objective: {np.min(objs):.8f}\n")
            f.write(f"  Max objective: {np.max(objs):.8f}\n")
            f.write(f"  Coefficient of variation: {(np.std(objs)/np.mean(objs))*100:.2f}%\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {results_file}")
    
    return all_results

#==============================================================================
# JUPYTER NOTEBOOK FRIENDLY INTERFACE
#==============================================================================

if __name__ == "__main__":
    # For running as script
    results = run_convergence_test(
        num_trials=TEST_CONFIG['num_trials'],
        output_dir='convergence_test_results'
    )
else:
    # For Jupyter notebook - just define functions
    print("Convergence test module loaded!")
    print("Run: results = run_convergence_test(num_trials=5)")
