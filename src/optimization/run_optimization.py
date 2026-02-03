"""
Main optimization script for star-shaped boundary optimization.

This script orchestrates the complete optimization workflow:
1. Initialize configuration
2. Set up initial guess and constraints
3. Run global optimization (differential_evolution)
4. Track and visualize progress
5. Generate final report and animation
"""

import numpy as np
import scipy.optimize
import sys
import os
from datetime import datetime

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import optimization modules
from optimization.config import (
    OPTIMIZATION_CONFIG,
    SIMULATION_CONFIG,
    INITIAL_GUESS,
    get_parameter_bounds,
    print_config_summary
)
from optimization.shape_optimizer import (
    objective_function,
    compute_area,
    evaluate_with_details
)
from optimization.constraints import (
    build_constraints,
    check_all_constraints
)
from optimization.viz_optimizer import (
    OptimizationTracker,
    visualize_iteration,
    create_optimization_animation
)


def run_optimization(opt_config=None, sim_params=None, 
                    initial_guess=None, max_iterations=None,
                    verbose=True):
    """
    Run the complete optimization workflow.
    
    Parameters
    ----------
    opt_config : dict, optional
        Optimization configuration (uses OPTIMIZATION_CONFIG if None)
    sim_params : dict, optional
        Simulation parameters (uses SIMULATION_CONFIG if None)
    initial_guess : dict, optional
        Initial guess configuration (uses INITIAL_GUESS if None)
    max_iterations : int, optional
        Override maximum iterations
    verbose : bool, optional
        Print detailed information (default: True)
        
    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Optimization result object
    tracker : OptimizationTracker
        Tracker with optimization history
    """
    
    # ========== STEP 1: Configuration ==========
    if verbose:
        print("\n" + "=" * 70)
        print("STAR-SHAPE BOUNDARY OPTIMIZATION")
        print("=" * 70)
        print(f"Start time: {datetime.now()}")
        print()
    
    # Use default configs if not provided
    if opt_config is None:
        opt_config = OPTIMIZATION_CONFIG.copy()
    if sim_params is None:
        sim_params = SIMULATION_CONFIG.copy()
    if initial_guess is None:
        initial_guess = INITIAL_GUESS.copy()
    
    # Override max iterations if specified
    if max_iterations is not None:
        opt_config['maxiter'] = max_iterations
    
    # Print configuration
    if verbose:
        print_config_summary()
        print()
    
    # ========== STEP 2: Initial Guess ==========
    M = opt_config['M']
    R = initial_guess['R']
    
    # Create initial coefficients: circle of radius R
    c0 = np.zeros(2 * M + 1)
    c0[0] = R  # Base radius
    
    if verbose:
        print("=" * 70)
        print("INITIAL GUESS")
        print("=" * 70)
        print(f"Initial shape: Circle with radius R = {R}")
        print(f"Initial coefficients: {c0}")
    
    # Compute initial area for constraint
    initial_area = compute_area(c0)
    opt_config['area_target'] = initial_area
    
    if verbose:
        print(f"Initial area: {initial_area:.4f}")
        print(f"Area constraint: {initial_area:.4f} ± {opt_config['area_tolerance']*100}%")
    
    # Check initial feasibility
    if verbose:
        print("\nChecking initial feasibility...")
        feasible, violations = check_all_constraints(c0, opt_config, verbose=True)
        if not feasible:
            print("\nWARNING: Initial guess is infeasible!")
            print("Consider adjusting initial radius or constraint parameters.")
        print()
    
    # ========== STEP 3: Set up Optimization ==========
    bounds = get_parameter_bounds(M)
    constraints = build_constraints(opt_config)
    
    if verbose:
        print("=" * 70)
        print("OPTIMIZATION SETUP")
        print("=" * 70)
        print(f"Optimizer: scipy.optimize.differential_evolution")
        print(f"Number of parameters: {len(c0)}")
        print(f"Number of constraints: {len(constraints)}")
        print(f"Max iterations: {opt_config['maxiter']}")
        print(f"Population size: {opt_config['popsize']}")
        print(f"Workers: {opt_config['workers']}")
        print()
    
    # ========== STEP 4: Initialize Tracker ==========
    tracker = OptimizationTracker(output_dir=opt_config['output_dir'])
    
    # Evaluation counter
    eval_count = [0]  # Use list to allow modification in nested function
    
    # ========== STEP 5: Define Callback ==========
    def callback(xk, convergence=None):
        """
        Callback function called after each iteration.
        
        Parameters
        ----------
        xk : ndarray
            Current parameter vector
        convergence : float, optional
            Convergence metric (provided by optimizer)
        """
        eval_count[0] += 1
        
        # Evaluate objective
        obj_val = objective_function(xk, sim_params, opt_config, verbose=False)
        
        # Check constraints
        _, constraint_vals = check_all_constraints(xk, opt_config, verbose=False)
        
        # Update tracker
        tracker.update(xk, obj_val, constraint_vals)
        
        # Print progress
        if verbose:
            print(f"\nIteration {tracker.iteration_count}:")
            print(f"  Objective: {obj_val:.6f}")
            print(f"  Best so far: {tracker.best_value:.6f}")
            if convergence is not None:
                print(f"  Convergence: {convergence:.6e}")
        
        # Visualize periodically
        if tracker.iteration_count % opt_config['viz_interval'] == 0:
            if verbose:
                print(f"  Creating visualization...")
            
            # Get detailed results
            results = evaluate_with_details(xk, sim_params, opt_config)
            
            if results is not None:
                visualize_iteration(xk, results, opt_config, 
                                  tracker.iteration_count, tracker)
                tracker.plot_progress(save=True)
            else:
                print(f"  WARNING: Could not create visualization (simulation failed)")
        
        return False  # Continue optimization
    
    # ========== STEP 6: Run Optimization ==========
    if verbose:
        print("=" * 70)
        print("RUNNING OPTIMIZATION")
        print("=" * 70)
        print("This may take a while...")
        print()
    
    # Define objective wrapper for scipy
    def objective_wrapper(c):
        """Wrapper to track evaluations and handle errors."""
        try:
            return objective_function(c, sim_params, opt_config, verbose=False)
        except Exception as e:
            print(f"ERROR in objective evaluation: {str(e)}")
            return 1e10  # Return high penalty on error
    
    # Run differential evolution
    result = scipy.optimize.differential_evolution(
        func=objective_wrapper,
        bounds=bounds,
        constraints=constraints,
        strategy=opt_config['strategy'],
        maxiter=opt_config['maxiter'],
        popsize=opt_config['popsize'],
        tol=opt_config['tol'],
        atol=opt_config['atol'],
        mutation=opt_config['mutation'],
        recombination=opt_config['recombination'],
        seed=opt_config['seed'],
        callback=callback,
        disp=verbose,
        polish=True,  # Local refinement at the end
        init='latinhypercube',  # Good initial population distribution
        workers=opt_config['workers'],
        updating='deferred' if opt_config['workers'] != 1 else 'immediate'
    )
    
    # ========== STEP 7: Post-Processing ==========
    if verbose:
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Total iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Best objective: {result.fun:.8f}")
        print(f"\nOptimal coefficients:")
        for i, c in enumerate(result.x):
            print(f"  c[{i}] = {c:12.8f}")
        print()
    
    # Check final constraints
    if verbose:
        print("Final constraint check:")
        feasible, _ = check_all_constraints(result.x, opt_config, verbose=True)
        print()
    
    # ========== STEP 8: Final Visualization ==========
    if verbose:
        print("Creating final visualizations...")
    
    # Run final detailed evaluation
    final_results = evaluate_with_details(result.x, sim_params, opt_config)
    
    if final_results is not None:
        visualize_iteration(result.x, final_results, opt_config, 
                          tracker.iteration_count + 1, tracker)
        tracker.plot_progress(save=True)
        
        if verbose:
            print(f"\nFinal Results:")
            print(f"  Total KE: {final_results['ke_total']:.6f}")
            print(f"  KE in square: {final_results['ke_in_square']:.6f}")
            print(f"  KE per particle: {final_results['ke_per_particle']:.6f}")
            print(f"  Regularization: {final_results['regularization']:.6f}")
            print(f"  Area: {final_results['area']:.4f}")
            print(f"  Particles in square: {final_results['num_particles_in_square']} / {final_results['num_particles_total']}")
    
    # ========== STEP 9: Save Summary ==========
    tracker.save_summary()
    
    # ========== STEP 10: Create Animation (if frames saved) ==========
    if opt_config['save_frames'] and len(os.listdir(tracker.frames_dir)) > 0:
        if verbose:
            print("\nCreating optimization animation...")
        try:
            create_optimization_animation(tracker, 'optimization_evolution.mp4')
        except Exception as e:
            print(f"  Could not create animation: {str(e)}")
            print("  (Install imageio and imageio-ffmpeg to enable animations)")
    
    # ========== STEP 11: Final Summary ==========
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Output directory: {tracker.output_dir}")
        print(f"Frames saved: {len(os.listdir(tracker.frames_dir))}")
        print(f"End time: {datetime.now()}")
        print("=" * 70)
        print()
    
    return result, tracker


def main():
    """Main entry point for the optimization script."""
    
    # Parse command line arguments (simple version)
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize star-shaped boundary to minimize heat in inscribed square'
    )
    parser.add_argument('--maxiter', type=int, default=None,
                       help='Maximum optimization iterations (default: from config)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--output-dir', type=str, default='optimization_results',
                       help='Output directory (default: optimization_results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--test', action='store_true',
                       help='Run short test (10 iterations)')
    
    args = parser.parse_args()
    
    # Modify config based on args
    opt_config = OPTIMIZATION_CONFIG.copy()
    sim_params = SIMULATION_CONFIG.copy()
    
    if args.workers:
        opt_config['workers'] = args.workers
    if args.output_dir:
        opt_config['output_dir'] = args.output_dir
    if args.test:
        opt_config['maxiter'] = 10
        print("\n*** TEST MODE: Running 10 iterations ***\n")
    
    # Run optimization
    result, tracker = run_optimization(
        opt_config=opt_config,
        sim_params=sim_params,
        max_iterations=args.maxiter,
        verbose=not args.quiet
    )
    
    print("\n✓ Optimization complete!")
    print(f"  Results saved to: {tracker.output_dir}")
    print(f"  Best objective: {result.fun:.6f}")
    
    return result, tracker


if __name__ == "__main__":
    result, tracker = main()
