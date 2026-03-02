"""
Quick Convergence Test Script
=============================

Simplified version for testing on GPU via Jupyter notebook.
Upload this file and run it in a notebook cell.

Key features:
- PERIMETER constraint (material cost)
- Multiple optimization runs
- Consistency analysis
- Visual comparison
"""

# To use in Jupyter notebook:
# 1. Upload this file to your notebook environment
# 2. Run: exec(open('quick_convergence_test.py').read())
# 3. Run: results = run_quick_test(num_trials=3)

import sys
import os

# UPDATE THESE PATHS FOR YOUR ENVIRONMENT
sys.path.append('/content/src')  # For Google Colab
sys.path.append('/content/src/2d/Arbitrary Shape')

# Import the full test module
from test_optimization_convergence import *

def run_quick_test(num_trials=3, output_dir='test_results'):
    """
    Quick convergence test with reduced settings.
    
    Parameters
    ----------
    num_trials : int
        Number of optimization runs (default: 3)
    output_dir : str
        Directory to save results
    """
    
    # Override with faster settings for quick testing
    quick_config = TEST_CONFIG.copy()
    quick_config['maxiter'] = 20      # Fewer iterations
    quick_config['popsize'] = 8       # Smaller population
    quick_config['M'] = 2             # Fewer Fourier modes (5 params)
    quick_config['num_trials'] = num_trials
    
    quick_sim = SIM_CONFIG.copy()
    quick_sim['N'] = 1500             # Fewer particles
    quick_sim['n_tot'] = 40           # Fewer timesteps
    quick_sim['mesh_size'] = 0.35     # Coarser mesh
    
    print("="*70)
    print("QUICK CONVERGENCE TEST")
    print("="*70)
    print(f"Trials: {num_trials}")
    print(f"Fourier modes: {quick_config['M']} (params: {2*quick_config['M']+1})")
    print(f"Max iterations: {quick_config['maxiter']}")
    print(f"Population: {quick_config['popsize'] * (2*quick_config['M']+1)}")
    print(f"Particles: {quick_sim['N']}")
    print(f"Timesteps: {quick_sim['n_tot']}")
    print(f"Estimated time: {num_trials * 2}-{num_trials * 5} minutes")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run trials
    all_results = []
    # Generate diverse seeds (supports up to 30 trials)
    base_seeds = [42, 123, 456, 789, 1011, 2048, 3141, 5926, 8192, 10000,
                  12345, 15000, 17777, 20000, 22222, 25000, 27182, 30000,
                  33333, 36000, 38888, 41000, 43210, 45678, 48000, 50000,
                  52525, 55555, 58008, 60000]
    seeds = base_seeds[:num_trials]
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"TRIAL {i}/{num_trials} - Seed: {seed}")
        print(f"{'='*70}")
        
        result = run_single_optimization(i, quick_config, quick_sim, seed)
        all_results.append(result)
        
        # Quick visualization
        if result['success']:
            save_path = os.path.join(output_dir, f'trial_{i}.png')
            visualize_trial_result(result, quick_config, quick_sim, save_path)
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    successful = [r for r in all_results if r['success']]
    
    if len(successful) > 0:
        objectives = [r['best_objective'] for r in successful]
        
        print(f"\nSuccessful: {len(successful)}/{len(all_results)}")
        print(f"\nObjective values:")
        for r in successful:
            print(f"  Trial {r['trial']}: {r['best_objective']:.8f}")
        
        if len(objectives) > 1:
            mean_obj = np.mean(objectives)
            std_obj = np.std(objectives)
            cv = (std_obj / mean_obj) * 100
            
            print(f"\nStatistics:")
            print(f"  Mean: {mean_obj:.8f}")
            print(f"  Std:  {std_obj:.8f}")
            print(f"  CV:   {cv:.2f}%")
            
            print(f"\nConsistency:")
            if cv < 1:
                print("  ✓ EXCELLENT")
            elif cv < 5:
                print("  ✓ GOOD")
            elif cv < 10:
                print("  ⚠ MODERATE")
            else:
                print("  ✗ POOR - Need more iterations")
        
        # Comparison plot
        comp_path = os.path.join(output_dir, 'comparison.png')
        compare_all_trials(successful, quick_config, comp_path)
    else:
        print("\nNo successful trials!")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")
    
    return all_results

# Auto-run if executed
if __name__ == "__main__":
    results = run_quick_test(num_trials=3)
