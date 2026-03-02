# Optimization Convergence Testing

This package tests whether the optimization architecture finds consistent, optimal solutions across multiple runs.

## Purpose

Test three key properties:
1. **Convergence** - Does it find a minimum?
2. **Consistency** - Do multiple runs give similar results?
3. **Stability** - How sensitive to random initialization?

## Key Changes from Original

1. **PERIMETER constraint** instead of area (represents material cost)
2. **Simplified configuration** for faster testing
3. **Multiple trials** with different random seeds
4. **Comprehensive visualization** and statistical analysis

---

## Files

### 1. `test_optimization_convergence.py`
**Full-featured testing module**
- Contains all functions for convergence testing
- Can run multiple optimization trials
- Generates visualizations and statistics
- ~800 lines, self-contained

### 2. `quick_convergence_test.py`
**Lightweight wrapper for Jupyter**
- Simplified interface
- Faster settings for quick tests
- Easy to use in notebooks
- Imports from main test file

### 3. `test_optimization.ipynb`
**Interactive Jupyter notebook**
- Step-by-step testing
- Cell-by-cell execution
- Analysis and visualization
- Good for exploration

---

## Usage

### Option A: Direct Python Execution (Local)

```bash
# Run full test locally
cd /path/to/project
python test_optimization_convergence.py
```

This will:
- Run 5 optimization trials
- Save results to `convergence_test_results/`
- Generate comparison plots
- Create summary report

---

### Option B: Jupyter Notebook (GPU/Remote)

#### Step 1: Upload Files

Upload to your Jupyter environment:
- `test_optimization_convergence.py` (main module)
- `quick_convergence_test.py` (wrapper)
- Your entire `src/` directory

#### Step 2: Set Paths

Edit `test_optimization_convergence.py` lines 31-32:

```python
sys.path.insert(0, '/path/to/src')  # Update!
sys.path.insert(0, '/path/to/src/2d/Arbitrary Shape')  # Update!
```

For Google Colab:
```python
sys.path.insert(0, '/content/src')
sys.path.insert(0, '/content/src/2d/Arbitrary Shape')
```

#### Step 3: Run in Notebook

**Quick Test (3 trials, ~10 minutes):**
```python
exec(open('quick_convergence_test.py').read())
results = run_quick_test(num_trials=3)
```

**Full Test (5 trials, ~30 minutes):**
```python
from test_optimization_convergence import *
results = run_convergence_test(num_trials=5)
```

---

## Configuration

### Test Settings (in `TEST_CONFIG`)

```python
# Test parameters
'num_trials': 30,          # Number of optimization runs
'num_workers': -1,         # Use all CPU cores (32 threads!)

# Fourier parameterization
'M': 2,                    # Fewer modes for speed (default: 4)
                           # Gives 2M+1 = 5 parameters

# Optimizer settings (reduced for testing)
'maxiter': 30,             # Max generations (default: 50)
'popsize': 10,             # Population multiplier (default: 15)

# PERIMETER constraint
'perimeter_tolerance': 0.05,  # 5% tolerance (like area)
'perimeter_target': None,      # Computed from initial guess
```

### Simulation Settings (in `SIM_CONFIG`)

```python
# Lightweight for testing
'N': 2000,                 # Particles (default: 3000)
'n_tot': 50,               # Timesteps (default: 100)
'mesh_size': 0.3,          # Coarser mesh (default: 0.25)
```

### Adjust for Your Hardware

**Current Configuration (RTX 3080, Ryzen 16c/32t):**
```python
TEST_CONFIG['num_trials'] = 30   # 30 trials for robust statistics
TEST_CONFIG['num_workers'] = -1  # Use all 32 threads!
TEST_CONFIG['M'] = 2             # 5 parameters (faster)
TEST_CONFIG['maxiter'] = 30      # 30 generations
TEST_CONFIG['popsize'] = 10      # Population: 10×5 = 50 shapes

# Expected time: ~2-3 hours for 30 trials
# Population evaluations: ~1,500 per trial
# Total simulations: ~45,000 DSMC runs (parallelized)
```

**Faster (for quick preliminary tests):**
```python
TEST_CONFIG['num_trials'] = 5
TEST_CONFIG['M'] = 2           # 5 parameters
TEST_CONFIG['maxiter'] = 20    # Fewer iterations
TEST_CONFIG['popsize'] = 8     # Smaller population
SIM_CONFIG['N'] = 1500         # Fewer particles
SIM_CONFIG['n_tot'] = 40       # Fewer timesteps
# Expected time: ~20-30 minutes
```

**Slower (for maximum thoroughness):**
```python
TEST_CONFIG['num_trials'] = 30
TEST_CONFIG['M'] = 3           # 7 parameters
TEST_CONFIG['maxiter'] = 50    # More iterations
TEST_CONFIG['popsize'] = 15    # Larger population
SIM_CONFIG['N'] = 3000         # More particles
SIM_CONFIG['n_tot'] = 100      # More timesteps
# Expected time: ~5-8 hours
```

---

## Interpreting Results

### Coefficient of Variation (CV)

The CV measures consistency across trials:

```
CV = (std_deviation / mean) × 100%
```

**Interpretation:**
- **CV < 1%**: ✓ **EXCELLENT** - Very consistent, algorithm highly reliable
- **CV < 5%**: ✓ **GOOD** - Reasonably consistent, publication-quality
- **CV < 10%**: ⚠ **MODERATE** - Some variation, consider more iterations
- **CV > 10%**: ✗ **POOR** - High variation, increase `maxiter` or `popsize`

**With 30 trials**, you get:
- 📊 Very robust statistics (narrow confidence intervals)
- 📈 Reliable detection of outliers
- 🎯 Can compute percentiles (10th, 25th, 50th, 75th, 90th)
- 📉 Distribution analysis (normal vs. skewed)
- ✅ Publication-quality results

### Visual Checks

1. **Shape Overlay Plot**
   - Shapes should be similar if consistent
   - Some variation is normal
   - Wildly different shapes → problem!

2. **Objective Bar Chart**
   - Bars should be close in height
   - All within mean ± 1 std dev is good

3. **Coefficient Comparison**
   - Error bars should be small
   - Similar patterns across trials

### What to Look For

**Good signs:**
- ✓ All trials succeed
- ✓ Objectives within 5% of each other
- ✓ Shapes visually similar
- ✓ Small error bars on coefficients

**Bad signs:**
- ✗ Some trials fail
- ✗ Objectives vary by >10%
- ✗ Shapes look very different
- ✗ Large error bars on coefficients

---

## Troubleshooting

### All Trials Fail

**Problem**: No successful optimizations

**Solutions:**
1. Check constraint feasibility - initial guess may violate constraints
2. Increase perimeter tolerance: `'perimeter_tolerance': 0.1`
3. Check simulation parameters - mesh might be too coarse
4. Verify paths are correct in imports

### High Variation (CV > 10%)

**Problem**: Results not consistent

**Solutions:**
1. Increase iterations: `'maxiter': 50` or `100`
2. Increase population: `'popsize': 15` or `20`
3. Try different DE strategy: `'strategy': 'best2bin'`
4. Reduce noise: Increase `n_tot` in simulation

### Runs Too Slow

**Problem**: Each trial takes > 30 minutes

**Solutions:**
1. Reduce Fourier modes: `'M': 2`
2. Reduce population: `'popsize': 8`
3. Reduce particles: `'N': 1500`
4. Reduce timesteps: `'n_tot': 40`
5. Coarser mesh: `'mesh_size': 0.4`
6. Enable parallelization: `'num_workers': -1` (if CPU)

### Out of Memory

**Problem**: GPU/RAM exhausted

**Solutions:**
1. Reduce particles: `'N': 1000`
2. Reduce population: `'popsize': 5`
3. Run trials sequentially (not in parallel)
4. Use CPU: `'num_workers': 1`

---

## Expected Output

After running, you should see:

### In Console:
```
======================================================================
TRIAL 1 - Seed: 42
======================================================================
Initial perimeter: 12.5664
Perimeter constraint: 12.5664 ± 5.0%
...
Trial 1 Complete:
  Objective: 0.428516
  Success: True
  Duration: 8.23 minutes
```

### Files Created:
```
convergence_test_results/
├── trial_1_seed_42.png          # Individual trial visualizations
├── trial_2_seed_123.png
├── trial_3_seed_456.png
├── ...
├── all_trials_comparison.png    # Comprehensive comparison
└── results_summary.txt          # Text summary with statistics
```

### Visualizations:

Each trial plot shows:
1. Optimized shape with constraints
2. Fourier coefficients
3. Objective and metrics

Comparison plot shows:
1. All shapes overlaid
2. Objective bar chart with statistics
3. Coefficient comparison across trials
4. Summary statistics and assessment

---

## Next Steps

After testing:

1. **If consistent (CV < 5%)**: 
   - ✓ Architecture works!
   - Can trust results
   - Scale up for production runs

2. **If moderate (CV 5-10%)**:
   - Increase iterations for better convergence
   - Still usable but may need tuning

3. **If poor (CV > 10%)**:
   - Investigate problem setup
   - Check constraints
   - Try different optimization strategy

---

## Questions to Answer

After running tests, you should know:

1. ✓ Does the algorithm find a minimum?
   - Check if objectives are reasonable
   - Compare to initial guess

2. ✓ Is it consistent across runs?
   - Look at CV percentage
   - Visual inspection of shapes

3. ✓ How long does it take?
   - Use to estimate full optimization runtime
   - Adjust settings for speed/quality trade-off

4. ✓ Is the architecture sound?
   - Can proceed with confidence
   - Or needs more tuning

---

## Support

For issues:
1. Check paths in import statements
2. Verify all `src/` files are uploaded
3. Check Python version (3.7+ required)
4. Verify scipy, numpy, matplotlib installed
5. Try simplified settings first

Good luck with testing! 🚀
