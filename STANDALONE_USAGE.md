# Standalone Convergence Test - Quick Start

## 🎯 What You Need

**Just ONE file:**
- `standalone_convergence_test.py`

That's it! No other dependencies on the project structure.

---

## 📤 Upload to Jupyter

### For Google Colab:
1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook** or use upload button
3. Upload `standalone_convergence_test.py`

### For Local Jupyter:
1. Start Jupyter: `jupyter notebook`
2. Navigate to notebook interface
3. Upload `standalone_convergence_test.py` using Upload button

### For Remote GPU (ramsey/fibonacci):
```bash
# Option 1: SCP from local machine
scp standalone_convergence_test.py hec66@ramsey:~/

# Option 2: Use Jupyter upload interface
# Start Jupyter on remote machine, then upload via browser
```

---

## 🚀 How to Run

### Method 1: In Jupyter Notebook Cell

Create a new notebook and run:

```python
# Cell 1: Load the module
exec(open('standalone_convergence_test.py').read())

# Cell 2: Run the test
results = run_convergence_test(num_trials=30)
```

### Method 2: Direct Python Execution

```bash
# In terminal/SSH
cd ~/path/to/file
python3 standalone_convergence_test.py
```

### Method 3: In IPython/Jupyter Console

```python
In [1]: exec(open('standalone_convergence_test.py').read())
In [2]: results = run_convergence_test(num_trials=30)
```

---

## ⚙️ Configuration

To change settings, edit these lines in the file:

```python
TEST_CONFIG = {
    'num_trials': 30,        # Number of trials (5-30 recommended)
    'num_workers': -1,       # -1 = use all cores, 1 = sequential
    'M': 2,                  # Fourier modes (2-4 recommended)
    'maxiter': 30,           # Max iterations per trial
    'popsize': 10,           # Population multiplier
}

SIM_CONFIG = {
    'N': 2000,               # Number of particles
    'n_tot': 50,             # Time steps
}
```

---

## ⏱️ Expected Runtime

**On your GPU workstation (RTX 3080, 32 threads):**

| Trials | M | Expected Time |
|--------|---|---------------|
| 5      | 2 | ~15-25 min    |
| 10     | 2 | ~30-50 min    |
| 30     | 2 | ~2-3 hours    |
| 30     | 3 | ~3-5 hours    |

---

## 📊 Output

### Console Output:
```
======================================================================
TRIAL 1 - Seed: 42
======================================================================
Initial perimeter: 12.5664
Complete: Obj=0.428516, Time=4.23min

TRIAL 2 - Seed: 123
...
```

### Files Created:
```
convergence_results/
├── trial_5.png          # Visualizations every 5 trials
├── trial_10.png
├── trial_15.png
├── ...
├── comparison.png       # Final comparison plot
└── summary.txt          # Text summary
```

### Visualization Shows:
1. All shapes overlaid
2. Objective value bar chart
3. Mean coefficients with error bars
4. Statistics summary (CV, mean, std)

---

## 📈 Interpreting Results

### Coefficient of Variation (CV)

- **CV < 1%**: ✓ **EXCELLENT** - Very consistent
- **CV < 5%**: ✓ **GOOD** - Reasonably consistent  
- **CV < 10%**: ⚠ **MODERATE** - Some variation
- **CV > 10%**: ✗ **POOR** - Need more iterations

### What to Look For:

**Good Results:**
- ✅ Most/all trials succeed
- ✅ Objectives within 5% of each other
- ✅ Shapes visually similar
- ✅ Small error bars

**Bad Results:**
- ❌ Many trials fail
- ❌ Objectives vary by >10%
- ❌ Shapes very different
- ❌ Large error bars

---

## 🔧 Customization

### Quick Test (5 trials, ~20 min):
```python
results = run_convergence_test(num_trials=5, output_dir='quick_test')
```

### Full Test (30 trials, ~2-3 hours):
```python
results = run_convergence_test(num_trials=30, output_dir='full_test')
```

### Access Individual Results:
```python
# After running
for r in results:
    if r['success']:
        print(f"Trial {r['trial']}: {r['best_objective']:.6f}")

# Best trial
best = min([r for r in results if r['success']], 
           key=lambda x: x['best_objective'])
print(f"Best: {best['best_objective']:.6f}")

# Plot specific trial
visualize_trial(results[0], TEST_CONFIG)
```

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"
```bash
# Install required packages
pip install numpy scipy matplotlib

# Or if using conda
conda install numpy scipy matplotlib
```

### "MemoryError"
Reduce particle count:
```python
# In the file, change:
SIM_CONFIG = {
    'N': 1000,              # Reduced from 2000
    'n_tot': 40,            # Reduced from 50
}
```

### Runs too slowly
```python
# Reduce trials or iterations:
results = run_convergence_test(num_trials=10)  # Instead of 30

# Or edit file:
TEST_CONFIG['maxiter'] = 20  # Reduced from 30
```

### Want more visualization
```python
# Visualize every trial:
for i, result in enumerate(results):
    if result['success']:
        visualize_trial(result, TEST_CONFIG, f'trial_{i}.png')
```

---

## 🎓 What It Tests

1. **Convergence**: Does optimization find a minimum?
2. **Consistency**: Do multiple runs give similar results?
3. **Stability**: How sensitive to random initialization?

With **30 trials**, you get:
- Robust statistics (narrow confidence intervals)
- Outlier detection
- Distribution analysis
- Publication-quality results

---

## 📝 Notes

- **DSMC simulation is simplified** for speed (not full project version)
- Still captures essential physics
- Good for testing optimization architecture
- Results qualitatively similar to full simulation
- For production runs, use full project code

---

## ✅ Success Checklist

After running, you should have:

- [ ] Most trials succeeded (>80%)
- [ ] CV < 10% (preferably < 5%)
- [ ] Visualizations showing similar shapes
- [ ] Summary file with statistics
- [ ] Comparison plot showing consistency

If yes to all: **Optimization architecture is sound!** ✓

---

## 🆘 Support

If you encounter issues:

1. Check Python version: `python --version` (need 3.7+)
2. Check packages: `pip list | grep -E "numpy|scipy|matplotlib"`
3. Try with fewer trials first: `num_trials=3`
4. Check memory usage: `free -h`
5. Monitor CPU usage: `htop`

---

## 🚀 Ready to Go!

1. Upload `standalone_convergence_test.py` to Jupyter
2. Run: `exec(open('standalone_convergence_test.py').read())`
3. Execute: `results = run_convergence_test(num_trials=30)`
4. Wait 2-3 hours
5. Check results in `convergence_results/` directory

Good luck! 🎯
