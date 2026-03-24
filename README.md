# DSMC with Arbitrary Boundary Conditions and Shape Optimization

A research implementation of the **Direct Simulation Monte Carlo (DSMC)** method using the **Nanbu-Babovsky** collision algorithm, supporting 2D and 3D geometries, arbitrary boundary shapes, and gradient-based shape optimization via a discrete adjoint method.

---

## Overview

DSMC is a particle-based stochastic method for simulating rarefied gas dynamics by modeling individual particle collisions and boundary interactions. This project extends standard DSMC with:

- **Arbitrary 2D boundary shapes** defined by Fourier (star-shape) parameterization and triangulated meshes
- **Multiple boundary condition types**: specular reflection and Maxwell thermal (diffuse) reflection
- **Shape optimization**: find the boundary shape that minimizes a kinetic energy objective in an inscribed region
- **Discrete adjoint DSMC**: analytically compute gradients of the objective with respect to Fourier shape coefficients

---

## Repository Structure

```
src/
├── 2d/
│   ├── Arbitrary Shape/
│   │   ├── arbitrary_helpers.py       # Mesh, boundary, KD-tree, cell utilities
│   │   ├── arbitrary_bc.py            # Specular and thermal boundary conditions
│   │   └── arbitrary_parameterized.py # Main DSMC loop for arbitrary shapes
│   ├── Basic Shapes/
│   │   ├── Circular/                  # Circular domain simulation
│   │   └── Grid/                      # Rectangular grid simulation
│   └── general_helpers.py             # Maxwellian velocity sampling
├── 3d/
│   ├── Box/                           # 3D box domain simulation
│   └── Spherical/                     # 3D spherical domain simulation
├── adjoint_dsmc/
│   ├── forward_wrapper.py             # Forward pass with history recording
│   ├── backward.py                    # Backward adjoint sweep
│   ├── gradient.py                    # Gradient accumulation (∂J/∂c)
│   ├── boundary.py                    # Boundary normal gradients
│   ├── kernels.py                     # Collision kernel (VHS, Maxwell)
│   └── records.py                     # Data structures for forward history
├── optimization/
│   ├── config.py                      # Optimization and simulation parameters
│   ├── shape_optimizer.py             # Objective function evaluation
│   ├── constraints.py                 # Geometric constraints
│   ├── run_optimization.py            # Main optimization driver
│   └── viz_optimizer.py               # Visualization and progress tracking
├── plots/                             # Plotting scripts for each geometry
├── cell_class.py                      # Triangular cell data structure
├── edge_class.py                      # Edge data structure
└── universal_sim_helpers.py           # VHS cross-section, probabilistic rounding

experiments/
├── batch_run.py                       # Run multiple simulations in batch
├── compare_results.py                 # Compare simulation outputs
└── visualize_batch.py                 # Batch visualization

notebooks/
└── jupyterhub_run.ipynb               # Interactive notebook for cluster runs

optimization_results/                  # Saved optimization outputs and frames
```

---

## Physics and Algorithms

### Nanbu-Babovsky Collision Scheme

At each time step, particles within each mesh cell are randomly paired. A collision between pair \((i, j)\) is **accepted** with probability proportional to the Variable Hard Sphere (VHS) cross-section:

$$\sigma_{ij} = C \cdot |v_i - v_j|^\alpha$$

Accepted pairs exchange momentum via a random unit vector on the collision sphere, conserving total momentum and kinetic energy exactly.

### Star-Shape Boundary Parameterization

The boundary is a **star-shaped curve** defined in polar coordinates by a truncated Fourier series:

$$r(\theta;\, c) = c_0 + \sum_{m=1}^{M} \left[ c_m \cos(m\theta) + c_{m+M} \sin(m\theta) \right]$$

where $c \in \mathbb{R}^{2M+1}$ are the Fourier coefficients. A circle is recovered with $c = [R, 0, \ldots, 0]$.

### Boundary Conditions

**Specular Reflection:** The velocity component normal to the boundary is reversed:

$$v' = v - 2(v \cdot \hat{n})\,\hat{n}$$

**Maxwell Thermal (Diffuse) Reflection:** With accommodation coefficient $\alpha \in [0,1]$:
- With probability $1 - \alpha$: specular reflection
- With probability $\alpha$: the particle is re-emitted with a speed drawn from a Maxwellian at wall temperature $T_\text{wall}$, in a direction sampled from Lambert's cosine law relative to the inward normal

**Overshoot correction:** When a particle exits the boundary, the overshoot vector is also reflected, placing the particle back inside at the correct mirrored position.

### Particle-to-Boundary Distance

For each escaped particle, the closest point on the boundary is found via:
1. A **KD-tree** on edge midpoints identifies the 3 nearest candidate edges
2. An **orthogonal projection** onto each candidate segment computes the exact distance, with the parameter clamped to $[0,1]$ to snap to endpoints when needed
3. The edge yielding minimum distance is selected

### Cell Assignment and Rebinning

The domain is triangulated using **pygmsh/gmsh**. Particles are assigned to triangular cells, and after each time step, any particle that has crossed a cell boundary is rebinned using:
1. A **KD-tree** over cell centroids to find the nearest starting cell
2. **Barycentric triangle walking** (`triangle_to_follow`) to iteratively follow the mesh topology to the true containing cell

---

## Shape Optimization

The optimization problem is:

> **Find the Fourier coefficients $c$ that minimize the total kinetic energy of gas particles in the inscribed square $[-a, a]^2$**, subject to area preservation, a bounding box constraint, and a minimum-radius constraint (star-shape validity).

The optimizer uses **differential evolution** (`scipy.optimize.differential_evolution`) with:
- Population-based global search over $2M+1$ parameters
- Parallelized objective evaluations across CPU cores
- A spectral regularization penalty $\lambda \|c_{m \geq 2}\|^2$ to suppress unphysical high-frequency shape modes

Each objective evaluation runs a full DSMC simulation and averages the kinetic energy inside the inscribed square over the final time steps.

### Discrete Adjoint for Gradient Computation

For gradient-based methods, the project implements the **discrete adjoint DSMC** from Yang, Silantyev & Caflisch (2023). The adjoint variable $\gamma[k]$ satisfies a backward recurrence:

- **Step A** (adjoint through specular BC): $\gamma_i \leftarrow \gamma_i - 2(\gamma_i \cdot \hat{n})\hat{n}$
- **Step B** (adjoint through collisions): applies the transpose of the collision Jacobian plus score-function corrections for accepted/rejected pairs

The shape gradient is then:

$$\frac{\partial J}{\partial c_m} = \sum_{k} \sum_{i \in \text{hits}_k} \gamma[k+1, i] \cdot \frac{\partial v'_{k,i}}{\partial c_m}$$

where $\partial v' / \partial c_m$ is derived analytically from the dependence of the boundary normal $\hat{n}$ on the Fourier coefficients.

---

## Installation

```bash
# Install gmsh first (required before pygmsh)
pip install gmsh

# Install remaining dependencies
pip install -r requirements.txt
```

### Requirements

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Core numerics and optimization |
| `matplotlib` | Plotting |
| `gmsh`, `pygmsh`, `meshio` | Mesh generation for arbitrary domains |
| `Pillow`, `imageio`, `imageio-ffmpeg` | Image I/O and animation export |

---

## Usage

### Running a Single DSMC Simulation

```python
from src.adjoint_dsmc.forward_wrapper import forward_pass_with_history
import numpy as np

# Star-shape: circle of radius 2 with 2 Fourier modes
c = np.array([2.0, 0.0, 0.0, 0.0, 0.0])  # [c0, c1, c2, c3, c4]

positions, velocities, temp_history, cells, boundary = \
    Arbitrary_Shape_Parameterized(
        N=1000, fourier_coefficients=c, num_boundary_points=100,
        T_x0=1.0, T_y0=1.0, dt=0.01, n_tot=100,
        e=1.0, mu=1.0, alpha=1.0, mesh_size=0.4,
        T_wall_x=0.2, T_wall_y=0.2, accommodation_coefficient=1.0
    )
```

### Running Shape Optimization

```bash
# Full optimization run
python src/optimization/run_optimization.py

# With options
python src/optimization/run_optimization.py --maxiter 50 --workers -1 --output-dir results/

# Quick test (10 iterations)
python src/optimization/run_optimization.py --test
```

### Running on JupyterHub

```bash
bash pack_for_jupyterhub.sh   # Packages source into dsmc-shape-opt.zip for upload
```

Then open `notebooks/jupyterhub_run.ipynb` on the cluster.

---

## Key Configuration Parameters

All parameters live in `src/optimization/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `M` | `2` | Number of Fourier modes ($2M+1$ total shape parameters) |
| `N` | `2500` | Number of simulation particles |
| `n_tot` | `80` | Number of time steps per simulation |
| `dt` | `0.01` | Time step size |
| `mesh_size` | `0.4` | Triangular cell size (larger = coarser = faster) |
| `T_wall_x/y` | `0.2` | Wall temperature for thermal BC |
| `accommodation_coefficient` | `1.0` | 0 = specular, 1 = fully diffuse |
| `a` | `0.5` | Half-side of inscribed square for objective |
| `lambda_reg` | `0.01` | Regularization weight |
| `workers` | `-1` | Parallel workers (-1 = all cores) |

---

## Output

The optimization saves results to `optimization_results/`:

- `optimization_progress.png` — objective value vs. iteration
- `best_coefficients.txt` — optimal Fourier coefficients
- `best_shape_extracted.png` — final optimal boundary shape
- `frames/iter_XXXX.png` — per-iteration shape snapshots
- `optimization_evolution.mp4` — animation of shape evolution (requires `imageio-ffmpeg`)

---

## References

- Nanbu, K. (1980). *Direct simulation scheme derived from the Boltzmann equation.* Journal of the Physical Society of Japan.
- Babovsky, H. (1989). *On a simulation scheme for the Boltzmann equation.* Mathematical Methods in the Applied Sciences.
- Yang, Silantyev & Caflisch (2023). *Discrete adjoint method for DSMC.*
- Caflisch, Silantyev & Yang (2021). *Adjoint DSMC for nonlinear Boltzmann equation constrained optimization.*
