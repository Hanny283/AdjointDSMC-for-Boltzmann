# Adjoint DSMC — Shape Optimisation Module

This package implements a **discrete adjoint method** for shape optimisation of
a 2-D Nanbu-Babovsky DSMC simulation with a Fourier-parameterised reflective
boundary.  Given a boundary shape **C** (a vector of Fourier coefficients), the
pipeline computes the exact gradient dL/d**C** of any smooth terminal objective
L through the full stochastic forward trajectory.

---

## Quick start

```python
import numpy as np
import sys, os
sys.path.insert(0, "/path/to/src")           # add src/ to the Python path

from adjoint import ForwardSimulation
from adjoint.shape_gradient import shape_gradient, area, area_gradient

# --- 1. Define boundary coefficients C = [c0, a1,...,aN, b1,...,bN] ---
N_FOURIER = 3
C = np.zeros(2 * N_FOURIER + 1)
C[0] = 0.75          # mean radius
C[3] = 0.20          # 3-lobe cosine mode
C[5] = 0.08          # 2-lobe sine mode

# --- 2. Initial particle ensemble ---
rng = np.random.default_rng(42)
N   = 1000
r   = rng.uniform(0.25, 0.65, N)
th  = rng.uniform(0, 2*np.pi, N)
x0  = np.column_stack([r*np.cos(th), r*np.sin(th)])
v0  = rng.standard_normal((N, 2))

# --- 3. Forward simulation ---
sim     = ForwardSimulation(C, dt=0.10, e=10.0,
                            mesh_size=0.25, num_boundary_points=100, seed=0)
history = sim.run(x0, v0, n_steps=60)

# --- 4. Terminal adjoint conditions (define your objective here) ---
# Example: L = mean |x_M|^2  (minimised by a circle under fixed area)
def term_beta(v, x):   return np.zeros_like(v)      # dL/dv_M = 0
def term_alpha(v, xM): return 2.0 * xM / len(xM)   # dL/dx_M

betas, alphas = history.backward_pass(term_beta, term_alpha)

# --- 5. Shape gradient ---
g = shape_gradient(history, betas, alphas)   # shape (2N+1,)

# --- 6. Gradient descent step with area penalty ---
A_TARGET  = area(C)
area_viol = area(C) - A_TARGET
lr        = 6e-3
lam_area  = 30.0
C_new     = C - lr * (g + lam_area * area_viol * area_gradient(C))
```

For full working experiment scripts see `experiments/simple/`, `experiments/2_mode_simple/`,
and `experiments/complex/` in the repository root.

---

## Validation scripts

Three self-contained scripts are included in this folder to verify correctness.
Run them from the repository root (or any directory — they resolve their own paths):

```bash
# 1. Pure-math geometry checks (~2 s, no simulation)
python src/adjoint/validate_geometry.py

# 2. Gradient correctness: analytical FD check + descent direction (~30 s)
python src/adjoint/validate_fd_gradient.py

# 3. End-to-end convergence to the provably optimal circle (~50 s)
python src/adjoint/validate_convergence.py
```

### What each script checks

| Script | Checks |
|---|---|
| `validate_geometry.py` | `‖n‖=1`, `n⊥γ'`, outward direction, `solve_theta_inter` returns a point on the boundary, closed-form area vs Green's theorem quadrature, `dr_dC` / `area_gradient` finite-difference match |
| `validate_fd_gradient.py` | (A) Expected L decreases after one gradient step; (B) `dv_reflected_dC` and `dx_reflected_dC` match numerical finite differences to machine precision (< 0.01% relative error) |
| `validate_convergence.py` | Full 60-iteration optimisation: L decreases, all Fourier modes shrink toward zero (circle), area preserved within 0.2% |

---

## Module map

```
                    ┌─────────────────────────────────────┐
                    │  experiments/*.py  (user-facing)     │
                    │  defines L, optimizer, constraints   │
                    └──────────────────┬──────────────────┘
                                       │ calls
                                       ▼
┌──────────────────────────────────────────────────────────┐
│  shape_gradient.py                                        │
│  Given β, α from the backward pass, form dL/dC           │
│  This is the only module whose output lives in C-space   │
└────────────────────────┬─────────────────────────────────┘
                         │ needs betas, alphas, history
                         ▼
┌──────────────────────────────────────────────────────────┐
│  forward_pass.py                                          │
│  Forward:  run particles step by step, record everything  │
│  Backward: given terminal ∂L/∂x, ∂L/∂v, propagate β, α  │
└──────────┬─────────────────────────────────┬─────────────┘
           │ uses (geometry)                 │ uses (Jacobians)
           ▼                                 ▼
┌──────────────────┐                ┌────────────────────────┐
│ boundary_geometry│                │ adjoint_jacobians       │
│ wall shape, n,   │                │ ∂ṽ/∂v, ∂x̃/∂x,        │
│ θ_hit, c̃        │                │ collision Jacobian 4×4  │
└──────────────────┘                └────────────────────────┘
```

---

## Module summaries

### `boundary_geometry.py`
*"Where is the wall and what does it look like?"*

Defines the Fourier star-shape γ(θ; C), computes the outward unit normal **n̂**,
solves for the angle θ where a particle ray hits the wall (scalar `solve_theta_inter`
and vectorised batch version `solve_theta_inter_batch`), and computes the
reflection offset c̃.  Nothing here knows about gradients — it is pure geometry.

### `adjoint_jacobians.py`
*"If I nudge x or v a little before a sub-step, how do x̃ and ṽ change?"*

Closed-form Jacobians of the reflection and collision maps — matrices M, N, G,
and the 4×4 Nanbu-Babovsky collision Jacobian — exactly as derived in the paper
(Lemmas 2.2–2.10).  The backward pass multiplies by the **transposes** of these
matrices to propagate sensitivity backward in time.

### `forward_pass.py`
*"Run the simulation and record the full tape; then rewind β and α."*

`ForwardSimulation.run()` executes the DSMC loop (collision + free-flight +
specular reflection) and stores a `SimulationHistory` — a complete record of
every particle trajectory, collision pair, and boundary hit.
`SimulationHistory.backward_pass()` is the discrete adjoint (Props. 2.1–2.2):
it consumes the tape and produces β (velocity adjoints) and α (position adjoints)
at every time step.

### `shape_gradient.py`
*"Given β and α, how does L change if I nudge the Fourier coefficients C?"*

Sums `β_{k+1}ᵀ ∂ṽ/∂C + α_{k+1}ᵀ ∂x̃/∂C` at every wall-hit event to produce
dL/d**C**.  Also contains:

- `area(C)` / `area_gradient(C)` — **exact closed form** via Parseval (no quadrature needed)
- `perimeter(C)` / `perimeter_gradient(C)` — numerical quadrature
- `project_step_perimeter_cap(...)` — hard perimeter cap on a gradient step

### `visualization.py`
Plots and animations only.  Contains `plot_convergence(obj_hist, ..., smooth_window=W)`
which overlays a rolling-mean curve on the raw stochastic objective history to
show the trend clearly.

---

## Data flow

```
C  →  mesh + geometry  →  forward trajectory  →  terminal ∂L/∂x, ∂L/∂v
   →  backward_pass (β, α)  →  shape_gradient  →  dL/dC  →  optimizer  →  C_new
```

---

## Dependencies

`forward_pass.py` relies on the mesh-generation and cell infrastructure from the
wider repository.  The expected layout relative to this file is:

```
src/
├── adjoint/          ← this folder
├── 2d/
│   └── Arbitrary Shape/
│       └── arbitrary_helpers.py
├── cell_class.py
├── edge_class.py
└── universal_sim_helpers.py
```

External packages: `numpy`, `scipy`, `pygmsh==7.1.17`, `meshio>=5.3`, `matplotlib`.
See `requirements.txt` for the full list.

---

## Recent updates

- **Optimizer**: switched from Adam (adaptive moments) to gradient descent with
  cosine-annealed learning rate across all experiment scripts.  Removes BETA1,
  BETA2, EPS_ADAM parameters; the update rule is now simply
  `C ← C − lr(t) · ∇L`.
- **Public API**: `area` and `area_gradient` added to `__init__.py` exports so
  they are reachable via `from adjoint import area, area_gradient`.
- **Convergence plot**: `plot_convergence` gained a `smooth_window` parameter;
  when set, it draws a rolling-mean overlay on the noisy raw curve so the
  descent trend is visible with small N_AVG.
- **Validation suite**: three scripts (`validate_geometry.py`,
  `validate_fd_gradient.py`, `validate_convergence.py`) added to this folder.
