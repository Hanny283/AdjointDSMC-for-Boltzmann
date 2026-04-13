# Adjoint DSMC package — module map

`src/adjoint/` contains five modules that form a fixed pipeline.  This file explains
**what each module does**, **how they connect**, and **what order to read them**.

---

## What the pipeline does in one sentence

Given a boundary shape (Fourier coefficients **C**), run particles forward, record
everything, run the adjoint backward to get sensitivities, then assemble the gradient
of the objective with respect to **C** — so an optimiser can improve the shape.

---

## The three layers

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
solves for the angle θ where a particle ray hits the wall, and computes the
reflection offset c̃.  **Nothing here knows about gradients or adjoints** — it is
pure geometry used by the forward step and by the derivative formulas.

### `adjoint_jacobians.py`
*"If I nudge x or v a little before a sub-step, how do x̃ and ṽ change?"*

Supplies the closed-form Jacobians of the reflection and collision maps — the
matrices M, N, G, and the 4×4 collision Jacobian — exactly as derived in the paper
(Lemmas 2.2–2.10).  The backward pass multiplies by the **transposes** of these
matrices to propagate sensitivity backward in time.

### `forward_pass.py`
*"Run the simulation and record the full tape; then rewind β and α."*

The **forward** half runs collisions, free flight, and reflections at each time step,
storing a complete `SimulationHistory` (every particle position, velocity,
collision pair, and boundary hit).  The **backward** half (`backward_pass`) is the
discrete adjoint (Props. 2.1–2.2): it consumes the tape and outputs arrays **β**
(velocity adjoints) and **α** (position adjoints) at every time step.

### `shape_gradient.py`
*"Given β and α, how does L change if I nudge the Fourier coefficients C?"*

Sums `β_k+1ᵀ ∂ṽ/∂C + α_k+1ᵀ ∂x̃/∂C` at every wall-hit event.  This is the only
module whose output is a vector in the same space as **C** — the gradient the
optimiser actually uses.  Also contains:
- `perimeter(C)` / `perimeter_gradient(C)` — numerical quadrature
- `area(C)` / `area_gradient(C)` — **exact closed form** via Parseval (no quadrature)
- `project_step_perimeter_cap(...)` — hard perimeter cap for a gradient step

### `visualization.py`
Plots and animations only.  No effect on the math pipeline.

---

## Data flow

```
C  →  mesh + geometry  →  forward trajectory  →  terminal ∂L/∂x, ∂L/∂v
   →  backward_pass (β, α)  →  shape_gradient  →  dL/dC  →  optimizer  →  C_new
```

---

## Recommended reading order

| Step | Read | Why |
|------|------|-----|
| 1 | `boundary_geometry.py` + `boundary_geometry.md` | Must picture the Fourier wall, normal, and ray–wall hit before anything else. |
| 2 | `forward_pass.py` forward half + `forward_pass.md` intro | Understand what is stored at each time step and in what order. |
| 3 | `adjoint_jacobians.py` + `adjoint_jacobians.md` | Learn what M, N, G, and the collision Jacobian mean as linear maps. |
| 4 | `forward_pass.py` `backward_pass` + rest of `forward_pass.md` | Now the backward loop is readable: you know what each matrix multiplies. |
| 5 | `shape_gradient.py` + `shape_gradient.md` | See how C enters only through the wall map and how β, α weight ∂ṽ/∂C, ∂x̃/∂C. |
| 6 | `visualization.py` | Optional — just plotting. |

**Paper alignment:** after step 1, skim Section 1 of the paper; after step 4, Section 2
(Props. 2.1–2.2); after step 5, Section 3.
