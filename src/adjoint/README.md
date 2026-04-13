# Adjoint DSMC package — high-level map

`src/adjoint/` is a **single folder** (no subfolders). These **modules** work together in a fixed pipeline. This file explains **how they connect** and **what order to read** for the clearest mental model.

---

## How the pieces interact

Think of **three layers**:

```
                    ┌─────────────────────────────────────┐
                    │  experiments/*.py (outside adjoint)  │
                    │  objective L, optimizer, penalties   │
                    └─────────────────┬───────────────────┘
                                      │
              uses                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  shape_gradient.py                                                 │
│  “Turn β, α into dL/dC”  (only wall hits matter for C)            │
└───────────────────────────────┬─────────────────────────────────┘
                                │ needs betas, alphas + history
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  forward_pass.py                                                 │
│  Forward: collisions + flight + reflection → SimulationHistory   │
│  Backward: backward_pass → β, α at every time step               │
└───────┬───────────────────────────────────────┬─────────────────┘
        │ uses                                  │ uses
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────────────┐
│ boundary_geometry │                 │ adjoint_jacobians          │
│ γ, r, n, θ_hit, c │                 │ Jacobians for reflection   │
│ (the wall in math)│                 │ + collision 4×4 + G, M, N  │
└───────────────────┘                 └───────────────────────────┘
```

- **`boundary_geometry`** — *Where is the wall?* Computes γ, outward unit normal **ñ**, intersection angle **θ_inter**, and offset **c̃** for reflection. **Nothing here knows about L or adjoints**; it is pure geometry used by the forward step and by derivative formulas.

- **`adjoint_jacobians`** — *If we nudge x, v a little before a sub-step, how do x̃, ṽ change?* Supplies the matrices/vectors from the paper (Lemmas 2.2–2.10, collision Jacobian). **`forward_pass.backward_pass`** calls into this layer when rewinding sensitivity.

- **`forward_pass`** — *Run the simulation and record everything; then rewind β, α.* The **forward** half moves particles and fills `CollisionRecord` / `BoundaryRecord`. The **backward** half is the discrete adjoint (Props. 2.1–2.2): it consumes those records and outputs **`betas`**, **`alphas`**.

- **`shape_gradient`** — *Given β, α, how does L change if we nudge the coefficient vector **C**?* Sums **βᵀ ∂ṽ/∂C + αᵀ ∂x̃/∂C** at each wall hit. **This is the only adjoint module that outputs a vector in the same space as your Fourier coefficients.**

- **`visualization.py`** — Plots and animations only; **not** part of the math pipeline.

- **`__init__.py`** — Re-exports the public API; read last or skip.

**Experiment scripts** (under `experiments/`) wire everything together: run forward → `backward_pass` → `shape_gradient` → gradient step on **C**.

---

## Recommended reading order (best understanding)

Read **code** and the matching **`.md`** in the same step when you want detail.

| Step | Read | Why this order |
|------|------|----------------|
| **1** | `boundary_geometry.py` + `boundary_geometry.md` | You must picture the **Fourier star**, **normal**, and **ray–wall hit** before anything else makes sense. |
| **2** | `forward_pass.py` — forward half only (`ForwardSimulation.run`, `_boundary_step`, `_reflect`, record classes) + start of `forward_pass.md` | See the **exact sequence** each time step: collisions, then move, then reflect. Understand what is **stored** in each record. |
| **3** | `adjoint_jacobians.py` + `adjoint_jacobians.md` | Learn the **building blocks**: what **M**, **N**, **G**, and the collision Jacobian **mean** as small linear maps through reflection and collision. |
| **4** | `forward_pass.py` — `SimulationHistory.backward_pass` + rest of `forward_pass.md` | Now the backward loop is readable: you know **what** each matrix is multiplying. Focus on **terminal α, β** → loop backward → output arrays. |
| **5** | `shape_gradient.py` + `shape_gradient.md` | See how **C** enters only through the **wall map** and how **β, α** weight **∂ṽ/∂C** and **∂x̃/∂C**. |
| **6** | `visualization.py` (optional) | How results are plotted; no effect on gradients. |
| **7** | `__init__.py` | Quick list of exported names. |

**Paper alignment (optional):** after step 1, skim **Section 1** of your PDF; after step 4, **Section 2** (Props. 2.1–2.2); after step 5, **Section 3**.

---

## One-sentence summary per module

| Module | One sentence |
|--------|----------------|
| `boundary_geometry` | Defines the moving wall and finds where a particle ray hits it. |
| `forward_pass` | Simulates particles with full tape, then runs the discrete adjoint to get β, α. |
| `adjoint_jacobians` | Closed-form Jacobians for “differentiate through bounce and collision.” |
| `shape_gradient` | Combines β, α with **∂(ṽ, x̃)/∂C** at hits → **dL/dC**. |
| `visualization` | Figures and GIFs for experiments. |

---

## Data flow in one line

**C** → (mesh + geometry) → **forward trajectory** → **terminal ∂L/∂x, ∂L/∂v** → **`backward_pass`(β, α)** → **`shape_gradient`** → **dL/dC** → optimizer updates **C**.
