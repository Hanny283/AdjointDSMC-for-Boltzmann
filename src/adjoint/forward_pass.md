# `forward_pass.py` — forward simulation, records, and backward adjoint pass

This module **runs** the particle simulation, **stores** a full trace, and implements **`backward_pass`**: computing **velocity adjoints β** and **position adjoints α** backward in time. It corresponds to the **discrete trajectory** and **Propositions 2.1–2.2** in **Section 2** of the paper.

## Role in the big picture

1. **Forward:** For each time step, apply **collision** sub-step then **free flight + optional wall reflection** (see class docstring for current ordering). Mutate global x, v and append **`StepRecord`**s with **`CollisionRecord`** and **`BoundaryRecord`** entries.

2. **Terminal condition:** The objective **L** at final time gives **∂L/∂v**<sub>M</sub> and **∂L/∂x**<sub>M</sub> — in code, user-supplied `terminal_beta_fn` and `terminal_alpha_fn` return arrays shaped `(N, 2)`.

3. **Backward:** Starting from step **M − 1** down to **0**, update **α** then **β** using the **same** ω, **θ_inter**, in/out flags as the forward run.

4. **Shape gradient** is **not** here: `shape_gradient.py` uses the resulting **β, α** and the forward records to form **dL/dC**.

## Data classes (what gets recorded)

| Class | Stores | Why |
|-------|--------|-----|
| `CollisionRecord` | Indices i, i′, pre/post velocities, ω | Backward pass must apply **J**<sub>coll</sub><sup>⊤</sup> with the **same** ω and pre-collision velocities. |
| `BoundaryRecord` | x<sub>k</sub>, v′, x′, inside/outside, **θ_inter** or `None`, final **x̃, ṽ** | Needed to rebuild reflection Jacobians and **G** in the backward pass. |
| `StepRecord` | One step’s start state, collision list, boundary list, end state | One time slice of the tape. |
| `SimulationHistory` | `C`, `dt`, list of `StepRecord` | Immutable forward trace; exposes `backward_pass` and `final_positions` / `final_velocities`. |

## `ForwardSimulation`

- **`__init__(C, dt, ...)`** — stores **C**, **Δt**, RNG; **builds** the collision mesh (triangle cells, Bird parameter **e**, etc.) in the current implementation so collisions match the standalone arbitrary-shape DSMC pattern.

- **`run(positions, velocities, n_steps)`** — returns a **`SimulationHistory`**.

## `SimulationHistory.backward_pass(terminal_beta_fn, terminal_alpha_fn)`

Returns **`betas`** and **`alphas`** of shape **`(M+1, N, 2)`** for steps **k = 0, …, M**.

### Intuition for α (**Prop. 2.2**)

- **α** answers: “If I nudge **position** at this time, how does **L** change at the end?”
- If the particle **did not** hit the wall at this step, position before flight feeds **linearly** into future positions without reflection branching → **α**<sub>k</sub> = **α**<sub>k+1</sub> (for that particle’s α update as implemented).
- If it **did** hit, the map from **pre-hit** position to **post-reflection** position is differentiated → **α**<sub>k</sub> = **G**<sup>⊤</sup>**α**<sub>k+1</sub> with **`compute_G_ki`** from `adjoint_jacobians`.

### Intuition for β (**Prop. 2.1**)

- **β** answers: “If I nudge **velocity**, how does **L** change?”
- **Free flight:** **x′ = x + Δt v′**, so sensitivity of future **L** to **v′** includes a **Δt α**<sub>k+1</sub> term (velocity pushes position).
- **Wall:** If outside, apply the **reflection** linearization: **M**<sup>⊤</sup>**β**<sub>k+1</sub> + **N**<sup>⊤</sup>**α**<sub>k+1</sub> (using `dv_reflected_dv` and `compute_N_ki`) — same physics as the paper, transpose form for adjoint.
- **Collision:** For each colliding pair, multiply the stacked **rhs** for two particles by **`collision_jacobian_transpose`** so **both** particles’ **β**<sub>k</sub> stay consistent with the **coupled** collision.

### Terminal functions

- For **L = (1/N) Σ<sub>i</sub> ‖x<sub>i</sub><sup>M</sup>‖²**, typically `terminal_beta_fn = 0` and `terminal_alpha_fn = 2 x_M / N` (sign convention matches the code’s Lagrangian / objective derivative).

## Relation to the paper

- State updates **ṽ, x̃** match **Section 2** (reflection when **x′ ∉ Ω**).
- **`backward_pass`** is the algorithmic form of **Propositions 2.1 and 2.2**.

## Helpers

- **`_is_inside` / `_is_inside_batch`** — star-shaped inside test using `radius_r`.
- **`_reflect`** — specular **ṽ, x̃** using `normal_n` and `compute_c_inter` from `boundary_geometry`.
