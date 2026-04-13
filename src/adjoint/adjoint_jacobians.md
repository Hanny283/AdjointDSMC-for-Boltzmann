# `adjoint_jacobians.py` — linearizations for reflection and collisions

This module contains the **matrices and vectors** that describe how **small changes** in pre-step velocities and positions affect **post-reflection** **ṽ**, **x̃**, and how collisions map **pre-collision** to **post-collision** velocities. They are the building blocks of **Section 2** of the paper (**Lemmas 2.2–2.10**, **Propositions 2.1–2.2**).

## Role in the big picture

Each simulation step applies a **nonlinear map**

**(new x, new v) = F(old x, old v).**

To run sensitivity **backward in time**, we need the **transpose** of the **Jacobian** of **F** (reverse-mode / adjoint). This file supplies those Jacobians (or their pieces) **in closed form** as derived in the paper.

## Convention

- In the **backward** pass, you often multiply by **M**<sup>⊤</sup>, **N**<sup>⊤</sup>, **J**<sub>coll</sub><sup>⊤</sup>, not **Mβ** in the naive form, when **M** is **not symmetric**. The code comments cite this; the implementation matches the **correct adjoint** transpose of the forward map.

## Functions (paper mapping)

### Normal and intersection sensitivity

| Function | Paper | Meaning (short) |
|----------|--------|-----------------|
| `dn_dtheta(theta, C)` | **Lemma 2.6** | **∂ñ / ∂θ̃** — normal rotates if hit angle shifts. |
| `dtheta_dv_prime(theta_inter, x_k, v_k, C)` | **Lemma 2.7** | **∂θ̃ / ∂v′** — aim changes → hit parameter changes. |
| `dtheta_dx(theta_inter, v_k, C)` | **Lemma 2.10** | **∂θ̃ / ∂x_k**. |
| `dc_dv_prime(...)` | **Lemma 2.8** | **∂c̃ / ∂v′** (chains through **θ̃**). |
| `dc_dtheta_scalar(theta, C)` | Used in **Lemma 2.9** | Scalar **F = ∂c̃ / ∂θ̃** for **G**<sub>k,i</sub>. |

### Reflection and flight Jacobians

| Function | Paper | Meaning (short) |
|----------|--------|-----------------|
| `dv_reflected_dv(...)` | **Lemma 2.2** with **J = I** | **∂ṽ / ∂v′** for specular map. |
| `compute_M_ki(..., J)` | **Lemma 2.2** | Full **∂ṽ / ∂v**<sub>pre</sub> = **(∂ṽ / ∂v′) J** with collision **J = ∂v′/∂v**. |
| `compute_N_ki(...)` | **Lemma 2.3** | **N**<sub>k,i</sub>: **∂x̃ / ∂v**<sub>pre</sub> through **Δt** flight + reflection. |
| `compute_G_ki(...)` | **Lemma 2.9** | **G**<sub>k,i</sub> = **∂x̃ / ∂x** at a wall hit — used in **Prop. 2.2** for **α**. |

### Collision pair

| Function | Paper | Meaning (short) |
|----------|--------|-----------------|
| `collision_jacobian(v_i, v_j, omega)` | **Prop. 2.1** | **4×4** **∂(v<sub>i</sub>′, v<sub>j</sub>′) / ∂(v<sub>i</sub>, v<sub>j</sub>)** for Nanbu–Babovsky with drawn ω. |
| `collision_jacobian_transpose(...)` | Same | Transpose for backward pass. |

### Packaged backward steps (optional use)

| Function | Paper |
|----------|--------|
| `apply_proposition_21(...)` | **Prop. 2.1** — one **β** update for a colliding pair (all in-domain / out-domain cases). |
| `apply_proposition_22(...)` | **Prop. 2.2** — **α**<sub>k</sub> = **G**<sup>⊤</sup>**α**<sub>k+1</sub> if wall hit, else **α**<sub>k</sub> = **α**<sub>k+1</sub>. |

`forward_pass.SimulationHistory.backward_pass` implements the same logic as **Props. 2.1–2.2** inline (using `dv_reflected_dv`, `compute_N_ki`, `collision_jacobian_transpose`, `apply_proposition_22`).

## Why each piece exists

- **Without** `dn_dtheta`, `dtheta_dv_prime`, etc., you cannot differentiate the reflection map when **θ̃** and **ñ** move with **v′** and **x**.
- **Without** `collision_jacobian_transpose`, you cannot propagate **β** backward **through** the randomized pairwise collision while keeping particles **i** and **i′** **coupled** correctly.

Together, these functions are the **chain rule**, pre-derived, so the backward pass is **correct** and **fast** (no finite differences on the whole trajectory).
