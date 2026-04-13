# `boundary_geometry.py` — boundary curve, normal, and wall intersection

This module describes the **Fourier star-shaped** wall and everything needed to **reflect** particles off it. It implements the geometry from **Section 1** of the adjoint write-up and **equations (7), (10), (11)** in the paper (outward normal, intersection angle, reflection offset).

## Role in the big picture

- The simulation needs to know **where the wall is**, **which way it faces** (outward normal), and **where a ray first hits** the wall.
- Later modules differentiate through reflection; they assume this geometry is **exact** and **smooth** in the angle parameter **θ**.

## Boundary model

The boundary is

**γ(θ; C) = r(θ; C) · (cos 2πθ, sin 2πθ),**

with **r** a **truncated Fourier series** in **C** = (c₀, a₁, …, a<sub>N</sub>, b₁, …, b<sub>N</sub>).

## Functions (goal and why)

| Function | Goal | Why we need it |
|----------|------|----------------|
| `radius_r(theta, C)` | Evaluate **r(θ; C)**. | Defines distance from origin to the wall along direction **θ**. |
| `radius_r_theta(theta, C)` | **∂r / ∂θ**. | Tangent / curvature-related terms; appears in **γ′** and the normal. |
| `radius_r_theta_theta(theta, C)` | **∂²r / ∂θ²**. | Derivative of the unnormalized normal w.r.t. **θ** (**Lemma 2.6** chain). |
| `gamma(theta, C)` | Point **γ(θ)** on the wall. | Plotting and geometric checks. |
| `gamma_prime(theta, C)` | Tangent vector **γ′(θ)**. | Orthogonal to the outward normal in 2D. |
| `f_unnormalized(theta, C)` | Vector proportional to outward normal before unit length. | Paper builds **eq. (7)** as **ñ = f / ‖f‖**. |
| `f_unnormalized_prime(theta, C)` | **∂f / ∂θ**. | How the normal direction changes when the hit parameter **θ̃** moves — **Lemma 2.6**. |
| `normal_n(theta, C)` | Unit outward normal **ñ(θ; C)**. | Specular reflection uses **ñ**. |
| `solve_theta_inter(x_k, v_k, C, n_grid)` | Solve **eq. (10)** for **θ_inter**: first intersection of the ray from **x**<sub>k</sub> along **v**<sub>k</sub> with the curve **r(θ)**. | Without **θ_inter** we do not know **which** normal and **which** **c̃** to use. Uses a coarse **θ**-scan plus Brent refinement and picks the **forward** hit. |
| `compute_c_inter(theta_inter, C)` | Scalar **c̃** in **eq. (11)**. | Defines the reflection line **⟨ñ, x⟩ = c̃**; used in the position reflection **x̃**. |

## Relation to the paper

- **Section 1:** Fourier parametrization **(1)–(2)**.
- **Eq. (7):** same formula as `normal_n` (via `f_unnormalized`).
- **Eq. (10)–(11):** implemented by `solve_theta_inter` and `compute_c_inter`.

## Relation to lemmas

**Lemmas 2.6–2.10** assume **ñ**, **θ̃**, and **c̃** are known functions of the state and **C**. This file **computes** those geometric quantities on the forward pass. It does **not** compute adjoints; `adjoint_jacobians.py` and `shape_gradient.py` differentiate **through** these maps.
