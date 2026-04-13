# `boundary_geometry.py` — wall shape, normal, and particle–wall intersection

This module answers one question: **given a Fourier boundary shape C and a particle
at position x moving with velocity v, where does the particle hit the wall, and
which way is the wall facing at that point?**  Everything in the file is pure geometry.
No gradients, no adjoints — those live in `adjoint_jacobians.py` and `shape_gradient.py`.

---

## The boundary model

The wall is the curve

    γ(θ; C) = r(θ; C) · (cos 2πθ,  sin 2πθ),    θ ∈ [0, 1)

where the radial distance r is a truncated Fourier series in the coefficient vector
**C** = (c₀, a₁, …, aₙ, b₁, …, bₙ):

    r(θ; C) = c₀ + Σₖ₌₁ᴺ [aₖ cos(2πkθ) + bₖ sin(2πkθ)]

As C changes (during optimisation), the entire wall shape moves.  The adjoint
computes how those movements affect the objective — but this file only evaluates
the wall at a fixed C.

---

## Functions

| Function | What it returns | Why it exists |
|----------|-----------------|---------------|
| `radius_r(theta, C)` | r(θ; C) — distance from origin to wall at angle θ | Defines the wall location. |
| `radius_r_theta(theta, C)` | ∂r/∂θ — how r changes as you move along the wall | Needed for the tangent vector and the normal direction. |
| `radius_r_theta_theta(theta, C)` | ∂²r/∂θ² | Derivative of the unnormalized normal w.r.t. θ (Lemma 2.6 in paper). |
| `gamma(theta, C)` | Point γ(θ) on the wall curve | Used for plotting and geometric checks. |
| `gamma_prime(theta, C)` | Tangent vector γ′(θ) | The outward normal is perpendicular to this in 2D. |
| `f_unnormalized(theta, C)` | Vector proportional to the outward normal (not yet unit length) | Paper's eq. (7): **n̂** = f / ‖f‖. |
| `f_unnormalized_prime(theta, C)` | ∂f/∂θ | How the normal direction rotates when the hit angle θ̃ changes — used in Lemma 2.6. |
| `normal_n(theta, C)` | Unit outward normal **n̂**(θ; C) | Specular reflection requires the unit normal. |
| `solve_theta_inter(x, v, C)` | θ_inter: the angle where the ray from x along v first hits the wall | Without this we do not know which normal or offset c̃ to use for reflection. Uses grid scan + Brent's method; picks the forward-facing hit. |
| `solve_theta_inter_batch(x, v, C)` | Same as above but vectorised for K particles simultaneously | 145× speedup over calling `solve_theta_inter` in a loop. |
| `compute_c_inter(theta_inter, C)` | Scalar c̃ — the reflection-plane offset (eq. (11)) | Defines the plane ⟨n̂, x⟩ = c̃ that the particle reflects across. |

---

## How the wall intersection is solved

The ray x(t) = x_start + t·v hits the wall when r(θ)·(cos 2πθ, sin 2πθ) = x_start + t·v
for some (t, θ).  Eliminating t leaves a single nonlinear equation in θ:

    F(θ) = r(θ)(vʸ cos 2πθ − vˣ sin 2πθ) − (vʸ xˢˣ − vˣ xˢʸ) = 0

`solve_theta_inter` finds this root in two stages:
1. **Grid scan** — evaluate F at 400 equally spaced θ values; record any sign flips.
   A sign flip guarantees a root between those two θ values.
2. **Brent's method** — zooms in on the bracketed interval to machine precision (1e-12)
   in a fixed number of iterations.

The function returns the root corresponding to a **forward** crossing (the particle
is actually heading into the wall, not through it from the other side).

---

## Near-tangential crossings

When a particle travels nearly parallel to the wall (nearly tangential incidence),
the denominator ∂F/∂θ becomes very small.  This makes the implicit-differentiation
formulas for ∂θ_inter/∂v and ∂θ_inter/∂C blow up.  These cases are handled by a
stability guard in `adjoint_jacobians.py` and `shape_gradient.py`: any contribution
whose denominator is smaller than a threshold (1e-2) is zeroed out, discarding that
particle's contribution to the gradient rather than letting it corrupt the direction.

---

## Relation to the paper

- **Section 1:** Fourier parametrisation (equations 1–2).
- **Eq. (7):** outward normal via `f_unnormalized` → `normal_n`.
- **Eqs. (10)–(11):** `solve_theta_inter` and `compute_c_inter`.
- **Lemmas 2.6–2.10** in `adjoint_jacobians.py` differentiate **through** the quantities
  computed here; `boundary_geometry.py` supplies the forward values.
