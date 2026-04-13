# Adjoint DSMC Shape Optimisation — Experimental Write-Up

## Boundary Reflection

When a particle moves far enough in one time step to exit the domain, it needs
to bounce off the wall.  To do this we need to know exactly where on the wall
it hit, and which direction the wall is facing at that point.

The wall is not a simple circle or polygon — it is a smooth curve described by
a mathematical formula (the Fourier star-shape).  So instead of reading off a
wall normal from a mesh edge (which would only be an approximation), we solve
directly for the exact point on the curve where the particle's straight-line
path crosses it.  Concretely, the particle travels along the line
x(t) = x_start + t·v, and the wall is the set of points γ(θ) for θ ∈ [0, 1).
Setting these equal and eliminating t gives a single equation in θ alone.  We
find the root of that equation numerically (grid scan + Brent's method) to
machine precision.

Finding that root works in two stages.  First, the grid scan: we evaluate the
equation at 400 evenly spaced values of θ around the boundary and look for
adjacent pairs where the sign flips from positive to negative (or vice versa).
A sign flip means the root — the actual crossing point — must lie somewhere
between those two θ values, because a continuous function that goes from
positive to negative has to pass through zero in between.  This tells us
roughly where the crossing is, but only to within 1/400th of the boundary.

Second, Brent's method zooms in on each bracketed interval to find the exact
root.  It works like a smart version of binary search: it repeatedly cuts the
interval in half, checks which half still contains the sign flip, and discards
the other half.  It also uses a faster interpolation step when the function
behaves smoothly enough.  After a handful of iterations it converges to the
crossing point to within 1e-12 — effectively exact.

Once θ is known, the exact outward wall normal at that point follows from the
derivative of the curve formula.  The particle velocity is then flipped across
that normal (specular reflection), and the particle is placed on the correct
side of the wall.

This exact approach is not optional: the gradient computation (adjoint) needs
to know how the reflection changes when the wall shape changes.  That
sensitivity can only be computed from the analytic curve formula — a mesh-edge
normal carries no information about how the wall moves when the shape
coefficients are adjusted.

## Method

We implemented an adjoint-based shape optimisation framework for a two-dimensional
Direct Simulation Monte Carlo (DSMC) system governed by Nanbu-Babovsky collision
dynamics.  The domain boundary is parameterised as a Fourier star-shape:

    r(θ; C) = c₀ + Σₖ₌₁ᴺ [aₖ cos(2πkθ) + bₖ sin(2πkθ)]

so the coefficient vector C ∈ ℝ^(2N+1) fully describes the boundary geometry.
At each discrete time step, the simulation performs (i) a Nanbu-Babovsky collision
sub-step that randomises particle velocities in energy-conserving pairs, followed by
(ii) a free-flight and specular-reflection sub-step that bounces particles off the
Fourier boundary.  The full forward trajectory is recorded so that the backward
(adjoint) pass can propagate sensitivity information from the terminal objective
back through every collision and reflection event.

The shape gradient dL/dC is assembled by accumulating, at each boundary-reflection
event, the product of the adjoint vectors (β for velocity, α for position) with the
analytic Jacobians ∂ṽ/∂C and ∂x̃/∂C.  These Jacobians are derived via implicit
differentiation of the boundary intersection equation and the chain rule through the
unit-normal map.  A finite-difference validation confirmed relative errors below 1e-5
on single-particle test cases, establishing that the adjoint implementation is correct.

## Constraint Design and Convergence Theory

### Why the constraint matters as much as the objective

A shape optimisation problem has two parts: an objective to minimise and a constraint
that prevents degenerate solutions (e.g. collapsing to a point).  The interaction
between these two components determines whether the expected optimal shape — a circle,
in our symmetric experiment — is actually the mathematical global minimum.

### The perimeter constraint does not give a circle (for N_FOURIER ≥ 3)

A natural physical constraint is a perimeter (material) budget: the boundary cannot
use more material than the initial shape.  This is intuitive — it models a construction
cost.  However, for the mean |x|² objective, a perimeter constraint does not make the
circle the global minimum when three or more Fourier modes are present.

To see why, consider adding a small k-lobe perturbation aₖ cos(2πkθ) to a circle of
radius R.  Maintaining fixed perimeter forces R to decrease slightly to absorb the extra
boundary length the lobes add.  The net effect on mean |x|² (measured at ergodic
equilibrium, where particles fill the domain uniformly) is:

    Δ(mean |x|²) ≈ aₖ² · (5 − k²) / 4

For k = 1 (elliptical) and k = 2 (two-lobe), the coefficient (5 − k²)/4 is positive,
so the perturbation increases mean |x|² and the circle is a local minimum.  But for
k ≥ 3 (three or more lobes), (5 − k²) < 0, so the perturbation decreases mean |x|².
The optimizer is doing the right thing when it moves away from the circle — a
three-lobe shape genuinely has lower mean |x|² under the perimeter constraint.  The
circle is not the global minimum, and no starting shape will converge to it.

### The area constraint provably gives a circle (for all N_FOURIER)

Replacing the perimeter constraint with a fixed-area constraint resolves this.  For
a Fourier star-shape, the enclosed area is given exactly in closed form:

    Area(C) = π c₀² + (π/2) Σₖ (aₖ² + bₖ²)

This follows from Parseval's theorem applied to r(θ; C)² integrated over [0, 1].  The
gradient is equally simple: ∂Area/∂c₀ = 2π c₀ and ∂Area/∂aₖ = π aₖ.

Under fixed area, the same perturbation analysis gives:

    Δ(mean |x|²) ≈ aₖ² > 0    for all k ≥ 1

Every Fourier perturbation — regardless of mode index — increases mean |x|².  This
follows from Jensen's inequality applied to the polar-coordinate integral:

    mean |x|² = (1/(2π)) ∫r²dθ / (1/(2π)) ∫r²dθ · (π/2) ∫r⁴ dθ / ∫r² dθ ≥ Area/(2π)

with equality only when r is constant (a circle).  The circle is the unique global
minimum for any number of Fourier modes, regardless of starting shape.

### Practical implementation

The area constraint is imposed as a quadratic penalty added to the objective gradient:

    ∇L_total = ∇L_DSMC + λ · (Area(C) − A_target) · ∇Area(C) + λ_box · ∇P_box

where A_target = Area(C_init) is the area of the initial shape (preserved throughout
optimisation) and λ = 30.  The box penalty prevents the boundary from expanding beyond
the outer box.  Because ∇Area(C) is available in closed form, no quadrature is needed
for the constraint gradient, and the penalty has negligible runtime cost.

### Two-phase experimental strategy

The two experiments serve complementary purposes:

| Phase | Experiment | Constraint | Objective | Provably optimal |
|-------|------------|------------|-----------|-----------------|
| 1 | Simple | Fixed area | mean \|x\|² | Circle (Jensen) |
| 2 | Complex | Fixed area | mean Gaussian overlap with L-region | Asymmetric shape biased toward L |

Phase 1 validates the adjoint method against a known, mathematically guaranteed
optimum.  Phase 2 demonstrates the method's capability on a genuinely asymmetric
problem where no closed-form optimal exists and the interesting result is the shape
the optimiser discovers.

## Simple Experiment

The first experiment uses N = 1000 particles initialised in the annulus r ∈ [0.25, 0.65],
a Fourier truncation of N_FOURIER = 3, and 60 time steps with dt = 0.10.  The objective
is the mean squared Euclidean distance from the origin at final time:

    L = (1/N) Σᵢ |xᵢᴹ|²

which is minimised by a circular boundary under the fixed-area constraint (proven
above via Jensen's inequality).

The initial boundary is a deliberately irregular, multi-lobe star shape
(c₀ = 0.75, a₁ = 0.05, a₂ = 0.10, a₃ = 0.20, b₂ = 0.08, b₃ = 0.06) with prominent
2- and 3-lobe distortions, chosen to be visibly far from circular so that convergence
behaviour is unambiguous.

Optimisation runs for 300 iterations using the Adam optimiser (lr = 3×10⁻³, β₁ = 0.9,
β₂ = 0.999) with N_AVG = 32 independent DSMC realisations per gradient estimate
parallelised across 16 CPU workers.  The Bird parameter e = 10 keeps collision rates
low enough that the adjoint signal survives to the terminal time.  The area penalty
coefficient λ = 30 enforces Area(C) ≈ Area(C_init) = 1.865 throughout optimisation.

Terminal adjoint conditions:  term_beta = 0 (no velocity dependence),
term_alpha = 2 x_M / N  (gradient of mean |x|² at final positions).

Expected result: a₃ and all other mode amplitudes decrease toward zero; c₀ converges
toward √(A_target/π) ≈ 0.770 (the radius of a circle with area A_target).

## Results and Interpretation — Simple Experiment

(To be updated after cluster run.)

The theoretical prediction is: L decreases monotonically (in expectation), all Fourier
mode amplitudes |aₖ|, |bₖ| shrink toward zero, and c₀ converges toward the
equivalent-circle radius.  The boundary evolution plot should show a clear progression
from the irregular multi-lobe initial shape toward a circle.

## Complex Experiment

The second experiment uses the same framework (N = 1000 particles, N_FOURIER = 5,
60 steps, area constraint) but with an asymmetric objective that breaks all rotational
symmetry.  The domain boundary is a degree-5 Fourier star-shape (11 coefficients),
giving the optimiser enough representational freedom to discover a genuinely irregular
optimal shape.

### Objective: maximise Gaussian overlap with the L-region

An L-shaped target region is defined by two arms:

- Vertical arm: x ∈ [−0.35, −0.05], y ∈ [−0.35, 0.35]
- Horizontal arm: x ∈ [−0.35, 0.35], y ∈ [−0.35, −0.05]

The objective is

    L = −(1/N) Σᵢ φ(xᵢᴹ),    φ(x) = Σⱼ exp(−|x − tⱼ|² / 2σ²)

where tⱼ are five Gaussian centres placed inside the L-region (three along the vertical
arm at x = −0.20, two along the horizontal arm at y = −0.20) with bandwidth σ = 0.13.
Minimising L is equivalent to maximising the mean Gaussian overlap of particle positions
with the L-region at the terminal time.

### Why the sum-Gaussian objective is preferred over an indicator function

The indicator 𝟙_{x ∈ L} is non-differentiable: its gradient is zero almost everywhere,
so the adjoint method would produce zero gradients and the optimiser could not move.
The Gaussian sum φ(x) is a smooth, differentiable surrogate: it is large when x is
near a Gaussian centre (inside L) and small elsewhere.  The gradient of the objective
with respect to terminal positions is:

    ∂L/∂xᵢᴹ = (1/N) Σⱼ exp(−|xᵢ − tⱼ|²/2σ²) · (xᵢ − tⱼ) / σ²

This is non-zero and smooth whenever xᵢ is on the "slope" of a Gaussian — that is,
whenever xᵢ has meaningful proximity to the L-region.

### Why a sum and not a max

The original formulation used max_j exp(−|x − tⱼ|²/2σ²), which is not
differentiable at the switching boundaries between nearest centres.  The sum
Σⱼ exp(−|x − tⱼ|²/2σ²) is globally smooth and produces a clean gradient for
all particle positions.

### Provably obvious optimum

At ergodic equilibrium (long-time limit), the particle distribution inside D is
uniform, so:

    E[L] ≈ −(1/Area(D)) ∫∫_D φ(x) dA

Under fixed area constraint, minimising E[L] requires maximising the integral
∫∫_D φ(x) dA — that is, placing the domain D where the Gaussian landscape φ is
largest.  Since φ peaks inside the L-region, the optimal domain shifts toward and
into the L-region.  Because the Fourier star-shape is parameterised around the
origin but the L-region is in the third quadrant (negative x, negative y), the
optimal boundary is an asymmetric shape that extends toward the L-region and is
compressed in the opposite direction.  No circle or ellipse achieves this; the
adjoint method must discover the shape.

## Results and Interpretation — Complex Experiment

(To be updated after cluster run.)

The expected result is a boundary that deforms asymmetrically: it extends toward
the L-region (lower-left quadrant) and compresses elsewhere.  The fraction of
particles inside the L-region should increase from the initial value, and the
Fourier modes b₁ (which controls left/right tilt) and a₂, b₃ (higher-order
asymmetries) should take on non-zero values reflecting the L-shape's geometry.
The fact that the optimal boundary is an irregular, non-circular shape demonstrates
that the adjoint method correctly encodes directional sensitivity from the
asymmetric objective.
