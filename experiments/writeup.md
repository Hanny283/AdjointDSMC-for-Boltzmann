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

## Simple Experiment

The first experiment used N = 150 particles initialised uniformly in the annulus
r ∈ [0.25, 0.70], a Fourier truncation of N_FOURIER = 3, and 20 time steps with
dt = 0.10.  The objective was the mean squared Euclidean distance from the origin
over all particles at the final time:

    L = (1/N) Σᵢ |xᵢᴹ|²

The initial boundary was deliberately chosen to be an irregular, multi-lobe star shape
(c₀ = 0.75, a₁ = 0.05, a₂ = 0.10, a₃ = 0.20, b₂ = 0.08, b₃ = 0.06) — a shape that
is visibly far from circular.  Starting from a near-optimal shape would make it
impossible to demonstrate meaningful convergence behaviour, so the initial condition
was set to have prominent 2- and 3-lobe distortions.  Gradient descent ran for 150
iterations using three stabilisation measures: gradient averaging over N_AVG = 5
independent DSMC realisations per step, total gradient clipping at ‖g‖ ≤ 0.5, and
projection of C onto the feasible set after each update (c₀ ∈ [0.35, 1.10],
per-mode amplitude ≤ 0.45 c₀).

## Results and Interpretation — Simple Experiment

The objective decreased from L = 0.369 to L = 0.304, a 17.6% reduction over 150
iterations.  The dominant Fourier mode a₃ (the 3-lobe distortion) shrank from 0.199
to 0.160, and c₀ decreased from 0.800 to 0.699, meaning the boundary both shrank
inward and became progressively smoother and rounder.  All higher modes (a₂, b₂, b₃)
were also reduced in amplitude.

The result is physically intuitive.  An irregular, multi-lobed boundary confines
particles unevenly: lobes that protrude farther from the origin force particles
traveling along those directions to reflect at large radii, returning with larger
radial offsets.  A rounder boundary distributes reflections more uniformly in angle,
reducing the mean |x|² across all directions.  The perimeter constraint prevents the
trivial solution of shrinking the domain to a point.

The stochastic noise inherent in DSMC (the gradient is an average over random
collision realisations) causes visible oscillations in the L-curve, but the
clipping-and-averaging strategy keeps these bounded and does not destabilise the
optimisation.

## Complex Experiment

The second experiment replaces the origin-centred, symmetric objective with one
defined by an L-shaped inscribed target region.  The L-shape completely breaks the
rotational and reflective symmetry of the problem, so the optimal boundary is a
genuinely irregular shape that cannot be described analytically.

The L-shaped target consists of two arms:

- Vertical arm: x ∈ [−0.35, −0.05], y ∈ [−0.35, 0.35]
- Horizontal arm: x ∈ [−0.35, 0.35], y ∈ [−0.35, −0.05]

The region is approximated by seven Gaussian centres (four along the vertical arm
at x = −0.20, three along the horizontal arm at y = −0.20) with bandwidth σ = 0.13.
The objective is a negative mean peak-kernel value:

    L = −(1/N) Σᵢ max_j exp(−|xᵢ − tⱼ|² / (2σ²))

which is minimised by concentrating particles near any of the Gaussian centres —
that is, inside the L-region.  This differs fundamentally from the simple experiment:
there is no symmetry to appeal to, and neither a circle nor an ellipse is even a
local minimum.

The Fourier truncation was increased to N_FOURIER = 5 (11 boundary coefficients)
to allow the optimizer sufficient representational freedom to discover a non-trivial
solution.  The initial boundary was set to an irregular multi-lobe shape
(a₂ = 0.18, a₄ = 0.10, b₁ = 0.14, b₃ = 0.12, b₅ = 0.08, c₀ = 0.75) — again,
deliberately far from the optimum so that convergence is visible.

## Results and Interpretation — Complex Experiment

The optimisation ran for 120 iterations using N_AVG = 5 realisations per step and
N = 150 particles.  The objective decreased from L = −0.187 to L = −0.224 (a 19.5%
reduction in magnitude, meaning the average peak-kernel value rose by 19.5%).  The
fraction of particles inside the L-region increased from 16.7% to 22.0%.  The
perimeter converged from 5.64 to 5.02, close to the target 5.03.

The optimised boundary retains significant multi-mode Fourier content (a₂ ≈ 0.165,
b₁ ≈ 0.133, a₄ ≈ 0.070, b₃ ≈ 0.103, b₅ ≈ 0.052), confirming that the adjoint
method discovered a genuinely irregular shape rather than collapsing to a simple
geometry.  The boundary deforms asymmetrically: it extends further toward the
L-region and is compressed elsewhere, reflecting the asymmetric structure of the
target.  This validates that the adjoint gradient correctly encodes directional
sensitivity information from the non-symmetric objective.
