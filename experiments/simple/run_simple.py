"""
Simple DSMC shape optimisation experiment.

Objective : minimise mean kinetic energy ("heat") in the inscribed square
            |x| ≤ INNER_HALF, |y| ≤ INNER_HALF at the final time:

                L = (1/N) Σᵢ 𝟙_inscribed(xᵢᴹ) · ‖vᵢᴹ‖²

            (Same units as ‖v‖²; proportional to translational temperature in
            that region.)  Terminal adjoint: β_M = (2/N) 𝟙 v_M, α_M = 0.

Initial C : an irregular multi-lobe star; optimisation nudges the boundary so
            fewer / slower particles end up in the inner box (lower heat there).

Perimeter : hard cap P_MAX (material budget). Each step uses the largest step in
            [0, LR] such that perimeter(project(C)) ≤ P_MAX after the box projector.

Run:
    python experiments/simple/run_simple.py
Outputs are saved to experiments/simple/.
"""

import sys, warnings, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adjoint import ForwardSimulation
from adjoint.boundary_geometry import gamma, radius_r
from adjoint.shape_gradient import (
    shape_gradient,
    perimeter,
    dr_dC,
    project_step_perimeter_cap,
)
from adjoint.visualization import (
    animate_comparison, plot_convergence,
    plot_objective_landscape_1d, plot_final_density,
)

OUT_DIR = os.path.dirname(__file__)

# ── Parameters ────────────────────────────────────────────────────────────────
N_PARTICLES = 150
N_FOURIER   = 3
DT          = 0.10
N_STEPS     = 20
N_COLL      = N_PARTICLES // 4
SEED_INIT   = 42

R_INIT      = 0.80
BOX_HALF    = 1.20
INNER_HALF  = 0.35
R_MAX       = BOX_HALF

N_ITER      = 150
LR          = 2e-3
LAM_BOX     = 8.0
# Material budget: perimeter must not exceed this (construction / first-wall length).
# Must be ≥ perimeter(C_init); ~20% slack over 2πR_init is a reasonable default.
P_MAX       = 2 * np.pi * R_INIT * 1.22

N_AVG       = 5
G_MAX       = 0.5
C0_MIN      = 0.35
C0_MAX      = 1.10
A_MAX_FRAC  = 0.45


# ── Initial condition: irregular multi-lobe shape ─────────────────────────────
# C = [c0, a1, a2, a3, b1, b2, b3]
# Large a3 creates a 3-lobe structure; a2 + b2 add asymmetric bumps.
# This is visually non-circular and far from the optimal circle.
C_init = np.zeros(2 * N_FOURIER + 1)
C_init[0] = 0.75                  # c0  — mean radius (slightly smaller than R_INIT)
C_init[1] = 0.05                  # a1  — slight x-elongation
C_init[2] = 0.10                  # a2  — 2-lobe mode
C_init[3] = 0.20                  # a3  — dominant 3-lobe mode (creates 3 bumps)
C_init[N_FOURIER + 2] = 0.08      # b2  — breaks left-right symmetry
C_init[N_FOURIER + 3] = 0.06      # b3  — rotates the 3-lobe pattern

# Verify shape is valid (r > 0 everywhere, inside box)
_rs = [radius_r(t, C_init) for t in np.linspace(0, 1, 400)]
print(f'Initial shape:  r ∈ [{min(_rs):.3f}, {max(_rs):.3f}]  '
      f'(min>0: {min(_rs)>0},  max<BOX: {max(_rs)<BOX_HALF})')
_p0 = perimeter(C_init)
assert _p0 <= P_MAX + 1e-6, (
    f'perimeter(C_init)={_p0:.4f} exceeds P_MAX={P_MAX:.4f}; '
    'increase P_MAX or shrink initial Fourier coefficients.')
print(f'Initial perimeter: {_p0:.4f}  (material cap P_MAX={P_MAX:.4f})')


# ── Particles ─────────────────────────────────────────────────────────────────
rng0 = np.random.default_rng(SEED_INIT)
r0   = rng0.uniform(0.25, 0.65, N_PARTICLES)
th0  = rng0.uniform(0, 2 * np.pi, N_PARTICLES)
x0   = np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)])
v0   = rng0.standard_normal((N_PARTICLES, 2))


# ── Objective and adjoint ─────────────────────────────────────────────────────
def _inscribed_mask_square(xM):
    """Particles whose final position lies in the inscribed square."""
    return (np.abs(xM[:, 0]) <= INNER_HALF) & (np.abs(xM[:, 1]) <= INNER_HALF)


def compute_L(history):
    """Mean ‖v‖² over particles, counting only those inside the inscribed square."""
    xM = history.final_positions
    vM = history.final_velocities
    m = _inscribed_mask_square(xM)
    return float(np.mean(m * np.sum(vM ** 2, axis=1)))


def term_beta(vM, xM):
    """∂L/∂v for L = (1/N) Σ 𝟙_i ‖v_i‖²  →  (2/N) 𝟙_i v_i."""
    N = len(xM)
    m = _inscribed_mask_square(xM)
    return (2.0 / N) * m[:, np.newaxis] * vM


def term_alpha(_v, _x):
    """Hard mask: no ∂L/∂x at final time (α_M = 0)."""
    return np.zeros_like(_x)


# ── Penalty helpers ───────────────────────────────────────────────────────────
def box_penalty_grad(C, n=200):
    ts = np.linspace(0, 1, n, endpoint=False)
    g  = np.zeros_like(C)
    for t in ts:
        v = max(radius_r(t, C) - R_MAX, 0.0)
        if v > 0:
            g += 2 * v * dr_dC(t, C)
    return g / n

def _project_C(C, c0_min, c0_max, a_max_frac):
    C = C.copy()
    C[0] = np.clip(C[0], c0_min, c0_max)
    N_f   = (len(C) - 1) // 2
    A_MAX = a_max_frac * C[0]
    for ki in range(1, N_f + 1):
        amp = np.sqrt(C[ki] ** 2 + C[N_f + ki] ** 2)
        if amp > A_MAX:
            scale = A_MAX / amp
            C[ki]       *= scale
            C[N_f + ki] *= scale
    return C

def boundary_pts(C, n=400):
    thetas = np.linspace(0, 1, n)
    return np.array([gamma(t, C) for t in thetas])


# ── Optimisation ──────────────────────────────────────────────────────────────
C = C_init.copy()
obj_hist, perim_hist, grad_norm_hist = [], [], []
C_snapshots = [C.copy()]


def _proj(Z):
    return _project_C(Z, C0_MIN, C0_MAX, A_MAX_FRAC)


print('\nRunning optimisation ...')
for it in range(N_ITER):
    L_vals, gL_vals = [], []
    for s in range(N_AVG):
        sim_  = ForwardSimulation(C, DT, n_coll_pairs=N_COLL, seed=it * N_AVG + s)
        hist_ = sim_.run(x0.copy(), v0.copy(), N_STEPS)
        L_vals.append(compute_L(hist_))
        b_, a_ = hist_.backward_pass(term_beta, term_alpha)
        gL_vals.append(shape_gradient(hist_, b_, a_))

    L   = float(np.mean(L_vals))
    g_L = np.mean(gL_vals, axis=0)
    obj_hist.append(L)
    perim_hist.append(perimeter(C))
    grad_norm_hist.append(float(np.linalg.norm(g_L)))

    P_now = perim_hist[-1]
    total = g_L + LAM_BOX * box_penalty_grad(C)
    g_norm = float(np.linalg.norm(total))
    if g_norm > G_MAX:
        total *= G_MAX / g_norm

    C = project_step_perimeter_cap(C, total, LR, _proj, P_MAX)

    if it % 20 == 0:
        C_snapshots.append(C.copy())
    if it % 25 == 0 or it == N_ITER - 1:
        print(f'  iter {it:3d}: L={L:.4f}  P={P_now:.3f}  '
              f'a3={C[3]:.4f}  b3={C[N_FOURIER+3]:.4f}')

C_snapshots.append(C.copy())
C_opt = C.copy()
pct   = 100 * (obj_hist[0] - obj_hist[-1]) / obj_hist[0]
print(f'\nDone.  L: {obj_hist[0]:.4f} → {obj_hist[-1]:.4f}  ({pct:.1f}% reduction)')
print(f'C_opt = {np.round(C_opt, 4)}')
_rs = [radius_r(t, C_opt) for t in np.linspace(0, 1, 400)]
print(f'r(θ): min={min(_rs):.4f}  max={max(_rs):.4f}')

# ── Final simulations ─────────────────────────────────────────────────────────
sim_init = ForwardSimulation(C_init, DT, n_coll_pairs=N_COLL, seed=99)
hist_init = sim_init.run(x0.copy(), v0.copy(), N_STEPS)
sim_opt   = ForwardSimulation(C_opt,  DT, n_coll_pairs=N_COLL, seed=99)
hist_opt  = sim_opt.run(x0.copy(), v0.copy(), N_STEPS)
L_init, L_opt = compute_L(hist_init), compute_L(hist_opt)
print(f'Eval: L_init={L_init:.4f}  L_opt={L_opt:.4f}')


# ── Figure 1: Convergence + evolution + comparison ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

plot_convergence(obj_hist, grad_norm_hist, ax=axes[0])

ax = axes[1]
ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
    fill=False, edgecolor='k', lw=2))
ax.add_patch(plt.Rectangle((-INNER_HALF, -INNER_HALF), 2*INNER_HALF, 2*INNER_HALF,
    facecolor='lightyellow', edgecolor='darkorange', lw=1.2, alpha=0.6))
cmap = plt.cm.plasma; n_s = len(C_snapshots)
for si, Cs in enumerate(C_snapshots):
    c = cmap(si / max(n_s - 1, 1))
    pts = boundary_pts(Cs)
    ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.4,
            alpha=0.3 + 0.7 * si / max(n_s - 1, 1))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_ITER))
plt.colorbar(sm, ax=ax, label='Iteration')
ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
ax.set_aspect('equal'); ax.set_title('Boundary evolution')

ax = axes[2]
ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
    fill=False, edgecolor='k', lw=2))
ax.add_patch(plt.Rectangle((-INNER_HALF, -INNER_HALF), 2*INNER_HALF, 2*INNER_HALF,
    facecolor='lightyellow', edgecolor='darkorange', lw=1.5, alpha=0.6))
ax.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial')
ax.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label='Optimised')
xf   = hist_opt.final_positions
mask = (np.abs(xf[:, 0]) <= INNER_HALF) & (np.abs(xf[:, 1]) <= INNER_HALF)
ax.scatter(xf[~mask, 0], xf[~mask, 1], s=9,  c='steelblue', alpha=0.45, zorder=3)
ax.scatter(xf[mask,  0], xf[mask,  1], s=16, c='crimson',   alpha=0.90, zorder=4)
ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
ax.set_aspect('equal')
ax.set_title(f'Heat L in inner sq: {L_init:.4f} → {L_opt:.4f}  ({pct:.1f}% ↓)')
ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Simple — minimise heat (mean ‖v‖²) in inscribed square',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'convergence.png'), dpi=140, bbox_inches='tight')
print('Saved convergence.png')

# ── Figure 2: Boundary close-up + polar plot ──────────────────────────────────
fig2, (ax1, ax2_cart) = plt.subplots(1, 2, figsize=(12, 5.5))

ax1.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
    fill=False, edgecolor='k', lw=2))
ax1.add_patch(plt.Rectangle((-INNER_HALF, -INNER_HALF), 2*INNER_HALF, 2*INNER_HALF,
    facecolor='lightyellow', edgecolor='darkorange', lw=1.2, alpha=0.5))
ax1.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial')
ax1.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label='Optimised')
ax1.set_xlim(-1.15, 1.15); ax1.set_ylim(-1.15, 1.15)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.25)
ax1.legend(fontsize=9); ax1.set_title('Boundary shape comparison')

thetas = np.linspace(0, 2 * np.pi, 400)
ax2 = fig2.add_subplot(122, projection='polar')
ax2.plot(thetas, [radius_r(t/(2*np.pi), C_init) for t in thetas],
         color='firebrick', lw=1.8, label='Initial')
ax2.plot(thetas, [radius_r(t/(2*np.pi), C_opt)  for t in thetas],
         color='royalblue', lw=2.2, label='Optimised')
ax2.set_title('Radial profile r(θ)', pad=15)
ax2.legend(fontsize=9, loc='lower right')

plt.suptitle('Simple experiment — boundary shape', fontsize=12, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'boundary_comparison.png'), dpi=140, bbox_inches='tight')
print('Saved boundary_comparison.png')

# ── Figure 3: Density maps ────────────────────────────────────────────────────
fig3 = plot_final_density(hist_init, hist_opt, BOX_HALF, INNER_HALF,
    label_a=f'Initial  (heat L={L_init:.4f})',
    label_b=f'Optimised (heat L={L_opt:.4f})')
fig3.savefig(os.path.join(OUT_DIR, 'density.png'), dpi=140, bbox_inches='tight')
print('Saved density.png')

# ── Figure 4: Landscape ───────────────────────────────────────────────────────
def quick_L(C_, seed=0, n_avg=4):
    vals = [compute_L(ForwardSimulation(C_, DT, n_coll_pairs=N_COLL, seed=seed+s)
                      .run(x0.copy(), v0.copy(), N_STEPS)) for s in range(n_avg)]
    return float(np.mean(vals))

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
d_a3 = np.zeros_like(C_opt); d_a3[3] = 1.0
plot_objective_landscape_1d(C_opt, d_a3, (-0.15, 0.15), quick_L, n_points=13, ax=axes4[0])
axes4[0].set_title('Sweep $a_3$ (3-lobe mode)  at $C_{\\mathrm{opt}}$')

sim_tmp = ForwardSimulation(C_opt, DT, n_coll_pairs=N_COLL, seed=0)
hist_tmp = sim_tmp.run(x0.copy(), v0.copy(), N_STEPS)
b_tmp, a_tmp = hist_tmp.backward_pass(term_beta, term_alpha)
g_opt = shape_gradient(hist_tmp, b_tmp, a_tmp)
if np.linalg.norm(g_opt) > 1e-10:
    d_grad = g_opt / np.linalg.norm(g_opt)
    plot_objective_landscape_1d(C_opt, d_grad, (-0.15, 0.15), quick_L, n_points=13, ax=axes4[1])
    axes4[1].set_title('Sweep gradient direction at $C_{\\mathrm{opt}}$')
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'landscape.png'), dpi=140, bbox_inches='tight')
print('Saved landscape.png')

# ── GIF ───────────────────────────────────────────────────────────────────────
print('Rendering GIF ...')
ani = animate_comparison(hist_init, C_init, hist_opt, C_opt, BOX_HALF, INNER_HALF,
    label_a=f'Initial (heat L={L_init:.4f})',
    label_b=f'Optimised (heat L={L_opt:.4f})', interval=200)
ani.save(os.path.join(OUT_DIR, 'comparison.gif'), writer='pillow', fps=5)
print('Saved comparison.gif\nAll done.')
