"""
Complex DSMC shape optimisation experiment.

Objective : minimise mean kinetic energy ("heat") inside the L-shaped inscribed
            polygon L_VERTS at the final time:

                L = (1/N) Σᵢ 𝟙_L(xᵢᴹ) · ‖vᵢᴹ‖²

            Terminal adjoint: β_M = (2/N) 𝟙_L v_M, α_M = 0 (hard mask).

            The L-shape breaks rotational symmetry, so interesting boundaries
            are typically non-circular.

Initial C : irregular multi-lobe (N_FOURIER = 5).

Perimeter : hard cap P_MAX (material budget); each step is shortened if needed so
            perimeter(project(C)) ≤ P_MAX.

Run:
    python experiments/complex/run_complex.py
Outputs are saved to experiments/complex/.
"""

import sys, warnings, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

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
N_FOURIER   = 5
DT          = 0.10
N_STEPS     = 20
N_COLL      = N_PARTICLES // 4
SEED_INIT   = 7

R_INIT      = 0.80
BOX_HALF    = 1.20
R_MAX       = BOX_HALF

N_ITER      = 120
LR          = 2e-3
LAM_BOX     = 8.0
P_MAX       = 2 * np.pi * R_INIT * 1.22   # material budget; must be ≥ perimeter(C_init)

N_AVG       = 5
G_MAX       = 0.5
C0_MIN      = 0.35
C0_MAX      = 1.10
A_MAX_FRAC  = 0.45

# ── L-shaped target region ────────────────────────────────────────────────────
# Vertical arm: x ∈ [−0.35, −0.05],  y ∈ [−0.35, 0.35]
# Horizontal arm: x ∈ [−0.35,  0.35],  y ∈ [−0.35, −0.05]
L_VERTS = [(-0.35, 0.35), (-0.05, 0.35), (-0.05, -0.05),
           ( 0.35,-0.05), ( 0.35,-0.35), (-0.35,-0.35)]  # L polygon (closed)


def _inscribed_mask_L_polygon(xM, verts=L_VERTS):
    """(N,) bool — positions inside the L polygon (even–odd ray rule)."""
    n = len(verts)
    pts = list(verts) + [verts[0]]
    out = np.zeros(len(xM), dtype=bool)
    for i, (x, y) in enumerate(xM):
        w = 0
        for k in range(n):
            x1, y1 = pts[k]; x2, y2 = pts[k + 1]
            if y1 <= y < y2 or y2 <= y < y1:
                xint = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                if x < xint:
                    w += 1
        out[i] = bool(w % 2)
    return out


# ── Objective and adjoint ─────────────────────────────────────────────────────
def compute_L(history):
    """Mean ‖v‖², counting only particles whose final position lies inside L."""
    xM = history.final_positions
    vM = history.final_velocities
    m = _inscribed_mask_L_polygon(xM)
    return float(np.mean(m * np.sum(vM ** 2, axis=1)))


def term_beta(vM, xM):
    N = len(xM)
    m = _inscribed_mask_L_polygon(xM)
    return (2.0 / N) * m[:, np.newaxis] * vM


def term_alpha(_v, _x):
    return np.zeros_like(_x)


def compute_L_count(history):
    """Fraction of particles inside the L polygon (diagnostic)."""
    return float(_inscribed_mask_L_polygon(history.final_positions).mean())


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
    return np.array([gamma(t, C) for t in np.linspace(0, 1, n)])


def _draw_L(ax, alpha=0.25, zorder=1):
    """Draw the L-shaped inscribed region as a filled polygon."""
    poly = MplPoly(L_VERTS, closed=True)
    pc   = PatchCollection([poly], facecolor='mediumseagreen',
                           edgecolor='green', lw=1.5, alpha=alpha, zorder=zorder)
    ax.add_collection(pc)


# ── Initial condition: irregular multi-lobe shape ─────────────────────────────
# Use several Fourier modes so the starting shape is clearly non-trivial.
# N_FOURIER=5 → C has 11 entries: [c0, a1..a5, b1..b5]
C_init = np.zeros(2 * N_FOURIER + 1)
C_init[0] = 0.75          # c0
C_init[2] = 0.18          # a2  — 2-lobe mode
C_init[4] = 0.10          # a4  — 4-lobe mode
C_init[N_FOURIER + 1] = 0.14   # b1  — asymmetric tilt
C_init[N_FOURIER + 3] = 0.12   # b3
C_init[N_FOURIER + 5] = 0.08   # b5  — fine ripple

_rs = [radius_r(t, C_init) for t in np.linspace(0, 1, 600)]
print(f'N_FOURIER={N_FOURIER}  (C length {2*N_FOURIER+1})')
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

    if it % 25 == 0:
        C_snapshots.append(C.copy())
    if it % 30 == 0 or it == N_ITER - 1:
        frac = compute_L_count(
            ForwardSimulation(C, DT, n_coll_pairs=N_COLL, seed=999)
            .run(x0.copy(), v0.copy(), N_STEPS))
        print(f'  iter {it:3d}: L={L:.4f}  P={P_now:.3f}  |g|={g_norm:.2f}  '
              f'in_L={100*frac:.1f}%')

C_snapshots.append(C.copy())
C_opt = C.copy()
pct   = 100 * (obj_hist[0] - obj_hist[-1]) / max(abs(obj_hist[0]), 1e-12)
print(f'\nDone.  L: {obj_hist[0]:.4f} → {obj_hist[-1]:.4f}  ({pct:.1f}% reduction)')
print(f'C_opt = {np.round(C_opt, 4)}')
_rs = [radius_r(t, C_opt) for t in np.linspace(0, 1, 400)]
print(f'r(θ): min={min(_rs):.4f}  max={max(_rs):.4f}  all_positive={min(_rs)>0}')
print(f'Perimeter: {perimeter(C_init):.4f} → {perimeter(C_opt):.4f}  (cap P_MAX={P_MAX:.4f})')


# ── Final evaluation ──────────────────────────────────────────────────────────
sim_init  = ForwardSimulation(C_init, DT, n_coll_pairs=N_COLL, seed=99)
hist_init = sim_init.run(x0.copy(), v0.copy(), N_STEPS)
sim_opt   = ForwardSimulation(C_opt,  DT, n_coll_pairs=N_COLL, seed=99)
hist_opt  = sim_opt.run(x0.copy(), v0.copy(), N_STEPS)
L_init    = compute_L(hist_init)
L_opt     = compute_L(hist_opt)
f_init    = compute_L_count(hist_init)
f_opt     = compute_L_count(hist_opt)
print(f'\nEval: L_init={L_init:.4f}  L_opt={L_opt:.4f}')
print(f'Fraction in L-shape: {100*f_init:.1f}% → {100*f_opt:.1f}%')


# ── Figure 1: Convergence + evolution + comparison ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

plot_convergence(obj_hist, grad_norm_hist, ax=axes[0])
axes[0].set_title('Convergence  (L = mean ‖v‖² in L-shaped region)')

ax = axes[1]
ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
    fill=False, edgecolor='k', lw=2))
_draw_L(ax, alpha=0.20)
cmap = plt.cm.plasma; n_s = len(C_snapshots)
for si, Cs in enumerate(C_snapshots):
    c   = cmap(si / max(n_s - 1, 1))
    pts = boundary_pts(Cs)
    ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.3,
            alpha=0.3 + 0.7 * si / max(n_s - 1, 1))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_ITER))
plt.colorbar(sm, ax=ax, label='Iteration')
ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
ax.set_aspect('equal'); ax.set_title('Boundary evolution\n(green = inscribed L, minimise heat there)')

ax = axes[2]
ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
    fill=False, edgecolor='k', lw=2))
_draw_L(ax, alpha=0.25)
ax.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label=f'Initial')
ax.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label=f'Optimised')
xf = hist_opt.final_positions
in_L = _inscribed_mask_L_polygon(xf)
ax.scatter(xf[~in_L, 0], xf[~in_L, 1], s=9,  c='steelblue', alpha=0.45, zorder=4)
ax.scatter(xf[in_L,  0], xf[in_L,  1], s=16, c='crimson',   alpha=0.90, zorder=5,
           label=f'In L (heat term) ({in_L.sum()})')
ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
ax.set_aspect('equal')
ax.set_title(f'Heat L: {L_init:.4f} → {L_opt:.4f}  |  in L: {100*f_init:.0f}% → {100*f_opt:.0f}%')
ax.legend(fontsize=7, loc='upper right')

plt.suptitle(f'Complex optimisation — minimise heat in L, N_FOURIER={N_FOURIER}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'convergence.png'), dpi=140, bbox_inches='tight')
print('Saved convergence.png')

# ── Figure 2: Boundary close-up + polar ──────────────────────────────────────
fig2 = plt.figure(figsize=(13, 5.5))

ax1 = fig2.add_subplot(121)
ax1.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
    fill=False, edgecolor='k', lw=2))
_draw_L(ax1, alpha=0.30)
ax1.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial (irregular)')
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

plt.suptitle('Complex experiment — optimised boundary', fontsize=12, fontweight='bold')
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'boundary_comparison.png'), dpi=140, bbox_inches='tight')
print('Saved boundary_comparison.png')

# ── Figure 3: Density maps ────────────────────────────────────────────────────
fig3 = plot_final_density(hist_init, hist_opt, BOX_HALF, 0.5,
    label_a=f'Initial  (in L: {100*f_init:.0f}%)',
    label_b=f'Optimised (in L: {100*f_opt:.0f}%)', n_bins=35)
for ax_ in fig3.axes[:2]:
    _draw_L(ax_, alpha=0.20, zorder=5)
fig3.savefig(os.path.join(OUT_DIR, 'density.png'), dpi=140, bbox_inches='tight')
print('Saved density.png')

# ── Figure 4: Landscape ───────────────────────────────────────────────────────
def quick_L(C_, seed=0, n_avg=4):
    vals = [compute_L(ForwardSimulation(C_, DT, n_coll_pairs=N_COLL, seed=seed+s)
                      .run(x0.copy(), v0.copy(), N_STEPS)) for s in range(n_avg)]
    return float(np.mean(vals))

fig4, axes4 = plt.subplots(1, 3, figsize=(17, 4))

d_b1 = np.zeros_like(C_opt); d_b1[N_FOURIER + 1] = 1.0
plot_objective_landscape_1d(C_opt, d_b1, (-0.12, 0.12), quick_L, n_points=13, ax=axes4[0])
axes4[0].set_title('Sweep $b_1$ (sin-1 mode)')

d_a2 = np.zeros_like(C_opt); d_a2[2] = 1.0
plot_objective_landscape_1d(C_opt, d_a2, (-0.12, 0.12), quick_L, n_points=13, ax=axes4[1])
axes4[1].set_title('Sweep $a_2$ (cos-2 mode)')

sim_tmp  = ForwardSimulation(C_opt, DT, n_coll_pairs=N_COLL, seed=0)
hist_tmp = sim_tmp.run(x0.copy(), v0.copy(), N_STEPS)
b_tmp, a_tmp = hist_tmp.backward_pass(term_beta, term_alpha)
g_opt = shape_gradient(hist_tmp, b_tmp, a_tmp)
if np.linalg.norm(g_opt) > 1e-10:
    d_grad = g_opt / np.linalg.norm(g_opt)
    plot_objective_landscape_1d(C_opt, d_grad, (-0.12, 0.12), quick_L,
                                n_points=13, ax=axes4[2])
    axes4[2].set_title('Sweep gradient direction at $C_{\\mathrm{opt}}$')

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'landscape.png'), dpi=140, bbox_inches='tight')
print('Saved landscape.png')

# ── GIF ───────────────────────────────────────────────────────────────────────
print('Rendering GIF ...')
ani = animate_comparison(hist_init, C_init, hist_opt, C_opt, BOX_HALF, 0.5,
    label_a=f'Initial (in L: {100*f_init:.0f}%)',
    label_b=f'Optimised (in L: {100*f_opt:.0f}%)', interval=200)
ani.save(os.path.join(OUT_DIR, 'comparison.gif'), writer='pillow', fps=5)
print('Saved comparison.gif\nAll done.')
