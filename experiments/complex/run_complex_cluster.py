"""
Complex DSMC shape optimisation — cluster version.

Objective : maximise mean Gaussian overlap with an L-shaped target region,
            implemented as minimising

                L = −(1/N) Σᵢ φ(xᵢᴹ),    φ(x) = Σⱼ exp(−|x − tⱼ|² / 2σ²)

            where tⱼ are centres distributed inside the L-region and σ = 0.13.

Constraint: area(C) ≈ A_TARGET  (quadratic penalty, coefficient LAM_AREA)

Theory (provably obvious optimum):
  At ergodic equilibrium the particle distribution is uniform in D, so
      E[L] ≈ −(1/Area(D)) ∫∫_D φ(x) dA.
  Minimising E[L] under fixed area = maximising ∫∫_D φ(x) dA, i.e., shifting
  the domain to overlap maximally with the L-shaped Gaussian landscape.
  The qualitatively obvious optimum is a boundary that extends into/toward
  the L-region and is compressed elsewhere — an asymmetric star-shape biased
  in the direction of the L-target.  No circle or ellipse achieves this; the
  adjoint method must discover the optimal irregular shape.

Run:
    python experiments/complex/run_complex_cluster.py
Outputs saved to experiments/complex/.
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
from concurrent.futures import ProcessPoolExecutor

from adjoint import ForwardSimulation
from adjoint.boundary_geometry import gamma, radius_r
from adjoint.shape_gradient import (
    shape_gradient,
    area,
    area_gradient,
    dr_dC,
)
from adjoint.visualization import (
    animate_comparison, plot_convergence,
    plot_objective_landscape_1d, plot_final_density,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parameters ────────────────────────────────────────────────────────────────
BIRD_E = 10.0
MESH_SIZE           = 0.25
NUM_BOUNDARY_POINTS = 100

N_PARTICLES = 1000
N_FOURIER   = 5
DT          = 0.10
N_STEPS     = 60
SEED_INIT   = 7

R_INIT   = 0.80
BOX_HALF = 1.20
R_MAX    = BOX_HALF

N_ITER    = 300
LR        = 5e-3          # Adam learning rate (decays to LR_MIN over training)
LR_MIN    = 5e-4          # cosine-annealing floor
LAM_BOX   = 8.0
LAM_AREA  = 30.0          # area-equality penalty: keeps Area(C) ≈ A_TARGET

N_AVG     = 32
N_WORKERS = 16

BETA1    = 0.9
BETA2    = 0.999
EPS_ADAM = 1e-8

C0_MIN     = 0.35
C0_MAX     = 1.10
A_MAX_FRAC = 0.45

# ── L-shaped target region ────────────────────────────────────────────────────
# Vertical arm:   x ∈ [−0.35, −0.05],  y ∈ [−0.35, 0.35]
# Horizontal arm: x ∈ [−0.35,  0.35],  y ∈ [−0.35, −0.05]
L_VERTS = [(-0.35,  0.35), (-0.05,  0.35), (-0.05, -0.05),
           ( 0.35, -0.05), ( 0.35, -0.35), (-0.35, -0.35)]

# Gaussian centres inside the L-region (3 along vertical arm, 2 along horizontal)
_TARGET_PTS = np.array([
    [-0.20,  0.20],   # vertical arm, top
    [-0.20,  0.00],   # vertical arm, middle
    [-0.20, -0.20],   # vertical arm, bottom (corner of L)
    [ 0.00, -0.20],   # horizontal arm
    [ 0.20, -0.20],   # horizontal arm, far end
])
_SIGMA = 0.13


def _phi(x):
    """Sum-Gaussian landscape over L-shape centres.  Shape (len(x),)."""
    diff = x[:, np.newaxis, :] - _TARGET_PTS[np.newaxis, :, :]   # (N, T, 2)
    sq   = np.sum(diff ** 2, axis=2)                              # (N, T)
    return np.sum(np.exp(-sq / (2 * _SIGMA ** 2)), axis=1)        # (N,)


def _inscribed_mask_L_polygon(xM, verts=L_VERTS):
    """(N,) bool — positions inside the L polygon (winding number rule)."""
    n   = len(verts)
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


# ── Initial condition: irregular multi-lobe shape ─────────────────────────────
# N_FOURIER=5 → C has 11 entries: [c0, a1..a5, b1..b5]
C_init = np.zeros(2 * N_FOURIER + 1)
C_init[0] = 0.75
C_init[2] = 0.18
C_init[4] = 0.10
C_init[N_FOURIER + 1] = 0.14
C_init[N_FOURIER + 3] = 0.12
C_init[N_FOURIER + 5] = 0.08

# ── Fixed particle ensemble ───────────────────────────────────────────────────
rng0 = np.random.default_rng(SEED_INIT)
r0   = rng0.uniform(0.25, 0.65, N_PARTICLES)
th0  = rng0.uniform(0, 2 * np.pi, N_PARTICLES)
x0   = np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)])
v0   = rng0.standard_normal((N_PARTICLES, 2))


# ── Objective and terminal adjoint ────────────────────────────────────────────
# L = −(1/N) Σᵢ φ(xᵢᴹ)   →   maximise Gaussian overlap with L-region.
def compute_L(history):
    xM = history.final_positions
    return -float(np.mean(_phi(xM)))


def term_beta(_v, _x):
    return np.zeros_like(_v)   # L has no velocity dependence


def term_alpha(_v, xM):
    """
    ∂L/∂xᵢᴹ = (1/N) Σⱼ exp(−|xᵢ−tⱼ|²/2σ²) · (xᵢ−tⱼ)/σ²
    Points away from each Gaussian centre, weighted by proximity.
    Gradient descent will push the domain toward the centres.
    """
    N    = len(xM)
    diff = xM[:, np.newaxis, :] - _TARGET_PTS[np.newaxis, :, :]  # (N, T, 2)
    sq   = np.sum(diff ** 2, axis=2)                              # (N, T)
    w    = np.exp(-sq / (2 * _SIGMA ** 2))                        # (N, T)
    return np.sum(w[:, :, np.newaxis] * diff, axis=1) / (_SIGMA ** 2 * N)


# ── Diagnostic: fraction of particles inside L ────────────────────────────────
def compute_L_count(history):
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

def _project_C(C):
    C = C.copy()
    C[0] = np.clip(C[0], C0_MIN, C0_MAX)
    N_f   = (len(C) - 1) // 2
    A_MAX = A_MAX_FRAC * C[0]
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
    poly = MplPoly(L_VERTS, closed=True)
    pc   = PatchCollection([poly], facecolor='mediumseagreen',
                           edgecolor='green', lw=1.5, alpha=alpha, zorder=zorder)
    ax.add_collection(pc)


# ── Parallel worker ───────────────────────────────────────────────────────────
def _forward_sim(C, seed):
    return ForwardSimulation(
        C,
        DT,
        n_coll_pairs=None,
        seed=seed,
        e=BIRD_E,
        mesh_size=MESH_SIZE,
        num_boundary_points=NUM_BOUNDARY_POINTS,
    )


def _run_one(args):
    C, seed = args
    sim  = _forward_sim(C, seed)
    hist = sim.run(x0.copy(), v0.copy(), N_STEPS)
    L_   = compute_L(hist)
    b_, a_ = hist.backward_pass(term_beta, term_alpha)
    g_   = shape_gradient(hist, b_, a_)
    return L_, g_


# ── Main optimisation ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    _rs = [radius_r(t, C_init) for t in np.linspace(0, 1, 600)]
    print(f'N_PARTICLES={N_PARTICLES}  N_FOURIER={N_FOURIER}  '
          f'N_STEPS={N_STEPS}  N_AVG={N_AVG}  N_WORKERS={N_WORKERS}  '
          f'BIRD_E={BIRD_E}  mesh={MESH_SIZE}')
    print(f'Initial shape:  r ∈ [{min(_rs):.3f}, {max(_rs):.3f}]  '
          f'(min>0: {min(_rs)>0},  max<BOX: {max(_rs)<BOX_HALF})')
    A_TARGET = area(C_init)
    print(f'Initial area: {A_TARGET:.4f}  (target for area penalty; '
          f'equivalent circle radius R≈{np.sqrt(A_TARGET/np.pi):.4f})')

    C      = C_init.copy()
    adam_m = np.zeros_like(C)
    adam_v = np.zeros_like(C)

    obj_hist, area_hist, grad_norm_hist = [], [], []
    C_snapshots = [C.copy()]
    t_start = time.time()

    print('\nRunning optimisation (Adam, parallelised) ...')
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for it in range(N_ITER):
            seeds = [(C, it * N_AVG + s) for s in range(N_AVG)]
            results = list(pool.map(_run_one, seeds))

            L   = float(np.mean([r[0] for r in results]))
            g_L = np.mean([r[1] for r in results], axis=0)
            obj_hist.append(L)
            A_now = area(C)
            area_hist.append(A_now)
            grad_norm_hist.append(float(np.linalg.norm(g_L)))

            # Area penalty keeps Area(C) ≈ A_TARGET
            area_viol = A_now - A_TARGET
            total = g_L + LAM_AREA * area_viol * area_gradient(C) + LAM_BOX * box_penalty_grad(C)

            # Adam update with cosine-annealed learning rate
            t_adam = it + 1
            lr_t   = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * it / N_ITER))
            adam_m = BETA1 * adam_m + (1 - BETA1) * total
            adam_v = BETA2 * adam_v + (1 - BETA2) * total ** 2
            m_hat  = adam_m / (1 - BETA1 ** t_adam)
            v_hat  = adam_v / (1 - BETA2 ** t_adam)
            step_dir = m_hat / (np.sqrt(v_hat) + EPS_ADAM)
            C = _project_C(C - lr_t * step_dir)

            if it % 30 == 0:
                C_snapshots.append(C.copy())
            if it % 20 == 0 or it == N_ITER - 1:
                elapsed = time.time() - t_start
                frac = compute_L_count(
                    _forward_sim(C, 9999).run(x0.copy(), v0.copy(), N_STEPS))
                print(f'  iter {it:3d}: L={L:.4f}  A={A_now:.4f}  '
                      f'in_L={100*frac:.1f}%  lr={lr_t:.4f}  [{elapsed:.0f}s elapsed]')

    C_snapshots.append(C.copy())
    C_opt = C.copy()
    pct   = 100 * (obj_hist[0] - obj_hist[-1]) / max(abs(obj_hist[0]), 1e-12)
    elapsed_total = time.time() - t_start
    print(f'\nDone in {elapsed_total:.0f}s.  '
          f'L: {obj_hist[0]:.4f} → {obj_hist[-1]:.4f}  ({pct:.1f}% reduction)')
    print(f'C_opt = {np.round(C_opt, 4)}')
    _rs = [radius_r(t, C_opt) for t in np.linspace(0, 1, 400)]
    print(f'r(θ): min={min(_rs):.4f}  max={max(_rs):.4f}  all_positive={min(_rs)>0}')

    # ── Final evaluation ──────────────────────────────────────────────────────
    sim_init  = _forward_sim(C_init, 99)
    hist_init = sim_init.run(x0.copy(), v0.copy(), N_STEPS)
    sim_opt   = _forward_sim(C_opt, 99)
    hist_opt  = sim_opt.run(x0.copy(), v0.copy(), N_STEPS)
    L_init    = compute_L(hist_init)
    L_opt     = compute_L(hist_opt)
    f_init    = compute_L_count(hist_init)
    f_opt     = compute_L_count(hist_opt)
    print(f'\nEval: L_init={L_init:.4f}  L_opt={L_opt:.4f}')
    print(f'Fraction in L-shape: {100*f_init:.1f}% → {100*f_opt:.1f}%')

    # ── Figure 1: Convergence + evolution + comparison ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    plot_convergence(obj_hist, grad_norm_hist, ax=axes[0])
    axes[0].set_title('Convergence  (L = −mean φ in L-region; Adam, 32 real./iter)')

    ax = axes[1]
    ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
        fill=False, edgecolor='k', lw=2))
    _draw_L(ax, alpha=0.18)
    cmap = plt.cm.plasma; n_s = len(C_snapshots)
    for si, Cs in enumerate(C_snapshots):
        c   = cmap(si / max(n_s - 1, 1))
        pts = boundary_pts(Cs)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.3,
                alpha=0.3 + 0.7 * si / max(n_s - 1, 1))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_ITER))
    plt.colorbar(sm, ax=ax, label='Iteration')
    ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
    ax.set_aspect('equal'); ax.set_title('Boundary evolution (green = L-region)')

    ax = axes[2]
    ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
        fill=False, edgecolor='k', lw=2))
    _draw_L(ax, alpha=0.25)
    ax.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial')
    ax.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label='Optimised')
    xf = hist_opt.final_positions
    inside = _inscribed_mask_L_polygon(xf)
    ax.scatter(xf[~inside, 0], xf[~inside, 1], s=9,  c='steelblue', alpha=0.45, zorder=3)
    ax.scatter(xf[ inside, 0], xf[ inside, 1], s=18, c='crimson', alpha=0.90, zorder=4,
               label=f'In L ({inside.sum()})')
    ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
    ax.set_aspect('equal')
    ax.set_title(f'in L: {100*f_init:.0f}% → {100*f_opt:.0f}%  |  '
                 f'φ: {-L_init:.3f} → {-L_opt:.3f}')
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Complex — maximise Gaussian overlap with L-region  (cluster, Adam)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'convergence.png'), dpi=140, bbox_inches='tight')
    print('Saved convergence.png')

    # ── Figure 2: Boundary comparison ────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (Cv, lbl, col) in zip(axes2,
            [(C_init, 'Initial', 'firebrick'), (C_opt, 'Optimised', 'royalblue')]):
        ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
            fill=False, edgecolor='k', lw=2))
        _draw_L(ax, alpha=0.22)
        pts = boundary_pts(Cv)
        ax.fill(pts[:, 0], pts[:, 1], alpha=0.10, color=col)
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.5, label=lbl)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9); ax.set_title(lbl)

    plt.suptitle('Complex — boundary before/after', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'boundary_comparison.png'), dpi=140, bbox_inches='tight')
    print('Saved boundary_comparison.png')

    # ── Figure 3: Density maps ────────────────────────────────────────────────
    fig3 = plot_final_density(hist_init, hist_opt, BOX_HALF, 0.35,
        label_a=f'Initial  (in_L={100*f_init:.0f}%)',
        label_b=f'Optimised (in_L={100*f_opt:.0f}%)')
    fig3.savefig(os.path.join(OUT_DIR, 'density.png'), dpi=140, bbox_inches='tight')
    print('Saved density.png')

    # ── Figure 4: Landscape along b₁ ─────────────────────────────────────────
    def quick_L(C_, seed=0, n_avg=8):
        vals = [
            compute_L(_forward_sim(C_, seed + s).run(x0.copy(), v0.copy(), N_STEPS))
            for s in range(n_avg)
        ]
        return float(np.mean(vals))

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
    d_b1 = np.zeros_like(C_opt); d_b1[N_FOURIER + 1] = 1.0
    plot_objective_landscape_1d(C_opt, d_b1, (-0.20, 0.20), quick_L, n_points=17, ax=axes4[0])
    axes4[0].set_title('Sweep $b_1$ (asymmetric tilt) at $C_{\\mathrm{opt}}$')

    sim_tmp  = _forward_sim(C_opt, 0)
    hist_tmp = sim_tmp.run(x0.copy(), v0.copy(), N_STEPS)
    b_tmp, a_tmp = hist_tmp.backward_pass(term_beta, term_alpha)
    g_opt = shape_gradient(hist_tmp, b_tmp, a_tmp)
    if np.linalg.norm(g_opt) > 1e-10:
        d_grad = g_opt / np.linalg.norm(g_opt)
        plot_objective_landscape_1d(C_opt, d_grad, (-0.20, 0.20), quick_L,
                                    n_points=17, ax=axes4[1])
        axes4[1].set_title('Sweep gradient direction at $C_{\\mathrm{opt}}$')
    plt.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, 'landscape.png'), dpi=140, bbox_inches='tight')
    print('Saved landscape.png')

    # ── GIF ───────────────────────────────────────────────────────────────────
    print('Rendering GIF ...')
    ani = animate_comparison(hist_init, C_init, hist_opt, C_opt, BOX_HALF, 0.35,
        label_a=f'Initial  (in_L={100*f_init:.0f}%)',
        label_b=f'Optimised (in_L={100*f_opt:.0f}%)', interval=200)
    ani.save(os.path.join(OUT_DIR, 'comparison.gif'), writer='pillow', fps=5)
    print('Saved comparison.gif\nAll done.')
