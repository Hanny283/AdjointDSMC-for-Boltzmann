"""
Simple DSMC shape optimisation — 2-mode local quick-test.

Identical to experiments/simple/run_simple_cluster.py but uses N_FOURIER=2
so the boundary is parameterised by only 5 coefficients:

    C = [c₀, a₁, a₂, b₁, b₂]

With fewer modes the problem is lower-dimensional and the adjoint gradient
signal must suppress only 2-lobe (a₂, b₂) and 1-lobe (a₁, b₁) distortions.
Convergence to a circle is faster and serves as a cleaner proof-of-concept.

This is the LOCAL quick-test version (N_PARTICLES=200, N_AVG=4, N_ITER=60).
Results will be noisy but should show the correct directional trend.

Objective : minimise mean squared radius  L = (1/N) Σᵢ |xᵢᴹ|²
Constraint: area(C) ≈ A_TARGET  (quadratic penalty, coefficient LAM_AREA)
Provably optimal shape: circle (Jensen's inequality, fixed area).

Run:
    python experiments/2_mode_simple/run_2mode_simple_cluster.py
Outputs saved to experiments/2_mode_simple/.
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
BIRD_E              = 10.0
MESH_SIZE           = 0.25
NUM_BOUNDARY_POINTS = 100

N_PARTICLES = 200
N_FOURIER   = 2           # only modes k=1,2 → C = [c0, a1, a2, b1, b2]
DT          = 0.10
N_STEPS     = 20
SEED_INIT   = 42

R_INIT     = 0.80
BOX_HALF   = 1.20
INNER_HALF = 0.35
R_MAX      = BOX_HALF

N_ITER    = 60
LR        = 6e-3          # Adam learning rate (decays to LR_MIN over training)
LR_MIN    = 5e-4          # cosine-annealing floor — fine-tunes near circle
LAM_BOX   = 8.0
LAM_AREA  = 30.0          # area-equality penalty: keeps Area(C) ≈ A_TARGET

N_AVG     = 4
N_WORKERS = 4

BETA1    = 0.9
BETA2    = 0.999
EPS_ADAM = 1e-8

C0_MIN     = 0.35
C0_MAX     = 1.10
A_MAX_FRAC = 0.45

# ── Initial condition: 2-lobe distorted shape ─────────────────────────────────
# C = [c0, a1, a2, b1, b2]
# Dominant 2-lobe (a2) + small 1-lobe (a1) + asymmetric break (b2).
C_init = np.zeros(2 * N_FOURIER + 1)
C_init[0] = 0.75
C_init[1] = 0.05          # a1 — slight 1-lobe tilt
C_init[2] = 0.18          # a2 — dominant 2-lobe (ellipse-like)
C_init[N_FOURIER + 2] = 0.10   # b2 — breaks left-right symmetry

# ── Fixed particle ensemble ───────────────────────────────────────────────────
rng0 = np.random.default_rng(SEED_INIT)
r0   = rng0.uniform(0.25, 0.65, N_PARTICLES)
th0  = rng0.uniform(0, 2 * np.pi, N_PARTICLES)
x0   = np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)])
v0   = rng0.standard_normal((N_PARTICLES, 2))


# ── Objective and terminal adjoint ────────────────────────────────────────────
def compute_L(history):
    xM = history.final_positions
    return float(np.mean(np.sum(xM ** 2, axis=1)))

def term_beta(_v, _x):
    return np.zeros_like(_v)

def term_alpha(_v, xM):
    return 2.0 * xM / len(xM)


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
    _rs = [radius_r(t, C_init) for t in np.linspace(0, 1, 400)]
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
                print(f'  iter {it:3d}: L={L:.4f}  A={A_now:.4f}  '
                      f'a2={C[2]:.4f}  b2={C[N_FOURIER+2]:.4f}  '
                      f'lr={lr_t:.4f}  [{elapsed:.0f}s elapsed]')

    C_snapshots.append(C.copy())
    C_opt = C.copy()
    pct   = 100 * (obj_hist[0] - obj_hist[-1]) / max(abs(obj_hist[0]), 1e-12)
    elapsed_total = time.time() - t_start
    print(f'\nDone in {elapsed_total:.0f}s.  '
          f'L: {obj_hist[0]:.4f} → {obj_hist[-1]:.4f}  ({pct:.1f}% reduction)')
    print(f'C_opt = {np.round(C_opt, 4)}')
    _rs = [radius_r(t, C_opt) for t in np.linspace(0, 1, 400)]
    print(f'r(θ): min={min(_rs):.4f}  max={max(_rs):.4f}')

    # ── Final evaluation ──────────────────────────────────────────────────────
    sim_init  = _forward_sim(C_init, 99)
    hist_init = sim_init.run(x0.copy(), v0.copy(), N_STEPS)
    sim_opt   = _forward_sim(C_opt, 99)
    hist_opt  = sim_opt.run(x0.copy(), v0.copy(), N_STEPS)
    L_init, L_opt = compute_L(hist_init), compute_L(hist_opt)
    print(f'Eval: L_init={L_init:.4f}  L_opt={L_opt:.4f}')

    # ── Figure 1: Convergence + evolution + comparison ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_convergence(obj_hist, grad_norm_hist, ax=axes[0])
    axes[0].set_title('Convergence  (Adam, 32 realisations/iter)')

    ax = axes[1]
    ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
        fill=False, edgecolor='k', lw=2))
    cmap = plt.cm.plasma; n_s = len(C_snapshots)
    for si, Cs in enumerate(C_snapshots):
        c   = cmap(si / max(n_s - 1, 1))
        pts = boundary_pts(Cs)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.3,
                alpha=0.3 + 0.7 * si / max(n_s - 1, 1))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_ITER))
    plt.colorbar(sm, ax=ax, label='Iteration')
    ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
    ax.set_aspect('equal'); ax.set_title('Boundary evolution')

    ax = axes[2]
    ax.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
        fill=False, edgecolor='k', lw=2))
    ax.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial')
    ax.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label='Optimised')
    xf = hist_opt.final_positions
    ax.scatter(xf[:, 0], xf[:, 1], s=9, c='steelblue', alpha=0.45, zorder=3)
    ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
    ax.set_aspect('equal')
    ax.set_title(f'mean |x|²: {L_init:.4f} → {L_opt:.4f}  ({pct:.1f}% ↓)')
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('2-mode simple — minimise mean |x|²  (area-constrained, Adam)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'convergence.png'), dpi=140, bbox_inches='tight')
    print('Saved convergence.png')

    # ── Figure 2: Boundary comparison + polar ────────────────────────────────
    fig2, (ax1, _) = plt.subplots(1, 2, figsize=(12, 5.5))
    ax2 = fig2.add_subplot(122, projection='polar')
    fig2.delaxes(_)

    ax1.add_patch(plt.Rectangle((-BOX_HALF, -BOX_HALF), 2*BOX_HALF, 2*BOX_HALF,
        fill=False, edgecolor='k', lw=2))
    ax1.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial')
    ax1.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label='Optimised')
    ax1.set_xlim(-1.15, 1.15); ax1.set_ylim(-1.15, 1.15)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9); ax1.set_title('Boundary shape comparison')

    thetas = np.linspace(0, 2 * np.pi, 400)
    ax2.plot(thetas, [radius_r(t / (2*np.pi), C_init) for t in thetas],
             color='firebrick', lw=1.8, label='Initial')
    ax2.plot(thetas, [radius_r(t / (2*np.pi), C_opt)  for t in thetas],
             color='royalblue', lw=2.2, label='Optimised')
    ax2.set_title('Radial profile r(θ)', pad=15)
    ax2.legend(fontsize=9, loc='lower right')

    plt.suptitle('2-mode simple — boundary shape', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'boundary_comparison.png'), dpi=140, bbox_inches='tight')
    print('Saved boundary_comparison.png')

    # ── Figure 3: Density maps ────────────────────────────────────────────────
    fig3 = plot_final_density(hist_init, hist_opt, BOX_HALF, INNER_HALF,
        label_a=f'Initial  (mean|x|²={L_init:.4f})',
        label_b=f'Optimised (mean|x|²={L_opt:.4f})')
    fig3.savefig(os.path.join(OUT_DIR, 'density.png'), dpi=140, bbox_inches='tight')
    print('Saved density.png')

    # ── Figure 4: Landscape sweep along a₂ ───────────────────────────────────
    def quick_L(C_, seed=0, n_avg=8):
        vals = [
            compute_L(_forward_sim(C_, seed + s).run(x0.copy(), v0.copy(), N_STEPS))
            for s in range(n_avg)
        ]
        return float(np.mean(vals))

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
    d_a2 = np.zeros_like(C_opt); d_a2[2] = 1.0
    plot_objective_landscape_1d(C_opt, d_a2, (-0.20, 0.20), quick_L, n_points=17, ax=axes4[0])
    axes4[0].set_title('Sweep $a_2$ (2-lobe mode) at $C_{\\mathrm{opt}}$')

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
    ani = animate_comparison(hist_init, C_init, hist_opt, C_opt, BOX_HALF, INNER_HALF,
        label_a=f'Initial (mean|x|²={L_init:.4f})',
        label_b=f'Optimised (mean|x|²={L_opt:.4f})', interval=200)
    ani.save(os.path.join(OUT_DIR, 'comparison.gif'), writer='pillow', fps=5)
    print('Saved comparison.gif\nAll done.')
