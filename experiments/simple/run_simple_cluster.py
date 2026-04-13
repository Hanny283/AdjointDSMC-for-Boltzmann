"""
Simple DSMC shape optimisation — cluster version.

Objective : minimise mean kinetic energy ("heat") in the inscribed square
            (|x|,|y| ≤ INNER_HALF) at final time:
            L = (1/N) Σᵢ 𝟙_inscribed(xᵢᴹ) ‖vᵢᴹ‖²

Improvements over the local script:
  • N_AVG gradient realisations parallelised across N_WORKERS CPU cores.
  • Adam optimiser with per-coefficient adaptive learning rates.
  • N_PARTICLES≈1000, N_STEPS≈50–80, and Bird's parameter e≈5–20 so collision
    rate is reduced (adjoint signal survives to the final time; see table in docs).
  • Explicit BIRD_E, MESH_SIZE, NUM_BOUNDARY_POINTS for ForwardSimulation.
  • Hard perimeter cap P_MAX (material budget); step length is reduced if needed.

Expected runtime: scales with N_PARTICLES·N_STEPS·N_AVG (e.g. ~10–30 min for
300 iterations on a 16-core CPU depending on mesh build cost).

Run:
    python experiments/simple/run_simple_cluster.py
Outputs saved to experiments/simple/.
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
    perimeter,
    dr_dC,
    project_step_perimeter_cap,
)
from adjoint.visualization import (
    animate_comparison, plot_convergence,
    plot_objective_landscape_1d, plot_final_density,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Parameters ────────────────────────────────────────────────────────────────
# Bird's parameter e: expected cell collisions scale as 1/e.  Larger e → fewer
# collisions per step (typical range 5–20 for boundary-dominated gradients).
BIRD_E = 10.0

# Triangle mesh for cell-based collisions (see ForwardSimulation).
MESH_SIZE           = 0.25   # smaller → more cells (good for N_PARTICLES≈1000)
NUM_BOUNDARY_POINTS = 100

N_PARTICLES = 1000
N_FOURIER   = 3
DT          = 0.10
N_STEPS     = 60           # 50–80 recommended for more boundary interactions
SEED_INIT   = 42

R_INIT     = 0.80
BOX_HALF   = 1.20
INNER_HALF = 0.35
R_MAX      = BOX_HALF

N_ITER    = 300
LR        = 3e-3          # Adam learning rate
LAM_BOX   = 8.0
P_MAX     = 2 * np.pi * R_INIT * 1.22   # material budget; must be ≥ perimeter(C_init)

N_AVG     = 32            # gradient realisations per iteration (32–64 typical)
N_WORKERS = 16            # parallel workers (physical cores)

# Adam hyper-parameters
BETA1    = 0.9
BETA2    = 0.999
EPS_ADAM = 1e-8

C0_MIN     = 0.35
C0_MAX     = 1.10
A_MAX_FRAC = 0.45

# ── Initial condition: irregular multi-lobe shape ─────────────────────────────
# C = [c0, a1, a2, a3, b1, b2, b3]
# Dominant 3-lobe (a3) + 2-lobe (a2) + asymmetric breaks (b2, b3).
C_init = np.zeros(2 * N_FOURIER + 1)
C_init[0] = 0.75
C_init[1] = 0.05
C_init[2] = 0.10
C_init[3] = 0.20
C_init[N_FOURIER + 2] = 0.08
C_init[N_FOURIER + 3] = 0.06

# ── Fixed particle ensemble ───────────────────────────────────────────────────
rng0 = np.random.default_rng(SEED_INIT)
r0   = rng0.uniform(0.25, 0.65, N_PARTICLES)
th0  = rng0.uniform(0, 2 * np.pi, N_PARTICLES)
x0   = np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)])
v0   = rng0.standard_normal((N_PARTICLES, 2))


# ── Objective and terminal adjoint ────────────────────────────────────────────
def _inscribed_mask_square(xM):
    return (np.abs(xM[:, 0]) <= INNER_HALF) & (np.abs(xM[:, 1]) <= INNER_HALF)


def compute_L(history):
    xM = history.final_positions
    vM = history.final_velocities
    m = _inscribed_mask_square(xM)
    return float(np.mean(m * np.sum(vM ** 2, axis=1)))


def term_beta(vM, xM):
    N = len(xM)
    m = _inscribed_mask_square(xM)
    return (2.0 / N) * m[:, np.newaxis] * vM


def term_alpha(_v, _x):
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


# ── Parallel worker (module-level so ProcessPoolExecutor can pickle it) ───────
# On Linux the default fork start method gives workers access to all globals
# (x0, v0, DT, N_STEPS, compute_L, BIRD_E, …) without re-passing them.
def _forward_sim(C, seed):
    return ForwardSimulation(
        C,
        DT,
        n_coll_pairs=None,  # unused; collision rate from Bird's formula + cells
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
    _p0 = perimeter(C_init)
    assert _p0 <= P_MAX + 1e-6, (
        f'perimeter(C_init)={_p0:.4f} exceeds P_MAX={P_MAX:.4f}; '
        'increase P_MAX or shrink initial Fourier coefficients.')
    print(f'Initial perimeter: {_p0:.4f}  (material cap P_MAX={P_MAX:.4f})')

    C      = C_init.copy()
    adam_m = np.zeros_like(C)
    adam_v = np.zeros_like(C)

    obj_hist, perim_hist, grad_norm_hist = [], [], []
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
            P_now = perimeter(C)
            perim_hist.append(P_now)
            grad_norm_hist.append(float(np.linalg.norm(g_L)))

            total = g_L + LAM_BOX * box_penalty_grad(C)

            # Adam update, then enforce hard perimeter cap (material budget)
            t_adam = it + 1
            adam_m = BETA1 * adam_m + (1 - BETA1) * total
            adam_v = BETA2 * adam_v + (1 - BETA2) * total ** 2
            m_hat  = adam_m / (1 - BETA1 ** t_adam)
            v_hat  = adam_v / (1 - BETA2 ** t_adam)
            step_dir = m_hat / (np.sqrt(v_hat) + EPS_ADAM)
            C = project_step_perimeter_cap(C, step_dir, LR, _project_C, P_MAX)

            if it % 30 == 0:
                C_snapshots.append(C.copy())
            if it % 20 == 0 or it == N_ITER - 1:
                elapsed = time.time() - t_start
                print(f'  iter {it:3d}: L={L:.4f}  P={P_now:.3f}  '
                      f'a3={C[3]:.4f}  b3={C[N_FOURIER+3]:.4f}  '
                      f'[{elapsed:.0f}s elapsed]')

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
    ax.add_patch(plt.Rectangle((-INNER_HALF, -INNER_HALF), 2*INNER_HALF, 2*INNER_HALF,
        facecolor='lightyellow', edgecolor='darkorange', lw=1.2, alpha=0.45, zorder=1))
    ax.plot(*boundary_pts(C_init).T, color='firebrick',  lw=2,   label='Initial')
    ax.plot(*boundary_pts(C_opt).T,  color='royalblue',  lw=2.5, label='Optimised')
    xf = hist_opt.final_positions
    mk = _inscribed_mask_square(xf)
    ax.scatter(xf[~mk, 0], xf[~mk, 1], s=9,  c='steelblue', alpha=0.45, zorder=3)
    ax.scatter(xf[mk,  0], xf[mk,  1], s=14, c='crimson',   alpha=0.85, zorder=4)
    ax.set_xlim(-BOX_HALF*1.1, BOX_HALF*1.1); ax.set_ylim(-BOX_HALF*1.1, BOX_HALF*1.1)
    ax.set_aspect('equal')
    ax.set_title(f'Heat L in inner sq: {L_init:.4f} → {L_opt:.4f}  ({pct:.1f}% ↓)')
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Simple — minimise heat in inscribed square  (cluster, Adam)',
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

    plt.suptitle('Simple — boundary shape', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, 'boundary_comparison.png'), dpi=140, bbox_inches='tight')
    print('Saved boundary_comparison.png')

    # ── Figure 3: Density maps ────────────────────────────────────────────────
    fig3 = plot_final_density(hist_init, hist_opt, BOX_HALF, INNER_HALF,
        label_a=f'Initial  (heat L={L_init:.4f})',
        label_b=f'Optimised (heat L={L_opt:.4f})')
    fig3.savefig(os.path.join(OUT_DIR, 'density.png'), dpi=140, bbox_inches='tight')
    print('Saved density.png')

    # ── Figure 4: Landscape sweep along a₃ ───────────────────────────────────
    def quick_L(C_, seed=0, n_avg=8):
        vals = [
            compute_L(_forward_sim(C_, seed + s).run(x0.copy(), v0.copy(), N_STEPS))
            for s in range(n_avg)
        ]
        return float(np.mean(vals))

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4))
    d_a3 = np.zeros_like(C_opt); d_a3[3] = 1.0
    plot_objective_landscape_1d(C_opt, d_a3, (-0.20, 0.20), quick_L, n_points=17, ax=axes4[0])
    axes4[0].set_title('Sweep $a_3$ (3-lobe mode) at $C_{\\mathrm{opt}}$')

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
        label_a=f'Initial (heat L={L_init:.4f})',
        label_b=f'Optimised (heat L={L_opt:.4f})', interval=200)
    ani.save(os.path.join(OUT_DIR, 'comparison.gif'), writer='pillow', fps=5)
    print('Saved comparison.gif\nAll done.')
