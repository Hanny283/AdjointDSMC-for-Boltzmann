"""
Validation 3: Convergence to the provably optimal circle.

Objective : L = (1/N) Σ |x_M|²
Constraint: area(C) ≈ A_TARGET  (quadratic penalty)

Theory (Jensen's inequality): for ANY fixed-area domain, the circle minimises
the time-averaged mean-squared radius of a uniformly distributed particle
ensemble at equilibrium.  Hence the gradient descent must drive the Fourier
modes a_k, b_k (k ≥ 1) toward zero while keeping c_0 ≈ sqrt(A_TARGET/π).

Checks
------
  1. Objective decreases: L_final < L_initial
  2. Shape approaches circle: ||C_opt[1:]||₂ < ||C_init[1:]||₂
     (all non-constant Fourier modes shrink)
  3. Area preserved: |area(C_opt) − A_TARGET| / A_TARGET < 5%

Run time: ~40 s  (N_PARTICLES=200, N_STEPS=20, N_ITER=60, N_AVG=8).
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # adds src/ to path
warnings.filterwarnings("ignore")

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from adjoint import ForwardSimulation
from adjoint.boundary_geometry import gamma, radius_r
from adjoint.shape_gradient import shape_gradient, area, area_gradient, dr_dC

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  [{tag}]  {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ── Parameters ────────────────────────────────────────────────────────────────
N_FOURIER   = 2
N_PARTICLES = 200
N_STEPS     = 20
DT          = 0.10
BIRD_E      = 10.0
MESH_SIZE   = 0.25
N_BP        = 80
N_ITER      = 60
LR          = 6e-3
LR_MIN      = 5e-4
LAM_AREA    = 30.0
LAM_BOX     = 8.0
BOX_HALF    = 1.20
R_MAX       = BOX_HALF
N_AVG       = 8
N_WORKERS   = 4

C0_MIN     = 0.35
C0_MAX     = 1.10
A_MAX_FRAC = 0.45

C_init = np.zeros(2 * N_FOURIER + 1)
C_init[0] = 0.75
C_init[1] = 0.05
C_init[2] = 0.18
C_init[N_FOURIER + 2] = 0.10

rng0 = np.random.default_rng(42)
r0   = rng0.uniform(0.25, 0.65, N_PARTICLES)
th0  = rng0.uniform(0, 2*np.pi, N_PARTICLES)
x0   = np.column_stack([r0*np.cos(th0), r0*np.sin(th0)])
v0   = rng0.standard_normal((N_PARTICLES, 2))


def compute_L(history):
    xM = history.final_positions
    return float(np.mean(np.sum(xM**2, axis=1)))

def term_beta(_v, _x):
    return np.zeros_like(_v)

def term_alpha(_v, xM):
    return 2.0 * xM / len(xM)

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
        amp = np.sqrt(C[ki]**2 + C[N_f + ki]**2)
        if amp > A_MAX:
            scale = A_MAX / amp
            C[ki]       *= scale
            C[N_f + ki] *= scale
    return C

def _forward_sim(C, seed):
    return ForwardSimulation(C, DT, seed=seed, e=BIRD_E,
                             mesh_size=MESH_SIZE, num_boundary_points=N_BP)

def _run_one(args):
    C, seed = args
    sim  = _forward_sim(C, seed)
    hist = sim.run(x0.copy(), v0.copy(), N_STEPS)
    L_   = compute_L(hist)
    b_, a_ = hist.backward_pass(term_beta, term_alpha)
    g_   = shape_gradient(hist, b_, a_)
    return L_, g_


def main():
    # ── Run optimisation ──────────────────────────────────────────────────────
    print("\n=== Convergence to circle (2-mode, N_ITER=60) ===")
    A_TARGET = area(C_init)
    C = C_init.copy()
    obj_hist = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for it in range(N_ITER):
            seeds   = [(C, it * N_AVG + s) for s in range(N_AVG)]
            results = list(pool.map(_run_one, seeds))

            L   = float(np.mean([r[0] for r in results]))
            g_L = np.mean([r[1] for r in results], axis=0)
            obj_hist.append(L)

            A_now     = area(C)
            area_viol = A_now - A_TARGET
            total = g_L + LAM_AREA * area_viol * area_gradient(C) + LAM_BOX * box_penalty_grad(C)

            lr_t = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * it / N_ITER))
            C = _project_C(C - lr_t * total)

            if it % 20 == 0 or it == N_ITER - 1:
                print(f"  iter {it:3d}: L={L:.4f}  A={A_now:.4f}  "
                      f"a2={C[2]:.4f}  b2={C[N_FOURIER+2]:.4f}  lr={lr_t:.4f}  "
                      f"[{time.time()-t0:.0f}s]")

    C_opt = C.copy()

    # Evaluate final L with a clean run
    sim_eval = _forward_sim(C_opt, 999)
    L_final  = np.mean([compute_L(sim_eval.run(x0.copy(), v0.copy(), N_STEPS))
                        for _ in range(4)])
    sim_init = _forward_sim(C_init, 998)
    L_initial = np.mean([compute_L(sim_init.run(x0.copy(), v0.copy(), N_STEPS))
                         for _ in range(4)])

    print(f"\n  L_initial (eval) = {L_initial:.4f}")
    print(f"  L_final   (eval) = {L_final:.4f}")
    print(f"  C_init modes ||C[1:]|| = {np.linalg.norm(C_init[1:]):.4f}")
    print(f"  C_opt  modes ||C[1:]|| = {np.linalg.norm(C_opt[1:]):.4f}")
    print(f"  A_TARGET = {A_TARGET:.4f}   area(C_opt) = {area(C_opt):.4f}")

    ALL_PASS = True
    print()

    ALL_PASS &= check("Objective decreased", L_final < L_initial,
                      f"{L_initial:.4f} → {L_final:.4f}")

    ALL_PASS &= check("Fourier modes shrank toward circle",
                      np.linalg.norm(C_opt[1:]) < np.linalg.norm(C_init[1:]),
                      f"||C[1:]||: {np.linalg.norm(C_init[1:]):.4f} → "
                      f"{np.linalg.norm(C_opt[1:]):.4f}")

    area_err = abs(area(C_opt) - A_TARGET) / A_TARGET
    ALL_PASS &= check("Area preserved within 5%", area_err < 0.05,
                      f"rel_err={area_err:.3f}")

    print()
    if ALL_PASS:
        print(f"[{PASS}]  Convergence checks passed.")
    else:
        print(f"[{FAIL}]  Some convergence checks failed — see above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
