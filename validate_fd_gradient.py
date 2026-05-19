"""
Validation 2: Finite-difference gradient check.

Tests two things:

  A. Gradient direction (stochastic sign check)
     Taking a step C ← C − lr·g should decrease the expected objective L.
     Verified by averaging N_AVG=8 realisations before and after the step.

  B. Analytical gradient correctness (deterministic FD)
     The functions dv_reflected_dC and dx_reflected_dC compute the exact
     Jacobians of the boundary reflection maps w.r.t. C (including the chain
     rule through θ_inter(C)).  We verify them against numerical finite
     differences by:
       1. Fixing a particle (x_k, v_prime) that intersects the boundary.
       2. For each perturbed C ± ε e_j:
            re-solve θ_inter, recompute the reflection.
       3. Compare (v_refl_+ − v_refl_−)/(2ε) and (x_refl_+ − x_refl_−)/(2ε)
          to the analytical column of dv_dC / dx_dC.
     This is deterministic (no randomness), so relative errors should be O(ε²).

Run time: ~ 30 s.
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np

from adjoint import ForwardSimulation
from adjoint.boundary_geometry import normal_n, solve_theta_inter, compute_c_inter
from adjoint.shape_gradient import (
    shape_gradient, dv_reflected_dC, dx_reflected_dC, area,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  [{tag}]  {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ── Shared problem parameters ─────────────────────────────────────────────────
N_FOURIER   = 2
N_PARTICLES = 80
N_STEPS     = 8
DT          = 0.10
BIRD_E      = 10.0
MESH_SIZE   = 0.25
N_BP        = 80
N_AVG       = 8

C = np.array([0.75, 0.05, 0.18, 0.0, 0.10])   # 2-mode distorted shape

rng0 = np.random.default_rng(42)
r0   = rng0.uniform(0.25, 0.55, N_PARTICLES)
th0  = rng0.uniform(0, 2*np.pi, N_PARTICLES)
x0   = np.column_stack([r0*np.cos(th0), r0*np.sin(th0)])
v0   = rng0.standard_normal((N_PARTICLES, 2))

def compute_L(history):
    xM = history.final_positions
    return float(np.mean(np.sum(xM**2, axis=1)))

def term_beta(_v, _x):   return np.zeros_like(_v)
def term_alpha(_v, xM):  return 2.0 * xM / len(xM)

def reflect_at(x_k, v_prime, C_):
    """Return (x_tilde, v_tilde) for the specular reflection, or None if no hit."""
    th = solve_theta_inter(x_k, v_prime, C_)
    if th is None:
        return None
    n  = normal_n(th, C_)
    c  = compute_c_inter(th, C_)
    x_prime = x_k + DT * v_prime
    v_tilde = v_prime - 2.0 * np.dot(n, v_prime) * n
    x_tilde = x_prime - 2.0 * (np.dot(n, x_prime) - c) * n
    return x_tilde, v_tilde, th, x_prime


ALL_PASS = True

# ── A. Gradient direction (stochastic) ───────────────────────────────────────
print("\n=== A. Gradient direction ===")

def avg_L_and_grad(C_, seed_base):
    sim = ForwardSimulation(C_, DT, seed=None, e=BIRD_E,
                            mesh_size=MESH_SIZE, num_boundary_points=N_BP)
    Ls, grads = [], []
    for s in range(N_AVG):
        sim._rng = np.random.default_rng(seed_base + s)
        hist = sim.run(x0.copy(), v0.copy(), N_STEPS)
        L = compute_L(hist)
        b, a = hist.backward_pass(term_beta, term_alpha)
        g = shape_gradient(hist, b, a)
        Ls.append(L); grads.append(g)
    return float(np.mean(Ls)), np.mean(grads, axis=0)

L_nom, g_nom = avg_L_and_grad(C, seed_base=100)
print(f"  Nominal L = {L_nom:.5f}   ||grad|| = {np.linalg.norm(g_nom):.4f}")

lr_test = 1e-2
C_step  = C - lr_test * g_nom
L_step, _ = avg_L_and_grad(C_step, seed_base=200)
delta = L_step - L_nom
ok_A = delta < 0
ALL_PASS &= check("L(C − lr·g) < L(C)", ok_A,
                  f"ΔL = {delta:+.5f}  (negative = correct descent)")

# ── B. Analytical FD check of dv_reflected_dC and dx_reflected_dC ────────────
print("\n=== B. Analytical gradient FD check (deterministic) ===")

eps_fd   = 1e-5
rng_test = np.random.default_rng(7)

# Find test particles that actually hit the boundary under C
test_cases = []
for _ in range(200):
    r_s  = rng_test.uniform(0.05, 0.40)
    th_s = rng_test.uniform(0, 2*np.pi)
    x_k  = r_s * np.array([np.cos(th_s), np.sin(th_s)])
    v_p  = rng_test.standard_normal(2)
    v_p /= np.linalg.norm(v_p)
    result = reflect_at(x_k, v_p, C)
    if result is not None:
        test_cases.append((x_k, v_p, result))
    if len(test_cases) == 5:
        break

print(f"  Found {len(test_cases)} test particles that hit the boundary")

for trial, (x_k, v_prime, (x_tilde, v_tilde, theta_nom, x_prime)) in enumerate(test_cases):
    # Analytical gradients
    dv_an = dv_reflected_dC(theta_nom, v_prime, C)   # (2, 2N+1)
    dx_an = dx_reflected_dC(theta_nom, x_prime, v_prime, C)  # (2, 2N+1)

    dv_fd = np.zeros_like(dv_an)
    dx_fd = np.zeros_like(dx_an)
    skip  = False

    for j in range(len(C)):
        Cp = C.copy(); Cm = C.copy()
        Cp[j] += eps_fd; Cm[j] -= eps_fd

        rp = reflect_at(x_k, v_prime, Cp)
        rm = reflect_at(x_k, v_prime, Cm)
        if rp is None or rm is None:
            skip = True; break

        x_tilde_p, v_tilde_p, _, x_prime_p = rp
        x_tilde_m, v_tilde_m, _, x_prime_m = rm
        dv_fd[:, j] = (v_tilde_p - v_tilde_m) / (2 * eps_fd)
        dx_fd[:, j] = (x_tilde_p - x_tilde_m) / (2 * eps_fd)

    if skip:
        print(f"    trial {trial+1}: skipped (no intersection on perturbed C)")
        continue

    norm_dv = np.linalg.norm(dv_fd)
    norm_dx = np.linalg.norm(dx_fd)
    err_dv  = np.linalg.norm(dv_an - dv_fd) / (norm_dv + 1e-12)
    err_dx  = np.linalg.norm(dx_an - dx_fd) / (norm_dx + 1e-12)

    ok_dv = err_dv < 0.02   # < 2% relative error
    ok_dx = err_dx < 0.02
    print(f"    trial {trial+1}: dv_dC rel_err={err_dv:.4f} {'OK' if ok_dv else 'FAIL'} | "
          f"dx_dC rel_err={err_dx:.4f} {'OK' if ok_dx else 'FAIL'}")
    ALL_PASS &= ok_dv
    ALL_PASS &= ok_dx

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if ALL_PASS:
    print(f"[{PASS}]  Gradient checks passed.")
else:
    print(f"[{FAIL}]  Some gradient checks failed — see above.")
    sys.exit(1)
