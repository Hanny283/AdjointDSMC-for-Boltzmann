"""
Validation 1: Boundary geometry — pure mathematical checks, no simulation.

Tests:
  1. Unit normal:       ||n(θ; C)|| = 1
  2. Orthogonality:     n(θ; C) · γ'(θ; C) = 0
  3. Outward normal:    n points outward (dot with γ(θ) > 0 for convex-ish shapes)
  4. Intersection:      solve_theta_inter returns a point that lies on the boundary
                        and is in the forward direction of travel
  5. Area formula:      closed-form area(C) matches numerical quadrature
  6. Perimeter:         perimeter(C) matches numerical quadrature of ||γ'||
  7. dr_dC finite diff: ∂r/∂C matches finite-difference approximation
  8. area_gradient FD:  ∂Area/∂C matches finite-difference approximation

Runs in < 5 seconds.  All checks print PASS / FAIL with tolerance information.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from adjoint.boundary_geometry import (
    radius_r, gamma, gamma_prime, normal_n,
    solve_theta_inter, f_unnormalized,
)
from adjoint.shape_gradient import (
    area, area_gradient, perimeter, perimeter_gradient, dr_dC,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  [{tag}]  {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ── Test shapes ───────────────────────────────────────────────────────────────
# Simple circle
C_circle = np.array([0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # c0=0.75, rest 0

# Distorted 3-mode shape (same as simple experiment init)
C_dist = np.array([0.75, 0.05, 0.10, 0.20, 0.0, 0.08, 0.06])

SHAPES = [("circle  (c0=0.75)", C_circle), ("distorted 3-mode", C_dist)]

ALL_PASS = True
eps_fd = 1e-6
thetas_test = np.linspace(0.05, 0.95, 20)   # avoid θ=0 corner

# ── 1 & 2: Unit normal and orthogonality ─────────────────────────────────────
print("\n=== 1–2. Unit normal and orthogonality ===")
for shape_name, C in SHAPES:
    errs_norm  = []
    errs_ortho = []
    for th in thetas_test:
        n  = normal_n(th, C)
        gp = gamma_prime(th, C)
        errs_norm.append(abs(np.linalg.norm(n) - 1.0))
        errs_ortho.append(abs(np.dot(n, gp)))
    ok1 = max(errs_norm)  < 1e-12
    ok2 = max(errs_ortho) < 1e-10
    ALL_PASS &= check(f"||n||=1          [{shape_name}]", ok1,
                      f"max_err={max(errs_norm):.2e}")
    ALL_PASS &= check(f"n ⊥ γ'           [{shape_name}]", ok2,
                      f"max_err={max(errs_ortho):.2e}")

# ── 3: Outward normal ─────────────────────────────────────────────────────────
print("\n=== 3. Outward normal ===")
for shape_name, C in SHAPES:
    dots = []
    for th in thetas_test:
        n = normal_n(th, C)
        g = gamma(th, C)
        dots.append(np.dot(n, g))
    ok = all(d > 0 for d in dots)
    ALL_PASS &= check(f"n · γ > 0        [{shape_name}]", ok,
                      f"min={min(dots):.4f}")

# ── 4: Intersection solve ─────────────────────────────────────────────────────
print("\n=== 4. Intersection solve ===")
rng = np.random.default_rng(0)
for shape_name, C in SHAPES:
    n_trials, n_ok = 30, 0
    for _ in range(n_trials):
        # Random interior point + random direction
        r_sample = rng.uniform(0.1, 0.5)
        th_sample = rng.uniform(0, 2 * np.pi)
        x = r_sample * np.array([np.cos(th_sample), np.sin(th_sample)])
        v = rng.standard_normal(2)
        v /= np.linalg.norm(v)

        th_inter = solve_theta_inter(x, v, C)
        if th_inter is None:
            continue
        hit = gamma(th_inter, C)
        # Check point is on the boundary: r(theta) * e_r == gamma
        r_at = radius_r(th_inter, C)
        expected = r_at * np.array([np.cos(2*np.pi*th_inter),
                                     np.sin(2*np.pi*th_inter)])
        if np.linalg.norm(hit - expected) < 1e-8:
            # Check forward direction
            if np.dot(hit - x, v) > 0:
                n_ok += 1
    ok = n_ok == n_trials
    ALL_PASS &= check(f"θ_inter on boundary & forward [{shape_name}]", ok,
                      f"{n_ok}/{n_trials} trials passed")

# ── 5: Area formula vs quadrature ─────────────────────────────────────────────
print("\n=== 5. Area formula vs quadrature ===")
for shape_name, C in SHAPES:
    # Exact closed-form
    A_exact = area(C)
    # Numerical: Green's theorem A = (1/2) ∮ (x dy - y dx)
    n_q = 10000
    ths = np.linspace(0, 1, n_q, endpoint=False)
    gs  = np.array([gamma(t, C) for t in ths])
    gps = np.array([gamma_prime(t, C) for t in ths])
    # ∫ (x * dy/dθ - y * dx/dθ) dθ / 2
    A_num = 0.5 * np.mean(gs[:,0]*gps[:,1] - gs[:,1]*gps[:,0])
    err = abs(A_exact - A_num) / A_num
    ok = err < 1e-4
    ALL_PASS &= check(f"area formula     [{shape_name}]", ok,
                      f"exact={A_exact:.6f}  quad={A_num:.6f}  rel_err={err:.2e}")

# ── 6: Perimeter vs quadrature ────────────────────────────────────────────────
print("\n=== 6. Perimeter vs quadrature ===")
for shape_name, C in SHAPES:
    P_fn  = perimeter(C, n_quad=2000)
    n_q   = 2000
    ths   = np.linspace(0, 1, n_q, endpoint=False)
    norms = np.array([np.linalg.norm(gamma_prime(t, C)) for t in ths])
    P_num = norms.mean()
    err   = abs(P_fn - P_num) / P_num
    ok    = err < 1e-8
    ALL_PASS &= check(f"perimeter        [{shape_name}]", ok,
                      f"fn={P_fn:.6f}  quad={P_num:.6f}  rel_err={err:.2e}")

# ── 7: dr_dC finite difference ────────────────────────────────────────────────
print("\n=== 7. dr_dC finite difference ===")
for shape_name, C in SHAPES:
    max_err = 0.0
    for th in thetas_test[:5]:
        grad_an = dr_dC(th, C)
        grad_fd = np.zeros_like(C)
        for j in range(len(C)):
            Cp = C.copy(); Cm = C.copy()
            Cp[j] += eps_fd; Cm[j] -= eps_fd
            grad_fd[j] = (radius_r(th, Cp) - radius_r(th, Cm)) / (2 * eps_fd)
        max_err = max(max_err, np.max(np.abs(grad_an - grad_fd)))
    ok = max_err < 1e-8
    ALL_PASS &= check(f"dr_dC finite diff [{shape_name}]", ok,
                      f"max_err={max_err:.2e}")

# ── 8: area_gradient finite difference ────────────────────────────────────────
print("\n=== 8. area_gradient finite difference ===")
for shape_name, C in SHAPES:
    grad_an = area_gradient(C)
    grad_fd = np.zeros_like(C)
    for j in range(len(C)):
        Cp = C.copy(); Cm = C.copy()
        Cp[j] += eps_fd; Cm[j] -= eps_fd
        grad_fd[j] = (area(Cp) - area(Cm)) / (2 * eps_fd)
    err = np.max(np.abs(grad_an - grad_fd))
    ok = err < 1e-8
    ALL_PASS &= check(f"area_gradient FD  [{shape_name}]", ok,
                      f"max_err={err:.2e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if ALL_PASS:
    print(f"[{PASS}]  All geometry checks passed.")
else:
    print(f"[{FAIL}]  Some checks failed — see above.")
    sys.exit(1)
