# `shape_gradient.py` — gradient of L with respect to Fourier coefficients **C**

This module computes **dL/dC** after a forward run and a **`backward_pass`**. It implements **Section 3** of the paper (**Lemmas 3.1–3.4** and the final sum over boundary hits).

## Role in the big picture

- **C** does **not** appear inside the random collision draw in the same way as x, v; it enters **strongly** at **wall reflections** (normal, hit angle, offset **c̃**).
- The adjoint variables **β**<sub>k+1</sub> and **α**<sub>k+1</sub> tell you how much **L** would change if you perturbed **post-bounce** **ṽ** and **x̃** at that event.
- The chain rule gives:

**dL/dC** = Σ over wall hits (k, i) of  
(β<sub>k+1,i</sub><sup>⊤</sup> ∂ṽ/∂**C** + α<sub>k+1,i</sub><sup>⊤</sup> ∂x̃/∂**C**)

(plus any **penalty** terms on **C** added **outside** this file, e.g. perimeter in the experiment scripts).

So: **β, α** carry sensitivity **through time**; **`shape_gradient`** turns that into sensitivity **in coefficient space**.

## How α and β connect to updating **C**

1. **`backward_pass`** → full arrays **`betas`**, **`alphas`**.
2. **`shape_gradient(history, betas, alphas)`** → vector **`g`** same length as **`C`**.
3. The **optimizer** (in `experiments/...`) does **`C ← C − η g`** (plus projection, Adam, perimeter terms, etc.).

**α and β do not update C by themselves** — they are **intermediate** quantities. Only **`shape_gradient`** (plus explicit penalty gradients) produces a direction in **C**.

## Functions (goal and paper link)

| Function | Goal | Paper |
|----------|------|--------|
| `dr_dC(theta, C)` | **∂r/∂C** at fixed θ. | Building block for **∂f/∂C**. |
| `drtheta_dC(theta, C)` | **∂**(**∂r/∂θ**)** / ∂C**. | Same. |
| `_df_dC_fixed`, `_dn_dC_fixed` | **∂f/∂C**, **∂ñ/∂C** **holding θ fixed**. | Toward **Lemma 3.3**. |
| `dtheta_inter_dC(theta_inter, v_prime, C)` | **∂θ̃/∂C** via implicit differentiation of **eq. (10)**. | **Section 3** (implicit **∂θ̃/∂C**). |
| `_dn_dC_total` | Full **∂ñ/∂C** = (**∂ñ/∂θ̃**)(**∂θ̃/∂C**) + (**∂ñ/∂C**) with θ fixed. | **Lemma 3.3** style. |
| `_dc_dC_total` | Full **∂c̃/∂C**. | **Lemma 3.4** style. |
| `dv_reflected_dC` | **∂ṽ/∂C** for **ṽ = v′ − 2⟨ñ, v′⟩ ñ**. | **Lemma 3.1** (with full **C** chain). |
| `dx_reflected_dC` | **∂x̃/∂C** for **x̃ = x′ − 2(⟨ñ, x′⟩ − c̃) ñ**. | **Lemma 3.2** (with full chain). |
| `shape_gradient(history, betas, alphas)` | Sum contributions over **`BoundaryRecord`** with **`in_domain == False`**. | Closing formula in **Section 3**. |
| `perimeter(C)` | Approximate perimeter **∫ ‖γ′‖ dθ**. | **Not** in core Boltzmann adjoint; used as **soft constraint** in optimization. |
| `perimeter_gradient(C)` | **dP/dC**. | Same — paired with penalty in experiment scripts. |

## Why implicit **∂θ̃/∂C**?

At a hit, **θ̃** is defined **implicitly** by the intersection of the ray with **r(θ; C)**. When **C** changes, the **curve** moves, so **θ̃** shifts. **`dtheta_inter_dC`** implements the standard rule: **∂θ̃/∂C = −(∂F/∂C) / (∂F/∂θ)** for the scalar equation **F(θ; C) = 0** (**eq. (10)**).

## End-to-end chain (one sentence)

**Forward tape** → **backward β, α** (sensitivity to v, x along the trajectory) → **`shape_gradient`** (at each wall hit, contract β, α with **∂ṽ/∂C**, **∂x̃/∂C**) → **gradient w.r.t. shape knobs C** for the optimizer.
